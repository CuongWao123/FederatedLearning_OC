import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import cv2
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import os, glob, math, random, argparse
import numpy as np

from datasets import load_dataset
from typing import Optional, Literal
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

class Conv2dIN(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=None, d=1, activate=True, use_norm=True, bias=False):
        super().__init__()
        if p is None:
            p = (k // 2) if isinstance(k, int) else 0
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, dilation=d, bias=bias)
        self.inorm = nn.InstanceNorm2d(out_c, affine=True) if use_norm else None
        self.act = nn.ReLU(inplace=True) if activate else None

    def forward(self, x):
        x = self.conv(x)
        if self.inorm is not None:
            x = self.inorm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DeConv2dIN(nn.Module):
    def __init__(self, in_c, out_c, k=2, s=2, p=0, activate=True, use_norm=True, bias=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p, output_padding=0, bias=bias)
        self.inorm = nn.InstanceNorm2d(out_c, affine=True) if use_norm else None
        self.act = nn.ReLU(inplace=True) if activate else None

    def forward(self, x):
        x = self.deconv(x)
        if self.inorm is not None:
            x = self.inorm(x)
        if self.act is not None:
            x = self.act(x)
        return x


# ------------------------------
# Scale Aggregation (SA) Module: 4 nhánh 1/3/5/7; từ module thứ 2 trở đi có 1x1 reduce
# ------------------------------
class SAModule(nn.Module):
    def __init__(self, in_c: int, out_c: int, use_reduction: bool):
        super().__init__()
        mid = out_c // 4  
        def make_branch(k: int, reduce: bool):
            if k == 1 or not reduce:
                return nn.Sequential(
                    Conv2dIN(in_c, mid, k, bias=False),
                )
            else:
                return nn.Sequential(
                    Conv2dIN(in_c, mid * 2, 1, bias=False),
                    Conv2dIN(mid * 2, mid, k, bias=False),
                )

        self.b1 = make_branch(1, use_reduction)
        self.b3 = make_branch(3, use_reduction)
        self.b5 = make_branch(5, use_reduction)
        self.b7 = make_branch(7, use_reduction)

        self.fuse = nn.Sequential(
            Conv2dIN(out_c, out_c, 1, bias=False),
        )

    def forward(self, x):
        y = torch.cat([self.b1(x), self.b3(x), self.b5(x), self.b7(x)], dim=1)
        y = self.fuse(y)
        return y


# ------------------------------
# SANet (ECCV 2018)
# ------------------------------
class SANet(nn.Module):
    def __init__(self, sa_channels=(16, 32, 32, 16)):
        super().__init__()
        c1, c2, c3, c4 = sa_channels

        # ----- FME (Encoder): 4 SA modules, pooling sau 3 module đầu -----
        self.sa1 = SAModule(in_c=3,  out_c=c1, use_reduction=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.sa2 = SAModule(in_c=c1, out_c=c2, use_reduction=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.sa3 = SAModule(in_c=c2, out_c=c3, use_reduction=True)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.sa4 = SAModule(in_c=c3, out_c=c4, use_reduction=True)  # không pool sau cùng

        # ----- DME (Decoder): Conv 9→7→5→3 xen kẽ với 3× Deconv×2 -----
        self.conv9 = Conv2dIN(c4, 64, 9, bias=False)
        self.up1   = DeConv2dIN(64, 64, k=2, s=2, p=0)   # ×2

        self.conv7 = Conv2dIN(64, 32, 7, bias=False)
        self.up2   = DeConv2dIN(32, 32, k=2, s=2, p=0)   # ×4

        self.conv5 = Conv2dIN(32, 16, 5, bias=False)
        self.up3   = DeConv2dIN(16, 16, k=2, s=2, p=0)   # ×8 (về full-res)

        self.conv3    = Conv2dIN(16, 16, 3, bias=False)
        self.out1x1   = nn.Conv2d(16, 1, kernel_size=1, bias=True)
        self.relu_final = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None: nn.init.constant_(m.weight, 1.0)
                if m.bias   is not None: nn.init.constant_(m.bias,   0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_size = x.shape[2:]  # (H, W)

        # ----- Encoder -----
        x = self.sa1(x); x = self.pool1(x)   # /2
        x = self.sa2(x); x = self.pool2(x)   # /4
        x = self.sa3(x); x = self.pool3(x)   # /8
        x = self.sa4(x)

        # ----- Decoder (interleaved) -----
        x = self.conv9(x); x = self.up1(x)   # refine + ×2
        x = self.conv7(x); x = self.up2(x)   # refine + ×2
        x = self.conv5(x); x = self.up3(x)   # refine + ×2
        x = self.conv3(x)
        x = self.out1x1(x)
        x = self.relu_final(x)               # đảm bảo density ≥ 0

        if x.shape[2:] != inp_size:
            x = F.interpolate(x, size=inp_size, mode='bilinear', align_corners=False)
        return x

# ------------------------------
# SSIM (single-scale) cho loss
# ------------------------------
def _gaussian(window_size: int, sigma: float):
    gauss = torch.tensor([math.exp(-(x - window_size//2)**2 / float(2 * sigma**2))
                          for x in range(window_size)], dtype=torch.float32)
    return gauss / gauss.sum()


def _create_window(window_size: int, sigma: float, channel: int, device, dtype):
    _1D = _gaussian(window_size, sigma).to(device=device, dtype=dtype).unsqueeze(1)
    _2D = _1D @ _1D.t()
    window = _2D.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_single_scale(img1: torch.Tensor,
                      img2: torch.Tensor,
                      window_size: int = 11,
                      sigma: float = 1.5,
                      C1: float = 0.01**2,
                      C2: float = 0.03**2) -> torch.Tensor:
    """SSIM single-scale cho ảnh xám/density (B,1,H,W). Không backprop qua window."""
    assert img1.shape == img2.shape
    b, c, h, w = img1.shape
    device, dtype = img1.device, img1.dtype
    window = _create_window(window_size, sigma, c, device, dtype)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=c)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=c)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=c) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=(1, 2, 3))  # (B,)


class SANetLoss(nn.Module):
    """
    L = MSE + alpha*(1 - SSIM) + beta*Count_Loss
    alpha=1e-3 (paper), beta=1e-3 (added for stability)
    """
    def __init__(self, alpha: float = 1e-3, beta: float = 1e-3, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B,1,H,W)
        
        # MSE loss (Euclidean loss)
        l2 = F.mse_loss(pred, target, reduction='mean')
        
        # SSIM loss
        ssim_val = ssim_single_scale(pred, target, window_size=self.window_size, sigma=self.sigma)  # (B,)
        l_ssim = 1.0 - ssim_val.mean()
        
        # Count loss (normalized by GT count to prevent explosion)
        pred_count = pred.view(pred.shape[0], -1).sum(dim=1)    # (B,)
        target_count = target.view(target.shape[0], -1).sum(dim=1)  # (B,)
        count_error = torch.abs(pred_count - target_count) / (target_count + 1.0)
        l_count = count_error.mean()
        
        # Total loss
        loss = l2 + self.alpha * l_ssim + self.beta * l_count
        
        return loss


# ========================= Load data, train, test =========================


# from datasets import load_dataset
# from flwr_datasets.partitioner import ChosenPartitioner

# # Single file
# data_files = "path-to-my-file.csv"

# # Multiple Files
# dataset = load_dataset("csv", data_files=data_files)

# partitioner = ChosenPartitioner(...)
# partitioner.dataset = dataset
# partition = partitioner.load_partition(partition_id=0)

def load_points_from_mat(mat_path):
    """
    Load annotation points từ .mat file (ShanghaiTech dataset format).
    
    Supported formats:
    1. mat['image_info'][0,0][0,0][0] - ShanghaiTech Part A/B format
    2. mat['annPoints'] - Alternative format
    
    Args:
        mat_path: Path to .mat file
        
    Returns:
        List of [x, y] coordinates: [[x1, y1], [x2, y2], ...]
        Empty list if no annotations
        
    Raises:
        KeyError: If neither 'image_info' nor 'annPoints' exists
        FileNotFoundError: If mat_path doesn't exist
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Ground truth file not found: {mat_path}")
    
    mat = sio.loadmat(mat_path)
    
    # Try format 1: ShanghaiTech standard format
    if 'image_info' in mat:
        points = mat['image_info'][0, 0][0, 0][0]
    # Try format 2: Alternative format
    elif 'annPoints' in mat:
        points = mat['annPoints']
    else:
        available_keys = list(mat.keys())
        raise KeyError(
            f"Không tìm thấy key 'image_info' hoặc 'annPoints' trong {mat_path}. "
            f"Keys có sẵn: {available_keys}"
        )
    
    # Convert to numpy array
    pts = np.array(points, dtype=np.float32)
    
    # Handle empty annotations
    if pts.size == 0:
        return []
    
    # Ensure shape is (N, 2)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    
    # Validate shape
    if pts.shape[1] != 2:
        raise ValueError(
            f"Expected points with shape (N, 2), got {pts.shape} from {mat_path}"
        )
    
    return pts.tolist()

def make_density_map_fixed_sigma(img_h, img_w, points_xy, sigma=4):
    """
    Generate density map với FIXED Gaussian kernel.
    
    Paper SANet (ECCV 2018) Section 4.1:
    - "We follow the same settings in [32] to generate ground truth density maps"
    - "We empirically set σ = 4 for all the datasets"
    - Fixed sigma (không phụ thuộc vào khoảng cách k-nearest neighbors)
    
    Args:
        img_h: Image height
        img_w: Image width
        points_xy: List of (x, y) coordinates [[x1, y1], [x2, y2], ...]
        sigma: Gaussian kernel standard deviation (default: 4)
        
    Returns:
        density_map: (H, W) numpy array with sum ≈ number of people
        
    Note:
        - Kernel size = 6*sigma + 1 để đảm bảo coverage đầy đủ
        - Sử dụng cv2.GaussianBlur với BORDER_CONSTANT để tránh edge artifacts
    """
    dm = np.zeros((img_h, img_w), dtype=np.float32)
    
    if len(points_xy) == 0:
        return dm
    
    # Place delta functions at each annotation point
    for x, y in points_xy:
        # Round to nearest integer and clamp to image bounds
        xi = min(img_w - 1, max(0, int(round(x))))
        yi = min(img_h - 1, max(0, int(round(y))))
        dm[yi, xi] += 1.0
    
    # Apply fixed Gaussian kernel
    # Kernel size: 6*sigma + 1 covers ~99.7% of Gaussian (3-sigma rule)
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0: 
        ksize += 1  # Ensure odd kernel size
    
    dm = cv2.GaussianBlur(
        dm, 
        (ksize, ksize), 
        sigmaX=sigma, 
        sigmaY=sigma, 
        borderType=cv2.BORDER_CONSTANT  # Zero padding at borders
    )
    
    return dm

class CrowdPatchDataset(Dataset):
    """
    Crowd Counting Dataset với patch extraction theo SANet paper (ECCV 2018).
    
    Paper specifications (Section 4.1):
    - Patch size: 1/4 of original image (H/2 × W/2)
    - Fixed Gaussian sigma = 4
    - Data augmentation: random cropping, mirroring, random brightness
    - Batch size = 1 nếu ảnh có size khác nhau
    
    Args:
        pairs: List of (image_path, gt_path) tuples
        sigma: Fixed Gaussian sigma for density map generation (default: 4)
        train: Training mode (True) or validation/test mode (False)
    """
    def __init__(self, pairs, sigma=4, train=True):
        self.pairs = pairs
        self.sigma = sigma
        self.train = train
        
    def __len__(self): 
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, gt_path = self.pairs[idx]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        
        # Load ground truth points and generate density map
        pts = load_points_from_mat(gt_path)
        dm = make_density_map_fixed_sigma(H, W, pts, sigma=self.sigma)

        # Paper Section 4.1: "patches of 1/4 size of the original images"
        # Patch size = (H/2, W/2) để tổng diện tích = 1/4 ảnh gốc
        ph, pw = max(2, H // 2), max(2, W // 2)
        
        if self.train:
            # 1. Random crop (paper: "random cropping")
            y0 = random.randint(0, max(0, H - ph))
            x0 = random.randint(0, max(0, W - pw))
            img_crop = img[y0:y0+ph, x0:x0+pw].copy()
            dm_crop = dm[y0:y0+ph, x0:x0+pw].copy()
            
            # 2. Horizontal flip / mirroring (paper: "mirroring")
            if random.random() < 0.5:
                img_crop = cv2.flip(img_crop, 1)  # flip along vertical axis
                dm_crop = np.fliplr(dm_crop).copy()
            
            # 3. Gray scale transformation (paper Section 4.1)
            # "We augment the training data using random cropping, mirroring, and random brightness"
            if random.random() < 0.5:
                gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
                img_crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            # 4. Random brightness adjustment (paper: "random brightness")
            # Brightness factor range: [0.7, 1.3] for ±30% variation
            if random.random() < 0.5:
                brightness_factor = 0.7 + random.random() * 0.6
                img_crop = np.clip(img_crop * brightness_factor, 0, 255).astype(np.uint8)
            
            img_out, dm_out = img_crop, dm_crop
            
        else:
            # Validation/Test: Center crop (deterministic)
            y0 = max(0, (H - ph) // 2)
            x0 = max(0, (W - pw) // 2)
            img_out = img[y0:y0+ph, x0:x0+pw]
            dm_out = dm[y0:y0+ph, x0:x0+pw]

        # Normalize image to [0, 1]
        img_tensor = torch.from_numpy(img_out.transpose(2, 0, 1).astype(np.float32) / 255.0)
        dm_tensor = torch.from_numpy(dm_out.astype(np.float32)).unsqueeze(0)
        
        return img_tensor, dm_tensor

_raw: Optional[Dataset] = None
_raw_source: Optional[str] = None
_partitioner = None
_num_partitions: Optional[int] = None


def _ensure_raw_loaded(csv_path: str):
    """Load CSV -> HF Dataset đúng 1 lần. Reset partitioner nếu nguồn thay đổi."""
    global _raw, _raw_source, _partitioner
    if (_raw is None) or (_raw_source != csv_path):
        ds = load_dataset("csv", data_files={"train": csv_path})["train"]
        cols = set(ds.column_names)
        must = {"image_path", "gt_path"}
        missing = must - cols
        if missing:
            raise KeyError(f"CSV thiếu cột {missing}. Cột hiện có: {sorted(cols)}")
        if len(ds) == 0:
            raise ValueError("CSV rỗng.")
        _raw = ds
        _raw_source = csv_path
        _partitioner = None 

def _ensure_partitioner(num_partitions: int,
                        partitioner_type: Literal["iid", "dirichlet"] = "iid",
                        dirichlet_alpha: float = 0.5):
    """Tạo/gán partitioner đúng 1 lần. Reset nếu num_partitions hoặc loại partitioner đổi."""
    global _partitioner, _raw, _num_partitions

    # Lưu loại partitioner hiện tại trong đối tượng (đính kèm attr tạm)
    current_kind = getattr(_partitioner, "_kind", None)
    current_alpha = getattr(_partitioner, "_alpha", None)

    need_new = (
        _partitioner is None
        or _num_partitions != num_partitions
        or current_kind != partitioner_type
        or (partitioner_type == "dirichlet" and current_alpha != dirichlet_alpha)
    )

    if need_new:
        if _raw is None:
            raise RuntimeError("Raw dataset chưa được load. Gọi _ensure_raw_loaded trước.")
        if partitioner_type == "iid":
            part = IidPartitioner(num_partitions=num_partitions)
            part._kind = "iid"         # gắn nhãn để so sánh lần sau
            part._alpha = None
        else:
            part = DirichletPartitioner(num_partitions=num_partitions, alpha=dirichlet_alpha, balanced=True)
            part._kind = "dirichlet"
            part._alpha = dirichlet_alpha

        part.dataset = _raw
        _num_partitions = num_partitions
        _partitioner = part

def load_data(
    partition_id: int,
    num_partitions: int,
    csv_path: str,
    batch_size: int = 1,  # Default = 1 vì patches có size khác nhau
    sigma: int = 4,
    partitioner_type: Literal["iid", "dirichlet"] = "iid",
    dirichlet_alpha: float = 0.5,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """
    Load data cho 1 client trong Federated Learning setup.
    
    Args:
        partition_id: ID của partition (client) hiện tại (0-indexed)
        num_partitions: Tổng số partitions (clients)
        csv_path: Path đến file CSV chứa [image_path, gt_path]
        batch_size: Batch size cho DataLoader
                   QUAN TRỌNG: Paper SANet sử dụng patches 1/4 ảnh gốc
                   - Nếu ảnh có size KHÁC NHAU → batch_size = 1 (bắt buộc)
                   - Nếu ảnh có size GIỐNG NHAU → có thể dùng batch_size > 1
        sigma: Fixed Gaussian sigma cho density map (paper: σ=4)
        partitioner_type: "iid" hoặc "dirichlet"
        dirichlet_alpha: Alpha parameter cho Dirichlet partitioner (0.5 = heterogeneous)
        num_workers: Số workers cho DataLoader
        pin_memory: Pin memory cho faster GPU transfer
        
    Returns:
        (trainloader, testloader): DataLoader cho train và test sets
        
    Note:
        - Paper training: batch_size=32, lr=1e-5, epochs=400
        - Mỗi partition được chia 80% train / 20% test
        - Global dataset được cache để tránh reload nhiều lần
    """
    global _partitioner, _raw
    
    # Ensure global dataset is loaded
    _ensure_raw_loaded(csv_path)
    _ensure_partitioner(num_partitions, partitioner_type, dirichlet_alpha)

    # Load partition for this client
    partition = _partitioner.load_partition(partition_id=partition_id)
    if len(partition) == 0:
        raise ValueError(f"Partition {partition_id} rỗng. Kiểm tra num_partitions và CSV.")

    # Split 80/20 train/test trên mỗi partition
    partition_tt = partition.train_test_split(test_size=0.2, seed=42)

    # Convert HuggingFace Dataset → List of (image_path, gt_path) pairs
    def to_pairs(ds_split):
        return list(zip(ds_split["image_path"], ds_split["gt_path"]))

    train_pairs = to_pairs(partition_tt["train"])
    test_pairs = to_pairs(partition_tt["test"])
    
    if not train_pairs or not test_pairs:
        raise ValueError(f"Partition {partition_id}: Split 80/20 tạo ra train hoặc test rỗng.")

    # Create PyTorch Datasets
    train_ds = CrowdPatchDataset(train_pairs, sigma=sigma, train=True)
    test_ds = CrowdPatchDataset(test_pairs, sigma=sigma, train=False)

    # Create DataLoaders
    trainloader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        drop_last=False  # Giữ batch cuối cùng dù không đủ size
    )
    testloader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return trainloader, testloader


def train(
    net,
    trainloader,
    epochs: int,
    lr: float,
    device: torch.device,
    amp: bool = True,
    grad_clip: float = 0.0,
    alpha: float = 1e-3,
    beta: float = 1e-3,
    ssim_window: int = 11,
    ssim_sigma: float = 1.5,
):
    """Train cho crowd counting."""
    net.to(device)
    net.train()

    criterion = SANetLoss(alpha=alpha, beta=beta, window_size=ssim_window, sigma=ssim_sigma).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and (device.type == "cuda"))

    running_loss = 0.0
    num_batches_total = 0

    for _ in range(epochs):
        for imgs, dms in trainloader:
            imgs = imgs.to(device, non_blocking=True)     
            dms  = dms.to(device, non_blocking=True)        

            optimizer.zero_grad(set_to_none=True)

            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    preds = net(imgs)
                    loss  = criterion(preds, dms)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = net(imgs)
                loss  = criterion(preds, dms)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                optimizer.step()

            running_loss += loss.item()
            num_batches_total += 1

    avg_trainloss = running_loss / max(1, num_batches_total)
    return avg_trainloss


@torch.no_grad()
def test_fn(
    net,
    valloader,
    device: torch.device,
    alpha: float = 1e-3,
    beta: float = 1e-3,
    ssim_window: int = 11,
    ssim_sigma: float = 1.5,
):
    """Evaluate cho crowd counting."""
    net.to(device)
    net.eval()

    criterion = SANetLoss(alpha=alpha, beta=beta, window_size=ssim_window, sigma=ssim_sigma).to(device)

    total_loss = 0.0
    total_mae  = 0.0
    total_mse  = 0.0
    n_samples  = 0
    n_batches  = 0

    for imgs, dms in valloader:
        imgs = imgs.to(device, non_blocking=True)
        dms  = dms.to(device, non_blocking=True)

        preds = net(imgs)                         # [B,1,H,W]
        loss  = criterion(preds, dms)

        pred_counts = preds.sum(dim=[1,2,3])      # [B]
        gt_counts   = dms.sum(dim=[1,2,3])        # [B]
        diff        = pred_counts - gt_counts

        total_loss += loss.item()
        total_mae  += diff.abs().sum().item()
        total_mse  += (diff ** 2).sum().item()
        n_samples  += imgs.size(0)
        n_batches  += 1

    avg_loss = total_loss / max(1, n_batches)
    mae      = total_mae  / max(1, n_samples)
    rmse     = math.sqrt(total_mse / max(1, n_samples))
    return avg_loss, mae, rmse
