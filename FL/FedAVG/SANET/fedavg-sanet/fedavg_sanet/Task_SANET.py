import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ================ Define SANet model and SANetLoss ================
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
