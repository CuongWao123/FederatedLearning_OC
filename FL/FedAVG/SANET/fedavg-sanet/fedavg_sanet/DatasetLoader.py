

import os, glob, math, random, argparse
import numpy as np
import cv2
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets import load_dataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Literal

def make_density_map_fixed_sigma(img_h, img_w, points_xy, sigma=4):
    """
    Generate density map với FIXED Gaussian sigma
    Paper SANet Section 4.1: σ = 4
    """
    dm = np.zeros((img_h, img_w), dtype=np.float32)
    if len(points_xy) == 0:
        return dm
    for x, y in points_xy:
        xi = min(img_w - 1, max(0, int(round(x))))
        yi = min(img_h - 1, max(0, int(round(y))))
        dm[yi, xi] += 1.0
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0: ksize += 1
    dm = cv2.GaussianBlur(dm, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_CONSTANT)
    return dm


def load_points_from_mat(mat_path):
    mat = sio.loadmat(mat_path)
    if 'image_info' in mat:
        points = mat['image_info'][0, 0][0, 0][0]
    elif 'annPoints' in mat:
        points = mat['annPoints']
    else:
        raise KeyError(f"Không tìm thấy key 'image_info' hay 'annPoints' trong {mat_path}")
    pts = np.array(points, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    return pts.tolist()

class CrowdPatchDataset(Dataset):
    def __init__(
        self,
        pairs,
        sigma=4,
        train=True,
        target_size=(384, 512),   # (H, W) — có thể đổi
        p_hflip=0.5,
        p_gray=0.5,
        p_brightness=0.5,
        brightness_range=(0.7, 1.3),
    ):
        self.pairs = pairs
        self.sigma = sigma
        self.train = train
        self.th, self.tw = target_size
        self.p_hflip = p_hflip
        self.p_gray = p_gray
        self.p_brightness = p_brightness
        self.brightness_range = brightness_range

    def __len__(self):
        return len(self.pairs)

    def _crop_pad_to_target(self, img, dm, train):
        """Đưa (img, dm) về đúng (th, tw) bằng crop rồi pad 0 nếu thiếu.
        - Train: random crop khi đủ lớn; Eval: center crop.
        - Nếu nhỏ hơn target: crop phần khả dụng rồi pad 0 để đạt (th, tw)."""
        H, W = img.shape[:2]
        th, tw = self.th, self.tw

        # 1) Chọn vị trí crop
        if H >= th and W >= tw:
            if train:
                y0 = random.randint(0, H - th)
                x0 = random.randint(0, W - tw)
            else:
                y0 = (H - th) // 2
                x0 = (W - tw) // 2
            h_crop, w_crop = th, tw
        else:
            # Ảnh nhỏ hơn target: crop tối đa và pad sau
            h_crop = min(H, th)
            w_crop = min(W, tw)
            if train:
                y0 = random.randint(0, max(0, H - h_crop))
                x0 = random.randint(0, max(0, W - w_crop))
            else:
                y0 = max(0, (H - h_crop)//2)
                x0 = max(0, (W - w_crop)//2)

        # 2) Crop
        img_c = img[y0:y0+h_crop, x0:x0+w_crop]
        dm_c  = dm [y0:y0+h_crop, x0:x0+w_crop]

        # 3) Pad về (th, tw) nếu cần (pad=0 không đổi tổng density)
        pad_h = th - img_c.shape[0]
        pad_w = tw - img_c.shape[1]
        if pad_h > 0 or pad_w > 0:
            img_c = np.pad(img_c, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            dm_c  = np.pad(dm_c,  ((0, pad_h), (0, pad_w)),        mode="constant")

        return img_c, dm_c

    def __getitem__(self, idx):
        img_path, gt_path = self.pairs[idx]

        # Load ảnh & density gốc
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape

        pts = load_points_from_mat(gt_path)
        dm  = make_density_map_fixed_sigma(H, W, pts, sigma=self.sigma)  # (H, W)

        # Chuẩn hoá size: crop + pad về (th, tw)
        img, dm = self._crop_pad_to_target(img, dm, train=self.train)

        # Augment (không đổi tổng density)
        if self.train:
            # Flip ngang
            if random.random() < self.p_hflip:
                img = img[:, ::-1].copy()
                dm  = dm [:, ::-1].copy()

            # Gray scale
            if random.random() < self.p_gray:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            # Brightness
            if random.random() < self.p_brightness:
                lo, hi = self.brightness_range
                bf = lo + random.random() * (hi - lo)
                img = np.clip(img * bf, 0, 255).astype(np.uint8)

        # To tensor
        img = (img / 255.0).astype(np.float32)
        img = torch.from_numpy(img.transpose(2, 0, 1))   # [3, th, tw]
        dm  = torch.from_numpy(dm).unsqueeze(0)          # [1, th, tw]
        return img, dm


# ====== Globals (giữ state để không load lại vô ích) ======
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
        # gán ra global
        _partitioner = part

def load_data(
    partition_id: int,
    num_partitions: int,
    csv_path: str,
    batch_size: int = 32,
    sigma: int = 4,
    partitioner_type: Literal["iid", "dirichlet"] = "iid",
    dirichlet_alpha: float = 0.5,
    num_workers: int = 0,
    pin_memory: bool = False,
    target_size=(384, 512),
):
    """Trả về (trainloader, testloader) cho 1 client từ CSV local."""
    global _partitioner, _raw
    _ensure_raw_loaded(csv_path)
    _ensure_partitioner(num_partitions, partitioner_type, dirichlet_alpha)

    partition = _partitioner.load_partition(partition_id=partition_id)
    if len(partition) == 0:
        raise ValueError(f"Partition rỗng: partition_id={partition_id}")

    # Chia 80/20 trên partition
    partition_tt = partition.train_test_split(test_size=0.2, seed=42)

    # Đổi sang list[(image_path, gt_path)]
    def to_pairs(ds_split):
        return list(zip(ds_split["image_path"], ds_split["gt_path"]))

    train_pairs = to_pairs(partition_tt["train"])
    test_pairs  = to_pairs(partition_tt["test"])
    if not train_pairs or not test_pairs:
        raise ValueError("Split 80/20 quá nhỏ: thiếu mẫu train hoặc test.")

    train_ds = CrowdPatchDataset(train_pairs, sigma=sigma, train=True,  target_size=target_size)
    test_ds  = CrowdPatchDataset(test_pairs,  sigma=sigma, train=False, target_size=target_size)

    trainloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    testloader  = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return trainloader, testloader

