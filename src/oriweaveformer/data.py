import random
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config


def _labelled_dir(root: Path, dataset: str, split: str) -> Path:
    if dataset == "deepglobe":
        for directory in [root / "train", root, *root.rglob("train")]:
            if directory.is_dir() and next(directory.glob("*_sat.jpg"), None):
                return directory
    else:
        name = "eval" if split == "test" else "training"
        for candidate in [root, *root.rglob(name)]:
            directory = candidate if candidate.name == name else candidate / name
            if (directory / "images").is_dir() and (directory / "masks").is_dir():
                return directory
    raise FileNotFoundError(f"Cannot find {dataset}/{split} below {root}")


def collect_pairs(cfg: Config, split: str):
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split}")
    if cfg.split_manifest:
        manifest = json.loads(Path(cfg.split_manifest).read_text(encoding="utf-8"))
        if manifest.get("dataset") != cfg.dataset:
            raise ValueError("Split manifest belongs to a different dataset")
        pairs = [(Path(cfg.data_root) / item["image"], Path(cfg.data_root) / item["mask"])
                 for item in manifest["splits"][split]]
        if any(not image.is_file() or not mask.is_file() for image, mask in pairs):
            raise FileNotFoundError("Missing file in frozen split manifest")
        return pairs
    directory = _labelled_dir(Path(cfg.data_root), cfg.dataset, split)
    if cfg.dataset == "deepglobe":
        pairs = [(image, image.with_name(image.name.replace("_sat.jpg", "_mask.png")))
                 for image in sorted(directory.glob("*_sat.jpg"))]
        pairs = [pair for pair in pairs if pair[1].exists()]
        random.Random(cfg.split_seed).shuffle(pairs)
        return (pairs[:5000] if split == "train" else
                pairs[5000:5600] if split == "val" else pairs[5600:])
    pairs = []
    for image in sorted((directory / "images").glob("*")):
        if image.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            continue
        stem = image.stem.removesuffix("_sat")
        masks = [directory / "masks" / f"{stem}{suffix}{extension}"
                 for suffix in ("_mask", "")
                 for extension in (".png", ".jpg", ".tif")]
        mask = next((path for path in masks if path.exists()), None)
        if mask:
            pairs.append((image, mask))
    if split in ("train", "val"):
        random.Random(cfg.split_seed).shuffle(pairs)
        count = max(1, round(len(pairs) * cfg.val_fraction))
        return pairs[:count] if split == "val" else pairs[count:]
    return pairs


class RoadDataset(Dataset):
    def __init__(self, cfg: Config, split: str) -> None:
        import albumentations as A
        self.cfg, self.split = cfg, split
        self.samples = collect_pairs(cfg, split)
        if not self.samples:
            raise RuntimeError(f"No {cfg.dataset}/{split} pairs found")
        common = A.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5), A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 15, 10, p=0.3),
            A.GaussNoise(std_range=(0.05, 0.15), p=0.2), common,
        ] if split == "train" else [common])

    def __len__(self):
        return len(self.samples)

    def _crop(self, image, mask):
        size, (height, width) = self.cfg.crop_size, mask.shape
        if height < size or width < size:
            ph, pw = max(0, size - height), max(0, size - width)
            image = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT)
            height, width = mask.shape
        if self.split != "train":
            y, x = (height - size) // 2, (width - size) // 2
            return image[y:y + size, x:x + size], mask[y:y + size, x:x + size]
        require_road = random.random() < self.cfg.road_crop_probability
        chosen = None
        for _ in range(12 if require_road else 1):
            y, x = random.randint(0, height - size), random.randint(0, width - size)
            chosen = image[y:y + size, x:x + size], mask[y:y + size, x:x + size]
            if not require_road or chosen[1].mean() >= self.cfg.min_road_fraction:
                break
        return chosen

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        raw_image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if raw_image is None or mask is None:
            raise RuntimeError(f"Cannot read image/mask: {image_path.name}, {mask_path.name}")
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask shape mismatch: {image_path.name}")
        image, mask = self._crop(image, (mask > 127).astype(np.float32))
        item = self.transform(image=image, mask=mask)
        return (torch.from_numpy(item["image"].transpose(2, 0, 1)).float(),
                torch.from_numpy(item["mask"][None]).float())
