from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage import io

from src.features.read_image import read_image


class PillsDataset(Dataset):
    """PyTorch Dataset for pill images with optional bounding box annotations."""

    def __init__(self, images_dir: Path | str, bounding_box_csv: Path | str | None = None):
        self.images_dir = Path(images_dir)
        self.image_paths = sorted(self.images_dir.glob("*.jp*g"))

        self.bbox_data = {}
        if bounding_box_csv is not None:
            df = pd.read_csv(bounding_box_csv)
            for _, row in df.iterrows():
                filename = row['name']
                self.bbox_data[filename] = {
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'width': float(row['width']),
                    'height': float(row['height'])
                }

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image_name = image_path.stem

        if image_name in self.bbox_data:
            try:
                raw = io.imread(str(image_path))
                orig_h, orig_w = raw.shape[:2]
            except Exception:
                orig_h, orig_w = 256, 256

            bbox = self.bbox_data[image_name]
            bbox_tensor = torch.tensor([
                bbox['x'] / orig_w,
                bbox['y'] / orig_h,
                bbox['width'] / orig_w,
                bbox['height'] / orig_h,
            ], dtype=torch.float32).clamp(0.0, 1.0)
        else:
            bbox_tensor = None

        image = read_image(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW

        return image_tensor, bbox_tensor


def custom_collate_fn(batch):
    """
    Custom collate function that handles mixed batch items.
    Some items may have bounding boxes, others may not.
    
    Returns:
        Either (images,) or (images, bboxes, has_bbox_mask) as a tuple
    """
    images = []
    bboxes = []
    has_bbox_mask = []
    
    for item in batch:
        if isinstance(item, tuple):
            img, bbox = item
            images.append(img)
            bboxes.append(bbox)
            has_bbox_mask.append(bbox is not None)
        else:
            images.append(item)
            bboxes.append(None)
            has_bbox_mask.append(False)
    
    images_stacked = torch.stack(images)
    
    if any(has_bbox_mask):
        # Stack bboxes, using zeros for None values
        bboxes_stacked = torch.stack([
            bbox if bbox is not None else torch.zeros(4, dtype=torch.float32)
            for bbox in bboxes
        ])
        has_bbox_mask = torch.tensor(has_bbox_mask, dtype=torch.bool)
        return (images_stacked, bboxes_stacked, has_bbox_mask)
    else:
        return (images_stacked,)

 
