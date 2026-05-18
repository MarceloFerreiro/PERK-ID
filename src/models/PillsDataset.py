from pathlib import Path
from collections.abc import Callable
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

from src.features.read_image import read_image


class PillsDataset(Dataset):
    """PyTorch Dataset for pill images with optional bounding box annotations."""
    
    def __init__(
        self,
        images_dir: Path | str,
        bounding_box_csv: Path | str | None = None,
        transform: Callable[[np.ndarray, np.ndarray | None], tuple[np.ndarray, np.ndarray | None]] | None = None,
    ):
        """
        Args:
            images_dir: Directory containing pill images (.jpg, .jpeg)
            bounding_box_csv: Path to CSV with columns [name, x, y, width, height]
                             where 'name' is the image filename (without extension)
            transform: Callable that accepts (image, bbox_xywh) and returns (image, bbox_xywh)
        """
        self.images_dir = Path(images_dir)
        self.image_paths = sorted(self.images_dir.glob("*.jp*g"))
        self.transform = transform
        
        # Load bounding box data if provided
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
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
        #return 1000
    
    def __getitem__(self, idx: int):
        """
        Get an image and optionally its bounding box.
        
        Returns:
            If bounding box available: (image_tensor, bbox_tensor)
                where bbox_tensor = [x, y, width, height]
            If no bounding box: (image_tensor, None)
        """
        image_path = self.image_paths[idx]
        image = read_image(image_path)
        
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Check if this image has bounding box data
        image_name = image_path.stem  # filename without extension
        bbox_array: np.ndarray | None = None
        if image_name in self.bbox_data:
            bbox = self.bbox_data[image_name]
            bbox_array = np.array(
                [bbox['x'], bbox['y'], bbox['width'], bbox['height']],
                dtype=np.float32,
            )

        if self.transform is not None:
            image, bbox_array = self.transform(image, bbox_array)

        # Convert to tensor (C, H, W) format
        # image is already (H, W, 3) from read_image
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW

        if bbox_array is not None:
            bbox_tensor = torch.from_numpy(bbox_array).float()
            return image_tensor, bbox_tensor

        return image_tensor, None


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

 
