import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse
import random
from PIL import Image, ImageDraw
import numpy as np

from src.features.read_image import read_image
from src.features.transform import _make_grid


class MaskedAutoencoder(nn.Module):
    """
    A relatively small CNN-based masked autoencoder for pill images.
    - Uses reconstruction loss (MSE) for all images
    - If bounding box is available, also uses bounding box regression loss
    """
    
    def __init__(self, in_channels: int = 3, latent_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # Encoder: Conv layers to reduce spatial dimensions and extract features
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),           # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),          # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, latent_dim, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
        )
        
        # Decoder: Transpose conv layers to reconstruct the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 32, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),          # 32 -> 64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),           # 64 -> 128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Sigmoid(),  # Output in [0, 1] range
        )
        
        # Bounding box regression head (separate from reconstruction)
        # Input: flattened latent features (latent_dim * 16 * 16)
        self.bbox_head = nn.Sequential(
            nn.Linear(latent_dim * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 4),  # Output: [x, y, width, height]
        )
    
    
    def forward(self, x: torch.Tensor, return_latent: bool = False):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        
        if return_latent:
            return reconstruction, latent
        return reconstruction
    
    def predict_bbox(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.size(0)
        latent_flat = latent.view(batch_size, -1)
        bbox_pred = self.bbox_head(latent_flat)
        return bbox_pred


def compute_loss(
    model: MaskedAutoencoder,
    batch: tuple,
    device: torch.device,
    lambda_bbox: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute loss for a batch of data.
    
    Args:
        model: The masked autoencoder model
        batch: Either (image_tensor,) or (image_tensor, bbox_tensor, has_bbox_mask)
        device: Device to run on
        lambda_bbox: Weight for bounding box loss
    
    Returns:
        loss: Total loss
        loss_dict: Dictionary with individual loss components
    """
    has_bbox = len(batch) == 3
    
    if has_bbox:
        images, bboxes, bbox_mask = batch
        images = images.to(device)
        bboxes = bboxes.to(device)
        bbox_mask = bbox_mask.to(device)
    else:
        images = batch[0].to(device)
    
    # Reconstruction loss
    recon, latent = model(images, return_latent=True)
    recon_loss = F.mse_loss(recon, images)
    
    loss_dict = {"recon_loss": recon_loss.item()}
    total_loss = recon_loss
    
    # Bounding box loss (if available)
    if has_bbox and bbox_mask.sum() > 0:
        bbox_pred = model.predict_bbox(latent)
        # Normalize bounding box coordinates to [0, 1] for better training stability
        # Assuming bbox coordinates are in pixel space (0-256)
        bboxes_norm = bboxes / 256.0
        # Only compute loss for items that have bboxes
        bbox_loss = F.smooth_l1_loss(bbox_pred[bbox_mask], bboxes_norm[bbox_mask])
        loss_dict["bbox_loss"] = bbox_loss.item()
        total_loss = recon_loss + lambda_bbox * bbox_loss
    
    return total_loss, loss_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize autoencoder reconstructions", add_help=True)
    parser.add_argument("--params", type=Path, required=True, help="Path to model weights (.pt file)")
    parser.add_argument("--dir", type=Path, default=Path("data/imagenes"), help="Directory containing images")
    parser.add_argument("--h", type=int, default=3, dest="height", help="Grid height (number of rows)")
    parser.add_argument("--w", type=int, default=4, dest="width", help="Grid width (number of columns)")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension of model")
    parser.add_argument("--output", type=Path, default=Path("data/reconstruction_grid.png"), help="Output image path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Check device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    print(f"Loading model from {args.params}")
    
    # Load model
    model = MaskedAutoencoder(in_channels=3, latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(args.params, map_location=device))
    model.eval()
    
    # Load random images
    print(f"Loading images from {args.dir}")
    paths = sorted(args.dir.glob("*.jp*g")) + sorted(args.dir.glob("*.png"))
    if not paths:
        print(f"ERROR: No images found in {args.dir}")
        exit(1)
    
    random.seed(args.seed)
    num_images = args.height * args.width
    picks = random.sample(paths, k=min(num_images, len(paths)))
    
    print(f"Processing {len(picks)} images...")
    pairs = []
    
    with torch.no_grad():
        for path in picks:
            # Read original image
            img_array = read_image(path)
            if img_array is None:
                continue
            
            # Convert to PIL for grid
            orig_pil = Image.fromarray(img_array)
            
            # Convert to tensor and reconstruct
            img_tensor = torch.from_numpy(img_array).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # CHW -> BCHW
            
            recon_tensor, latent = model(img_tensor, return_latent=True)
            recon_array = (recon_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            recon_pil = Image.fromarray(recon_array)
            
            # Predict and draw bounding box
            bbox_pred = model.predict_bbox(latent)  # normalized [0, 1]
            bbox_pixels = (bbox_pred[0].cpu().numpy() * 256).astype(int)  # denormalize to pixel coords
            x, y, w, h = bbox_pixels
            
            # Draw bounding box on reconstructed image
            draw = ImageDraw.Draw(recon_pil)
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)  # Green box
            
            pairs.append((orig_pil, recon_pil))
    
    if not pairs:
        print("ERROR: No valid image pairs created")
        exit(1)
    
    print(f"Creating grid ({args.height}x{args.width})...")
    grid = _make_grid(pairs, args.height, args.width)
    
    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(f"✓ Grid saved to {args.output}")
