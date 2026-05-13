import argparse
import csv
from pathlib import Path
from time import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.PillsDataset import PillsDataset, custom_collate_fn
from src.models.net import MaskedAutoencoder, compute_loss


def train_epoch(
    model: MaskedAutoencoder,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_bbox: float = 1.0,
) -> dict:

    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_bbox_loss = 0.0
    num_with_bbox = 0
    
    with tqdm(train_loader, desc="Entrenando una época", unit='MiniBatch', leave=False) as pbar:
        for batch in pbar:
            optimizer.zero_grad()
            loss, loss_dict = compute_loss(model, batch, device, lambda_bbox=lambda_bbox)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += loss_dict["recon_loss"]
            if "bbox_loss" in loss_dict:
                total_bbox_loss += loss_dict["bbox_loss"]
                num_with_bbox += 1
            
            pbar.update(1)
            pbar.set_postfix({"perdida": loss.item()})
    
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_bbox_loss = total_bbox_loss / num_with_bbox if num_with_bbox > 0 else 0.0
    
    return {
        "loss": avg_loss,
        "recon_loss": avg_recon_loss,
        "bbox_loss": avg_bbox_loss,
    }


def main():
    parser = argparse.ArgumentParser(description="Train masked autoencoder on pill images")
    parser.add_argument( "--images-dir", type=Path, default=Path("data/imagenes"), help="Directory containing pill images",)
    parser.add_argument( "--bbox-csv", type=Path, default=Path("tmp.csv"), help="Path to CSV with bounding box annotations",)
    parser.add_argument( "--batch-size", type=int, default=32, help="Batch size for training",)
    parser.add_argument( "--epochs", type=int, default=50, help="Number of epochs to train",)
    parser.add_argument( "--lr", type=float, default=1e-3, help="Learning rate",)
    parser.add_argument( "--lambda-bbox", type=float, default=1.0, help="Weight for bounding box loss",)
    parser.add_argument( "--output-dir", type=Path, default=Path("data/params"), help="Directory to save model checkpoints",)
    parser.add_argument( "--checkpoint-interval", type=int, default=5, help="Save checkpoint every N epochs",)
    parser.add_argument( "--latent-dim", type=int, default=128, help="Dimensionality of latent space",)
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-select best available device: cuda > mps > cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    print(f"Loading dataset from {args.images_dir}")
    dataset = PillsDataset(args.images_dir, args.bbox_csv)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
        collate_fn=custom_collate_fn,
    )
    print(f"Número de imágenes a proceser: {len(dataset)}")
    
    model = MaskedAutoencoder(in_channels=3, latent_dim=args.latent_dim).to(device)
    print(f"#parametros: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create CSV for logging losses
    csv_path = args.output_dir / "training_losses.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["epoch", "loss", "recon_loss", "bbox_loss"])
    csv_writer.writeheader()
    
    with tqdm(range(args.epochs), desc='Entrenando', unit="Epocas") as epoch_pbar:
        for epoch in epoch_pbar:
            epoch_losses = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                lambda_bbox=args.lambda_bbox,
            )
            scheduler.step()
            
            # Write losses to CSV
            csv_writer.writerow({
                "epoch": epoch + 1,
                "loss": f"{epoch_losses['loss']:.6f}",
                "recon_loss": f"{epoch_losses['recon_loss']:.6f}",
                "bbox_loss": f"{epoch_losses['bbox_loss']:.6f}",
            })
            csv_file.flush()
            
            # Update progress bar
            loss_str = f"loss={epoch_losses['loss']:.4f}"
            if epoch_losses["bbox_loss"] > 0:
                loss_str += f", bbox={epoch_losses['bbox_loss']:.4f}"
            epoch_pbar.set_postfix_str(loss_str)
            
            # Save checkpoint
            if (epoch + 1) % args.checkpoint_interval == 0:
                checkpoint_path = args.output_dir / f"model_epoch_{epoch+1:03d}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"[*] Guardados pesos tempoarmente en: {checkpoint_path}")
    
    csv_file.close()
    print(f"[*] Pérdidas guardadas en: {csv_path}")
    
    
    # Save final model
    final_path = args.output_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"[*] Guardados pesos en: {final_path}")


if __name__ == "__main__":
    main()
