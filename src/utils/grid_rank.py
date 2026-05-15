import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def _convert_to_imshow_format(img):
    """Convert tensor or array to format suitable for imshow (numpy array with HxWxC)"""
    if hasattr(img, 'permute'):  # PyTorch tensor
        return img.permute(1, 2, 0).cpu().numpy()
    elif isinstance(img, np.ndarray):
        # If channels-first (CxHxW), transpose to HxWxC
        if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
            return np.transpose(img, (1, 2, 0))
        return img
    return img

def grid_rank(pills_images, rank_images, dists=None):
    num_pills_grid = len(pills_images)
    num_rank_grid = len(rank_images[0]) if rank_images else 0

    fig, axes = plt.subplots(num_pills_grid, num_rank_grid + 1, figsize=(15, 3 * num_pills_grid))

    for i in range(num_pills_grid):
        axes[i, 0].imshow(_convert_to_imshow_format(pills_images[i]))
        axes[i, 0].set_title("Imagen\ntransformada")
        axes[i, 0].axis('off')

        for j in range(num_rank_grid):
            if i < len(rank_images) and j < len(rank_images[i]):
                axes[i, j + 1].imshow(_convert_to_imshow_format(rank_images[i][j]))
                axes[i, j + 1].set_title(f"Rank {j + 1}\ndist={dists[i][j]:.2f}" if dists else f"Rank {j + 1}")
                axes[i, j + 1].axis('off')
            else:
                axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.show()