import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


def _convert_to_imshow_format(img):
    if hasattr(img, 'permute'):  # PyTorch tensor
        return img.permute(1, 2, 0).cpu().numpy()
    elif isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
            return np.transpose(img, (1, 2, 0))
        return img
    return img


def grid_rank(pills_images, rank_images, dists=None, true_rank_pos=None, save_path=None):
    """
    pills_images  : list of query images (augmented)
    rank_images   : list of lists of retrieved images
    dists         : list of lists of distances (optional)
    true_rank_pos : list of ints — 0-based rank of the true match per query, -1 if not in ranking
    save_path     : Path to save the figure (None = don't save)
    """
    num_pills_grid = len(pills_images)
    num_rank_grid = len(rank_images[0]) if rank_images else 0

    fig, axes = plt.subplots(num_pills_grid, num_rank_grid + 1,
                             figsize=(15, 3 * num_pills_grid))

    # ensure axes is always 2-D
    if num_pills_grid == 1:
        axes = axes[np.newaxis, :]

    MATCH_COLOR = "#00e676"

    for i in range(num_pills_grid):
        axes[i, 0].imshow(_convert_to_imshow_format(pills_images[i]))
        axes[i, 0].set_title("Query\n(augmented)")
        axes[i, 0].axis('off')

        for j in range(num_rank_grid):
            ax = axes[i, j + 1]
            if i < len(rank_images) and j < len(rank_images[i]):
                ax.imshow(_convert_to_imshow_format(rank_images[i][j]))
                ax.axis('off')

                is_match = (true_rank_pos is not None
                            and i < len(true_rank_pos)
                            and true_rank_pos[i] == j)

                dist_str = f"\ndist={dists[i][j]:.2f}" if dists else ""
                if is_match:
                    title = f"Rank {j + 1} ★{dist_str}"
                    ax.set_title(title, color=MATCH_COLOR, fontweight='bold')
                    ax.add_patch(mpatches.Rectangle(
                        (0, 0), 1, 1,
                        transform=ax.transAxes,
                        fill=False,
                        edgecolor=MATCH_COLOR,
                        linewidth=4,
                        clip_on=False,
                    ))
                else:
                    ax.set_title(f"Rank {j + 1}{dist_str}")
            else:
                ax.axis('off')

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    plt.close(fig)