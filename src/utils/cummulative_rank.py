from pathlib import Path

import matplotlib.pyplot as plt
from uniplot import plot


def cummulative_rank(ranks, save_path: Path | None = None):

    valid_ranks = [r for r in ranks if r != -1]

    if not valid_ranks:
        print("No hay rankings dentro del k. O hiciste un bug o --eval-size es muy pequeño.")
        return

    max_rank = max(valid_ranks) + 1
    cumulative_acc = []
    for k in range(1, max_rank + 1):
        acc_at_k = sum(1 for r in valid_ranks if r < k) / len(ranks)
        cumulative_acc.append(acc_at_k)

    plot(cumulative_acc, title='P(Ranking <= X)', lines=True)

    # matplotlib version for saving
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, max_rank + 1), cumulative_acc, color='#4c9be8', linewidth=2)
    ax.axhline(0.9, color='#e05c5c', linewidth=1.2, linestyle='--', label='0.9')
    ax.set_xlabel('Rank k')
    ax.set_ylabel('P(rank ≤ k)')
    ax.set_title('Cumulative rank accuracy')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()

    out = Path(save_path) if save_path else Path('tmp/cumulative_rank.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Cumulative rank plot saved to {out}")
