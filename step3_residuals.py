"""
IS Experiment Step 3: Residual computation.

Residual = each frame - background tensor.
This should isolate the moving objects.
The residuals are in the raw signal space (pixel values).

Key observation: residuals should show the object at its position in each frame,
with the background completely removed.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def compute_residuals(frames: np.ndarray, background: np.ndarray) -> np.ndarray:
    """
    Compute residuals: each frame minus background tensor.

    Args:
        frames: (n_frames, H, W, 3) uint8
        background: (H, W, 3) float64
    Returns:
        residuals: (n_frames, H, W, 3) float64 (can be negative)
    """
    return frames.astype(np.float64) - background[None]


def visualize_residuals(
    residuals: np.ndarray,
    video_name: str,
    save_path: Path,
):
    """Visualize residual frames as a strip. Show absolute magnitude."""
    n_show = min(10, len(residuals))
    indices = np.linspace(0, len(residuals) - 1, n_show, dtype=int)

    # Compute magnitude across color channels
    magnitudes = np.sqrt(np.sum(residuals ** 2, axis=-1))  # (n_frames, H, W)

    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4))

    vmax = magnitudes.max()

    for i, idx in enumerate(indices):
        # Top row: signed residual (shifted to [0,1] for display)
        res_display = residuals[idx] / 255.0 * 0.5 + 0.5  # center at 0.5
        axes[0, i].imshow(np.clip(res_display, 0, 1))
        axes[0, i].set_title(f't={idx}')
        axes[0, i].axis('off')

        # Bottom row: magnitude (heatmap)
        axes[1, i].imshow(magnitudes[idx], cmap='hot', vmin=0, vmax=vmax)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Signed\nresidual', rotation=0, labelpad=60, va='center')
    axes[1, 0].set_ylabel('Magnitude', rotation=0, labelpad=60, va='center')

    fig.suptitle(f'{video_name}: Residuals (frame - background)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def residual_stats(residuals: np.ndarray) -> dict:
    """Compute summary statistics of residuals."""
    magnitudes = np.sqrt(np.sum(residuals ** 2, axis=-1))
    return {
        'mean_magnitude': magnitudes.mean(),
        'max_magnitude': magnitudes.max(),
        'nonzero_fraction': (magnitudes > 1.0).mean(),
        'energy': np.sum(residuals ** 2),
    }


def main():
    data_dir = Path(__file__).parent / 'data'
    output_dir = Path(__file__).parent / 'output'

    video_names = [p.stem for p in sorted(data_dir.glob('v*.npy'))
                   if '_bg' not in p.stem and '_residuals' not in p.stem]

    for name in video_names:
        frames = np.load(data_dir / f'{name}.npy')
        background = np.load(data_dir / f'{name}_bg.npy')

        # Compute residuals
        residuals = compute_residuals(frames, background)

        # Save
        np.save(data_dir / f'{name}_residuals.npy', residuals)

        # Visualize
        visualize_residuals(residuals, name, output_dir / f'{name}_residuals.png')

        # Stats
        stats = residual_stats(residuals)
        print(f'=== {name} ===')
        print(f'  Mean magnitude:    {stats["mean_magnitude"]:.2f}')
        print(f'  Max magnitude:     {stats["max_magnitude"]:.1f}')
        print(f'  Non-zero fraction: {stats["nonzero_fraction"]:.4f}')
        print(f'  Total energy:      {stats["energy"]:.0f}')
        print()


if __name__ == '__main__':
    main()
