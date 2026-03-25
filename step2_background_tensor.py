"""
IS Experiment Step 2: Background tensor extraction.

The first tensor is the full-frame average across all frames.
This should naturally produce a clear background with blurred objects,
because the background is static (identical across frames) while
objects move (averaged out / blurred).

This tests the prediction: the most compressive tensor emerges first,
and it is the background.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def extract_background_tensor(frames: np.ndarray) -> np.ndarray:
    """
    Extract the background tensor by averaging all frames.

    Args:
        frames: (n_frames, H, W, 3) uint8 array
    Returns:
        background: (H, W, 3) float64 array (preserving sub-pixel precision)
    """
    return frames.astype(np.float64).mean(axis=0)


def visualize_background(
    frames: np.ndarray,
    background: np.ndarray,
    video_name: str,
    save_path: Path,
):
    """Visualize: first frame | background tensor | last frame."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].imshow(frames[0])
    axes[0].set_title('Frame 0 (input)')
    axes[0].axis('off')

    # Show background as uint8 for display
    axes[1].imshow(np.clip(background, 0, 255).astype(np.uint8))
    axes[1].set_title('Background tensor\n(frame average)')
    axes[1].axis('off')

    axes[2].imshow(frames[-1])
    axes[2].set_title(f'Frame {len(frames)-1} (input)')
    axes[2].axis('off')

    fig.suptitle(f'{video_name}: Background Tensor Extraction', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def compute_background_stats(
    frames: np.ndarray,
    background: np.ndarray,
) -> dict:
    """
    Compute how well the background tensor explains the input.

    Returns dict with:
        - total_variance: total pixel variance across all frames
        - explained_variance: variance explained by background
        - compression_ratio: fraction of variance explained
    """
    frames_f = frames.astype(np.float64)
    n_frames = len(frames)

    # Total variance: sum of squared differences from global mean
    total_ss = np.sum((frames_f - background[None]) ** 2)

    # Total possible variance: sum of squared pixel values
    # (using variance relative to background as the denominator)
    total_pixel_variance = np.var(frames_f, axis=0).sum() * n_frames

    # Per-pixel explanation
    bg_uint8 = np.clip(background, 0, 255).astype(np.uint8)
    reconstruction_error = np.mean((frames_f - background[None]) ** 2)
    original_variance = np.var(frames_f)

    return {
        'mean_squared_error': reconstruction_error,
        'pixel_variance': original_variance,
        'variance_explained_ratio': 1.0 - (reconstruction_error / original_variance) if original_variance > 0 else 1.0,
        'mean_abs_error': np.mean(np.abs(frames_f - background[None])),
        'max_abs_error': np.max(np.abs(frames_f - background[None])),
    }


def main():
    data_dir = Path(__file__).parent / 'data'
    output_dir = Path(__file__).parent / 'output'

    video_files = sorted(data_dir.glob('*.npy'))
    print(f'Found {len(video_files)} videos\n')

    for vf in video_files:
        name = vf.stem
        frames = np.load(vf)

        # Extract background tensor
        background = extract_background_tensor(frames)

        # Save background tensor
        np.save(data_dir / f'{name}_bg.npy', background)

        # Visualize
        visualize_background(frames, background, name, output_dir / f'{name}_bg.png')

        # Compute stats
        stats = compute_background_stats(frames, background)
        print(f'=== {name} ===')
        print(f'  Variance explained by background: {stats["variance_explained_ratio"]:.4f}')
        print(f'  Mean squared error:               {stats["mean_squared_error"]:.2f}')
        print(f'  Mean absolute error:              {stats["mean_abs_error"]:.2f}')
        print(f'  Max absolute error:               {stats["max_abs_error"]:.1f}')
        print()


if __name__ == '__main__':
    main()
