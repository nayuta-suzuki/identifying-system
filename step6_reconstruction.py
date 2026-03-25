"""
IS Experiment Step 6: Reconstruction and compression evaluation.

Reconstruct each frame using:
  - Background tensor (full-frame average)
  - Object tensor (aligned average of residual crops) placed at detected position

Evaluation:
  - Compression ratio: how much of the original signal is explained
  - Comparison of: raw frames vs background-only vs background+object reconstruction
  - This tests the core IS prediction: tensor hierarchy (background → object → position)
    naturally emerges from compression pressure

The representation of each frame becomes:
  background_tensor + object_tensor @ position(t)
This is the "weighted placement list" described in IS theory.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


def reconstruct_frames(
    background: np.ndarray,
    object_tensor: np.ndarray,
    centroids: np.ndarray,
    n_frames: int,
    frame_size: int = 64,
) -> np.ndarray:
    """
    Reconstruct frames using background tensor + object tensor at detected positions.

    For each frame t:
        reconstruction[t] = background + object_tensor centered at centroid[t]

    Returns: (n_frames, frame_size, frame_size, 3) float64
    """
    H, W = frame_size, frame_size
    th, tw = object_tensor.shape[:2]
    half_h, half_w = th // 2, tw // 2

    reconstructed = np.zeros((n_frames, H, W, 3), dtype=np.float64)

    for t in range(n_frames):
        # Start with background
        frame = background.copy()

        if not np.isnan(centroids[t, 0]):
            cy, cx = int(round(centroids[t, 0])), int(round(centroids[t, 1]))

            # Compute placement bounds
            y0 = cy - half_h
            x0 = cx - half_w
            y1 = y0 + th
            x1 = x0 + tw

            # Clip to frame boundaries
            sy0, sx0 = max(0, y0), max(0, x0)
            sy1, sx1 = min(H, y1), min(W, x1)

            # Corresponding region in tensor
            ty0 = sy0 - y0
            tx0 = sx0 - x0
            ty1 = ty0 + (sy1 - sy0)
            tx1 = tx0 + (sx1 - sx0)

            # Add object tensor to background
            frame[sy0:sy1, sx0:sx1] += object_tensor[ty0:ty1, tx0:tx1]

        reconstructed[t] = frame

    return reconstructed


def evaluate_reconstruction(
    original: np.ndarray,
    background: np.ndarray,
    reconstructed: np.ndarray,
) -> dict:
    """
    Evaluate reconstruction quality.

    Returns metrics comparing:
    - Original vs background-only (how much background explains)
    - Original vs full reconstruction (background + object)
    - Improvement from adding object tensor
    """
    orig_f = original.astype(np.float64)

    # Total variance in original
    total_variance = np.var(orig_f)
    total_ss = np.sum((orig_f - orig_f.mean()) ** 2)

    # Background-only reconstruction error
    bg_error = np.sum((orig_f - background[None]) ** 2)
    bg_mse = np.mean((orig_f - background[None]) ** 2)

    # Full reconstruction error
    recon_error = np.sum((orig_f - reconstructed) ** 2)
    recon_mse = np.mean((orig_f - reconstructed) ** 2)

    # Variance explained
    bg_var_explained = 1.0 - bg_error / total_ss if total_ss > 0 else 1.0
    full_var_explained = 1.0 - recon_error / total_ss if total_ss > 0 else 1.0

    # Improvement from adding object tensor
    improvement = (bg_error - recon_error) / bg_error if bg_error > 0 else 0.0

    # Per-pixel max error
    bg_max_error = np.max(np.abs(orig_f - background[None]))
    recon_max_error = np.max(np.abs(orig_f - reconstructed))

    # Compression ratio: how compact is the representation?
    # Original: n_frames * H * W * 3 values
    # Compressed: H*W*3 (background) + th*tw*3 (object tensor) + n_frames*2 (positions)
    n_frames, H, W, C = original.shape
    th, tw = reconstructed.shape[1], reconstructed.shape[2]  # approximate
    original_params = n_frames * H * W * C
    compressed_params = H * W * C + np.prod(background.shape) + n_frames * 2
    param_compression = original_params / compressed_params

    return {
        'total_variance': total_variance,
        'bg_mse': bg_mse,
        'bg_var_explained': bg_var_explained,
        'bg_max_error': bg_max_error,
        'recon_mse': recon_mse,
        'recon_var_explained': full_var_explained,
        'recon_max_error': recon_max_error,
        'improvement_from_object': improvement,
        'param_compression_ratio': param_compression,
    }


def visualize_reconstruction(
    original: np.ndarray,
    background: np.ndarray,
    reconstructed: np.ndarray,
    metrics: dict,
    video_name: str,
    save_path: Path,
):
    """Compare original, background-only, and full reconstruction."""
    n_show = 5
    indices = np.linspace(0, len(original) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(4, n_show, figsize=(3 * n_show, 12))

    for i, idx in enumerate(indices):
        # Original
        axes[0, i].imshow(original[idx])
        axes[0, i].set_title(f't={idx}')
        axes[0, i].axis('off')

        # Background only
        bg_display = np.clip(background, 0, 255).astype(np.uint8)
        axes[1, i].imshow(bg_display)
        axes[1, i].axis('off')

        # Full reconstruction
        recon_display = np.clip(reconstructed[idx], 0, 255).astype(np.uint8)
        axes[2, i].imshow(recon_display)
        axes[2, i].axis('off')

        # Error heatmap (original - reconstruction)
        error = np.sqrt(np.sum((original[idx].astype(np.float64) - reconstructed[idx]) ** 2, axis=-1))
        axes[3, i].imshow(error, cmap='hot', vmin=0, vmax=200)
        axes[3, i].axis('off')

    # Row labels
    axes[0, 0].set_ylabel('Original', rotation=0, labelpad=70, va='center', fontsize=11)
    axes[1, 0].set_ylabel('Background\nonly', rotation=0, labelpad=70, va='center', fontsize=11)
    axes[2, 0].set_ylabel('BG + Object\ntensor', rotation=0, labelpad=70, va='center', fontsize=11)
    axes[3, 0].set_ylabel('Error\n(residual)', rotation=0, labelpad=70, va='center', fontsize=11)

    # Summary text
    summary = (
        f'Background var. explained: {metrics["bg_var_explained"]:.1%}\n'
        f'BG+Object var. explained:  {metrics["recon_var_explained"]:.1%}\n'
        f'Improvement from object:   {metrics["improvement_from_object"]:.1%}\n'
        f'Parameter compression:     {metrics["param_compression_ratio"]:.1f}x'
    )
    fig.text(0.02, 0.02, summary, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle(f'{video_name}: Reconstruction Evaluation', fontsize=14)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    data_dir = Path(__file__).parent / 'data'
    output_dir = Path(__file__).parent / 'output'

    video_names = [p.stem for p in sorted(data_dir.glob('v*.npy'))
                   if '_' not in p.stem or p.stem.startswith('v') and
                   '_bg' not in p.stem and '_residuals' not in p.stem and
                   '_motion' not in p.stem and '_accumulated' not in p.stem and
                   '_object' not in p.stem and '_centroids' not in p.stem]

    # More robust: just list the base video names
    video_names = ['v1_triangle_right', 'v2_square_diag', 'v3_circle_down', 'v4_two_objects']

    all_metrics = {}

    for name in video_names:
        frames = np.load(data_dir / f'{name}.npy')
        background = np.load(data_dir / f'{name}_bg.npy')
        object_tensor = np.load(data_dir / f'{name}_object_tensor.npy')
        centroids = np.load(data_dir / f'{name}_centroids.npy')

        print(f'=== {name} ===')
        print(f'  Object tensor shape: {object_tensor.shape}')

        # Reconstruct
        reconstructed = reconstruct_frames(
            background, object_tensor, centroids, len(frames)
        )

        # Evaluate
        metrics = evaluate_reconstruction(frames, background, reconstructed)
        all_metrics[name] = metrics

        print(f'  Background var. explained: {metrics["bg_var_explained"]:.4f}')
        print(f'  BG+Object var. explained:  {metrics["recon_var_explained"]:.4f}')
        print(f'  Improvement from object:   {metrics["improvement_from_object"]:.4f}')
        print(f'  BG MSE:                    {metrics["bg_mse"]:.2f}')
        print(f'  Full recon MSE:            {metrics["recon_mse"]:.2f}')
        print(f'  BG max error:              {metrics["bg_max_error"]:.1f}')
        print(f'  Full recon max error:      {metrics["recon_max_error"]:.1f}')
        print(f'  Parameter compression:     {metrics["param_compression_ratio"]:.1f}x')
        print()

        # Visualize
        visualize_reconstruction(
            frames, background, reconstructed, metrics,
            name, output_dir / f'{name}_reconstruction.png'
        )

    # Summary comparison
    print('=== SUMMARY ===')
    print(f'{"Video":<25} {"BG only":>10} {"BG+Obj":>10} {"Improvement":>12} {"Compression":>12}')
    print('-' * 70)
    for name, m in all_metrics.items():
        print(f'{name:<25} {m["bg_var_explained"]:>9.1%} {m["recon_var_explained"]:>9.1%} '
              f'{m["improvement_from_object"]:>11.1%} {m["param_compression_ratio"]:>10.1f}x')


if __name__ == '__main__':
    main()
