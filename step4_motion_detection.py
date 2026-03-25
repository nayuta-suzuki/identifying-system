"""
IS Experiment Step 4: Motion detection via frame differencing.

"Dynamic route" for tensor discovery: consecutive frame differences
reveal moving regions. This is computationally cheap and does not
require any external criteria — pixel differences are intrinsic.

Key prediction: frame differencing naturally separates moving objects
from static background, providing the "cut-out" that narrows the
search space for object tensor extraction (Step 5).
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage


def compute_frame_diffs(frames: np.ndarray) -> np.ndarray:
    """
    Compute consecutive frame differences.

    Args:
        frames: (n_frames, H, W, 3) uint8
    Returns:
        diffs: (n_frames-1, H, W) float64 — magnitude of difference
    """
    frames_f = frames.astype(np.float64)
    # Difference between consecutive frames, then magnitude across channels
    raw_diffs = frames_f[1:] - frames_f[:-1]  # (n-1, H, W, 3)
    magnitudes = np.sqrt(np.sum(raw_diffs ** 2, axis=-1))  # (n-1, H, W)
    return magnitudes


def compute_motion_mask(diff_magnitudes: np.ndarray, threshold_fraction: float = 0.2) -> np.ndarray:
    """
    Create a binary motion mask from accumulated frame differences.

    The threshold is relative to the max accumulated difference
    (no external absolute threshold needed).

    Args:
        diff_magnitudes: (n_diffs, H, W) frame difference magnitudes
        threshold_fraction: fraction of max to use as threshold
    Returns:
        motion_mask: (H, W) bool — True where motion detected
        accumulated: (H, W) float64 — accumulated diff magnitudes
    """
    # Accumulate across all frame pairs
    accumulated = diff_magnitudes.sum(axis=0)  # (H, W)

    # Relative threshold: fraction of maximum
    threshold = accumulated.max() * threshold_fraction
    motion_mask = accumulated > threshold

    return motion_mask, accumulated


def extract_motion_blobs(motion_mask: np.ndarray) -> list[dict]:
    """
    Find connected regions in the motion mask.
    Each blob is a candidate "moving object".

    Returns list of dicts with 'label', 'bbox', 'area', 'centroid'.
    """
    labeled, n_labels = ndimage.label(motion_mask)
    blobs = []
    for i in range(1, n_labels + 1):
        ys, xs = np.where(labeled == i)
        blobs.append({
            'label': i,
            'bbox': (ys.min(), xs.min(), ys.max(), xs.max()),
            'area': len(ys),
            'centroid': (ys.mean(), xs.mean()),
        })
    return blobs


def visualize_motion(
    frames: np.ndarray,
    diff_magnitudes: np.ndarray,
    accumulated: np.ndarray,
    motion_mask: np.ndarray,
    blobs: list[dict],
    video_name: str,
    save_path: Path,
):
    """Visualize frame diffs, accumulated motion, mask, and detected blobs."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Top row: sample frame diffs
    n_diffs = len(diff_magnitudes)
    vmax_diff = diff_magnitudes.max()
    for i, idx in enumerate([0, n_diffs // 3, 2 * n_diffs // 3, n_diffs - 1]):
        axes[0, i].imshow(diff_magnitudes[idx], cmap='hot', vmin=0, vmax=vmax_diff)
        axes[0, i].set_title(f'|frame[{idx+1}] - frame[{idx}]|')
        axes[0, i].axis('off')

    # Bottom row
    # Accumulated motion
    axes[1, 0].imshow(accumulated, cmap='hot')
    axes[1, 0].set_title('Accumulated motion')
    axes[1, 0].axis('off')

    # Motion mask
    axes[1, 1].imshow(motion_mask, cmap='gray')
    axes[1, 1].set_title('Motion mask')
    axes[1, 1].axis('off')

    # Detected blobs overlaid on first frame
    axes[1, 2].imshow(frames[0])
    for blob in blobs:
        y0, x0, y1, x1 = blob['bbox']
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                              linewidth=2, edgecolor='lime', facecolor='none')
        axes[1, 2].add_patch(rect)
        axes[1, 2].text(x0, y0 - 2, f'blob {blob["label"]}\n{blob["area"]}px',
                        color='lime', fontsize=8)
    axes[1, 2].set_title(f'Detected blobs ({len(blobs)})')
    axes[1, 2].axis('off')

    # Motion region cropped from a middle residual frame
    if blobs:
        biggest = max(blobs, key=lambda b: b['area'])
        y0, x0, y1, x1 = biggest['bbox']
        # Add small padding
        pad = 3
        y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
        y1, x1 = min(frames.shape[1], y1 + pad), min(frames.shape[2], x1 + pad)
        mid = len(frames) // 2
        crop = frames[mid, y0:y1+1, x0:x1+1]
        axes[1, 3].imshow(crop)
        axes[1, 3].set_title(f'Largest blob crop (t={mid})')
    else:
        axes[1, 3].text(0.5, 0.5, 'No blobs', ha='center', va='center')
    axes[1, 3].axis('off')

    fig.suptitle(f'{video_name}: Motion Detection (frame differencing)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    data_dir = Path(__file__).parent / 'data'
    output_dir = Path(__file__).parent / 'output'

    video_names = [p.stem for p in sorted(data_dir.glob('v*.npy'))
                   if '_bg' not in p.stem and '_residuals' not in p.stem]

    for name in video_names:
        frames = np.load(data_dir / f'{name}.npy')

        # Frame differencing
        diff_mags = compute_frame_diffs(frames)

        # Accumulate and threshold
        motion_mask, accumulated = compute_motion_mask(diff_mags)

        # Find blobs
        blobs = extract_motion_blobs(motion_mask)

        # Save motion data
        np.save(data_dir / f'{name}_motion_mask.npy', motion_mask)
        np.save(data_dir / f'{name}_accumulated_motion.npy', accumulated)

        # Visualize
        visualize_motion(frames, diff_mags, accumulated, motion_mask,
                         blobs, name, output_dir / f'{name}_motion.png')

        print(f'=== {name} ===')
        print(f'  Frame diffs computed: {len(diff_mags)}')
        print(f'  Motion mask coverage: {motion_mask.mean():.4f}')
        print(f'  Blobs detected: {len(blobs)}')
        for b in blobs:
            print(f'    Blob {b["label"]}: area={b["area"]}px, '
                  f'bbox={b["bbox"]}, centroid=({b["centroid"][0]:.1f}, {b["centroid"][1]:.1f})')
        print()


if __name__ == '__main__':
    main()
