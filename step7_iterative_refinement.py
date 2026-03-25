"""
IS Experiment Step 7: Iterative tensor refinement.

Tests the IS learning loop: reconstruct → residual → update tensor → repeat.

The core prediction: each iteration should improve compression because:
1. Background tensor gets purer (ghost trajectory removed)
2. Object tensor gets sharper (better alignment on cleaner residuals)
3. The loop converges when no further compression is possible

Algorithm (derived from IS principles):
  Iteration 0: background = frame average (Step 2 result)
  For each iteration:
    1. Compute residuals (frames - background)
    2. Detect motion regions from residuals
    3. Mask out motion regions from frames
    4. Recompute background from masked frames (only static pixels)
    5. Re-extract object tensor from new residuals
    6. Reconstruct and measure compression

This is the IS "read → residual → tensor update" loop in its simplest form.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import fftconvolve


# --- Reuse core functions from previous steps ---

def compute_frame_diffs(frames: np.ndarray) -> np.ndarray:
    frames_f = frames.astype(np.float64)
    raw_diffs = frames_f[1:] - frames_f[:-1]
    return np.sqrt(np.sum(raw_diffs ** 2, axis=-1))


def compute_motion_mask(diff_magnitudes: np.ndarray, threshold_fraction: float = 0.2):
    accumulated = diff_magnitudes.sum(axis=0)
    threshold = accumulated.max() * threshold_fraction
    motion_mask = accumulated > threshold
    return motion_mask, accumulated


def detect_object_per_frame(frames: np.ndarray, threshold_fraction: float = 0.3):
    n_frames = len(frames)
    frames_f = frames.astype(np.float64)
    centroids = np.full((n_frames, 2), np.nan)
    bboxes = [None] * n_frames

    for t in range(n_frames):
        diffs = []
        if t > 0:
            d = np.sqrt(np.sum((frames_f[t] - frames_f[t-1]) ** 2, axis=-1))
            diffs.append(d)
        if t < n_frames - 1:
            d = np.sqrt(np.sum((frames_f[t+1] - frames_f[t]) ** 2, axis=-1))
            diffs.append(d)
        if not diffs:
            continue

        avg_diff = np.mean(diffs, axis=0)
        thresh = avg_diff.max() * threshold_fraction
        mask = avg_diff > thresh
        if mask.sum() == 0:
            continue

        labeled, n_labels = ndimage.label(mask)
        if n_labels == 0:
            continue

        sizes = ndimage.sum(mask, labeled, range(1, n_labels + 1))
        largest = np.argmax(sizes) + 1
        blob_mask = labeled == largest

        ys, xs = np.where(blob_mask)
        centroids[t] = [ys.mean(), xs.mean()]
        bboxes[t] = (ys.min(), xs.min(), ys.max(), xs.max())

    return centroids, bboxes


def extract_object_tensor(residuals, centroids, crop_half_size=15):
    crop_size = 2 * crop_half_size + 1
    H, W = residuals.shape[1], residuals.shape[2]
    aligned_crops = []

    for t in range(len(residuals)):
        if np.isnan(centroids[t, 0]):
            continue
        cy, cx = int(round(centroids[t, 0])), int(round(centroids[t, 1]))
        y0, x0 = cy - crop_half_size, cx - crop_half_size
        y1, x1 = cy + crop_half_size + 1, cx + crop_half_size + 1

        crop = np.zeros((crop_size, crop_size, 3), dtype=np.float64)
        sy0, sx0 = max(0, y0), max(0, x0)
        sy1, sx1 = min(H, y1), min(W, x1)
        dy0, dx0 = sy0 - y0, sx0 - x0
        dy1, dx1 = dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)
        crop[dy0:dy1, dx0:dx1] = residuals[t, sy0:sy1, sx0:sx1]
        aligned_crops.append(crop)

    aligned_crops = np.array(aligned_crops)
    object_tensor = aligned_crops.mean(axis=0)
    return object_tensor, aligned_crops


def refine_with_cross_correlation(residuals, centroids, initial_tensor,
                                  crop_half_size=15, n_iterations=3):
    H, W = residuals.shape[1], residuals.shape[2]
    current_tensor = initial_tensor.copy()
    current_centroids = centroids.copy()

    for iteration in range(n_iterations):
        refined_centroids = current_centroids.copy()
        for t in range(len(residuals)):
            if np.isnan(current_centroids[t, 0]):
                continue
            cy, cx = int(round(current_centroids[t, 0])), int(round(current_centroids[t, 1]))
            search_pad = 5
            sy0 = max(0, cy - crop_half_size - search_pad)
            sx0 = max(0, cx - crop_half_size - search_pad)
            sy1 = min(H, cy + crop_half_size + search_pad + 1)
            sx1 = min(W, cx + crop_half_size + search_pad + 1)

            region = residuals[t, sy0:sy1, sx0:sx1]
            region_gray = region.mean(axis=-1)
            tensor_gray = current_tensor.mean(axis=-1)

            if (region_gray.shape[0] < tensor_gray.shape[0] or
                    region_gray.shape[1] < tensor_gray.shape[1]):
                continue

            region_norm = region_gray - region_gray.mean()
            tensor_norm = tensor_gray - tensor_gray.mean()
            if tensor_norm.std() < 1e-6 or region_norm.std() < 1e-6:
                continue

            corr = fftconvolve(region_norm, tensor_norm[::-1, ::-1], mode='valid')
            if corr.size == 0:
                continue

            peak = np.unravel_index(corr.argmax(), corr.shape)
            refined_centroids[t] = [sy0 + peak[0] + crop_half_size,
                                    sx0 + peak[1] + crop_half_size]

        current_tensor, _ = extract_object_tensor(residuals, refined_centroids, crop_half_size)
        current_centroids = refined_centroids

    return current_tensor, current_centroids


def reconstruct_frames(background, object_tensor, centroids, n_frames, frame_size=64):
    H, W = frame_size, frame_size
    th, tw = object_tensor.shape[:2]
    half_h, half_w = th // 2, tw // 2
    reconstructed = np.zeros((n_frames, H, W, 3), dtype=np.float64)

    for t in range(n_frames):
        frame = background.copy()
        if not np.isnan(centroids[t, 0]):
            cy, cx = int(round(centroids[t, 0])), int(round(centroids[t, 1]))
            y0, x0 = cy - half_h, cx - half_w
            y1, x1 = y0 + th, x0 + tw
            sy0, sx0 = max(0, y0), max(0, x0)
            sy1, sx1 = min(H, y1), min(W, x1)
            ty0, tx0 = sy0 - y0, sx0 - x0
            ty1, tx1 = ty0 + (sy1 - sy0), tx0 + (sx1 - sx0)
            frame[sy0:sy1, sx0:sx1] += object_tensor[ty0:ty1, tx0:tx1]
        reconstructed[t] = frame
    return reconstructed


# --- New: Iterative refinement ---

def refine_background_with_mask(
    frames: np.ndarray,
    motion_mask: np.ndarray,
) -> np.ndarray:
    """
    Recompute background tensor using only static (non-moving) pixels.

    For pixels in the motion region, we use the median across frames
    (more robust to object presence than mean). For static pixels,
    we use the mean (they're identical across frames anyway).

    This removes the ghost trajectory from the background tensor.
    """
    frames_f = frames.astype(np.float64)
    H, W = frames.shape[1], frames.shape[2]

    # For motion region: use median (robust to outliers = object presence)
    # For static region: use mean (exact)
    background = frames_f.mean(axis=0)

    # Overwrite motion region with median
    if motion_mask.any():
        motion_pixels = frames_f[:, motion_mask]  # (n_frames, n_motion_pixels, 3)
        background[motion_mask] = np.median(motion_pixels, axis=0)

    return background


def iterative_refinement(
    frames: np.ndarray,
    n_iterations: int = 5,
    crop_half: int = None,
) -> dict:
    """
    Run the IS learning loop: reconstruct → residual → update tensors → repeat.

    Returns history of metrics per iteration.
    """
    frames_f = frames.astype(np.float64)
    n_frames, H, W, C = frames.shape
    total_ss = np.sum((frames_f - frames_f.mean()) ** 2)

    # Initial state: simple frame average
    background = frames_f.mean(axis=0)

    history = []

    for iteration in range(n_iterations):
        # 1. Compute residuals
        residuals = frames_f - background[None]

        # 2. Detect motion
        diff_mags = compute_frame_diffs(frames)
        motion_mask, _ = compute_motion_mask(diff_mags)

        # 3. Detect object per frame
        centroids, bboxes = detect_object_per_frame(frames)
        n_detected = np.sum(~np.isnan(centroids[:, 0]))

        # Determine crop size
        if crop_half is None:
            valid_bboxes = [b for b in bboxes if b is not None]
            if valid_bboxes:
                heights = [b[2] - b[0] for b in valid_bboxes]
                widths = [b[3] - b[1] for b in valid_bboxes]
                c_half = max(int(np.median(heights)), int(np.median(widths))) + 3
                c_half = min(c_half, 20)
            else:
                c_half = 15
        else:
            c_half = crop_half

        # 4. Extract object tensor
        obj_tensor, _ = extract_object_tensor(residuals, centroids, c_half)

        # 5. Refine with cross-correlation
        obj_tensor, refined_centroids = refine_with_cross_correlation(
            residuals, centroids, obj_tensor, c_half, n_iterations=2
        )

        # 6. Reconstruct and evaluate
        reconstructed = reconstruct_frames(background, obj_tensor, refined_centroids, n_frames)

        bg_error = np.sum((frames_f - background[None]) ** 2)
        full_error = np.sum((frames_f - reconstructed) ** 2)

        bg_var_explained = 1.0 - bg_error / total_ss
        full_var_explained = 1.0 - full_error / total_ss

        # Ghost metric: how much energy is in the motion region of the background?
        bg_motion_energy = 0.0
        if motion_mask.any():
            # In a perfect background, motion region should be pure background color
            # The ghost energy = deviation of background in motion region from
            # the background color in static region
            static_mean = background[~motion_mask].mean(axis=0) if (~motion_mask).any() else background.mean(axis=(0, 1))
            ghost = background[motion_mask] - static_mean[None]
            bg_motion_energy = np.sum(ghost ** 2)

        history.append({
            'iteration': iteration,
            'bg_var_explained': bg_var_explained,
            'full_var_explained': full_var_explained,
            'bg_mse': np.mean((frames_f - background[None]) ** 2),
            'full_mse': np.mean((frames_f - reconstructed) ** 2),
            'ghost_energy': bg_motion_energy,
            'n_detected': n_detected,
            'background': background.copy(),
            'obj_tensor': obj_tensor.copy(),
            'centroids': refined_centroids.copy(),
        })

        print(f'  Iter {iteration}: BG={bg_var_explained:.4f}, '
              f'BG+Obj={full_var_explained:.4f}, '
              f'ghost_energy={bg_motion_energy:.0f}')

        # 7. Update background: mask out motion region and recompute
        background = refine_background_with_mask(frames, motion_mask)

    return history


def visualize_refinement(
    frames: np.ndarray,
    history: list[dict],
    video_name: str,
    save_path: Path,
):
    """Visualize the iterative refinement process."""
    n_iters = len(history)

    fig, axes = plt.subplots(3, n_iters, figsize=(4 * n_iters, 10))

    for i, h in enumerate(history):
        # Row 1: Background tensor
        bg_display = np.clip(h['background'], 0, 255).astype(np.uint8)
        axes[0, i].imshow(bg_display)
        axes[0, i].set_title(f'Iter {i}\nBG var={h["bg_var_explained"]:.3f}')
        axes[0, i].axis('off')

        # Row 2: Object tensor magnitude
        tensor_mag = np.sqrt(np.sum(h['obj_tensor'] ** 2, axis=-1))
        axes[1, i].imshow(tensor_mag, cmap='hot')
        axes[1, i].set_title(f'Obj tensor\nfull var={h["full_var_explained"]:.3f}')
        axes[1, i].axis('off')

        # Row 3: Reconstruction of middle frame
        mid = len(frames) // 2
        recon = reconstruct_frames(
            h['background'], h['obj_tensor'], h['centroids'], len(frames)
        )
        recon_display = np.clip(recon[mid], 0, 255).astype(np.uint8)
        axes[2, i].imshow(recon_display)
        axes[2, i].set_title(f'Recon (t={mid})\nghost={h["ghost_energy"]:.0f}')
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel('Background\ntensor', rotation=0, labelpad=60, va='center')
    axes[1, 0].set_ylabel('Object\ntensor', rotation=0, labelpad=60, va='center')
    axes[2, 0].set_ylabel('Reconstruction\n(middle frame)', rotation=0, labelpad=60, va='center')

    fig.suptitle(f'{video_name}: Iterative Refinement ({n_iters} iterations)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def visualize_convergence(
    all_histories: dict[str, list[dict]],
    save_path: Path,
):
    """Plot convergence curves for all videos."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for name, history in all_histories.items():
        iters = [h['iteration'] for h in history]
        bg_var = [h['bg_var_explained'] for h in history]
        full_var = [h['full_var_explained'] for h in history]
        ghost = [h['ghost_energy'] for h in history]

        axes[0].plot(iters, bg_var, '.-', label=name)
        axes[1].plot(iters, full_var, '.-', label=name)
        axes[2].plot(iters, ghost, '.-', label=name)

    axes[0].set_title('Background var. explained')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Variance explained')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('BG + Object var. explained')
    axes[1].set_xlabel('Iteration')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title('Ghost energy in background')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Energy')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Iterative Refinement: Convergence', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    data_dir = Path(__file__).parent / 'data'
    output_dir = Path(__file__).parent / 'output'

    video_names = ['v1_triangle_right', 'v2_square_diag', 'v3_circle_down', 'v4_two_objects']
    all_histories = {}

    for name in video_names:
        frames = np.load(data_dir / f'{name}.npy')
        print(f'=== {name} ===')

        history = iterative_refinement(frames, n_iterations=5)
        all_histories[name] = history

        # Visualize per-video refinement
        visualize_refinement(frames, history, name,
                             output_dir / f'{name}_refinement.png')

        # Report improvement
        h0, h_last = history[0], history[-1]
        print(f'  Improvement: BG {h0["bg_var_explained"]:.4f} → {h_last["bg_var_explained"]:.4f}')
        print(f'  Improvement: Full {h0["full_var_explained"]:.4f} → {h_last["full_var_explained"]:.4f}')
        print(f'  Ghost reduction: {h0["ghost_energy"]:.0f} → {h_last["ghost_energy"]:.0f}')
        print()

    # Convergence plot
    visualize_convergence(all_histories, output_dir / 'refinement_convergence.png')
    print('Done. Visualizations saved.')


if __name__ == '__main__':
    main()
