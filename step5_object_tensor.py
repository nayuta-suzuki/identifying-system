"""
IS Experiment Step 5: Object tensor extraction.

Algorithm:
1. Use motion mask to identify the region of interest
2. For each frame, extract the object from the residual using per-frame
   motion detection (frame diff thresholding)
3. Find object centroid in each frame
4. Align (center) all object crops using centroids
5. Average the aligned crops → object tensor

This combines both routes from IS theory:
- Dynamic route: frame differencing locates the object in each frame
- Static route: aligned averaging extracts the common shape (tensor)

Key principle: "motion cuts out the object, shape describes it"
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import fftconvolve


def detect_object_per_frame(frames: np.ndarray, threshold_fraction: float = 0.3):
    """
    Detect the moving object in each frame using frame differencing.

    For each frame t, we use |frame[t] - frame[t-1]| and |frame[t+1] - frame[t]|
    to find pixels that changed. The object is at the intersection of these regions.

    Returns:
        centroids: (n_frames, 2) array of (y, x) centroids, NaN if not found
        bboxes: list of (y0, x0, y1, x1) or None per frame
    """
    n_frames = len(frames)
    frames_f = frames.astype(np.float64)
    centroids = np.full((n_frames, 2), np.nan)
    bboxes = [None] * n_frames

    for t in range(n_frames):
        # Compute local frame differences
        diffs = []
        if t > 0:
            d = np.sqrt(np.sum((frames_f[t] - frames_f[t-1]) ** 2, axis=-1))
            diffs.append(d)
        if t < n_frames - 1:
            d = np.sqrt(np.sum((frames_f[t+1] - frames_f[t]) ** 2, axis=-1))
            diffs.append(d)

        if not diffs:
            continue

        # Average the diffs (for interior frames, this averages forward and backward)
        avg_diff = np.mean(diffs, axis=0)

        # Threshold: relative to max
        thresh = avg_diff.max() * threshold_fraction
        mask = avg_diff > thresh

        if mask.sum() == 0:
            continue

        # Find the largest connected component
        labeled, n_labels = ndimage.label(mask)
        if n_labels == 0:
            continue

        # Find largest blob
        sizes = ndimage.sum(mask, labeled, range(1, n_labels + 1))
        largest = np.argmax(sizes) + 1
        blob_mask = labeled == largest

        ys, xs = np.where(blob_mask)
        centroids[t] = [ys.mean(), xs.mean()]
        bboxes[t] = (ys.min(), xs.min(), ys.max(), xs.max())

    return centroids, bboxes


def extract_object_tensor(
    residuals: np.ndarray,
    centroids: np.ndarray,
    crop_half_size: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract object tensor by aligning residual crops at detected centroids
    and averaging.

    Args:
        residuals: (n_frames, H, W, 3) float64
        centroids: (n_frames, 2) — (y, x) centroids
        crop_half_size: half-width of the crop window

    Returns:
        object_tensor: (crop_size, crop_size, 3) float64 — the averaged aligned object
        aligned_crops: (n_valid, crop_size, crop_size, 3) — all valid aligned crops
    """
    crop_size = 2 * crop_half_size + 1
    H, W = residuals.shape[1], residuals.shape[2]

    aligned_crops = []

    for t in range(len(residuals)):
        if np.isnan(centroids[t, 0]):
            continue

        cy, cx = int(round(centroids[t, 0])), int(round(centroids[t, 1]))

        # Compute crop bounds with padding for out-of-frame objects
        y0 = cy - crop_half_size
        x0 = cx - crop_half_size
        y1 = cy + crop_half_size + 1
        x1 = cx + crop_half_size + 1

        # Create padded crop (zeros for out-of-bounds)
        crop = np.zeros((crop_size, crop_size, 3), dtype=np.float64)

        # Source bounds (clipped to frame)
        sy0 = max(0, y0)
        sx0 = max(0, x0)
        sy1 = min(H, y1)
        sx1 = min(W, x1)

        # Destination bounds in crop
        dy0 = sy0 - y0
        dx0 = sx0 - x0
        dy1 = dy0 + (sy1 - sy0)
        dx1 = dx0 + (sx1 - sx0)

        crop[dy0:dy1, dx0:dx1] = residuals[t, sy0:sy1, sx0:sx1]
        aligned_crops.append(crop)

    aligned_crops = np.array(aligned_crops)
    object_tensor = aligned_crops.mean(axis=0)

    return object_tensor, aligned_crops


def refine_with_cross_correlation(
    residuals: np.ndarray,
    centroids: np.ndarray,
    initial_tensor: np.ndarray,
    crop_half_size: int = 15,
    n_iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine the object tensor using cross-correlation for sub-pixel alignment.

    For each frame:
    1. Slide the current tensor estimate across the residual in the motion region
    2. Find the position of best match (cross-correlation peak)
    3. Re-extract the crop at that refined position
    4. Re-average to get updated tensor

    This is the IS "convolution-like operation" — sliding the tensor
    template across the raw signal to find where it best matches.
    """
    H, W = residuals.shape[1], residuals.shape[2]
    crop_size = 2 * crop_half_size + 1
    current_tensor = initial_tensor.copy()
    current_centroids = centroids.copy()

    for iteration in range(n_iterations):
        refined_centroids = current_centroids.copy()

        for t in range(len(residuals)):
            if np.isnan(current_centroids[t, 0]):
                continue

            cy, cx = int(round(current_centroids[t, 0])), int(round(current_centroids[t, 1]))

            # Define search region around current centroid
            search_pad = 5
            sy0 = max(0, cy - crop_half_size - search_pad)
            sx0 = max(0, cx - crop_half_size - search_pad)
            sy1 = min(H, cy + crop_half_size + search_pad + 1)
            sx1 = min(W, cx + crop_half_size + search_pad + 1)

            region = residuals[t, sy0:sy1, sx0:sx1]

            # Cross-correlate each channel and sum
            # Use grayscale for correlation
            region_gray = region.mean(axis=-1)
            tensor_gray = current_tensor.mean(axis=-1)

            # Ensure region is larger than tensor in both dimensions
            if (region_gray.shape[0] < tensor_gray.shape[0] or
                    region_gray.shape[1] < tensor_gray.shape[1]):
                continue

            # Normalize
            region_norm = region_gray - region_gray.mean()
            tensor_norm = tensor_gray - tensor_gray.mean()

            if tensor_norm.std() < 1e-6 or region_norm.std() < 1e-6:
                continue

            # Cross-correlation via FFT
            corr = fftconvolve(region_norm, tensor_norm[::-1, ::-1], mode='valid')

            if corr.size == 0:
                continue

            # Find peak
            peak = np.unravel_index(corr.argmax(), corr.shape)

            # Convert peak position back to frame coordinates
            refined_cy = sy0 + peak[0] + crop_half_size
            refined_cx = sx0 + peak[1] + crop_half_size

            refined_centroids[t] = [refined_cy, refined_cx]

        # Re-extract tensor with refined centroids
        current_tensor, aligned_crops = extract_object_tensor(
            residuals, refined_centroids, crop_half_size
        )
        current_centroids = refined_centroids

        # Report convergence
        shift = np.nanmean(np.abs(refined_centroids - centroids), axis=0)
        # print(f'  Iteration {iteration+1}: mean shift = ({shift[0]:.2f}, {shift[1]:.2f})')

    return current_tensor, current_centroids


def visualize_object_tensor(
    frames: np.ndarray,
    object_tensor: np.ndarray,
    centroids: np.ndarray,
    aligned_crops: np.ndarray,
    video_name: str,
    save_path: Path,
):
    """Visualize the extracted object tensor and alignment process."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # Top row: sample aligned crops
    n_show = min(5, len(aligned_crops))
    indices = np.linspace(0, len(aligned_crops) - 1, n_show, dtype=int)
    vmax = np.abs(aligned_crops).max()
    for i, idx in enumerate(indices):
        # Display as shifted to [0,1]
        display = aligned_crops[idx] / (2 * vmax) + 0.5
        axes[0, i].imshow(np.clip(display, 0, 1))
        axes[0, i].set_title(f'Crop {idx}')
        axes[0, i].axis('off')

    # Bottom row
    # Object tensor (the result)
    vmax_t = np.abs(object_tensor).max()
    tensor_display = object_tensor / (2 * vmax_t) + 0.5 if vmax_t > 0 else object_tensor
    axes[1, 0].imshow(np.clip(tensor_display, 0, 1))
    axes[1, 0].set_title('Object tensor\n(aligned average)')
    axes[1, 0].axis('off')

    # Object tensor magnitude
    tensor_mag = np.sqrt(np.sum(object_tensor ** 2, axis=-1))
    axes[1, 1].imshow(tensor_mag, cmap='hot')
    axes[1, 1].set_title('Tensor magnitude')
    axes[1, 1].axis('off')

    # Trajectory: centroids on first frame
    axes[1, 2].imshow(frames[0])
    valid = ~np.isnan(centroids[:, 0])
    axes[1, 2].plot(centroids[valid, 1], centroids[valid, 0], 'g.-', markersize=4, linewidth=1)
    axes[1, 2].set_title('Object trajectory')
    axes[1, 2].axis('off')

    # Frame with tensor overlay at detected position
    mid = len(frames) // 2
    if not np.isnan(centroids[mid, 0]):
        axes[1, 3].imshow(frames[mid])
        cy, cx = int(centroids[mid, 0]), int(centroids[mid, 1])
        half = object_tensor.shape[0] // 2
        rect = plt.Rectangle((cx - half, cy - half),
                              object_tensor.shape[1], object_tensor.shape[0],
                              linewidth=2, edgecolor='lime', facecolor='none')
        axes[1, 3].add_patch(rect)
        axes[1, 3].set_title(f'Detection (t={mid})')
    axes[1, 3].axis('off')

    # Tensor as color image (normalized for display)
    if vmax_t > 0:
        tensor_color = np.clip(object_tensor / vmax_t, 0, 1)
    else:
        tensor_color = np.zeros_like(object_tensor)
    axes[1, 4].imshow(tensor_color)
    axes[1, 4].set_title('Tensor (normalized color)')
    axes[1, 4].axis('off')

    fig.suptitle(f'{video_name}: Object Tensor Extraction', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    data_dir = Path(__file__).parent / 'data'
    output_dir = Path(__file__).parent / 'output'

    video_names = [p.stem for p in sorted(data_dir.glob('v*.npy'))
                   if '_bg' not in p.stem and '_residuals' not in p.stem
                   and '_motion' not in p.stem and '_accumulated' not in p.stem]

    for name in video_names:
        frames = np.load(data_dir / f'{name}.npy')
        residuals = np.load(data_dir / f'{name}_residuals.npy')

        print(f'=== {name} ===')

        # Step 1: Detect object per frame
        centroids, bboxes = detect_object_per_frame(frames)
        n_detected = np.sum(~np.isnan(centroids[:, 0]))
        print(f'  Detected object in {n_detected}/{len(frames)} frames')

        # Step 2: Initial object tensor (aligned average of residual crops)
        # Determine crop size from median bbox size
        valid_bboxes = [b for b in bboxes if b is not None]
        if valid_bboxes:
            heights = [b[2] - b[0] for b in valid_bboxes]
            widths = [b[3] - b[1] for b in valid_bboxes]
            crop_half = max(int(np.median(heights)), int(np.median(widths))) + 3
            crop_half = min(crop_half, 20)  # cap
        else:
            crop_half = 15

        print(f'  Crop half-size: {crop_half}')

        object_tensor, aligned_crops = extract_object_tensor(
            residuals, centroids, crop_half
        )

        print(f'  Initial tensor shape: {object_tensor.shape}')
        print(f'  Aligned crops: {len(aligned_crops)}')

        # Step 3: Refine with cross-correlation
        refined_tensor, refined_centroids = refine_with_cross_correlation(
            residuals, centroids, object_tensor, crop_half
        )

        # Compute shift from refinement
        valid = ~np.isnan(centroids[:, 0]) & ~np.isnan(refined_centroids[:, 0])
        if valid.any():
            shifts = np.abs(refined_centroids[valid] - centroids[valid])
            print(f'  Refinement mean shift: ({shifts[:, 0].mean():.2f}, {shifts[:, 1].mean():.2f})')

        # Save
        np.save(data_dir / f'{name}_object_tensor.npy', refined_tensor)
        np.save(data_dir / f'{name}_centroids.npy', refined_centroids)

        # Visualize
        _, aligned_final = extract_object_tensor(residuals, refined_centroids, crop_half)
        visualize_object_tensor(
            frames, refined_tensor, refined_centroids, aligned_final,
            name, output_dir / f'{name}_object_tensor.png'
        )

        # Stats
        tensor_energy = np.sum(refined_tensor ** 2)
        tensor_peak = np.abs(refined_tensor).max()
        print(f'  Tensor energy: {tensor_energy:.0f}')
        print(f'  Tensor peak value: {tensor_peak:.1f}')
        print()


if __name__ == '__main__':
    main()
