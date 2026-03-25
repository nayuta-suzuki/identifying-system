"""
IS Experiment 8: Sequential Processing — IS's Learning Loop (First Implementation)

Unlike experiments 1-7 which accessed all frames simultaneously ("god's eye view"),
this experiment processes frames one at a time, maintaining only a "hand" of tensors
between frames. Raw pixels are discarded after each frame is processed.

Design principle correction:
  "Color segments the image into parts first; motion then identifies which parts
   are objects." (replaces the prior "motion first, then shape describes")
  At the principle level, the one with higher compression rate goes first;
  color vs motion priority is input-dependent. However, with only 1 frame,
  motion information is zero, so color necessarily comes first.

Operations given externally (would ideally be learned):
  - Color-based region segmentation (connected components + color proximity threshold)
  - Centroid computation (mean of pixel coordinates in a same-color region)

Author: IS Project (Session 31)
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage


# ---------------------------------------------------------------------------
# Tensor representation
# ---------------------------------------------------------------------------

class Tensor:
    """A single tensor in the IS hand.

    Attributes:
        color:       (3,) float64 — mean RGB of the region
        shape_mask:  (h, w) bool — shape relative to bounding box
        shape_pixels:(h, w, 3) float64 — pixel values within bounding box
        centroid:    (2,) float64 — (x, y) position in frame coordinates
        pixel_count: int — number of True pixels in shape_mask
        id:          int — unique identifier
        age:         int — number of frames this tensor has existed
    """

    _next_id = 0

    def __init__(self, color, shape_mask, shape_pixels, centroid, pixel_count):
        self.color = color.astype(np.float64)
        self.shape_mask = shape_mask.copy()
        self.shape_pixels = shape_pixels.astype(np.float64)
        self.centroid = np.array(centroid, dtype=np.float64)
        self.pixel_count = pixel_count
        self.id = Tensor._next_id
        Tensor._next_id += 1
        self.age = 0
        self.centroid_history = [self.centroid.copy()]

    @property
    def bbox_h(self):
        return self.shape_mask.shape[0]

    @property
    def bbox_w(self):
        return self.shape_mask.shape[1]


# ---------------------------------------------------------------------------
# Color-based segmentation (externally given operation)
# ---------------------------------------------------------------------------

def segment_by_color(frame, color_threshold=30.0, min_region_size=5):
    """Segment a frame into regions of similar color using flood-fill.

    Returns list of dicts with keys: mask, color, centroid, pixel_count.
    """
    H, W = frame.shape[:2]
    visited = np.zeros((H, W), dtype=bool)
    regions = []

    for start_y in range(H):
        for start_x in range(W):
            if visited[start_y, start_x]:
                continue

            seed_color = frame[start_y, start_x].copy()
            mask = np.zeros((H, W), dtype=bool)
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            pixels_yx = []

            while stack:
                py, px = stack.pop()
                mask[py, px] = True
                pixels_yx.append((py, px))

                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                        dist = np.sqrt(np.sum((frame[ny, nx] - seed_color) ** 2))
                        if dist < color_threshold:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            if len(pixels_yx) < min_region_size:
                continue

            ys = np.array([p[0] for p in pixels_yx])
            xs = np.array([p[1] for p in pixels_yx])
            mean_color = frame[mask].mean(axis=0)
            centroid = np.array([xs.mean(), ys.mean()])

            regions.append({
                'mask': mask,
                'color': mean_color,
                'centroid': centroid,
                'pixel_count': len(pixels_yx),
            })

    return regions


def create_tensor_from_region(frame, region):
    """Create a Tensor object from a segmented region."""
    mask = region['mask']
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    bbox_mask = mask[y0:y1, x0:x1]
    bbox_pixels = frame[y0:y1, x0:x1].copy()
    bbox_pixels[~bbox_mask] = 0

    return Tensor(
        color=region['color'],
        shape_mask=bbox_mask,
        shape_pixels=bbox_pixels,
        centroid=region['centroid'],
        pixel_count=region['pixel_count'],
    )


# ---------------------------------------------------------------------------
# Tensor matching — local grid search
# ---------------------------------------------------------------------------

def compute_match_score(tensor, frame, cy, cx):
    """Compute match score of placing tensor centroid at (cx, cy) in frame.

    Returns (score, n_valid).  score = -mean_SSD over valid (masked) pixels.
    Higher (less negative) = better match.
    """
    H, W = frame.shape[:2]
    th, tw = tensor.bbox_h, tensor.bbox_w

    # Tensor's internal centroid (within its bounding box)
    mask_ys, mask_xs = np.where(tensor.shape_mask)
    if len(mask_ys) == 0:
        return -1e18, 0
    bbox_cy = mask_ys.mean()
    bbox_cx = mask_xs.mean()

    # Top-left of bbox in frame coords.
    # Use floor (not round) to avoid alignment errors with even-sized bboxes.
    # For a 10x10 bbox, internal centroid is 4.5. At frame position 12,
    # floor(12-4.5)=7 is correct; round(12-4.5)=round(7.5)=8 is wrong.
    top = int(np.floor(cy - bbox_cy))
    left = int(np.floor(cx - bbox_cx))

    # Overlap region
    src_y0 = max(0, -top)
    src_x0 = max(0, -left)
    src_y1 = min(th, H - top)
    src_x1 = min(tw, W - left)

    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return -1e18, 0

    dst_y0 = top + src_y0
    dst_x0 = left + src_x0
    dst_y1 = top + src_y1
    dst_x1 = left + src_x1

    t_patch = tensor.shape_pixels[src_y0:src_y1, src_x0:src_x1]
    t_mask = tensor.shape_mask[src_y0:src_y1, src_x0:src_x1]
    f_patch = frame[dst_y0:dst_y1, dst_x0:dst_x1]

    n_valid = int(t_mask.sum())
    if n_valid == 0:
        return -1e18, 0

    diff = t_patch[t_mask].astype(np.float64) - f_patch[t_mask].astype(np.float64)
    mean_ssd = np.sum(diff ** 2) / n_valid
    return -mean_ssd, n_valid


def local_search(tensor, frame, search_radius=7):
    """Grid search within ±search_radius of current centroid.

    The search area scales with expected movement, NOT with image size.
    Returns (best_x, best_y, best_score, scores_map).
    """
    H, W = frame.shape[:2]
    cx_int = int(round(tensor.centroid[0]))
    cy_int = int(round(tensor.centroid[1]))

    best_score = -1e18
    best_pos = (cx_int, cy_int)
    scores_map = {}

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            nx = cx_int + dx
            ny = cy_int + dy
            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                continue
            score, n_valid = compute_match_score(tensor, frame, ny, nx)
            scores_map[(nx, ny)] = score
            if score > best_score and n_valid > 0:
                best_score = score
                best_pos = (nx, ny)

    return best_pos[0], best_pos[1], best_score, scores_map


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def place_tensor_on_canvas(canvas, tensor, cx, cy):
    """Place tensor's masked pixels on canvas, centroid aligned to (cx, cy)."""
    H, W = canvas.shape[:2]
    th, tw = tensor.bbox_h, tensor.bbox_w

    mask_ys, mask_xs = np.where(tensor.shape_mask)
    if len(mask_ys) == 0:
        return
    bbox_cy = mask_ys.mean()
    bbox_cx = mask_xs.mean()

    top = int(np.floor(cy - bbox_cy))
    left = int(np.floor(cx - bbox_cx))

    src_y0 = max(0, -top)
    src_x0 = max(0, -left)
    src_y1 = min(th, H - top)
    src_x1 = min(tw, W - left)

    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return

    dst_y0 = top + src_y0
    dst_x0 = left + src_x0
    dst_y1 = top + src_y1
    dst_x1 = left + src_x1

    t_patch = tensor.shape_pixels[src_y0:src_y1, src_x0:src_x1]
    t_mask = tensor.shape_mask[src_y0:src_y1, src_x0:src_x1]

    region = canvas[dst_y0:dst_y1, dst_x0:dst_x1]
    region[t_mask] = t_patch[t_mask]


def reconstruct_from_hand(hand, frame_shape):
    """Reconstruct a frame from the current hand of tensors.

    The canvas is pre-filled with the largest tensor's mean color
    (the most compressive single explanation for the whole image).
    Then tensors are layered: largest first, smallest on top.
    """
    H, W = frame_shape[:2]

    # Sort by pixel count descending (largest = background first)
    sorted_tensors = sorted(hand, key=lambda t: t.pixel_count, reverse=True)

    # Pre-fill canvas with the largest tensor's color.
    # This is principled: the background tensor's color is the single value
    # that best explains the entire canvas. The "hole" left by objects in
    # frame 1 is filled by this color, which IS the correct background.
    if sorted_tensors:
        canvas = np.full((H, W, 3), sorted_tensors[0].color, dtype=np.float64)
    else:
        canvas = np.zeros((H, W, 3), dtype=np.float64)

    # Layer tensors: largest (background) first, then objects on top
    for tensor in sorted_tensors:
        place_tensor_on_canvas(canvas, tensor, tensor.centroid[0], tensor.centroid[1])

    return canvas


# ---------------------------------------------------------------------------
# Main sequential processing loop
# ---------------------------------------------------------------------------

def process_video_sequentially(
    frames,
    video_name,
    color_threshold=30.0,
    min_region_size=5,
    search_radius=7,
    residual_threshold=30.0,
):
    """Process a video frame by frame, building tensors sequentially.

    Returns dict with results.
    """
    N, H, W, C = frames.shape
    frames_f = frames.astype(np.float64)

    hand = []
    frame_results = []
    search_viz_data = None

    for t in range(N):
        frame = frames_f[t]

        if t == 0:
            # --- Frame 1: hand is empty → segment by color ---
            regions = segment_by_color(frame, color_threshold, min_region_size)

            for region in regions:
                tensor = create_tensor_from_region(frame, region)
                hand.append(tensor)

            recon = reconstruct_from_hand(hand, (H, W, C))
            residual = frame - recon
            total_var = np.var(frame)
            var_explained = 1.0 - np.var(residual) / total_var if total_var > 0 else 1.0

            frame_results.append({
                'frame': t,
                'n_tensors': len(hand),
                'mse': np.mean(residual ** 2),
                'var_explained': var_explained,
                'new_tensors': len(hand),
                'recon': recon.copy(),
                'residual': residual.copy(),
            })

        else:
            # --- Frame 2+: match existing tensors via local search ---
            new_tensors_this_frame = 0

            for tensor in hand:
                bx, by, score, scores_map = local_search(
                    tensor, frame, search_radius=search_radius
                )

                # Save search viz for one object tensor
                if (search_viz_data is None and t >= 1
                        and tensor.pixel_count < H * W * 0.5):
                    search_viz_data = {
                        'tensor_id': tensor.id,
                        'old_centroid': tensor.centroid.copy(),
                        'new_centroid': np.array([bx, by], dtype=np.float64),
                        'scores_map': scores_map,
                        'frame': frame.copy(),
                        'frame_idx': t,
                    }

                tensor.centroid = np.array([bx, by], dtype=np.float64)
                tensor.centroid_history.append(tensor.centroid.copy())
                tensor.age += 1

            # Reconstruct with updated positions
            recon = reconstruct_from_hand(hand, (H, W, C))
            residual = frame - recon

            # Check for large unexplained regions
            residual_mag = np.sqrt(np.sum(residual ** 2, axis=-1))
            large_residual_mask = residual_mag > residual_threshold

            if large_residual_mask.sum() > min_region_size:
                # Segment the unexplained part of the ACTUAL frame
                labeled, n_labels = ndimage.label(large_residual_mask)
                for label_id in range(1, n_labels + 1):
                    blob_mask = (labeled == label_id)
                    blob_size = blob_mask.sum()
                    if blob_size < min_region_size:
                        continue

                    mean_color = frame[blob_mask].mean(axis=0)

                    # Skip if already covered by an existing tensor of similar color
                    already_covered = False
                    for existing in hand:
                        color_dist = np.sqrt(np.sum(
                            (existing.color - mean_color) ** 2))
                        if color_dist < color_threshold:
                            already_covered = True
                            break
                    if already_covered:
                        continue

                    ys, xs = np.where(blob_mask)
                    region = {
                        'mask': blob_mask,
                        'color': mean_color,
                        'centroid': np.array([xs.mean(), ys.mean()]),
                        'pixel_count': int(blob_size),
                    }
                    new_tensor = create_tensor_from_region(frame, region)
                    hand.append(new_tensor)
                    new_tensors_this_frame += 1

                if new_tensors_this_frame > 0:
                    recon = reconstruct_from_hand(hand, (H, W, C))
                    residual = frame - recon

            total_var = np.var(frame)
            var_explained = (1.0 - np.var(residual) / total_var
                            if total_var > 0 else 1.0)

            frame_results.append({
                'frame': t,
                'n_tensors': len(hand),
                'mse': np.mean(residual ** 2),
                'var_explained': var_explained,
                'new_tensors': new_tensors_this_frame,
                'recon': recon.copy(),
                'residual': residual.copy(),
            })

    return {
        'hand': hand,
        'frame_results': frame_results,
        'search_viz_data': search_viz_data,
        'video_name': video_name,
        'frames': frames_f,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_reconstruction(result, output_dir):
    """3-column visualization: actual | reconstruction | residual."""
    frames = result['frames']
    fr = result['frame_results']
    name = result['video_name']
    N = len(frames)

    n_show = min(10, N)
    indices = np.linspace(0, N - 1, n_show, dtype=int)

    fig, axes = plt.subplots(3, n_show, figsize=(2.2 * n_show, 7))
    if n_show == 1:
        axes = axes.reshape(3, 1)

    for col, idx in enumerate(indices):
        actual = frames[idx]
        recon = fr[idx]['recon']
        residual = fr[idx]['residual']

        axes[0, col].imshow(np.clip(actual, 0, 255).astype(np.uint8))
        axes[0, col].set_title(f't={idx}', fontsize=8)
        axes[0, col].axis('off')

        axes[1, col].imshow(np.clip(recon, 0, 255).astype(np.uint8))
        ve = fr[idx]['var_explained']
        axes[1, col].set_title(f'VE={ve:.3f}', fontsize=7)
        axes[1, col].axis('off')

        res_mag = np.sqrt(np.sum(residual ** 2, axis=-1))
        axes[2, col].imshow(res_mag, cmap='hot', vmin=0,
                            vmax=max(res_mag.max(), 1))
        axes[2, col].set_title(f'MSE={fr[idx]["mse"]:.1f}', fontsize=7)
        axes[2, col].axis('off')

    axes[0, 0].set_ylabel('Actual', fontsize=10)
    axes[1, 0].set_ylabel('Recon', fontsize=10)
    axes[2, 0].set_ylabel('Residual', fontsize=10)

    fig.suptitle(f'{name} — Sequential Processing (Exp.8)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_exp8_recon.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_tensor_inventory(result, output_dir):
    """Show all tensors in the hand with their shapes, colors, and metadata."""
    hand = result['hand']
    name = result['video_name']

    if not hand:
        return

    n = len(hand)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    if n == 1:
        axes = [axes]

    for i, tensor in enumerate(hand):
        display = np.full((tensor.bbox_h, tensor.bbox_w, 3), 128, dtype=np.uint8)
        display[tensor.shape_mask] = np.clip(
            tensor.shape_pixels[tensor.shape_mask], 0, 255
        ).astype(np.uint8)

        axes[i].imshow(display)
        axes[i].set_title(
            f'T{tensor.id}\n{tensor.pixel_count}px\nage={tensor.age}',
            fontsize=7
        )
        axes[i].axis('off')

    fig.suptitle(f'{name} — Tensor Inventory', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_exp8_tensors.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_trajectories(result, output_dir):
    """Plot centroid trajectories of object tensors over time."""
    hand = result['hand']
    frames = result['frames']
    name = result['video_name']
    H, W = frames.shape[1], frames.shape[2]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(np.clip(frames[0], 0, 255).astype(np.uint8), alpha=0.3)

    colors_list = plt.cm.tab10(np.linspace(0, 1, max(len(hand), 1)))

    for i, tensor in enumerate(hand):
        history = np.array(tensor.centroid_history)
        if len(history) < 2 or tensor.pixel_count > H * W * 0.5:
            continue
        ax.plot(history[:, 0], history[:, 1], '-o', color=colors_list[i],
                markersize=2, linewidth=1.5,
                label=f'T{tensor.id} ({tensor.pixel_count}px)')
        ax.plot(history[0, 0], history[0, 1], 's',
                color=colors_list[i], markersize=6)
        ax.plot(history[-1, 0], history[-1, 1], '*',
                color=colors_list[i], markersize=8)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title(f'{name} — Tensor Trajectories', fontsize=11)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_exp8_trajectories.png', dpi=150,
                bbox_inches='tight')
    plt.close()


def visualize_search_process(result, output_dir):
    """Visualize the local search process for one frame."""
    viz = result.get('search_viz_data')
    if viz is None:
        return

    name = result['video_name']
    frame = viz['frame']
    scores_map = viz['scores_map']
    old_c = viz['old_centroid']
    new_c = viz['new_centroid']
    t_id = viz['tensor_id']
    frame_idx = viz['frame_idx']
    H, W = frame.shape[:2]

    # Build scores image
    scores_img = np.full((H, W), np.nan)
    for (x, y), score in scores_map.items():
        if 0 <= y < H and 0 <= x < W:
            scores_img[y, x] = score

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Frame with centroids
    axes[0].imshow(np.clip(frame, 0, 255).astype(np.uint8))
    axes[0].plot(old_c[0], old_c[1], 'ro', markersize=8, label='prev')
    axes[0].plot(new_c[0], new_c[1], 'g*', markersize=10, label='new')
    axes[0].set_title(f'Frame {frame_idx} (T{t_id})', fontsize=10)
    axes[0].legend(fontsize=7)
    axes[0].axis('off')

    # Zoomed scores heatmap
    cx_int = int(round(old_c[0]))
    cy_int = int(round(old_c[1]))
    r = 10
    y0, y1 = max(0, cy_int - r), min(H, cy_int + r + 1)
    x0, x1 = max(0, cx_int - r), min(W, cx_int + r + 1)

    zoomed = scores_img[y0:y1, x0:x1]
    valid = zoomed[~np.isnan(zoomed)]

    if len(valid) > 0:
        vmin, vmax = np.percentile(valid, 5), np.percentile(valid, 95)
        im = axes[1].imshow(zoomed, cmap='RdYlGn', vmin=vmin, vmax=vmax,
                            extent=[x0, x1, y1, y0])
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        axes[1].plot(old_c[0], old_c[1], 'ro', markersize=6)
        axes[1].plot(new_c[0], new_c[1], 'g*', markersize=8)
    axes[1].set_title('Match scores (zoomed)', fontsize=10)

    # Search coverage
    coverage = np.zeros((H, W), dtype=np.uint8)
    for (x, y) in scores_map.keys():
        if 0 <= y < H and 0 <= x < W:
            coverage[y, x] = 255
    axes[2].imshow(coverage[y0:y1, x0:x1], cmap='gray',
                   extent=[x0, x1, y1, y0])
    axes[2].plot(old_c[0], old_c[1], 'ro', markersize=6)
    axes[2].plot(new_c[0], new_c[1], 'g*', markersize=8)
    axes[2].set_title(f'Coverage ({len(scores_map)} pos)', fontsize=10)

    fig.suptitle(
        f'{name} — Local Search (Frame {frame_idx}, T{t_id})', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_exp8_search.png', dpi=150,
                bbox_inches='tight')
    plt.close()


def visualize_metrics_over_time(result, output_dir):
    """Plot var_explained and tensor count over frames."""
    fr = result['frame_results']
    name = result['video_name']

    frames_idx = [r['frame'] for r in fr]
    var_ex = [r['var_explained'] for r in fr]
    n_tensors = [r['n_tensors'] for r in fr]
    new_t = [r['new_tensors'] for r in fr]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    ax1.plot(frames_idx, var_ex, 'b-o', markersize=3)
    ax1.set_ylabel('Variance Explained')
    ax1.set_ylim(min(0, min(var_ex) - 0.05), 1.05)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax1.set_title(f'{name} — Metrics Over Time (Exp.8)', fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(frames_idx, n_tensors, 'r-o', markersize=3, label='total')
    new_frames = [frames_idx[i] for i in range(len(new_t)) if new_t[i] > 0]
    new_counts = [new_t[i] for i in range(len(new_t)) if new_t[i] > 0]
    if new_frames:
        ax2.bar(new_frames, new_counts, alpha=0.4, color='green', label='new')
    ax2.set_ylabel('Tensor Count')
    ax2.set_xlabel('Frame')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_exp8_metrics.png', dpi=150,
                bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = Path(__file__).parent / 'data'
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    video_names = [
        'v1_triangle_right',
        'v2_square_diag',
        'v3_circle_down',
        'v4_two_objects',
    ]

    print("=" * 70)
    print("IS Experiment 8: Sequential Processing")
    print("  IS's learning loop — first implementation")
    print("=" * 70)
    print()
    print("Operations given externally:")
    print("  - Color-based region segmentation (flood fill + color threshold)")
    print("  - Centroid computation (mean of pixel coordinates)")
    print()

    all_results = {}

    for vname in video_names:
        path = data_dir / f'{vname}.npy'
        if not path.exists():
            print(f"  {vname}: file not found, skipping")
            continue

        frames = np.load(path)
        print(f"--- {vname} ({frames.shape}) ---")

        Tensor._next_id = 0

        result = process_video_sequentially(
            frames, vname,
            color_threshold=30.0,
            min_region_size=5,
            search_radius=7,
            residual_threshold=30.0,
        )

        all_results[vname] = result

        # Print summary
        fr = result['frame_results']
        hand = result['hand']

        print(f"  Tensors in hand: {len(hand)}")
        for tensor in hand:
            print(f"    T{tensor.id}: {tensor.pixel_count}px, "
                  f"color=({tensor.color[0]:.0f},{tensor.color[1]:.0f},"
                  f"{tensor.color[2]:.0f}), "
                  f"bbox={tensor.bbox_h}x{tensor.bbox_w}, age={tensor.age}")

        final_ve = fr[-1]['var_explained']
        mean_ve = np.mean([r['var_explained'] for r in fr])
        print(f"  Final VE: {final_ve:.4f}")
        print(f"  Mean VE:  {mean_ve:.4f}")
        print(f"  Final MSE: {fr[-1]['mse']:.2f}")

        H, W = frames.shape[1], frames.shape[2]
        obj_tensors = [t for t in hand if t.pixel_count < H * W * 0.5]
        print(f"  Object tensors: {len(obj_tensors)}")
        for ot in obj_tensors:
            hist = np.array(ot.centroid_history)
            if len(hist) >= 2:
                n_steps = len(hist) - 1
                if n_steps > 0:
                    dx = (hist[-1, 0] - hist[0, 0]) / n_steps
                    dy = (hist[-1, 1] - hist[0, 1]) / n_steps
                    print(f"    T{ot.id}: velocity ~ ({dx:.2f}, {dy:.2f})")

        print()

        # Generate all visualizations
        visualize_reconstruction(result, output_dir)
        visualize_tensor_inventory(result, output_dir)
        visualize_trajectories(result, output_dir)
        visualize_search_process(result, output_dir)
        visualize_metrics_over_time(result, output_dir)

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY: Exp.8 (sequential) vs Exp.4 (all-frames-at-once)")
    print("=" * 70)
    exp4_ve = {
        'v1_triangle_right': 1.0000,
        'v2_square_diag': 1.0000,
        'v3_circle_down': 1.0000,
        'v4_two_objects': 1.0000,
    }
    print(f"{'Video':<22} {'Exp8 FinalVE':>12} {'Exp8 MeanVE':>12} "
          f"{'Exp4 VE':>8} {'#Tensors':>9}")
    print("-" * 70)

    for vname in video_names:
        if vname not in all_results:
            continue
        fr = all_results[vname]['frame_results']
        final_ve = fr[-1]['var_explained']
        mean_ve = np.mean([r['var_explained'] for r in fr])
        n_t = len(all_results[vname]['hand'])
        e4 = exp4_ve.get(vname, '-')
        print(f"{vname:<22} {final_ve:>12.4f} {mean_ve:>12.4f} "
              f"{e4:>8.4f} {n_t:>9}")

    print()
    print("Exp.4 had access to all 30 frames simultaneously.")
    print("Exp.8 processes one frame at a time; raw pixels are discarded.")


if __name__ == '__main__':
    main()
