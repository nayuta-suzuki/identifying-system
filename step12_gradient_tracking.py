"""
IS Experiment 8b: Gradient-Based Tracking — Alignment with IS Design Principle

This experiment replaces the grid search (±7px) in Experiment 8 with a
gradient-based local search, aligning the implementation with IS's design
principle:

  Tensor position updates use gradient descent from the previous position.
  Probe 2 linearly independent directions (right and down) to compute the
  gradient, then follow steepest descent until convergence. No external
  search_radius parameter is needed.

Changes from Experiment 8 (step11_sequential_processing.py):
  - local_search (±7px grid) → gradient_search (gradient-based, no radius)
  - Videos are 9 frames (reduced from 30); all other parameters identical
  - Everything else (segmentation, tensor generation, reconstruction,
    evaluation) is unchanged

Author: IS Project (Experiment 8b)
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage


# ---------------------------------------------------------------------------
# Video generation (same parameters as generate_videos.py, n_frames=9)
# ---------------------------------------------------------------------------

def draw_triangle(canvas, cx, cy, size, color):
    """Draw a filled equilateral triangle centered at (cx, cy)."""
    h, w = canvas.shape[:2]
    half = size // 2
    for y in range(max(0, cy - half), min(h, cy + half + 1)):
        t = (y - (cy - half)) / (size) if size > 0 else 0
        t = np.clip(t, 0, 1)
        x_left = cx - half * t
        x_right = cx + half * t
        for x in range(max(0, int(x_left)), min(w, int(x_right) + 1)):
            canvas[y, x] = color


def draw_square(canvas, cx, cy, size, color):
    """Draw a filled square centered at (cx, cy)."""
    h, w = canvas.shape[:2]
    half = size // 2
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    canvas[y0:y1, x0:x1] = color


def draw_circle(canvas, cx, cy, radius, color):
    """Draw a filled circle centered at (cx, cy)."""
    h, w = canvas.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    canvas[mask] = color


SHAPE_DRAWERS = {
    'triangle': draw_triangle,
    'square': draw_square,
    'circle': draw_circle,
}


def generate_video(shape_name, bg_color, obj_color, start_pos, velocity,
                   obj_size=10, n_frames=9, frame_size=64):
    """Generate a single-object video (identical to generate_videos.py)."""
    frames = np.zeros((n_frames, frame_size, frame_size, 3), dtype=np.uint8)
    draw_fn = SHAPE_DRAWERS[shape_name]
    cx, cy = start_pos
    vx, vy = velocity
    for t in range(n_frames):
        frame = np.full((frame_size, frame_size, 3), bg_color, dtype=np.uint8)
        draw_fn(frame, int(cx + vx * t), int(cy + vy * t), obj_size, obj_color)
        frames[t] = frame
    return frames


def generate_two_object_video(shapes, bg_color, n_frames=9, frame_size=64):
    """Generate a two-object video (identical to generate_videos.py)."""
    frames = np.zeros((n_frames, frame_size, frame_size, 3), dtype=np.uint8)
    for t in range(n_frames):
        frame = np.full((frame_size, frame_size, 3), bg_color, dtype=np.uint8)
        for s in shapes:
            draw_fn = SHAPE_DRAWERS[s['name']]
            cx = int(s['start_pos'][0] + s['velocity'][0] * t)
            cy = int(s['start_pos'][1] + s['velocity'][1] * t)
            draw_fn(frame, cx, cy, s['size'], s['color'])
        frames[t] = frame
    return frames


def generate_all_videos():
    """Generate all 4 videos with 9 frames. Parameters identical to Exp.8."""
    bg_blue = np.array([40, 60, 120], dtype=np.uint8)
    videos = {}
    true_velocities = {}

    videos['v1_triangle_right'] = generate_video(
        shape_name='triangle', bg_color=bg_blue,
        obj_color=np.array([240, 240, 240], dtype=np.uint8),
        start_pos=(10, 32), velocity=(1.5, 0), obj_size=12,
    )
    true_velocities['v1_triangle_right'] = [(1.5, 0)]

    videos['v2_square_diag'] = generate_video(
        shape_name='square', bg_color=bg_blue,
        obj_color=np.array([220, 60, 60], dtype=np.uint8),
        start_pos=(10, 10), velocity=(1.2, 1.0), obj_size=10,
    )
    true_velocities['v2_square_diag'] = [(1.2, 1.0)]

    videos['v3_circle_down'] = generate_video(
        shape_name='circle', bg_color=bg_blue,
        obj_color=np.array([60, 200, 80], dtype=np.uint8),
        start_pos=(32, 8), velocity=(0, 1.5), obj_size=7,
    )
    true_velocities['v3_circle_down'] = [(0, 1.5)]

    videos['v4_two_objects'] = generate_two_object_video(
        shapes=[
            {'name': 'triangle', 'start_pos': (10, 20), 'velocity': (1.5, 0.3),
             'size': 10, 'color': np.array([240, 240, 240], dtype=np.uint8)},
            {'name': 'circle', 'start_pos': (50, 50), 'velocity': (-1.0, -0.5),
             'size': 7, 'color': np.array([240, 200, 60], dtype=np.uint8)},
        ],
        bg_color=bg_blue,
    )
    true_velocities['v4_two_objects'] = [(1.5, 0.3), (-1.0, -0.5)]

    return videos, true_velocities


# ---------------------------------------------------------------------------
# Tensor representation (identical to step11)
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
# Color-based segmentation (identical to step11)
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
# Tensor matching — score computation (identical to step11)
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


# ---------------------------------------------------------------------------
# Tensor matching — gradient-based search (REPLACES local_search from step11)
# ---------------------------------------------------------------------------

def gradient_search(tensor, frame):
    """Gradient-based local search for tensor position update.

    IS design principle: probe 2 linearly independent directions (right, down)
    to compute gradient, then walk each axis independently until convergence.
    No external search_radius parameter is needed.

    Algorithm:
      1. Check if tensor hasn't moved (SSD=0 at current position → done).
      2. Compute gradient by probing (x+1, y) and (x, y+1).
      3. Determine axis directions dx, dy from gradient.
      4. X-axis line search: step (dx, 0) until score worsens.
      5. Y-axis line search: step (0, dy) until score worsens.
      6. If either axis improved → recompute gradient (step 2).
         If neither improved → stop.

    Each axis is walked independently. This handles arbitrary x:y movement
    ratios — unlike diagonal stepping which locks to 1:1.
    2 gradient probes (right and down) per iteration suffice because the
    axes are linearly independent and walked separately.

    Returns (best_x, best_y, best_score, n_evaluations, path).
    """
    H, W = frame.shape[:2]
    cx = int(round(tensor.centroid[0]))
    cy = int(round(tensor.centroid[1]))

    n_evals = 0
    path = [(cx, cy)]

    # Step 1: Check if tensor hasn't moved (SSD ≈ 0 at current position)
    score_here, n_valid = compute_match_score(tensor, frame, cy, cx)
    n_evals += 1

    if n_valid == 0:
        return cx, cy, score_here, n_evals, path

    if score_here >= -1e-6:  # SSD ≈ 0 → tensor hasn't moved
        return cx, cy, score_here, n_evals, path

    best_x, best_y = cx, cy
    best_score = score_here

    max_outer_iters = 50  # Safety limit

    for _ in range(max_outer_iters):
        # Step 2: Compute gradient — probe right (+x) and down (+y)
        score_right, _ = compute_match_score(tensor, frame, best_y, best_x + 1)
        score_down, _ = compute_match_score(tensor, frame, best_y + 1, best_x)
        n_evals += 2

        # Gradient in score space (higher = better = less negative SSD)
        delta_x = score_right - best_score  # positive → right improves
        delta_y = score_down - best_score    # positive → down improves

        # Axis directions from gradient
        dx = 1 if delta_x > 0 else (-1 if delta_x < 0 else 0)
        dy = 1 if delta_y > 0 else (-1 if delta_y < 0 else 0)

        if dx == 0 and dy == 0:
            break  # Gradient is zero → converged

        improved_this_iteration = False

        # Step 4: X-axis line search — step (dx, 0), y stays fixed
        if dx != 0:
            for _ in range(64):
                nx = best_x + dx
                if nx < 0 or nx >= W:
                    break

                score_new, nv = compute_match_score(tensor, frame, best_y, nx)
                n_evals += 1

                if score_new > best_score and nv > 0:
                    best_x = nx
                    best_score = score_new
                    path.append((best_x, best_y))
                    improved_this_iteration = True

                    if best_score >= -1e-6:
                        break
                else:
                    break

        if best_score >= -1e-6:
            break

        # Step 5: Y-axis line search — step (0, dy), x stays fixed
        #         (x is already updated from the x-axis search above)
        if dy != 0:
            for _ in range(64):
                ny = best_y + dy
                if ny < 0 or ny >= H:
                    break

                score_new, nv = compute_match_score(tensor, frame, ny, best_x)
                n_evals += 1

                if score_new > best_score and nv > 0:
                    best_y = ny
                    best_score = score_new
                    path.append((best_x, best_y))
                    improved_this_iteration = True

                    if best_score >= -1e-6:
                        break
                else:
                    break

        if best_score >= -1e-6:
            break

        # Step 6: If neither axis improved → converged
        if not improved_this_iteration:
            break

        # Otherwise → recompute gradient at new position

    return best_x, best_y, best_score, n_evals, path


# ---------------------------------------------------------------------------
# Reconstruction (identical to step11)
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
    residual_threshold=30.0,
):
    """Process a video frame by frame, building tensors sequentially.

    Uses gradient-based search instead of grid search for position updates.
    Returns dict with results.
    """
    N, H, W, C = frames.shape
    frames_f = frames.astype(np.float64)

    hand = []
    frame_results = []
    search_stats = []   # Track gradient search stats per frame
    search_paths = {}   # Track search paths for visualization

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
            search_stats.append({
                'frame': t,
                'tensor_evals': {},
                'total_evals': 0,
            })

        else:
            # --- Frame 2+: match existing tensors via gradient search ---
            new_tensors_this_frame = 0
            frame_evals = {}
            frame_paths = {}

            for tensor in hand:
                bx, by, score, n_evals, path = gradient_search(tensor, frame)
                frame_evals[tensor.id] = n_evals
                frame_paths[tensor.id] = path

                tensor.centroid = np.array([bx, by], dtype=np.float64)
                tensor.centroid_history.append(tensor.centroid.copy())
                tensor.age += 1

            search_paths[t] = frame_paths

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

                    # Skip if already covered by an existing tensor
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

            total_frame_evals = sum(frame_evals.values())
            search_stats.append({
                'frame': t,
                'tensor_evals': dict(frame_evals),
                'total_evals': total_frame_evals,
            })

    return {
        'hand': hand,
        'frame_results': frame_results,
        'search_stats': search_stats,
        'search_paths': search_paths,
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

    n_show = min(9, N)
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

    fig.suptitle(f'{name} — Gradient Tracking (Exp.8b)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_exp8b_recon.png', dpi=150,
                bbox_inches='tight')
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

    fig.suptitle(f'{name} — Tensor Inventory (Exp.8b)', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_exp8b_tensors.png', dpi=150,
                bbox_inches='tight')
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
    ax.set_title(f'{name} — Tensor Trajectories (Exp.8b)', fontsize=11)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_exp8b_trajectories.png', dpi=150,
                bbox_inches='tight')
    plt.close()


def visualize_metrics_over_time(result, output_dir):
    """Plot var_explained, tensor count, and gradient search evals."""
    fr = result['frame_results']
    ss = result['search_stats']
    name = result['video_name']

    frames_idx = [r['frame'] for r in fr]
    var_ex = [r['var_explained'] for r in fr]
    n_tensors = [r['n_tensors'] for r in fr]
    total_evals = [s['total_evals'] for s in ss]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

    ax1.plot(frames_idx, var_ex, 'b-o', markersize=3)
    ax1.set_ylabel('Variance Explained')
    ax1.set_ylim(min(0, min(var_ex) - 0.05), 1.05)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax1.set_title(f'{name} — Metrics Over Time (Exp.8b)', fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(frames_idx, n_tensors, 'r-o', markersize=3)
    ax2.set_ylabel('Tensor Count')
    ax2.grid(True, alpha=0.3)

    ax3.bar(frames_idx, total_evals, color='steelblue', alpha=0.7)
    ax3.set_ylabel('SSD Evaluations')
    ax3.set_xlabel('Frame')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_exp8b_metrics.png', dpi=150,
                bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    # Generate 9-frame videos (same parameters as Exp.8, fewer frames)
    videos, true_velocities = generate_all_videos()

    video_names = [
        'v1_triangle_right',
        'v2_square_diag',
        'v3_circle_down',
        'v4_two_objects',
    ]

    print("=" * 75)
    print("IS Experiment 8b: Gradient-Based Tracking")
    print("  Replaces grid search (±7px) with gradient-based local search")
    print("  Videos: 9 frames (same parameters as Exp.8)")
    print("=" * 75)
    print()
    print("Change from Exp.8:")
    print("  - Grid search (225 SSD evals per tensor per frame)")
    print("    → Gradient search (2-direction gradient + line search)")
    print()

    all_results = {}

    for vname in video_names:
        frames = videos[vname]
        print(f"--- {vname} ({frames.shape}) ---")

        Tensor._next_id = 0

        result = process_video_sequentially(
            frames, vname,
            color_threshold=30.0,
            min_region_size=5,
            residual_threshold=30.0,
        )

        all_results[vname] = result

        # Print summary
        fr = result['frame_results']
        hand = result['hand']
        ss = result['search_stats']

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

        # Print gradient search efficiency
        print()
        print(f"  Gradient search steps (SSD evaluations per frame):")
        n_tensors_at_frame = len(hand)
        grid_per_tensor = (2 * 7 + 1) ** 2  # 225
        for stat in ss:
            t = stat['frame']
            if t == 0:
                print(f"    Frame {t}: -- (initial segmentation, no search)")
                continue
            total = stat['total_evals']
            detail = ", ".join(
                f"T{tid}:{ev}" for tid, ev in sorted(stat['tensor_evals'].items())
            )
            grid_total = grid_per_tensor * len(stat['tensor_evals'])
            print(f"    Frame {t}: {total:>4d} evals ({detail})"
                  f"  [grid would be {grid_total}]")

        total_all_frames = sum(s['total_evals'] for s in ss)
        n_search_frames = sum(1 for s in ss if s['frame'] > 0)
        grid_total_all = grid_per_tensor * n_tensors_at_frame * n_search_frames
        print(f"  Total: {total_all_frames} evals"
              f"  (grid would be {grid_total_all},"
              f" {grid_total_all / max(total_all_frames, 1):.0f}x reduction)")

        print()

        # Generate visualizations
        visualize_reconstruction(result, output_dir)
        visualize_tensor_inventory(result, output_dir)
        visualize_trajectories(result, output_dir)
        visualize_metrics_over_time(result, output_dir)

    # --- Summary table ---
    print("\n" + "=" * 75)
    print("SUMMARY: Exp.8b (gradient) vs Exp.8 (grid)")
    print("=" * 75)
    exp8_ve = {
        'v1_triangle_right': 1.0000,
        'v2_square_diag': 1.0000,
        'v3_circle_down': 1.0000,
        'v4_two_objects': 1.0000,
    }
    exp8_velocities = {
        'v1_triangle_right': '(1.49, 0.00)',
        'v2_square_diag': '(1.19, 1.02)',
        'v3_circle_down': '(0.00, 1.48)',
        'v4_two_objects': '(1.49,0.29) (-1.00,-0.52)',
    }

    print(f"{'Video':<22} {'8b FinalVE':>10} {'8b MeanVE':>10} "
          f"{'8 VE':>6} {'#T':>4}  {'8b velocity':>30}  {'8 velocity':>30}")
    print("-" * 120)

    for vname in video_names:
        if vname not in all_results:
            continue
        fr = all_results[vname]['frame_results']
        hand = all_results[vname]['hand']
        final_ve = fr[-1]['var_explained']
        mean_ve = np.mean([r['var_explained'] for r in fr])
        n_t = len(hand)
        e8 = exp8_ve.get(vname, '-')

        H, W = videos[vname].shape[1], videos[vname].shape[2]
        obj_ts = [t for t in hand if t.pixel_count < H * W * 0.5]
        vel_strs = []
        for ot in obj_ts:
            hist = np.array(ot.centroid_history)
            if len(hist) >= 2:
                ns = len(hist) - 1
                ddx = (hist[-1, 0] - hist[0, 0]) / ns
                ddy = (hist[-1, 1] - hist[0, 1]) / ns
                vel_strs.append(f"({ddx:.2f},{ddy:.2f})")
        vel_8b = " ".join(vel_strs) if vel_strs else "-"
        vel_8 = exp8_velocities.get(vname, '-')

        print(f"{vname:<22} {final_ve:>10.4f} {mean_ve:>10.4f} "
              f"{e8:>6.4f} {n_t:>4}  {vel_8b:>30}  {vel_8:>30}")

    # True velocities
    print()
    print("True velocities:")
    for vname in video_names:
        vels = true_velocities[vname]
        vel_str = " ".join(f"({v[0]:.1f},{v[1]:.1f})" for v in vels)
        print(f"  {vname}: {vel_str}")

    print()
    print("Exp.8 used ±7px grid search (225 evals/tensor/frame), 30 frames.")
    print("Exp.8b uses gradient-based search (no radius parameter), 9 frames.")


if __name__ == '__main__':
    main()
