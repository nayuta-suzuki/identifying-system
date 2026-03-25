"""
IS Experiment 9: New Tensor from Residual

First verification of IS's core learning loop:
  "Residuals that cannot be explained by existing tensors generate new tensors."

In experiments 8-8b, all objects existed from frame 1, so the new-tensor-
from-residual path never fired. Experiment 9 introduces object B at frame 4,
forcing the system to detect unexplained residuals and generate a new tensor
mid-video.

Algorithm (per frame, t >= 1):
  1. Update all existing tensors via gradient search (independent-axis, Exp.8b)
  2. Reconstruct with updated positions
  3. Compute residual (actual frame - reconstruction)
  4. For regions with large residuals: apply color-based segmentation to the
     actual frame → generate new tensors → add to hand
  5. Re-reconstruct and evaluate

The new-tensor generation uses the same operation as frame 1: color-based
region detection + create_tensor_from_region. No new algorithm is introduced.

Base code: step12_gradient_tracking.py (Exp.8b)
Change: video only (object B appears at frame 4). Processing loop unchanged.

Author: IS Project (Experiment 9)
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage


# ---------------------------------------------------------------------------
# Shape drawing (identical to step12)
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


# ---------------------------------------------------------------------------
# Video generation — Experiment 9 specific
# ---------------------------------------------------------------------------

def generate_experiment9_video():
    """Generate a video where object B appears at frame 4.

    Object A (white triangle): present from frame 0.
      start=(10, 32), velocity=(1.5, 0), size=12
      Same as Exp.8 v1.

    Object B (red square): appears at frame 4.
      position at frame 4 = (48, 15), velocity=(-1.0, 0.5), size=8

    No overlap between A and B across all 9 frames:
      A occupies x:4-28, y:26-38  (across all frames)
      B occupies x:40-52, y:11-21 (across frames 4-8)
    """
    bg_color = np.array([40, 60, 120], dtype=np.uint8)
    n_frames = 9
    frame_size = 64

    # Object A parameters
    a_start = (10, 32)
    a_vel = (1.5, 0.0)
    a_size = 12
    a_color = np.array([240, 240, 240], dtype=np.uint8)

    # Object B parameters
    b_pos_at_appear = (48, 15)   # position when it first appears
    b_vel = (-1.0, 0.5)
    b_size = 8
    b_color = np.array([200, 50, 50], dtype=np.uint8)
    b_appear_frame = 4

    frames = np.zeros((n_frames, frame_size, frame_size, 3), dtype=np.uint8)

    for t in range(n_frames):
        frame = np.full((frame_size, frame_size, 3), bg_color, dtype=np.uint8)

        # Draw A (present on all frames)
        ax = int(a_start[0] + a_vel[0] * t)
        ay = int(a_start[1] + a_vel[1] * t)
        draw_triangle(frame, ax, ay, a_size, a_color)

        # Draw B (only from frame 4 onwards)
        if t >= b_appear_frame:
            dt = t - b_appear_frame
            bx = int(b_pos_at_appear[0] + b_vel[0] * dt)
            by = int(b_pos_at_appear[1] + b_vel[1] * dt)
            draw_square(frame, bx, by, b_size, b_color)

        frames[t] = frame

    video_info = {
        'a_vel': a_vel,
        'b_vel': b_vel,
        'b_appear_frame': b_appear_frame,
        'bg_color': bg_color,
    }

    return frames, video_info


# ---------------------------------------------------------------------------
# Tensor representation (identical to step12)
# ---------------------------------------------------------------------------

class Tensor:
    """A single tensor in the IS hand."""

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
        self.born_at_frame = -1  # set externally

    @property
    def bbox_h(self):
        return self.shape_mask.shape[0]

    @property
    def bbox_w(self):
        return self.shape_mask.shape[1]


# ---------------------------------------------------------------------------
# Color-based segmentation (identical to step12)
# ---------------------------------------------------------------------------

def segment_by_color(frame, color_threshold=30.0, min_region_size=5):
    """Segment a frame into regions of similar color using flood-fill."""
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
# Tensor matching (identical to step12)
# ---------------------------------------------------------------------------

def compute_match_score(tensor, frame, cy, cx):
    """Compute match score: -mean_SSD. Higher = better."""
    H, W = frame.shape[:2]
    th, tw = tensor.bbox_h, tensor.bbox_w

    mask_ys, mask_xs = np.where(tensor.shape_mask)
    if len(mask_ys) == 0:
        return -1e18, 0
    bbox_cy = mask_ys.mean()
    bbox_cx = mask_xs.mean()

    top = int(np.floor(cy - bbox_cy))
    left = int(np.floor(cx - bbox_cx))

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


def gradient_search(tensor, frame):
    """Gradient-based local search — independent axis walking (from Exp.8b)."""
    H, W = frame.shape[:2]
    cx = int(round(tensor.centroid[0]))
    cy = int(round(tensor.centroid[1]))

    n_evals = 0
    path = [(cx, cy)]

    score_here, n_valid = compute_match_score(tensor, frame, cy, cx)
    n_evals += 1

    if n_valid == 0:
        return cx, cy, score_here, n_evals, path

    if score_here >= -1e-6:
        return cx, cy, score_here, n_evals, path

    best_x, best_y = cx, cy
    best_score = score_here

    for _ in range(50):
        score_right, _ = compute_match_score(tensor, frame, best_y, best_x + 1)
        score_down, _ = compute_match_score(tensor, frame, best_y + 1, best_x)
        n_evals += 2

        delta_x = score_right - best_score
        delta_y = score_down - best_score

        dx = 1 if delta_x > 0 else (-1 if delta_x < 0 else 0)
        dy = 1 if delta_y > 0 else (-1 if delta_y < 0 else 0)

        if dx == 0 and dy == 0:
            break

        improved_this_iteration = False

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

        if not improved_this_iteration:
            break

    return best_x, best_y, best_score, n_evals, path


# ---------------------------------------------------------------------------
# Reconstruction (identical to step12)
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
    """Reconstruct a frame from the current hand of tensors."""
    H, W = frame_shape[:2]
    sorted_tensors = sorted(hand, key=lambda t: t.pixel_count, reverse=True)

    if sorted_tensors:
        canvas = np.full((H, W, 3), sorted_tensors[0].color, dtype=np.float64)
    else:
        canvas = np.zeros((H, W, 3), dtype=np.float64)

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

    Per-frame processing (t >= 1):
      1. Gradient search to update all existing tensor positions
      2. Reconstruct with updated positions
      3. Find large residual regions (unexplained by existing tensors)
      4. Apply color-based region detection to actual frame in those regions
         → generate new tensors (same operation as frame 1)
      5. Re-reconstruct and evaluate
    """
    N, H, W, C = frames.shape
    frames_f = frames.astype(np.float64)

    hand = []
    frame_results = []
    search_stats = []
    new_tensor_events = []  # Track when new tensors are born

    for t in range(N):
        frame = frames_f[t]

        if t == 0:
            # --- Frame 0: hand is empty → segment by color ---
            regions = segment_by_color(frame, color_threshold, min_region_size)

            for region in regions:
                tensor = create_tensor_from_region(frame, region)
                tensor.born_at_frame = t
                hand.append(tensor)

            recon = reconstruct_from_hand(hand, (H, W, C))
            residual = frame - recon
            total_var = np.var(frame)
            var_explained = 1.0 - np.var(residual) / total_var if total_var > 0 else 1.0

            # Record event
            for tensor in hand:
                new_tensor_events.append({
                    'frame': t,
                    'tensor_id': tensor.id,
                    'pixel_count': tensor.pixel_count,
                    'color': tensor.color.copy(),
                    'source': 'initial segmentation',
                })

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
            # --- Frame t >= 1: gradient search + residual detection ---
            new_tensors_this_frame = 0
            frame_evals = {}

            # Step 1: Update all existing tensor positions via gradient search
            for tensor in hand:
                bx, by, score, n_evals, path = gradient_search(tensor, frame)
                frame_evals[tensor.id] = n_evals

                tensor.centroid = np.array([bx, by], dtype=np.float64)
                tensor.centroid_history.append(tensor.centroid.copy())
                tensor.age += 1

            # Step 2: Reconstruct with updated positions
            recon = reconstruct_from_hand(hand, (H, W, C))
            residual = frame - recon

            # Record VE before new tensor generation (for frame 4 comparison)
            total_var = np.var(frame)
            ve_before = (1.0 - np.var(residual) / total_var
                         if total_var > 0 else 1.0)

            # Step 3-4: Find large residual regions → generate new tensors
            residual_mag = np.sqrt(np.sum(residual ** 2, axis=-1))
            large_residual_mask = residual_mag > residual_threshold

            if large_residual_mask.sum() > min_region_size:
                # Find connected regions of large residual
                labeled, n_labels = ndimage.label(large_residual_mask)
                for label_id in range(1, n_labels + 1):
                    blob_mask = (labeled == label_id)
                    blob_size = blob_mask.sum()
                    if blob_size < min_region_size:
                        continue

                    # Get color from ACTUAL frame (not residual)
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

                    # Create new tensor from actual frame pixels
                    # (same operation as frame 1's create_tensor_from_region)
                    ys, xs = np.where(blob_mask)
                    region = {
                        'mask': blob_mask,
                        'color': mean_color,
                        'centroid': np.array([xs.mean(), ys.mean()]),
                        'pixel_count': int(blob_size),
                    }
                    new_tensor = create_tensor_from_region(frame, region)
                    new_tensor.born_at_frame = t
                    hand.append(new_tensor)
                    new_tensors_this_frame += 1

                    new_tensor_events.append({
                        'frame': t,
                        'tensor_id': new_tensor.id,
                        'pixel_count': new_tensor.pixel_count,
                        'color': new_tensor.color.copy(),
                        'source': 'residual',
                        've_before': ve_before,
                    })

                # Step 5: Re-reconstruct with new tensors
                if new_tensors_this_frame > 0:
                    recon = reconstruct_from_hand(hand, (H, W, C))
                    residual = frame - recon

            var_explained = (1.0 - np.var(residual) / total_var
                             if total_var > 0 else 1.0)

            frame_results.append({
                'frame': t,
                'n_tensors': len(hand),
                'mse': np.mean(residual ** 2),
                'var_explained': var_explained,
                've_before_new_tensor': ve_before if new_tensors_this_frame > 0 else None,
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
        'new_tensor_events': new_tensor_events,
        'video_name': video_name,
        'frames': frames_f,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_reconstruction(result, output_dir):
    """3-row visualization: actual | reconstruction | residual."""
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
        title = f't={idx}'
        if fr[idx]['new_tensors'] > 0 and idx > 0:
            title += f'\n+{fr[idx]["new_tensors"]}T'
        axes[0, col].set_title(title, fontsize=8)
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

    fig.suptitle(f'{name} — New Tensor from Residual (Exp.9)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_exp9_recon.png', dpi=150,
                bbox_inches='tight')
    plt.close()


def visualize_tensor_inventory(result, output_dir):
    """Show all tensors in the hand with birth frame."""
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
        born = f'born@f{tensor.born_at_frame}'
        axes[i].set_title(
            f'T{tensor.id}\n{tensor.pixel_count}px\n{born}',
            fontsize=7
        )
        axes[i].axis('off')

    fig.suptitle(f'{name} — Tensor Inventory (Exp.9)', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_exp9_tensors.png', dpi=150,
                bbox_inches='tight')
    plt.close()


def visualize_trajectories(result, output_dir):
    """Plot centroid trajectories, showing when tensors were born."""
    hand = result['hand']
    frames = result['frames']
    name = result['video_name']
    H, W = frames.shape[1], frames.shape[2]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(np.clip(frames[-1], 0, 255).astype(np.uint8), alpha=0.3)

    colors_list = plt.cm.tab10(np.linspace(0, 1, max(len(hand), 1)))

    for i, tensor in enumerate(hand):
        history = np.array(tensor.centroid_history)
        if len(history) < 2 or tensor.pixel_count > H * W * 0.5:
            continue
        ax.plot(history[:, 0], history[:, 1], '-o', color=colors_list[i],
                markersize=2, linewidth=1.5,
                label=f'T{tensor.id} ({tensor.pixel_count}px, born@f{tensor.born_at_frame})')
        ax.plot(history[0, 0], history[0, 1], 's',
                color=colors_list[i], markersize=6)
        ax.plot(history[-1, 0], history[-1, 1], '*',
                color=colors_list[i], markersize=8)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title(f'{name} — Tensor Trajectories (Exp.9)', fontsize=11)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_exp9_trajectories.png', dpi=150,
                bbox_inches='tight')
    plt.close()


def visualize_metrics_over_time(result, output_dir):
    """Plot VE, tensor count, and search evals, marking new tensor events."""
    fr = result['frame_results']
    ss = result['search_stats']
    name = result['video_name']

    frames_idx = [r['frame'] for r in fr]
    var_ex = [r['var_explained'] for r in fr]
    n_tensors = [r['n_tensors'] for r in fr]
    total_evals = [s['total_evals'] for s in ss]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

    ax1.plot(frames_idx, var_ex, 'b-o', markersize=4)
    ax1.set_ylabel('Variance Explained')
    ax1.set_ylim(min(0, min(var_ex) - 0.05), 1.05)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax1.set_title(f'{name} — Metrics Over Time (Exp.9)', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Mark new tensor events
    for r in fr:
        if r.get('ve_before_new_tensor') is not None:
            ax1.annotate(
                f"VE before: {r['ve_before_new_tensor']:.4f}",
                xy=(r['frame'], r['var_explained']),
                xytext=(r['frame'] - 1.5, r['var_explained'] - 0.15),
                fontsize=7, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8),
            )

    ax2.step(frames_idx, n_tensors, 'r-o', markersize=4, where='mid')
    ax2.set_ylabel('Tensor Count')
    ax2.set_yticks(range(max(n_tensors) + 2))
    ax2.grid(True, alpha=0.3)

    ax3.bar(frames_idx, total_evals, color='steelblue', alpha=0.7)
    ax3.set_ylabel('SSD Evaluations')
    ax3.set_xlabel('Frame')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_exp9_metrics.png', dpi=150,
                bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    # Generate experiment 9 video
    frames, video_info = generate_experiment9_video()
    video_name = 'exp9_new_tensor'

    print("=" * 75)
    print("IS Experiment 9: New Tensor from Residual")
    print("  First test of the core learning loop:")
    print('  "Residuals unexplained by existing tensors → new tensor generation"')
    print("=" * 75)
    print()
    print("Video:")
    print(f"  {frames.shape[0]} frames, {frames.shape[1]}x{frames.shape[2]}px")
    print(f"  Object A (white triangle): present from frame 0, vel={video_info['a_vel']}")
    print(f"  Object B (red square):     appears at frame {video_info['b_appear_frame']}, vel={video_info['b_vel']}")
    print()
    print("Expected behavior:")
    print("  Frame 0-3: 2 tensors (background + triangle A). VE=1.0000")
    print("  Frame 4:   B appears → residual detected → new tensor T2 born. VE=1.0000")
    print("  Frame 5-8: 3 tensors tracked. VE=1.0000")
    print()

    Tensor._next_id = 0

    result = process_video_sequentially(
        frames, video_name,
        color_threshold=30.0,
        min_region_size=5,
        residual_threshold=30.0,
    )

    fr = result['frame_results']
    hand = result['hand']
    ss = result['search_stats']
    events = result['new_tensor_events']

    # --- Per-frame results table ---
    print("-" * 75)
    print(f"{'Frame':>5}  {'VE':>8}  {'MSE':>8}  {'#T':>3}  {'Event'}")
    print("-" * 75)

    for r in fr:
        t = r['frame']
        ve = r['var_explained']
        mse = r['mse']
        n_t = r['n_tensors']
        new = r['new_tensors']

        event_str = ""
        if t == 0:
            event_str = f"+{new} (initial: "
            parts = []
            for tensor in hand:
                if tensor.born_at_frame == 0:
                    H, W = frames.shape[1], frames.shape[2]
                    label = "bg" if tensor.pixel_count > H * W * 0.5 else "obj"
                    parts.append(f"T{tensor.id}={label}")
            event_str += ", ".join(parts) + ")"
        elif new > 0:
            ve_before = r.get('ve_before_new_tensor')
            new_ids = [e for e in events if e['frame'] == t]
            parts = []
            for e in new_ids:
                c = e['color']
                parts.append(f"T{e['tensor_id']} ({e['pixel_count']}px, "
                             f"color=({c[0]:.0f},{c[1]:.0f},{c[2]:.0f}))")
            event_str = f"+{new} from residual: {', '.join(parts)}"
            if ve_before is not None:
                event_str += f"  [VE before: {ve_before:.4f}]"
        else:
            event_str = "--"

        print(f"{t:>5}  {ve:>8.4f}  {mse:>8.2f}  {n_t:>3}  {event_str}")

    print("-" * 75)

    # --- Summary ---
    final_ve = fr[-1]['var_explained']
    mean_ve = np.mean([r['var_explained'] for r in fr])
    tensor_counts = [r['n_tensors'] for r in fr]

    print()
    print("Summary:")
    print(f"  Final VE:  {final_ve:.4f}")
    print(f"  Mean VE:   {mean_ve:.4f}")
    print(f"  Tensor count progression: {' → '.join(str(c) for c in tensor_counts)}")
    print()

    # --- Tensor inventory ---
    print("Tensor inventory:")
    H, W = frames.shape[1], frames.shape[2]
    for tensor in hand:
        label = "background" if tensor.pixel_count > H * W * 0.5 else "object"
        print(f"  T{tensor.id}: {tensor.pixel_count}px, "
              f"color=({tensor.color[0]:.0f},{tensor.color[1]:.0f},{tensor.color[2]:.0f}), "
              f"bbox={tensor.bbox_h}x{tensor.bbox_w}, "
              f"born@frame {tensor.born_at_frame}, age={tensor.age}, "
              f"type={label}")

    # --- Velocity estimates for object tensors ---
    print()
    print("Velocity estimates (object tensors only):")
    obj_tensors = [t for t in hand if t.pixel_count < H * W * 0.5]
    for ot in obj_tensors:
        hist = np.array(ot.centroid_history)
        if len(hist) >= 2:
            n_steps = len(hist) - 1
            if n_steps > 0:
                dx = (hist[-1, 0] - hist[0, 0]) / n_steps
                dy = (hist[-1, 1] - hist[0, 1]) / n_steps
                print(f"  T{ot.id}: ({dx:.2f}, {dy:.2f})"
                      f"  (tracked for {n_steps} frames, born@f{ot.born_at_frame})")

    print()
    print("True velocities:")
    print(f"  Object A (triangle): {video_info['a_vel']}")
    print(f"  Object B (square):   {video_info['b_vel']}  (appears at frame {video_info['b_appear_frame']})")

    # --- Gradient search efficiency ---
    print()
    print("Gradient search steps (SSD evaluations per frame):")
    for stat in ss:
        t = stat['frame']
        if t == 0:
            print(f"  Frame {t}: -- (initial segmentation)")
            continue
        total = stat['total_evals']
        detail = ", ".join(
            f"T{tid}:{ev}" for tid, ev in sorted(stat['tensor_evals'].items())
        )
        print(f"  Frame {t}: {total:>4d} evals ({detail})")

    # --- Generate visualizations ---
    visualize_reconstruction(result, output_dir)
    visualize_tensor_inventory(result, output_dir)
    visualize_trajectories(result, output_dir)
    visualize_metrics_over_time(result, output_dir)

    print()
    print(f"Visualizations saved to {output_dir}/")


if __name__ == '__main__':
    main()
