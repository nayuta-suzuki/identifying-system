"""
IS Experiment 7: Coordinate-Shift Separation ("目で追う" operation).

Core insight: A baby tracks a moving object by moving its eyes/head,
making the tracked object STATIONARY in its visual field. Everything
else (background, other objects) becomes "the thing that moves."

This is NOT a new operation. It reuses the same separation from experiments 4-5
("動くもの/動かないもの" separation) but in SHIFTED coordinate systems.

Algorithm:
  Step 0: Velocity (0,0) — raw frames. Separate static (background) from moving (residual).
  Step 1: Scan velocity space on residual frames.
    - For each candidate (dx, dy): shift residual frame i by (-dx*i, -dy*i),
      then measure the ENERGY of pixels that become static.
    - Score = sum of magnitudes of static pixels (not count — avoids zero-bg trap).
    - The (dx, dy) with the highest score = the velocity of the first object.
    - In that shifted coordinate system, apply the same separation.
    - Map back to original coordinates, subtract from residual.
  Step 2: Repeat on the remaining residual → second object tensor.
  Stop when residual variance is below threshold.

No new operations invented. No object count given in advance.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# Core separation (same as experiments 4-5)
# ──────────────────────────────────────────────

def separate_static_moving(frames: np.ndarray, diff_thresh: float = 5.0, obj_thresh: float = 15.0):
    """
    The fundamental operation: separate static pixels from moving pixels.
    Same logic as experiments 4-5.
    """
    n_frames, H, W, C = frames.shape
    frames_f = frames.astype(np.float64)

    ever_changed = np.zeros((H, W), dtype=bool)
    for t in range(1, n_frames):
        diff = np.sqrt(np.sum((frames_f[t] - frames_f[t - 1]) ** 2, axis=-1))
        ever_changed |= (diff > diff_thresh)

    never_changed = ~ever_changed
    n_static = int(never_changed.sum())

    if n_static > 0:
        bg_color = frames_f[0, never_changed].mean(axis=0)
    else:
        bg_color = np.median(frames_f.reshape(-1, C), axis=0)

    object_masks = np.zeros((n_frames, H, W), dtype=bool)
    for t in range(n_frames):
        diff = np.sqrt(np.sum((frames_f[t] - bg_color) ** 2, axis=-1))
        object_masks[t] = diff > obj_thresh

    bg_image = np.full((H, W, C), bg_color, dtype=np.float64)
    static_count = (~object_masks).sum(axis=0)
    safe_count = np.maximum(static_count, 1)
    bg_sum = np.zeros((H, W, C), dtype=np.float64)
    for t in range(n_frames):
        mask = ~object_masks[t]
        bg_sum[mask] += frames_f[t, mask]
    bg_image = bg_sum / safe_count[:, :, None]
    always_moving = static_count == 0
    if always_moving.any():
        bg_image[always_moving] = bg_color

    return bg_color, bg_image, object_masks, n_static


# ──────────────────────────────────────────────
# Coordinate shift operations
# ──────────────────────────────────────────────

def shift_frames(frames: np.ndarray, velocity: tuple[float, float]) -> np.ndarray:
    """
    Shift each frame by (-velocity * t) so that an object moving at `velocity`
    becomes stationary. Out-of-bounds filled with NaN to distinguish from real zeros.
    """
    n_frames, H, W, C = frames.shape
    dx, dy = velocity
    shifted = np.full_like(frames, np.nan, dtype=np.float64)

    for t in range(n_frames):
        shift_x = -int(round(dx * t))
        shift_y = -int(round(dy * t))

        src_y0 = max(0, -shift_y)
        src_x0 = max(0, -shift_x)
        src_y1 = min(H, H - shift_y)
        src_x1 = min(W, W - shift_x)

        dst_y0 = max(0, shift_y)
        dst_x0 = max(0, shift_x)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        if dst_y1 > dst_y0 and dst_x1 > dst_x0:
            shifted[t, dst_y0:dst_y1, dst_x0:dst_x1] = frames[t, src_y0:src_y1, src_x0:src_x1].astype(np.float64)

    return shifted


def find_static_with_energy(shifted: np.ndarray, diff_thresh: float = 8.0):
    """
    Find pixels that are static across shifted frames AND have significant energy.

    Returns:
        static_mask: (H, W) bool — pixels that are static and have content
        avg_values: (H, W, C) float64 — average value of static pixels
        total_energy: float — sum of magnitudes of static content pixels
    """
    n_frames, H, W, C = shifted.shape

    # Valid = not NaN in ALL frames
    valid = np.ones((H, W), dtype=bool)
    for t in range(n_frames):
        valid &= ~np.isnan(shifted[t, :, :, 0])

    if valid.sum() == 0:
        return np.zeros((H, W), dtype=bool), np.zeros((H, W, C)), 0.0

    # Among valid pixels, find those that never changed
    ever_changed = np.zeros((H, W), dtype=bool)
    for t in range(1, n_frames):
        # Only compare where both frames are valid
        both_valid = valid  # already ensured all frames valid
        d = np.zeros((H, W))
        for c in range(C):
            d += (shifted[t, :, :, c] - shifted[t - 1, :, :, c]) ** 2
        d = np.sqrt(d)
        d[~both_valid] = 0
        ever_changed |= (d > diff_thresh)

    static = valid & ~ever_changed

    # Average values of static pixels
    avg_values = np.zeros((H, W, C), dtype=np.float64)
    if static.sum() > 0:
        for c in range(C):
            vals = shifted[:, :, :, c]  # (n_frames, H, W)
            # Take mean only over valid frames (all are valid for static pixels)
            avg_values[:, :, c] = np.nanmean(vals, axis=0)
        avg_values[~static] = 0

    # Energy = sum of magnitudes of static pixels (this ignores zero-background)
    magnitudes = np.sqrt(np.sum(avg_values ** 2, axis=-1))
    total_energy = float(magnitudes[static].sum())

    return static, avg_values, total_energy


def scan_velocity_space(frames: np.ndarray, vel_candidates: list[tuple[float, float]],
                        diff_thresh: float = 8.0):
    """
    Scan velocity candidates and return scores.
    Score = total energy of static pixels (not count).
    """
    results = []
    for vel in vel_candidates:
        shifted = shift_frames(frames, vel)
        _, _, energy = find_static_with_energy(shifted, diff_thresh)
        results.append((vel, energy))

    return results


def extract_object_at_velocity(
    residual: np.ndarray,
    velocity: tuple[float, float],
    diff_thresh: float = 8.0,
    mag_thresh: float = 10.0,
) -> dict:
    """
    Shift residual by velocity, find static high-energy pixels = object tensor.
    """
    n_frames, H, W, C = residual.shape

    shifted = shift_frames(residual, velocity)
    static_mask, avg_values, energy = find_static_with_energy(shifted, diff_thresh)

    # Filter: only keep pixels with significant magnitude
    magnitudes = np.sqrt(np.sum(avg_values ** 2, axis=-1))
    object_mask = static_mask & (magnitudes > mag_thresh)

    # tensor_full: the object in shifted coordinates
    tensor_full = np.zeros((H, W, C), dtype=np.float64)
    tensor_full[object_mask] = avg_values[object_mask]

    n_obj_pixels = int(object_mask.sum())

    return {
        'tensor_full': tensor_full,
        'velocity': velocity,
        'object_mask': object_mask,
        'n_pixels': n_obj_pixels,
        'energy': energy,
    }


def subtract_object(residual: np.ndarray, obj: dict) -> np.ndarray:
    """Subtract object tensor from residual in original coordinates."""
    n_frames, H, W, C = residual.shape
    new_res = residual.copy()
    dx, dy = obj['velocity']
    tensor = obj['tensor_full']

    ys, xs = np.where(np.any(tensor != 0, axis=-1))
    if len(ys) == 0:
        return new_res

    tensor_vals = tensor[ys, xs]

    for t in range(n_frames):
        shift_y = int(round(dy * t))
        shift_x = int(round(dx * t))
        oys = ys + shift_y
        oxs = xs + shift_x
        valid = (oys >= 0) & (oys < H) & (oxs >= 0) & (oxs < W)
        if valid.any():
            new_res[t, oys[valid], oxs[valid]] -= tensor_vals[valid]

    return new_res


def reconstruct_from_objects(bg_image, objects, n_frames):
    """Reconstruct all frames from background + objects."""
    H, W, C = bg_image.shape
    recon = np.zeros((n_frames, H, W, C), dtype=np.float64)

    for t in range(n_frames):
        frame = bg_image.copy()
        for obj in objects:
            tensor = obj['tensor_full']
            dx, dy = obj['velocity']
            shift_y = int(round(dy * t))
            shift_x = int(round(dx * t))
            ys, xs = np.where(np.any(tensor != 0, axis=-1))
            if len(ys) == 0:
                continue
            oys = ys + shift_y
            oxs = xs + shift_x
            valid = (oys >= 0) & (oys < H) & (oxs >= 0) & (oxs < W)
            if valid.any():
                frame[oys[valid], oxs[valid]] += tensor[ys[valid], xs[valid]]
        recon[t] = frame
    return recon


def evaluate_all(original, reconstructed):
    orig = original.astype(np.float64)
    error = orig - reconstructed
    mse = np.mean(error ** 2)
    max_err = np.max(np.abs(error))
    total_ss = np.sum((orig - orig.mean()) ** 2)
    recon_ss = np.sum(error ** 2)
    var_explained = 1.0 - recon_ss / total_ss if total_ss > 0 else 1.0
    return {'mse': mse, 'max_error': max_err, 'var_explained': var_explained}


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def visualize_velocity_scan(scan_results, best_vel, step_idx, video_name, save_path):
    """Visualize velocity scan as heatmap."""
    # Build grid from results
    vels = [r[0] for r in scan_results]
    dxs = sorted(set(v[0] for v in vels))
    dys = sorted(set(v[1] for v in vels))
    score_map = {v: e for v, e in scan_results}

    grid = np.zeros((len(dys), len(dxs)))
    for j, dy in enumerate(dys):
        for i, dx in enumerate(dxs):
            grid[j, i] = score_map.get((dx, dy), 0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(grid, cmap='viridis', origin='lower',
                   extent=[dxs[0] - 0.25, dxs[-1] + 0.25, dys[0] - 0.25, dys[-1] + 0.25],
                   aspect='auto')
    ax.set_xlabel('dx (pixels/frame)')
    ax.set_ylabel('dy (pixels/frame)')
    best_score = score_map.get(best_vel, 0)
    ax.set_title(f'{video_name} — Step {step_idx}: Velocity Scan\n'
                 f'Best: dx={best_vel[0]}, dy={best_vel[1]} (energy={best_score:.0f})')
    ax.plot(best_vel[0], best_vel[1], 'r*', markersize=15)
    plt.colorbar(im, label='Static pixel energy')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def visualize_full_result(frames, bg_image, objects, final_recon, metrics,
                          video_name, save_path):
    """Visualize the full result."""
    n_obj = len(objects)
    n_show = 5
    indices = np.linspace(0, len(frames) - 1, n_show, dtype=int)
    n_rows = 3 + n_obj

    fig, axes = plt.subplots(n_rows, n_show, figsize=(3 * n_show, 3 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    # Row 0: Original
    for i, idx in enumerate(indices):
        axes[0, i].imshow(frames[idx])
        axes[0, i].set_title(f't={idx}')
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('Original', rotation=0, labelpad=55, va='center', fontsize=10)

    # Rows 1..n_obj: Each object on background
    for obj_idx, obj in enumerate(objects):
        row = 1 + obj_idx
        dx, dy = obj['velocity']
        tensor = obj['tensor_full']
        H, W = frames[0].shape[:2]
        ys, xs = np.where(np.any(tensor != 0, axis=-1))

        for i, idx in enumerate(indices):
            display = bg_image.copy()
            if len(ys) > 0:
                shift_y = int(round(dy * idx))
                shift_x = int(round(dx * idx))
                oys = ys + shift_y
                oxs = xs + shift_x
                valid = (oys >= 0) & (oys < H) & (oxs >= 0) & (oxs < W)
                if valid.any():
                    display[oys[valid], oxs[valid]] += tensor[ys[valid], xs[valid]]
            axes[row, i].imshow(np.clip(display, 0, 255).astype(np.uint8))
            axes[row, i].axis('off')
        axes[row, 0].set_ylabel(f'Obj {obj_idx + 1}\nv=({dx},{dy})\n{obj["n_pixels"]}px',
                                rotation=0, labelpad=55, va='center', fontsize=9)

    # Reconstruction row
    rr = 1 + n_obj
    for i, idx in enumerate(indices):
        axes[rr, i].imshow(np.clip(final_recon[idx], 0, 255).astype(np.uint8))
        axes[rr, i].axis('off')
    axes[rr, 0].set_ylabel('Recon', rotation=0, labelpad=55, va='center', fontsize=10)

    # Error row
    er = 2 + n_obj
    for i, idx in enumerate(indices):
        error = np.sqrt(np.sum((frames[idx].astype(float) - final_recon[idx]) ** 2, axis=-1))
        axes[er, i].imshow(error, cmap='hot', vmin=0, vmax=max(error.max(), 1))
        axes[er, i].set_title(f'max={error.max():.1f}')
        axes[er, i].axis('off')
    axes[er, 0].set_ylabel('Error', rotation=0, labelpad=55, va='center', fontsize=10)

    summary = (f'Var={metrics["var_explained"]:.4f}  MSE={metrics["mse"]:.2f}  '
               f'MaxErr={metrics["max_error"]:.1f}  Objects={n_obj}')
    fig.text(0.02, 0.01, summary, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    fig.suptitle(f'{video_name}: Coordinate-Shift Separation', fontsize=14)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(save_path, dpi=120)
    plt.close()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    data_dir = Path(__file__).parent / 'data'
    output_dir = Path(__file__).parent / 'output'

    video_names = ['v1_triangle_right', 'v2_square_diag', 'v3_circle_down', 'v4_two_objects']

    # Velocity candidates: 0.5 step from -3 to 3 (covers all object velocities with margin)
    vel_values = [v * 0.5 for v in range(-6, 7)]  # -3.0 to 3.0 in 0.5 steps
    vel_candidates = [(dx, dy) for dy in vel_values for dx in vel_values]
    # Exclude (0,0) — handled separately as Step 0
    vel_candidates_no_zero = [(dx, dy) for dx, dy in vel_candidates if not (dx == 0 and dy == 0)]

    print('=' * 80)
    print('IS Experiment 7: Coordinate-Shift Separation')
    print('  "目で追う" = 座標系を物体の速度に合わせて、同じ分離操作を繰り返す')
    print(f'  Velocity candidates: {len(vel_candidates_no_zero)} (0.5 step, -3.0 to 3.0)')
    print('=' * 80)
    print()

    for name in video_names:
        filepath = data_dir / f'{name}.npy'
        if not filepath.exists():
            continue

        frames = np.load(filepath)
        n_frames, H, W, C = frames.shape
        print(f'=== {name} ({n_frames} frames, {H}x{W}) ===')

        # ── Step 0: Separate background (velocity = 0) ──
        print(f'  Step 0: Background separation...')
        bg_color, bg_image, _, n_static = separate_static_moving(frames)
        print(f'    BG color: ({bg_color[0]:.1f}, {bg_color[1]:.1f}, {bg_color[2]:.1f})')

        residual = frames.astype(np.float64) - bg_image[None]
        residual_var = np.var(residual)
        print(f'    Residual variance: {residual_var:.2f}')

        # ── Iterative object extraction ──
        objects = []
        step = 1
        max_steps = 5
        used_velocities = set()
        used_velocities.add((0.0, 0.0))

        while step <= max_steps and residual_var > 1.0:
            print(f'  Step {step}: Scanning {len(vel_candidates_no_zero)} velocities...')

            # Filter out already-used velocities
            candidates = [v for v in vel_candidates_no_zero if v not in used_velocities]
            if not candidates:
                print(f'    No more velocity candidates. Stopping.')
                break

            scan_results = scan_velocity_space(residual, candidates, diff_thresh=8.0)

            # Find best
            scan_results.sort(key=lambda x: x[1], reverse=True)
            best_vel, best_energy = scan_results[0]

            print(f'    Best velocity: dx={best_vel[0]}, dy={best_vel[1]} (energy={best_energy:.0f})')

            if best_energy < 100:
                print(f'    Energy too low. Stopping.')
                break

            # Visualize scan
            visualize_velocity_scan(scan_results, best_vel, step,
                                    name, output_dir / f'{name}_coordshift_vscan_step{step}.png')

            # Extract object at this velocity
            obj = extract_object_at_velocity(residual, best_vel, diff_thresh=8.0, mag_thresh=10.0)
            print(f'    Object pixels: {obj["n_pixels"]}')

            if obj['n_pixels'] == 0:
                print(f'    No object pixels found. Trying next velocity...')
                used_velocities.add(best_vel)
                continue

            objects.append(obj)
            used_velocities.add(best_vel)

            # Subtract from residual
            residual = subtract_object(residual, obj)
            residual_var = np.var(residual)
            print(f'    Residual variance after subtraction: {residual_var:.2f}')

            step += 1

        # ── Final reconstruction ──
        final_recon = reconstruct_from_objects(bg_image, objects, n_frames)
        metrics = evaluate_all(frames, final_recon)

        print(f'  --- Final Result ---')
        print(f'    Objects found: {len(objects)}')
        for i, obj in enumerate(objects):
            dx, dy = obj['velocity']
            print(f'    Object {i + 1}: velocity=({dx}, {dy}), pixels={obj["n_pixels"]}')
        print(f'    Var Explained: {metrics["var_explained"]:.4f}')
        print(f'    MSE:           {metrics["mse"]:.2f}')
        print(f'    Max Error:     {metrics["max_error"]:.1f}')

        true_bg = np.array([40, 60, 120], dtype=np.float64)
        ghost = np.sqrt(np.sum((bg_image - true_bg) ** 2, axis=-1)).max()
        print(f'    BG Ghost max:  {ghost:.1f}')
        print()

        visualize_full_result(frames, bg_image, objects, final_recon, metrics,
                              name, output_dir / f'{name}_coordshift.png')

    print('=' * 80)
    print('KEY PRINCIPLES:')
    print('  1. No new operations: reuse static/moving separation in shifted coordinates')
    print('  2. No object count given: loop until residual energy is small')
    print('  3. Score = energy of static pixels (not count — avoids zero-bg trap)')
    print('  4. "目で追う" = shift coordinate system to match object velocity')
    print('=' * 80)


if __name__ == '__main__':
    main()
