"""
IS Experiment Step 8: Object-Removal Background Extraction.

Core idea (from N's observation):
"動いているものを発見したとき、その後ろを補完する。
動いている物体を分離できたのならば、各時点の画像について、
動いているものを取り除いた部分が、背景だ。"

Algorithm:
1. Detect moving regions per frame via frame differencing
2. For each frame, mask out the moving pixels → remaining pixels ARE background
3. For masked-out (object) pixels, fill from other frames where that pixel was static
4. The result is a pure background with zero ghost contamination

This is conceptually different from:
- Experiment 1 (frame average → ghost contamination)
- Experiment 2 (iterative refinement → needs multiple passes to remove ghosts)
- Both of those START from the background estimate. This STARTS from motion detection.

Key prediction: This approach should produce a background with ZERO ghost artifacts
in a single pass, because it never averages object pixels into the background.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage


def detect_motion_per_frame(frames: np.ndarray, threshold_fraction: float = 0.3):
    """
    Detect moving pixels in each frame using forward/backward frame differences.

    For frame t, a pixel is "moving" if it changed significantly from t-1 or t+1.

    Returns:
        motion_masks: (n_frames, H, W) bool — True where motion detected
    """
    n_frames, H, W, C = frames.shape
    frames_f = frames.astype(np.float64)
    motion_masks = np.zeros((n_frames, H, W), dtype=bool)

    for t in range(n_frames):
        diffs = []
        if t > 0:
            d = np.sqrt(np.sum((frames_f[t] - frames_f[t - 1]) ** 2, axis=-1))
            diffs.append(d)
        if t < n_frames - 1:
            d = np.sqrt(np.sum((frames_f[t + 1] - frames_f[t]) ** 2, axis=-1))
            diffs.append(d)

        if not diffs:
            continue

        avg_diff = np.mean(diffs, axis=0)
        thresh = avg_diff.max() * threshold_fraction
        motion_masks[t] = avg_diff > thresh

    return motion_masks


def detect_object_per_frame_from_static(frames: np.ndarray, threshold: float = 15.0):
    """
    Detect object in each frame by comparing against a background color
    estimated from pixels that NEVER changed across all frames.

    Algorithm:
    1. Frame differencing → find pixels that ever changed
    2. Pixels that NEVER changed across all frames → definitely background
    3. Average color of those pixels → background color estimate
    4. For each frame, pixels that differ from background color → object

    This captures the FULL BODY of the object, not just its motion edges.

    Returns:
        object_masks: (n_frames, H, W) bool — True where object detected
        bg_color_estimate: (3,) float64 — estimated background color
        ever_changed: (H, W) bool — pixels that changed at least once
    """
    n_frames, H, W, C = frames.shape
    frames_f = frames.astype(np.float64)

    # Step 1: Find pixels that ever changed (accumulated frame difference)
    ever_changed = np.zeros((H, W), dtype=bool)
    for t in range(1, n_frames):
        diff = np.sqrt(np.sum((frames_f[t] - frames_f[t - 1]) ** 2, axis=-1))
        ever_changed |= (diff > 5.0)  # small absolute threshold for noise

    # Step 2: Pixels that NEVER changed → definitely background
    never_changed = ~ever_changed
    n_static = never_changed.sum()

    if n_static == 0:
        # Fallback: use median
        bg_color_estimate = np.median(frames_f.reshape(-1, C), axis=0)
    else:
        # Average color of never-changed pixels (should be identical across frames,
        # so just take from first frame)
        bg_color_estimate = frames_f[0, never_changed].mean(axis=0)

    # Step 3: For each frame, find pixels that differ from background color
    object_masks = np.zeros((n_frames, H, W), dtype=bool)
    for t in range(n_frames):
        diff = np.sqrt(np.sum((frames_f[t] - bg_color_estimate) ** 2, axis=-1))
        object_masks[t] = diff > threshold

    return object_masks, bg_color_estimate, ever_changed


def extract_background_by_removal(
    frames: np.ndarray,
    motion_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract background by removing moving objects from each frame.

    For each pixel (y, x):
    - Collect its value from all frames where it was NOT moving
    - Average those values → pure background pixel
    - If a pixel was moving in ALL frames, fall back to median of all frames

    Returns:
        background: (H, W, 3) float64 — pure background tensor
        static_count: (H, W) int — number of frames where each pixel was static
    """
    n_frames, H, W, C = frames.shape
    frames_f = frames.astype(np.float64)

    # For each pixel, count how many frames it was static
    static_masks = ~motion_masks  # (n_frames, H, W)
    static_count = static_masks.sum(axis=0)  # (H, W)

    # Weighted sum: only include static frames
    background = np.zeros((H, W, C), dtype=np.float64)
    for t in range(n_frames):
        mask = static_masks[t]  # (H, W)
        background[mask] += frames_f[t, mask]

    # Average by number of static frames (avoid division by zero)
    safe_count = np.maximum(static_count, 1)
    background /= safe_count[:, :, None]

    # For pixels that were NEVER static (always moving), use median
    always_moving = static_count == 0
    if always_moving.any():
        median_vals = np.median(frames_f, axis=0)
        background[always_moving] = median_vals[always_moving]

    return background, static_count


def extract_objects_per_frame(
    frames: np.ndarray,
    motion_masks: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    """
    Extract object pixels per frame: frame - background, masked to motion region.

    Returns:
        object_layers: (n_frames, H, W, 3) float64 — object pixels only, zero elsewhere
    """
    frames_f = frames.astype(np.float64)
    object_layers = np.zeros_like(frames_f)

    for t in range(len(frames)):
        mask = motion_masks[t]
        object_layers[t, mask] = frames_f[t, mask] - background[mask]

    return object_layers


def reconstruct_frames(
    background: np.ndarray,
    object_layers: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct each frame as: background + object_layer[t].

    Returns:
        reconstructed: (n_frames, H, W, 3) float64
    """
    return background[None] + object_layers


def evaluate(
    original: np.ndarray,
    background: np.ndarray,
    reconstructed: np.ndarray,
) -> dict:
    """Evaluate reconstruction quality."""
    orig_f = original.astype(np.float64)
    total_ss = np.sum((orig_f - orig_f.mean()) ** 2)

    bg_error = np.sum((orig_f - background[None]) ** 2)
    recon_error = np.sum((orig_f - reconstructed) ** 2)

    bg_var = 1.0 - bg_error / total_ss if total_ss > 0 else 1.0
    full_var = 1.0 - recon_error / total_ss if total_ss > 0 else 1.0

    bg_mse = np.mean((orig_f - background[None]) ** 2)
    recon_mse = np.mean((orig_f - reconstructed) ** 2)
    recon_max = np.max(np.abs(orig_f - reconstructed))

    return {
        'bg_var': bg_var,
        'full_var': full_var,
        'bg_mse': bg_mse,
        'recon_mse': recon_mse,
        'recon_max_error': recon_max,
    }


def compare_with_frame_average(frames: np.ndarray) -> np.ndarray:
    """Experiment 1 method: simple frame average as background."""
    return frames.astype(np.float64).mean(axis=0)


def compute_ghost_energy(background: np.ndarray, true_bg_color: np.ndarray) -> float:
    """
    Measure ghost contamination: sum of squared deviations from the true background color
    for pixels that should be pure background.

    Since we know the true background color (from video generation), we can measure this directly.
    """
    diff = background - true_bg_color[None, None, :].astype(np.float64)
    return np.sum(diff ** 2)


def visualize_comparison(
    frames: np.ndarray,
    bg_removal: np.ndarray,
    bg_average: np.ndarray,
    motion_masks: np.ndarray,
    static_count: np.ndarray,
    metrics_removal: dict,
    metrics_average: dict,
    video_name: str,
    save_path: Path,
):
    """Compare object-removal background vs frame-average background."""
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    # Row 0: Original frames
    n_show = 5
    indices = np.linspace(0, len(frames) - 1, n_show, dtype=int)
    for i, idx in enumerate(indices):
        axes[0, i].imshow(frames[idx])
        axes[0, i].set_title(f't={idx}')
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('Original', rotation=0, labelpad=60, va='center', fontsize=11)

    # Row 1: Motion masks per frame + static count + backgrounds
    for i, idx in enumerate(indices[:3]):
        axes[1, i].imshow(motion_masks[idx], cmap='gray')
        axes[1, i].set_title(f'Motion mask t={idx}')
        axes[1, i].axis('off')
    axes[1, 0].set_ylabel('Motion\nmasks', rotation=0, labelpad=60, va='center', fontsize=11)

    axes[1, 3].imshow(static_count, cmap='viridis')
    axes[1, 3].set_title(f'Static count\n(min={static_count.min()}, max={static_count.max()})')
    axes[1, 3].axis('off')

    # Always-moving pixels
    always_moving = (static_count == 0).astype(float)
    axes[1, 4].imshow(always_moving, cmap='hot')
    axes[1, 4].set_title(f'Always moving\n({int(always_moving.sum())} px)')
    axes[1, 4].axis('off')

    # Row 2: Background comparison
    axes[2, 0].imshow(np.clip(bg_removal, 0, 255).astype(np.uint8))
    axes[2, 0].set_title('BG: Object Removal')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(np.clip(bg_average, 0, 255).astype(np.uint8))
    axes[2, 1].set_title('BG: Frame Average')
    axes[2, 1].axis('off')

    # Difference between the two backgrounds
    diff = np.sqrt(np.sum((bg_removal - bg_average) ** 2, axis=-1))
    axes[2, 2].imshow(diff, cmap='hot')
    axes[2, 2].set_title('|Removal - Average|')
    axes[2, 2].axis('off')
    axes[2, 0].set_ylabel('Background\ncomparison', rotation=0, labelpad=60, va='center', fontsize=11)

    # Ghost contamination heatmaps
    true_bg = np.array([40, 60, 120], dtype=np.float64)  # Known background color
    ghost_removal = np.sqrt(np.sum((bg_removal - true_bg) ** 2, axis=-1))
    ghost_average = np.sqrt(np.sum((bg_average - true_bg) ** 2, axis=-1))
    vmax_ghost = max(ghost_removal.max(), ghost_average.max(), 1)

    axes[2, 3].imshow(ghost_removal, cmap='hot', vmin=0, vmax=vmax_ghost)
    axes[2, 3].set_title(f'Ghost (Removal)\nmax={ghost_removal.max():.1f}')
    axes[2, 3].axis('off')

    axes[2, 4].imshow(ghost_average, cmap='hot', vmin=0, vmax=vmax_ghost)
    axes[2, 4].set_title(f'Ghost (Average)\nmax={ghost_average.max():.1f}')
    axes[2, 4].axis('off')

    # Summary text
    summary = (
        f'Object Removal:  BG Var={metrics_removal["bg_var"]:.4f}  '
        f'Full Var={metrics_removal["full_var"]:.4f}  '
        f'Recon MSE={metrics_removal["recon_mse"]:.2f}\n'
        f'Frame Average:   BG Var={metrics_average["bg_var"]:.4f}  '
        f'Full Var={metrics_average["full_var"]:.4f}  '
        f'Recon MSE={metrics_average["recon_mse"]:.2f}'
    )
    fig.text(0.02, 0.01, summary, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle(f'{video_name}: Object-Removal vs Frame-Average Background', fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    data_dir = Path(__file__).parent / 'data'
    output_dir = Path(__file__).parent / 'output'

    video_names = ['v1_triangle_right', 'v2_square_diag', 'v3_circle_down', 'v4_two_objects']
    true_bg_color = np.array([40, 60, 120], dtype=np.float64)

    print('=' * 80)
    print('IS Experiment: Object-Removal Background Extraction')
    print('  "動いているものを取り除いた部分が、背景だ"')
    print('=' * 80)
    print()

    all_results = {}

    for name in video_names:
        filepath = data_dir / f'{name}.npy'
        if not filepath.exists():
            print(f'  {name}: not found, skipping')
            continue

        frames = np.load(filepath)
        n_frames, H, W, C = frames.shape

        print(f'=== {name} ({n_frames} frames, {H}x{W}) ===')

        # --- Method A: Frame-diff motion masks (edge-only, for comparison) ---
        motion_masks_diff = detect_motion_per_frame(frames)

        # --- Method B: Object masks from static-pixel background estimation ---
        object_masks, bg_color_est, ever_changed = detect_object_per_frame_from_static(frames)
        print(f'  Estimated BG color: ({bg_color_est[0]:.1f}, {bg_color_est[1]:.1f}, {bg_color_est[2]:.1f})')
        print(f'  True BG color:      (40, 60, 120)')
        print(f'  Never-changed pixels: {(~ever_changed).sum()} / {H * W}')
        print(f'  Object mask coverage: {object_masks.mean():.4f}')

        # --- Object-Removal Background (using full-body masks) ---
        bg_removal, static_count = extract_background_by_removal(frames, object_masks)
        always_moving = (static_count == 0).sum()
        print(f'  Static count range: {static_count.min()}-{static_count.max()}')
        print(f'  Always-moving pixels: {always_moving}')

        # Reconstruct
        object_layers = extract_objects_per_frame(frames, object_masks, bg_removal)
        reconstructed_removal = reconstruct_frames(bg_removal, object_layers)
        metrics_removal = evaluate(frames, bg_removal, reconstructed_removal)
        ghost_removal = compute_ghost_energy(bg_removal, true_bg_color)

        # --- Frame Average (Experiment 1 baseline) ---
        bg_average = compare_with_frame_average(frames)
        object_layers_avg = extract_objects_per_frame(frames, object_masks, bg_average)
        reconstructed_avg = reconstruct_frames(bg_average, object_layers_avg)
        metrics_average = evaluate(frames, bg_average, reconstructed_avg)
        ghost_average = compute_ghost_energy(bg_average, true_bg_color)

        # --- Results ---
        print(f'  --- Object Removal (full-body masks) ---')
        print(f'    BG Var Explained:    {metrics_removal["bg_var"]:.4f}')
        print(f'    Full Var Explained:  {metrics_removal["full_var"]:.4f}')
        print(f'    Recon MSE:           {metrics_removal["recon_mse"]:.4f}')
        print(f'    Recon Max Error:     {metrics_removal["recon_max_error"]:.1f}')
        print(f'    Ghost Energy:        {ghost_removal:.0f}')
        print(f'  --- Frame Average (baseline) ---')
        print(f'    BG Var Explained:    {metrics_average["bg_var"]:.4f}')
        print(f'    Full Var Explained:  {metrics_average["full_var"]:.4f}')
        print(f'    Recon MSE:           {metrics_average["recon_mse"]:.4f}')
        print(f'    Recon Max Error:     {metrics_average["recon_max_error"]:.1f}')
        print(f'    Ghost Energy:        {ghost_average:.0f}')
        print()

        all_results[name] = {
            'removal': metrics_removal,
            'average': metrics_average,
            'ghost_removal': ghost_removal,
            'ghost_average': ghost_average,
        }

        # Visualize
        visualize_comparison(
            frames, bg_removal, bg_average, object_masks, static_count,
            metrics_removal, metrics_average,
            name, output_dir / f'{name}_obj_removal.png'
        )

    # --- Summary Table ---
    print('=' * 90)
    print(f'{"Video":<22} {"Method":<16} {"BG Var":>8} {"Full Var":>10} '
          f'{"Recon MSE":>10} {"Ghost":>10}')
    print('-' * 90)
    for name, r in all_results.items():
        print(f'{name:<22} {"Obj Removal":<16} {r["removal"]["bg_var"]:>8.4f} '
              f'{r["removal"]["full_var"]:>10.4f} {r["removal"]["recon_mse"]:>10.4f} '
              f'{r["ghost_removal"]:>10.0f}')
        print(f'{"":<22} {"Frame Average":<16} {r["average"]["bg_var"]:>8.4f} '
              f'{r["average"]["full_var"]:>10.4f} {r["average"]["recon_mse"]:>10.4f} '
              f'{r["ghost_average"]:>10.0f}')
    print('=' * 90)


if __name__ == '__main__':
    main()
