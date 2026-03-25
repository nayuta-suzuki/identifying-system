"""
IS Experiment Step 9: Two-Frame Object Separation.

Core insight (from N):
"連続する2枚の画像だけから、動いているものが分離できるのでは？
背景は、ほとんどわからないが、それはわからなくて当然であり、
再構築の上でも問題はない。"

Algorithm:
1. Take two consecutive frames A, B
2. Compute |B - A| → changed pixels
3. Changed pixels = where the object was (in A) + where it is now (in B)
4. Unchanged pixels → definitely background → background color estimate
5. Each frame: pixels differing from background color → full object body
6. Background behind object is unknown (occluded) — and that's fine
7. Reconstruction: background (where visible) + object on top

This is the most minimal version of object-background separation.
No need for 30 frames. Two frames suffice.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def separate_from_two_frames(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    diff_threshold: float = 5.0,
    obj_threshold: float = 15.0,
) -> dict:
    """
    Separate moving object from background using only two consecutive frames.

    Returns dict with:
        - changed_mask: (H, W) bool — pixels that changed between A and B
        - unchanged_mask: (H, W) bool — pixels that didn't change
        - bg_color: (3,) float64 — estimated background color from unchanged pixels
        - object_mask_a: (H, W) bool — object body in frame A
        - object_mask_b: (H, W) bool — object body in frame B
        - bg_visible_a: (H, W, 3) float64 — background as seen in frame A (object region = NaN)
        - bg_visible_b: (H, W, 3) float64 — background as seen in frame B (object region = NaN)
        - object_pixels_a: (H, W, 3) float64 — object pixels in A (background region = 0)
        - object_pixels_b: (H, W, 3) float64 — object pixels in B (background region = 0)
    """
    a = frame_a.astype(np.float64)
    b = frame_b.astype(np.float64)
    H, W, C = a.shape

    # Step 1: What changed?
    diff = np.sqrt(np.sum((b - a) ** 2, axis=-1))
    changed_mask = diff > diff_threshold
    unchanged_mask = ~changed_mask

    # Step 2: Unchanged pixels → background color
    n_unchanged = unchanged_mask.sum()
    if n_unchanged > 0:
        bg_color = a[unchanged_mask].mean(axis=0)
    else:
        bg_color = np.median(np.concatenate([a, b]).reshape(-1, C), axis=0)

    # Step 3: Object = pixels that differ from background color
    diff_a = np.sqrt(np.sum((a - bg_color) ** 2, axis=-1))
    diff_b = np.sqrt(np.sum((b - bg_color) ** 2, axis=-1))
    object_mask_a = diff_a > obj_threshold
    object_mask_b = diff_b > obj_threshold

    # Step 4: Separate
    # Background: what we can see (non-object pixels). Object region → unknown (NaN)
    bg_visible_a = np.full((H, W, C), np.nan)
    bg_visible_a[~object_mask_a] = a[~object_mask_a]

    bg_visible_b = np.full((H, W, C), np.nan)
    bg_visible_b[~object_mask_b] = b[~object_mask_b]

    # Object: just the object pixels
    object_pixels_a = np.zeros((H, W, C))
    object_pixels_a[object_mask_a] = a[object_mask_a]

    object_pixels_b = np.zeros((H, W, C))
    object_pixels_b[object_mask_b] = b[object_mask_b]

    return {
        'changed_mask': changed_mask,
        'unchanged_mask': unchanged_mask,
        'bg_color': bg_color,
        'object_mask_a': object_mask_a,
        'object_mask_b': object_mask_b,
        'bg_visible_a': bg_visible_a,
        'bg_visible_b': bg_visible_b,
        'object_pixels_a': object_pixels_a,
        'object_pixels_b': object_pixels_b,
    }


def reconstruct_from_separation(
    bg_color: np.ndarray,
    object_mask: np.ndarray,
    object_pixels: np.ndarray,
    frame_shape: tuple,
) -> np.ndarray:
    """
    Reconstruct a frame from separated components.

    Background (where visible) = bg_color.
    Object region = object pixels on top.
    The background behind the object is unknown, but reconstruction
    doesn't need it — the object covers it.
    """
    H, W, C = frame_shape
    recon = np.full((H, W, C), bg_color, dtype=np.float64)
    recon[object_mask] = object_pixels[object_mask]
    return recon


def evaluate_reconstruction(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Evaluate per-pixel reconstruction accuracy."""
    orig = original.astype(np.float64)
    error = orig - reconstructed
    mse = np.mean(error ** 2)
    max_err = np.max(np.abs(error))
    total_ss = np.sum((orig - orig.mean()) ** 2)
    recon_ss = np.sum(error ** 2)
    var_explained = 1.0 - recon_ss / total_ss if total_ss > 0 else 1.0
    return {'mse': mse, 'max_error': max_err, 'var_explained': var_explained}


def visualize_two_frame(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    result: dict,
    recon_a: np.ndarray,
    recon_b: np.ndarray,
    metrics_a: dict,
    metrics_b: dict,
    t_a: int,
    t_b: int,
    video_name: str,
    save_path: Path,
):
    """Visualize two-frame separation results."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    # Row 0: Input frames and diff
    axes[0, 0].imshow(frame_a)
    axes[0, 0].set_title(f'Frame A (t={t_a})')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(frame_b)
    axes[0, 1].set_title(f'Frame B (t={t_b})')
    axes[0, 1].axis('off')

    diff = np.sqrt(np.sum((frame_b.astype(float) - frame_a.astype(float)) ** 2, axis=-1))
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('|B - A|')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(result['changed_mask'], cmap='gray')
    n_changed = result['changed_mask'].sum()
    n_total = result['changed_mask'].size
    axes[0, 3].set_title(f'Changed pixels\n{n_changed}/{n_total} ({n_changed/n_total:.1%})')
    axes[0, 3].axis('off')

    # Row 1: Object masks (full body)
    axes[1, 0].imshow(result['object_mask_a'], cmap='gray')
    axes[1, 0].set_title(f'Object in A\n({result["object_mask_a"].sum()} px)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(result['object_mask_b'], cmap='gray')
    axes[1, 1].set_title(f'Object in B\n({result["object_mask_b"].sum()} px)')
    axes[1, 1].axis('off')

    # Object pixels
    obj_a_display = result['object_pixels_a'].copy()
    obj_a_display[~result['object_mask_a']] = 0
    axes[1, 2].imshow(np.clip(obj_a_display, 0, 255).astype(np.uint8))
    axes[1, 2].set_title('Object pixels (A)')
    axes[1, 2].axis('off')

    obj_b_display = result['object_pixels_b'].copy()
    obj_b_display[~result['object_mask_b']] = 0
    axes[1, 3].imshow(np.clip(obj_b_display, 0, 255).astype(np.uint8))
    axes[1, 3].set_title('Object pixels (B)')
    axes[1, 3].axis('off')

    # Row 2: Background (visible parts)
    bg_a_display = result['bg_visible_a'].copy()
    bg_a_display[np.isnan(bg_a_display)] = 0  # Show unknown as black
    axes[2, 0].imshow(np.clip(bg_a_display, 0, 255).astype(np.uint8))
    axes[2, 0].set_title('BG visible (A)\nblack = unknown')
    axes[2, 0].axis('off')

    bg_b_display = result['bg_visible_b'].copy()
    bg_b_display[np.isnan(bg_b_display)] = 0
    axes[2, 1].imshow(np.clip(bg_b_display, 0, 255).astype(np.uint8))
    axes[2, 1].set_title('BG visible (B)\nblack = unknown')
    axes[2, 1].axis('off')

    # Unknown regions highlighted
    unknown_a = result['object_mask_a'].astype(float)
    unknown_b = result['object_mask_b'].astype(float)
    axes[2, 2].imshow(unknown_a, cmap='Reds', vmin=0, vmax=1)
    axes[2, 2].set_title('Unknown BG (A)')
    axes[2, 2].axis('off')

    axes[2, 3].imshow(unknown_b, cmap='Reds', vmin=0, vmax=1)
    axes[2, 3].set_title('Unknown BG (B)')
    axes[2, 3].axis('off')

    # Row 3: Reconstruction
    axes[3, 0].imshow(np.clip(recon_a, 0, 255).astype(np.uint8))
    axes[3, 0].set_title(f'Recon A\nVar={metrics_a["var_explained"]:.4f}')
    axes[3, 0].axis('off')

    axes[3, 1].imshow(np.clip(recon_b, 0, 255).astype(np.uint8))
    axes[3, 1].set_title(f'Recon B\nVar={metrics_b["var_explained"]:.4f}')
    axes[3, 1].axis('off')

    # Error heatmaps
    err_a = np.sqrt(np.sum((frame_a.astype(float) - recon_a) ** 2, axis=-1))
    err_b = np.sqrt(np.sum((frame_b.astype(float) - recon_b) ** 2, axis=-1))
    vmax_err = max(err_a.max(), err_b.max(), 1)

    axes[3, 2].imshow(err_a, cmap='hot', vmin=0, vmax=vmax_err)
    axes[3, 2].set_title(f'Error A\nmax={err_a.max():.1f}')
    axes[3, 2].axis('off')

    axes[3, 3].imshow(err_b, cmap='hot', vmin=0, vmax=vmax_err)
    axes[3, 3].set_title(f'Error B\nmax={err_b.max():.1f}')
    axes[3, 3].axis('off')

    # Summary
    bg = result['bg_color']
    summary = (
        f'BG color estimate: ({bg[0]:.1f}, {bg[1]:.1f}, {bg[2]:.1f})   '
        f'True: (40, 60, 120)\n'
        f'Recon A: Var={metrics_a["var_explained"]:.4f}  MSE={metrics_a["mse"]:.4f}  '
        f'MaxErr={metrics_a["max_error"]:.1f}\n'
        f'Recon B: Var={metrics_b["var_explained"]:.4f}  MSE={metrics_b["mse"]:.4f}  '
        f'MaxErr={metrics_b["max_error"]:.1f}'
    )
    fig.text(0.02, 0.01, summary, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle(f'{video_name}: Two-Frame Separation (t={t_a}, t={t_b})', fontsize=14)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    data_dir = Path(__file__).parent / 'data'
    output_dir = Path(__file__).parent / 'output'

    video_names = ['v1_triangle_right', 'v2_square_diag', 'v3_circle_down', 'v4_two_objects']

    print('=' * 80)
    print('IS Experiment: Two-Frame Object Separation')
    print('  "連続する2枚の画像だけから、動いているものが分離できる"')
    print('=' * 80)
    print()

    # Test with multiple frame pairs to verify consistency
    pair_offsets = [
        (0, 1),     # first pair
        (14, 15),   # middle pair
        (28, 29),   # last pair
    ]

    for name in video_names:
        filepath = data_dir / f'{name}.npy'
        if not filepath.exists():
            print(f'  {name}: not found, skipping')
            continue

        frames = np.load(filepath)
        n_frames, H, W, C = frames.shape
        print(f'=== {name} ({n_frames} frames, {H}x{W}) ===')

        for t_a, t_b in pair_offsets:
            if t_b >= n_frames:
                continue

            frame_a = frames[t_a]
            frame_b = frames[t_b]

            # Separate
            result = separate_from_two_frames(frame_a, frame_b)

            # Reconstruct both frames
            recon_a = reconstruct_from_separation(
                result['bg_color'], result['object_mask_a'],
                result['object_pixels_a'], (H, W, C)
            )
            recon_b = reconstruct_from_separation(
                result['bg_color'], result['object_mask_b'],
                result['object_pixels_b'], (H, W, C)
            )

            # Evaluate
            metrics_a = evaluate_reconstruction(frame_a, recon_a)
            metrics_b = evaluate_reconstruction(frame_b, recon_b)

            print(f'  Pair (t={t_a}, t={t_b}):')
            print(f'    BG color estimate: ({result["bg_color"][0]:.1f}, '
                  f'{result["bg_color"][1]:.1f}, {result["bg_color"][2]:.1f})')
            print(f'    Changed pixels: {result["changed_mask"].sum()} / {H * W}')
            print(f'    Object A: {result["object_mask_a"].sum()} px  '
                  f'Object B: {result["object_mask_b"].sum()} px')
            print(f'    Recon A: Var={metrics_a["var_explained"]:.4f}  '
                  f'MSE={metrics_a["mse"]:.4f}  MaxErr={metrics_a["max_error"]:.1f}')
            print(f'    Recon B: Var={metrics_b["var_explained"]:.4f}  '
                  f'MSE={metrics_b["mse"]:.4f}  MaxErr={metrics_b["max_error"]:.1f}')

            # Visualize (only middle pair for brevity)
            if t_a == 14:
                visualize_two_frame(
                    frame_a, frame_b, result, recon_a, recon_b,
                    metrics_a, metrics_b, t_a, t_b,
                    name, output_dir / f'{name}_two_frame.png'
                )

        print()

    # --- The key point ---
    print('=' * 80)
    print('KEY INSIGHT:')
    print('  2 consecutive frames suffice to separate object from background.')
    print('  The background behind the object is unknown — and that is correct.')
    print('  Reconstruction is exact because the object covers the unknown region.')
    print('  No 30-frame accumulation needed. No iterative refinement needed.')
    print('=' * 80)


if __name__ == '__main__':
    main()
