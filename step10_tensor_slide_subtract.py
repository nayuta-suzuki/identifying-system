"""
IS Experiment Step 10: Tensor Slide-and-Subtract.

The IS-native approach to object-background separation:
1. From 2 consecutive frames, detect motion → extract the moving region as a TENSOR (average shape)
2. For each frame in the video, SLIDE the tensor across the image → find the position of best match
3. SUBTRACT the tensor at that position → what remains is the background for that frame
4. Combine all per-frame backgrounds → clean overall background

This is fundamentally different from experiments 4-5 which used raw pixels.
Here the object is represented as a single tensor (template), and separation
happens through the IS operations: slide (convolution), match, subtract.

This directly implements:
- "テンソルを画像上でスライドさせながら最も一致する場所を探す"
- "再構築は「テンソルの重み付き配置のリスト」として表現される"
- "残差＝元の画像から、一致したテンソルをその一致度の分だけ引いたもの"
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


def extract_tensor_from_two_frames(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    diff_threshold: float = 5.0,
    obj_threshold: float = 15.0,
) -> dict:
    """
    From two consecutive frames, extract the moving object as a tensor.

    Steps:
    1. |B - A| → changed pixels → background color from unchanged pixels
    2. In each frame, pixels differing from bg color → object mask (full body)
    3. Crop the object region from both frames, align by centroid, average → tensor

    Returns dict with:
        - tensor: (th, tw, 3) float64 — the object tensor (average shape)
        - bg_color: (3,) float64 — estimated background color
        - centroid_a: (y, x) — object centroid in frame A
        - centroid_b: (y, x) — object centroid in frame B
        - mask_a, mask_b: (H, W) bool — object masks
    """
    a = frame_a.astype(np.float64)
    b = frame_b.astype(np.float64)
    H, W, C = a.shape

    # Step 1: Background color from unchanged pixels
    diff = np.sqrt(np.sum((b - a) ** 2, axis=-1))
    unchanged = diff <= diff_threshold
    if unchanged.sum() > 0:
        bg_color = a[unchanged].mean(axis=0)
    else:
        bg_color = np.median(np.concatenate([a, b]).reshape(-1, C), axis=0)

    # Step 2: Object masks (full body)
    diff_a = np.sqrt(np.sum((a - bg_color) ** 2, axis=-1))
    diff_b = np.sqrt(np.sum((b - bg_color) ** 2, axis=-1))
    mask_a = diff_a > obj_threshold
    mask_b = diff_b > obj_threshold

    # Step 3: Extract object crops, align by centroid, average → tensor
    tensors = []
    centroids = []
    for mask, frame in [(mask_a, a), (mask_b, b)]:
        if mask.sum() == 0:
            continue
        ys, xs = np.where(mask)
        cy, cx = ys.mean(), xs.mean()
        centroids.append((cy, cx))

        # Crop bounding box with padding
        pad = 2
        y0 = max(0, ys.min() - pad)
        x0 = max(0, xs.min() - pad)
        y1 = min(H, ys.max() + 1 + pad)
        x1 = min(W, xs.max() + 1 + pad)

        crop = frame[y0:y1, x0:x1].copy()
        # Zero out non-object pixels (replace with bg_color then subtract)
        crop_mask = mask[y0:y1, x0:x1]
        # Store as deviation from background
        crop_deviation = np.zeros_like(crop)
        crop_deviation[crop_mask] = crop[crop_mask] - bg_color
        tensors.append(crop_deviation)

    if len(tensors) == 0:
        # No object found
        return {
            'tensor': np.zeros((1, 1, C)),
            'bg_color': bg_color,
            'centroid_a': (H // 2, W // 2),
            'centroid_b': (H // 2, W // 2),
            'mask_a': mask_a,
            'mask_b': mask_b,
        }

    # Align tensors to same size (take the larger bounding box)
    max_h = max(t.shape[0] for t in tensors)
    max_w = max(t.shape[1] for t in tensors)

    aligned = []
    for t in tensors:
        padded = np.zeros((max_h, max_w, C))
        dy = (max_h - t.shape[0]) // 2
        dx = (max_w - t.shape[1]) // 2
        padded[dy:dy + t.shape[0], dx:dx + t.shape[1]] = t
        aligned.append(padded)

    # Average → tensor
    tensor = np.mean(aligned, axis=0)

    centroid_a = centroids[0] if len(centroids) > 0 else (H // 2, W // 2)
    centroid_b = centroids[1] if len(centroids) > 1 else centroid_a

    return {
        'tensor': tensor,
        'bg_color': bg_color,
        'centroid_a': centroid_a,
        'centroid_b': centroid_b,
        'mask_a': mask_a,
        'mask_b': mask_b,
    }


def slide_and_match(
    frame: np.ndarray,
    tensor: np.ndarray,
    bg_color: np.ndarray,
) -> tuple[tuple[int, int], float]:
    """
    Slide the tensor across the frame and find the position of best match.

    This is the IS operation: "テンソルを画像上でスライドさせながら最も一致する場所を探す"

    Uses normalized cross-correlation for matching.
    The frame is first converted to deviation-from-background space
    (same space as the tensor).

    Returns:
        position: (y, x) — top-left corner of best match
        score: float — match score (normalized cross-correlation peak)
    """
    frame_f = frame.astype(np.float64)
    # Convert to deviation from background
    frame_dev = frame_f - bg_color

    th, tw = tensor.shape[:2]
    H, W = frame_f.shape[:2]

    if th > H or tw > W:
        return (0, 0), 0.0

    # Cross-correlation across all channels, summed
    corr = np.zeros((H - th + 1, W - tw + 1))
    for c in range(3):
        t_chan = tensor[:, :, c]
        f_chan = frame_dev[:, :, c]

        # Use FFT-based correlation
        full_corr = fftconvolve(f_chan, t_chan[::-1, ::-1], mode='valid')
        corr += full_corr

    # Normalize by tensor energy
    tensor_energy = np.sum(tensor ** 2)
    if tensor_energy > 0:
        corr /= np.sqrt(tensor_energy)

    # Find peak
    peak_idx = np.unravel_index(corr.argmax(), corr.shape)
    score = corr[peak_idx]

    return peak_idx, score


def subtract_tensor_at_position(
    frame: np.ndarray,
    tensor: np.ndarray,
    position: tuple[int, int],
    bg_color: np.ndarray,
) -> np.ndarray:
    """
    Subtract the tensor from the frame at the given position.

    残差＝元の画像から、一致したテンソルをその位置で引いたもの

    Returns:
        residual: (H, W, 3) float64 — frame with tensor subtracted
    """
    frame_f = frame.astype(np.float64)
    residual = frame_f.copy()

    y0, x0 = position
    th, tw = tensor.shape[:2]
    H, W = frame_f.shape[:2]

    # Clip to frame bounds
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1 = min(H, y0 + th)
    sx1 = min(W, x0 + tw)

    ty0 = sy0 - y0
    tx0 = sx0 - x0
    ty1 = ty0 + (sy1 - sy0)
    tx1 = tx0 + (sx1 - sx0)

    # Subtract tensor (which is in deviation-from-bg space)
    residual[sy0:sy1, sx0:sx1] -= tensor[ty0:ty1, tx0:tx1]

    return residual


def reconstruct_frame(
    bg_color: np.ndarray,
    tensor: np.ndarray,
    position: tuple[int, int],
    frame_shape: tuple,
) -> np.ndarray:
    """
    Reconstruct a frame from: background_color + tensor @ position.

    再構築 = 背景テンソル + 物体テンソル×位置(t)
    """
    H, W, C = frame_shape
    recon = np.full((H, W, C), bg_color, dtype=np.float64)

    y0, x0 = position
    th, tw = tensor.shape[:2]

    sy0, sx0 = max(0, y0), max(0, x0)
    sy1 = min(H, y0 + th)
    sx1 = min(W, x0 + tw)

    ty0 = sy0 - y0
    tx0 = sx0 - x0
    ty1 = ty0 + (sy1 - sy0)
    tx1 = tx0 + (sx1 - sx0)

    recon[sy0:sy1, sx0:sx1] += tensor[ty0:ty1, tx0:tx1]

    return recon


def evaluate(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    orig = original.astype(np.float64)
    error = orig - reconstructed
    mse = np.mean(error ** 2)
    max_err = np.max(np.abs(error))
    total_ss = np.sum((orig - orig.mean()) ** 2)
    recon_ss = np.sum(error ** 2)
    var_explained = 1.0 - recon_ss / total_ss if total_ss > 0 else 1.0
    return {'mse': mse, 'max_error': max_err, 'var_explained': var_explained}


def visualize_results(
    frames: np.ndarray,
    tensor: np.ndarray,
    bg_color: np.ndarray,
    positions: list,
    scores: list,
    residuals: np.ndarray,
    combined_bg: np.ndarray,
    metrics_per_frame: list,
    video_name: str,
    save_path: Path,
):
    """Visualize tensor extraction, slide-match, subtraction, and reconstruction."""
    n_show = 5
    indices = np.linspace(0, len(frames) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(5, n_show, figsize=(3 * n_show, 15))

    # Row 0: Original frames with detected position
    for i, idx in enumerate(indices):
        axes[0, i].imshow(frames[idx])
        if positions[idx] is not None:
            y0, x0 = positions[idx]
            th, tw = tensor.shape[:2]
            rect = plt.Rectangle((x0, y0), tw, th,
                                  linewidth=2, edgecolor='lime', facecolor='none')
            axes[0, i].add_patch(rect)
        axes[0, i].set_title(f't={idx}\nscore={scores[idx]:.0f}')
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('Original\n+ match', rotation=0, labelpad=60, va='center', fontsize=10)

    # Row 1: Residuals (frame - tensor)
    for i, idx in enumerate(indices):
        res_display = np.clip(residuals[idx], 0, 255).astype(np.uint8)
        axes[1, i].imshow(res_display)
        axes[1, i].axis('off')
    axes[1, 0].set_ylabel('Residual\n(frame-tensor)', rotation=0, labelpad=60, va='center', fontsize=10)

    # Row 2: Reconstruction (bg_color + tensor @ position)
    for i, idx in enumerate(indices):
        recon = reconstruct_frame(bg_color, tensor, positions[idx], frames[0].shape)
        recon_display = np.clip(recon, 0, 255).astype(np.uint8)
        axes[2, i].imshow(recon_display)
        var = metrics_per_frame[idx]['var_explained']
        axes[2, i].set_title(f'Var={var:.4f}')
        axes[2, i].axis('off')
    axes[2, 0].set_ylabel('Recon\n(bg+tensor)', rotation=0, labelpad=60, va='center', fontsize=10)

    # Row 3: Error heatmap
    for i, idx in enumerate(indices):
        recon = reconstruct_frame(bg_color, tensor, positions[idx], frames[0].shape)
        error = np.sqrt(np.sum((frames[idx].astype(float) - recon) ** 2, axis=-1))
        axes[3, i].imshow(error, cmap='hot', vmin=0, vmax=max(error.max(), 1))
        axes[3, i].set_title(f'max={error.max():.1f}')
        axes[3, i].axis('off')
    axes[3, 0].set_ylabel('Error', rotation=0, labelpad=60, va='center', fontsize=10)

    # Row 4: Tensor, combined background, ghost check
    # Tensor display
    t_display = tensor.copy()
    vmax = np.abs(t_display).max()
    if vmax > 0:
        t_display = t_display / (2 * vmax) + 0.5
    axes[4, 0].imshow(np.clip(t_display, 0, 1))
    axes[4, 0].set_title(f'Object Tensor\n{tensor.shape[0]}x{tensor.shape[1]}')
    axes[4, 0].axis('off')

    # Tensor as actual color (bg + tensor deviation)
    t_color = bg_color + tensor
    axes[4, 1].imshow(np.clip(t_color, 0, 255).astype(np.uint8))
    axes[4, 1].set_title('Tensor\n(actual color)')
    axes[4, 1].axis('off')

    # Combined background
    axes[4, 2].imshow(np.clip(combined_bg, 0, 255).astype(np.uint8))
    axes[4, 2].set_title('Combined BG\n(avg residuals)')
    axes[4, 2].axis('off')

    # Ghost check
    true_bg = np.array([40, 60, 120], dtype=np.float64)
    ghost = np.sqrt(np.sum((combined_bg - true_bg) ** 2, axis=-1))
    axes[4, 3].imshow(ghost, cmap='hot', vmin=0, vmax=max(ghost.max(), 1))
    axes[4, 3].set_title(f'Ghost\nmax={ghost.max():.1f}')
    axes[4, 3].axis('off')

    # Position trajectory
    valid_pos = [(p[1], p[0]) for p in positions if p is not None]
    if valid_pos:
        xs, ys = zip(*valid_pos)
        axes[4, 4].imshow(frames[0])
        axes[4, 4].plot(xs, ys, 'g.-', markersize=3, linewidth=1)
        axes[4, 4].set_title('Trajectory')
    axes[4, 4].axis('off')

    axes[4, 0].set_ylabel('Tensor &\nBackground', rotation=0, labelpad=60, va='center', fontsize=10)

    fig.suptitle(f'{video_name}: Tensor Slide-and-Subtract', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    data_dir = Path(__file__).parent / 'data'
    output_dir = Path(__file__).parent / 'output'

    video_names = ['v1_triangle_right', 'v2_square_diag', 'v3_circle_down', 'v4_two_objects']

    print('=' * 80)
    print('IS Experiment: Tensor Slide-and-Subtract')
    print('  2フレームからテンソルを抽出 → 各フレームでスライドマッチング → 引き算 → 背景')
    print('=' * 80)
    print()

    for name in video_names:
        filepath = data_dir / f'{name}.npy'
        if not filepath.exists():
            print(f'  {name}: not found, skipping')
            continue

        frames = np.load(filepath)
        n_frames, H, W, C = frames.shape
        print(f'=== {name} ({n_frames} frames, {H}x{W}) ===')

        # Step 1: Extract tensor from first two frames
        result = extract_tensor_from_two_frames(frames[0], frames[1])
        tensor = result['tensor']
        bg_color = result['bg_color']

        print(f'  Tensor shape: {tensor.shape}')
        print(f'  Tensor energy: {np.sum(tensor ** 2):.0f}')
        print(f'  BG color estimate: ({bg_color[0]:.1f}, {bg_color[1]:.1f}, {bg_color[2]:.1f})')
        print(f'  True BG color:     (40, 60, 120)')

        # Step 2: For each frame, slide tensor and find best match position
        positions = []
        scores = []
        residuals = np.zeros_like(frames, dtype=np.float64)
        metrics_per_frame = []

        for t in range(n_frames):
            pos, score = slide_and_match(frames[t], tensor, bg_color)
            positions.append(pos)
            scores.append(score)

            # Step 3: Subtract tensor at matched position → residual (per-frame background)
            residuals[t] = subtract_tensor_at_position(frames[t], tensor, pos, bg_color)

            # Evaluate reconstruction
            recon = reconstruct_frame(bg_color, tensor, pos, (H, W, C))
            metrics = evaluate(frames[t], recon)
            metrics_per_frame.append(metrics)

        # Step 4: Combine per-frame residuals → overall background
        combined_bg = residuals.mean(axis=0)

        # Overall metrics
        all_var = [m['var_explained'] for m in metrics_per_frame]
        all_mse = [m['mse'] for m in metrics_per_frame]
        all_max = [m['max_error'] for m in metrics_per_frame]

        print(f'  --- Per-frame reconstruction ---')
        print(f'    Var explained: min={min(all_var):.4f}  mean={np.mean(all_var):.4f}  max={max(all_var):.4f}')
        print(f'    MSE:           min={min(all_mse):.2f}  mean={np.mean(all_mse):.2f}  max={max(all_mse):.2f}')
        print(f'    Max error:     min={min(all_max):.1f}  mean={np.mean(all_max):.1f}  max={max(all_max):.1f}')

        # Ghost check on combined background
        true_bg = np.array([40, 60, 120], dtype=np.float64)
        ghost = np.sum((combined_bg - true_bg[None, None, :]) ** 2)
        print(f'  --- Combined background ---')
        print(f'    Ghost energy: {ghost:.0f}')
        ghost_max = np.sqrt(np.sum((combined_bg - true_bg) ** 2, axis=-1)).max()
        print(f'    Ghost max:    {ghost_max:.1f}')

        # Score range
        print(f'  --- Match scores ---')
        print(f'    min={min(scores):.0f}  mean={np.mean(scores):.0f}  max={max(scores):.0f}')

        # Position trajectory
        if positions:
            ys = [p[0] for p in positions]
            xs = [p[1] for p in positions]
            print(f'  --- Trajectory ---')
            print(f'    Y: {min(ys)}-{max(ys)}  X: {min(xs)}-{max(xs)}')

        print()

        # Visualize
        visualize_results(
            frames, tensor, bg_color, positions, scores,
            residuals, combined_bg, metrics_per_frame,
            name, output_dir / f'{name}_tensor_slide.png'
        )

    print('=' * 80)
    print('This experiment uses ISの3つの操作:')
    print('  1. テンソル抽出: 2フレームの動く部分 → 平均像としてのテンソル')
    print('  2. スライドマッチング: テンソルを画像上でスライド → 最も一致する場所')
    print('  3. 引き算: フレーム − テンソル@位置 → 残差（＝各時点の背景）')
    print('=' * 80)


if __name__ == '__main__':
    main()
