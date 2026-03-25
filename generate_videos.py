"""
IS Experiment Step 1: Generate synthetic video data.

Generates short videos (64x64, 30 frames) of simple geometric shapes
(triangle, square, circle) moving via translation on a solid background.
Each video has 1-2 objects. No rotation or scaling.

Output: numpy arrays saved as .npy files + visualization as PNG strips.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def draw_triangle(canvas, cx, cy, size, color):
    """Draw a filled equilateral triangle centered at (cx, cy)."""
    h, w = canvas.shape[:2]
    half = size // 2
    # Vertices of equilateral triangle (pointing up)
    pts = [
        (cx, cy - half),                          # top
        (cx - half, cy + half),                    # bottom-left
        (cx + half, cy + half),                    # bottom-right
    ]
    # Simple scanline fill
    for y in range(max(0, cy - half), min(h, cy + half + 1)):
        # Interpolate x range at this y
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


def generate_video(
    shape_name: str,
    bg_color: np.ndarray,
    obj_color: np.ndarray,
    start_pos: tuple[int, int],
    velocity: tuple[int, int],
    obj_size: int = 10,
    n_frames: int = 30,
    frame_size: int = 64,
) -> np.ndarray:
    """
    Generate a single video of a shape moving on a solid background.

    Returns: (n_frames, frame_size, frame_size, 3) uint8 array
    """
    frames = np.zeros((n_frames, frame_size, frame_size, 3), dtype=np.uint8)
    draw_fn = SHAPE_DRAWERS[shape_name]

    cx, cy = start_pos
    vx, vy = velocity

    for t in range(n_frames):
        # Fill background
        frame = np.full((frame_size, frame_size, 3), bg_color, dtype=np.uint8)
        # Draw shape at current position
        draw_fn(frame, int(cx + vx * t), int(cy + vy * t), obj_size, obj_color)
        frames[t] = frame

    return frames


def generate_two_object_video(
    shapes: list[dict],
    bg_color: np.ndarray,
    n_frames: int = 30,
    frame_size: int = 64,
) -> np.ndarray:
    """Generate a video with two objects."""
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


def visualize_video(frames: np.ndarray, title: str, save_path: Path):
    """Save a strip of selected frames as PNG."""
    n_show = min(10, len(frames))
    indices = np.linspace(0, len(frames) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(1, n_show, figsize=(2 * n_show, 2))
    for i, idx in enumerate(indices):
        axes[i].imshow(frames[idx])
        axes[i].set_title(f't={idx}')
        axes[i].axis('off')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def main():
    data_dir = Path(__file__).parent / 'data'
    output_dir = Path(__file__).parent / 'output'
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    bg_blue = np.array([40, 60, 120], dtype=np.uint8)

    videos = {}

    # Video 1: White triangle moving right
    videos['v1_triangle_right'] = generate_video(
        shape_name='triangle',
        bg_color=bg_blue,
        obj_color=np.array([240, 240, 240], dtype=np.uint8),
        start_pos=(10, 32),
        velocity=(1.5, 0),
        obj_size=12,
    )

    # Video 2: Red square moving diagonally
    videos['v2_square_diag'] = generate_video(
        shape_name='square',
        bg_color=bg_blue,
        obj_color=np.array([220, 60, 60], dtype=np.uint8),
        start_pos=(10, 10),
        velocity=(1.2, 1.0),
        obj_size=10,
    )

    # Video 3: Green circle moving down
    videos['v3_circle_down'] = generate_video(
        shape_name='circle',
        bg_color=bg_blue,
        obj_color=np.array([60, 200, 80], dtype=np.uint8),
        start_pos=(32, 8),
        velocity=(0, 1.5),
        obj_size=7,
    )

    # Video 4: Two objects — triangle and circle, different directions
    videos['v4_two_objects'] = generate_two_object_video(
        shapes=[
            {
                'name': 'triangle',
                'start_pos': (10, 20),
                'velocity': (1.5, 0.3),
                'size': 10,
                'color': np.array([240, 240, 240], dtype=np.uint8),
            },
            {
                'name': 'circle',
                'start_pos': (50, 50),
                'velocity': (-1.0, -0.5),
                'size': 7,
                'color': np.array([240, 200, 60], dtype=np.uint8),
            },
        ],
        bg_color=bg_blue,
    )

    # Save all videos
    for name, frames in videos.items():
        np.save(data_dir / f'{name}.npy', frames)
        visualize_video(frames, name, output_dir / f'{name}_strip.png')
        print(f'{name}: shape={frames.shape}, saved')

    print(f'\nAll videos saved to {data_dir}')
    print(f'Visualizations saved to {output_dir}')


if __name__ == '__main__':
    main()
