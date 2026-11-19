import os

import numpy as np
import imageio

from mrplot.media import write_mp4_frames, write_png
from mrplot.rendering import RenderConfig, render_image_frame


def test_write_png_creates_file(tmp_path):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    path = write_png(str(tmp_path / "out" / "frame.png"), frame)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_write_mp4_frames_creates_file(tmp_path):
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
    path = write_mp4_frames(str(tmp_path / "out" / "video.mp4"), frames, fps=5)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_colorbar_mp4_keeps_rendered_frame_size(tmp_path):
    frame = render_image_frame(
        np.zeros((20, 20)), (1.0, 1.0), RenderConfig(show_colorbar=True),
        (0.0, 1.0), "title",
    )
    path = write_mp4_frames(str(tmp_path / "colorbar.mp4"), [frame], fps=5)
    with imageio.get_reader(path) as reader:
        assert reader.get_meta_data()["size"] == (frame.shape[1], frame.shape[0])
