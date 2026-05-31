import datetime
import os

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np


class Camera:
    def __init__(self, args, model, data, cam_name: str = "", save_dir="data/img/"):
        """Initialize Camera instance.

        Args:
        - args: Arguments containing camera width and height.
        - model: Mujoco model.
        - data: Mujoco data.
        - cam_name: Name of the camera.
        - save_dir: Directory to save captured images.
        """
        self._args = args
        self._cam_name = cam_name
        self._model = model
        self._data = data
        self._save_dir = save_dir + self._cam_name + "/"

        self._width = self._args['cam_width']
        self._height = self._args['cam_height']
        self._cam_id = self._data.cam(self._cam_name).id

        self._renderer = mujoco.Renderer(self._model, self._height, self._width)
        self._camera = mujoco.MjvCamera()

        self._image = np.zeros((self._height, self._width, 3), dtype=np.uint8)

        # Video recording buffer
        self._frame_buffer = []

        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

    @property
    def height(self) -> int:
        """
        Get the height of the camera.

        Returns:
                int: The height of the camera.
        """
        return self._height

    @property
    def width(self) -> int:
        """
        Get the width of the camera.

        Returns:
                int: The width of the camera.
        """
        return self._width

    @property
    def save_dir(self) -> str:
        """
        Get the directory where images captured by the camera are saved.

        Returns:
                str: The directory where images captured by the camera are saved.
        """
        return self._save_dir

    @property
    def name(self) -> str:
        """
        Get the name of the camera.

        Returns:
                str: The name of the camera.
        """
        return self._cam_name

    @property
    def image(self) -> np.ndarray:
        """Return the captured RGB image (H, W, 3) uint8."""
        self._renderer.update_scene(self._data, camera=self.name)
        self._image = self._renderer.render()
        return self._image

    def record_frame(self) -> None:
        """Capture the current frame and add it to the video buffer."""
        self._frame_buffer.append(self.image.copy())

    def save_video(self, path: str, fps: int = 30) -> None:
        """Write buffered frames to an MP4 video file.

        Args:
        - path: Output file path for the MP4 video.
        - fps: Frames per second for the output video.
        """
        if not self._frame_buffer:
            print(f"[{self.name}] No frames to save.")
            return

        out_dir = os.path.dirname(path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        writer = imageio.get_writer(path, fps=fps, codec='libx264', quality=8)
        for frame in self._frame_buffer:
            writer.append_data(frame)
        writer.close()
        print(f"[{self.name}] Saved {len(self._frame_buffer)} frames to {path}")

    def reset_recording(self) -> None:
        """Clear the frame buffer for a new episode."""
        self._frame_buffer.clear()

    def save(self, img_name: str = "") -> None:
        """Saves the captured image.

        Args:
        - img_name: Name for the saved image file.
        """
        print(f"saving rgb image to {self.save_dir}")

        if img_name == "":
            timestamp = datetime.datetime.now()
            cv2.imwrite(
                self._save_dir + f"{timestamp}_rgb.png",
                cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR),
            )
        else:
            cv2.imwrite(
                self._save_dir + f"{img_name}_rgb.png",
                cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR),
            )