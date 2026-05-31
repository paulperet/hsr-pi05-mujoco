import gymnasium as gym
from gymnasium import spaces

import logging
import numpy as np
import mujoco
import mujoco.viewer

from camera import Camera


class HSREnv(gym.Env):
    """HSR MuJoCo Environment for PI0.5 policy evaluation.

    Matches the training schema:
    - observation.image.head: 480x640x3 from 'rgbd' camera
    - observation.image.hand: 480x640x3 from 'hand' camera
    - observation.state: 8-dim [arm_lift, arm_flex, arm_roll, wrist_flex,
                                wrist_roll, hand_motor, head_pan, head_tilt]
    - action: 11-dim [8 absolute joint positions + 3 delta base (x, y, theta)]
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # Joint names matching the training schema observation.state ordering
    STATE_JOINT_NAMES = [
        "arm_lift_joint",
        "arm_flex_joint",
        "arm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
        "hand_motor_joint",
        "head_pan_joint",
        "head_tilt_joint",
    ]

    # Base joint names for delta action control
    BASE_JOINT_NAMES = [
        "slide_x",
        "slide_y",
        "base_roll_joint",
    ]

    # Actuator names in world.xml, ordered to match schema action indices
    ACTUATOR_NAMES = [
        "arm_lift_motor",    # 0
        "arm_flex_motor",    # 1
        "arm_roll_motor",    # 2
        "wrist_flex_motor",  # 3
        "wrist_roll_motor",  # 4
        "hand_motor",        # 5
        "head_pan_motor",    # 6
        "head_tilt_motor",   # 7
        "base_x_motor",      # 8
        "base_y_motor",      # 9
        "base_roll_motor",   # 10
    ]

    def __init__(self, args: dict):
        """Initialize HSR Environment for PI0.5 policy evaluation.

        Args:
        - args: dict with keys 'logging', 'render'.
        """
        path = 'hsr/models/world.xml'

        # Camera resolution matching training schema (480h x 640w)
        self.cam_height = 480
        self.cam_width = 640

        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cam_height, self.cam_width, 3))

        # Control frequency matching training data (30 FPS)
        self.sample_hz = 30
        self.timestep = 0.002
        self.iters = int((1 / self.sample_hz) / self.timestep)

        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)

        self.paused = False
        self.logging = args.get('logging', False)
        self.render = args.get('render', False)

        # Look up joint qpos indices for state observation
        self._state_joint_ids = []
        for name in self.STATE_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qadr = self.model.jnt_qposadr[jid]
            self._state_joint_ids.append(qadr)

        # Look up joint qpos indices for base (delta action)
        self._base_joint_ids = []
        for name in self.BASE_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qadr = self.model.jnt_qposadr[jid]
            self._base_joint_ids.append(qadr)

        # Look up actuator indices
        self._actuator_ids = []
        for name in self.ACTUATOR_NAMES:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self._actuator_ids.append(aid)

        # Dual cameras matching training schema
        cam_args = {'cam_width': self.cam_width, 'cam_height': self.cam_height}
        self.camera_head = Camera(cam_args, self.model, self.data, "rgbd")
        self.camera_hand = Camera(cam_args, self.model, self.data, "hand")

        if self.logging:
            self.logger = logging.basicConfig(filename='data/log/app.log',
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

        for i in range(self.model.ncam):
            print(f"Camera ID {i}: name = '{self.model.cam(i).name}'")

        if self.render:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                key_callback=self.key_callback,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.sync()

        # Store initial qpos for reset
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

    def get_head_image(self) -> np.ndarray:
        """Get image from the head (rgbd) camera.

        Returns:
                np.ndarray: (480, 640, 3) uint8 RGB image.
        """
        return self.camera_head.image

    def get_hand_image(self) -> np.ndarray:
        """Get image from the hand camera.

        Returns:
                np.ndarray: (480, 640, 3) uint8 RGB image.
        """
        return self.camera_hand.image

    def get_state(self) -> np.ndarray:
        """Get the 8-dim observation state vector matching the training schema.

        Returns:
                np.ndarray: (8,) float32 vector of joint positions:
                    [arm_lift, arm_flex, arm_roll, wrist_flex,
                     wrist_roll, hand_motor, head_pan, head_tilt]
        """
        state = np.array(
            [self.data.qpos[idx] for idx in self._state_joint_ids],
            dtype=np.float32
        )
        return state

    def _get_info(self):
        """Get current information about all joints."""

        observations = {
            "slide_x": self.data.qpos[self._base_joint_ids[0]],
            "slide_y": self.data.qpos[self._base_joint_ids[1]],
            "base_roll_joint": self.data.qpos[self._base_joint_ids[2]],
        }
        for i, name in enumerate(self.STATE_JOINT_NAMES):
            observations[name] = self.data.qpos[self._state_joint_ids[i]]

        return observations

    def step(self, action: np.ndarray):
        """Step the environment with an 11-dim action.

        Actions [0:8] are absolute joint positions for arm/head joints.
        Actions [8:11] are delta values for base (x, y, theta).

        Args:
        - action: np.ndarray of shape (11,) matching the training schema.
        """
        # Set absolute position targets for arm/head joint actuators (indices 0-7)
        for i in range(8):
            self.data.ctrl[self._actuator_ids[i]] = action[i]

        # Apply delta base actions: add to current position for base_x/base_y
        current_base_x = self.data.qpos[self._base_joint_ids[0]]
        current_base_y = self.data.qpos[self._base_joint_ids[1]]
        self.data.ctrl[self._actuator_ids[8]] = current_base_x + action[8]
        self.data.ctrl[self._actuator_ids[9]] = current_base_y + action[9]

        # base_t is velocity-controlled (delta rotation)
        self.data.ctrl[self._actuator_ids[10]] = action[10]

        # Couple finger proximal joints to hand_motor (simple proportional mapping)
        hand_motor_val = action[5]
        # Map hand_motor range to proximal joint range [0, 0.349066]
        grip_frac = np.clip((hand_motor_val + 0.798) / (1.239 + 0.798), 0, 1)
        grip_val = grip_frac * 0.349066
        hand_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hand_l_proximal_motor")
        hand_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hand_r_proximal_motor")
        self.data.ctrl[hand_l_id] = grip_val
        self.data.ctrl[hand_r_id] = grip_val

        # Simulate and record
        for i in range(self.iters):
            mujoco.mj_step(self.model, self.data)
            if self.logging:
                logging.info(f"action: {action} \r\n")
                logging.info(f"results: {self._get_info()} \r\n")
            if self.render:
                self.viewer.sync()

        # Record frames from both cameras after stepping
        self.camera_head.record_frame()
        self.camera_hand.record_frame()

    def reset(self):
        """Reset the environment to the initial state.

        Returns:
                tuple: (observation_state, info_dict)
        """
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)

        # Reset video recording buffers
        self.camera_head.reset_recording()
        self.camera_hand.reset_recording()

        if self.render:
            self.viewer.sync()

        return self.get_state(), self._get_info()

    def save_episode_video(self, output_dir: str, episode_idx: int, fps: int = 30):
        """Save recorded episode frames as MP4 videos.

        Args:
        - output_dir: Directory to save videos.
        - episode_idx: Episode index for filename.
        - fps: Frames per second for the output video.
        """
        head_path = f"{output_dir}/episode_{episode_idx:03d}_head.mp4"
        hand_path = f"{output_dir}/episode_{episode_idx:03d}_hand.mp4"
        self.camera_head.save_video(head_path, fps=fps)
        self.camera_hand.save_video(hand_path, fps=fps)

    def close(self):
        """
        Close the renderer if option.
        """
        if self.render:
            self.viewer.close()

    def key_callback(self, keycode):
        """
        Callback for viewer pause.
        """
        key = chr(keycode)
        if key == ' ':
            self.paused = not self.paused