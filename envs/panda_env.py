import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os

from utils.kinematics import get_ee_pose

class FrankaPandaEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}
    
    def __init__(self, xml_path: str, frame_name: str = "hand") -> None:
        super().__init__()
        
        self.xml_path = xml_path
        self.frame_name = frame_name
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML file not found at: {xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # We limit the torque commands based on typical physical limits (-80 to 80 Nm for safety)
        # panda robot is 7 dof
        self.action_space = spaces.Box(
            low=-80.0,
            high=80.0,
            shape=(7,),
            dtype=np.float32
        )
        
        # Observation Space size calculation:
        # - Joint positions (qpos): 7 elements (ignoring gripper fingers for now)
        # - Joint velocities (qvel): 7 elements
        # - End-effector position: 3 elements (X, Y, Z)
        # Total size = 7 + 7 + 3 = 17
        obs_shape = (17,)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
        
        # save the intial home position of the joints
        self.q_home = self.data.qpos[:7].copy()
        
    def _get_obs(self) -> np.ndarray:
        """Extracts the state observation from the simulator"""
        pass
    
    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        pass
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advances the simulation by one environment step given an action"""
        pass
        