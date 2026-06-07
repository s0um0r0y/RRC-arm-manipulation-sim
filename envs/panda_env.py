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
        
    def _get_obs(self) -> np.ndarray:
        """Extracts the state observation from the simulator"""
        pass
    
    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        pass
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advances the simulation by one environment step given an action"""
        pass
        