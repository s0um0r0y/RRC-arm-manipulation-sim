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
        # 7 joints of the arm
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:7].copy()
        
        # get end effector cartesian position
        ee_pos, _ = get_ee_pose(self.model, self.data, self.frame_name)
        
        # Concatenate into a single flat observation vector
        return np.concatenate([qpos, qvel, ee_pos]).astype(np.float32)
    
    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # reset the physics simulation data structure
        mujoco.mj_resetData(self.model, self.data)
        
        # adding initialization for robustness
        noise = self.np_random.uniform(low=-0.05, high=0.05, size=(7,))
        self.data.qpos[:7] = self.q_home + noise
        self.data.qvel[:7] = 0.0  # Start from a complete stop
        
        # forward kinematics to update the spatial positions after after modifying qpos
        mujoco.mj_forward(self.model, self.data)
        observation = self._get_obs()
        info = {}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advances the simulation by one environment step given an action"""
        # Clip actions to ensure they stay within physical torque bounds
        torque_action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # We only command the first 7 joints; the gripper fingers are kept passive (0 torque)
        self.data.ctrl[:7] = torque_action
        self.data.ctrl[7:] = 0.0
        
        # MuJoCo internal timestep is typically 0.002s (500Hz). 
        # Running 10 substeps gives an env step of 0.02s (50Hz).
        substeps = 10
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
            
        # Get updated observation
        observation = self._get_obs()
        
        reward = 0.0
        terminated = False
        truncated = False
        
        info = {
            "ee_position": observation[-3:].tolist() # Track EE pos for debugging
        }
        
        return observation, reward, terminated, truncated, info