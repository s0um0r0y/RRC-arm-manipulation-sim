import numpy as np
import cvxpy as cp 
from scipy.spatial.transform import Rotation as R

class QPImpedanceController:
    def __init__(self, nv: int, Kp_pos: np.ndarray, Kd_pos: np.ndarray, Kp_ori: np.ndarray, Kd_ori: np.ndarray) -> None:
        self.nv = nv
        self.Kp_pos = Kp_pos
        self.Kd_pos = Kd_pos
        self.Kp_ori = Kp_ori
        self.Kd_ori = Kd_ori
        
    def compute_torques():
        pass