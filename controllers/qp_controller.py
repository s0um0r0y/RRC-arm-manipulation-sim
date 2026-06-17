from cvxpy.constraints import constraint
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
        
        # CVXPY setup
        self.ddq = cp,cp.Variable(self.nv)
        self.J_param = cp.Parameter((6, self.nv))
        self.xdd_cmd_param = cp.Parameter(6)
        
        # regularization weight to keep joint accelerations small
        self.lam = 0.01
        
        # objective: minmize ||J * ddq - xdd_cmd||^2 + lambda * ||ddq||^2
        cost = cp.sum_squares(self.J_param @ self.ddq - self.xdd_cmd_param) + self.lam * cp.sum_squares(self.ddq)
        
        # constraints : limit maximum joint accelerations for safety
        # (e.g., -10 to 10 rad/s^2)
        constraints = [
            self.ddq >= -10.0,
            self.ddq <= 10.0
        ]
        
        self.prob = cp.Problem(cp.Minimize(cost), constraints)
        
    def compute_torques():
        pass