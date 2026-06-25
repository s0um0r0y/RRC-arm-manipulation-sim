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
        self.ddq = cp.Variable(self.nv)
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
        
    def compute_torques(
        self, 
        J: np.ndarray, M: np.ndarray, qfrc_bias: np.ndarray,
        pos_curr: np.ndarray, mat_curr: np.ndarray, vel_curr: np.ndarray,
        pos_des: np.ndarray, mat_des: np.ndarray, vel_des: np.ndarray
    ) -> np.ndarray:
        
        # position error
        error_pos = pos_des - pos_curr
        error_vel_pos = vel_des[:3] - vel_curr[:3]
        
        # orientation error
        R_c = R.from_matrix(mat_curr)
        R_d = R.from_matrix(mat_des)
        error_ori = (R_d * R_c.inv()).as_rotvec()
        error_vel_ori = vel_des[3:] - vel_curr[3:]
        
        # commanded task space accelerations (impedance law)
        acc_pos = self.Kp_pos @ error_pos + self.Kd_pos @ error_vel_pos
        acc_ori = self.Kp_ori @ error_ori + self.Kd_ori @ error_vel_ori
        xdd_cmd = np.concatenate([acc_pos, acc_ori])
        
        # solve the QP for optimal joint acceleration (ddq)
        self.J_param.value = J
        self.xdd_cmd_param.value = xdd_cmd
        
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            ddq_opt = self.ddq.value
            if ddq_opt is None:
                raise ValueError("QP failed to find a solution.")
        except Exception:
            ddq_opt = np.zeros(self.nv)
            
        # Inverse dynamics: convert optimal accelerations into torques    
        tau = M @ ddq_opt + qfrc_bias
        
        return tau