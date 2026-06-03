import numpy as np
from scipy.spatial.transform import Rotation as R

class TaskSpacePDController:
    def __init__(self, 
                 Kp_pos: np.ndarray, 
                 Kd_pos: np.ndarray, 
                 Kp_ori: np.ndarray, 
                 Kd_ori: np.ndarray) -> None:
        """
        Initializes the controller with stiffness (Kp) and damping (Kd) matrices.
        Matrices should be 3x3 diagonal matrices.
        """
        self.Kp_pos = Kp_pos
        self.Kd_pos = Kd_pos
        self.Kp_ori = Kp_ori
        self.Kd_ori = Kd_ori
        
    def compute_torques(
        self,
        J: np.ndarray,
        pos_curr: np.ndarray,
        mat_curr: np.ndarray,
        vel_curr: np.ndarray,
        pos_des: np.ndarray,
        mat_des: np.ndarray,
        vel_des: np.ndarray,
        qfrd_bias: np.ndarray
    ) -> np.ndarray:
        """Computes joint torques to track a desired task space trajectory

        Args:
            J (np.ndarray): Jacobian matrix
            pos_curr (np.ndarray): current position (3,)
            mat_curr (np.ndarray): orientation (3x3)
            vel_curr (np.ndarray): spatial velocity (6,)
            pos_des (np.ndarray): desired position (3,)
            mat_des (np.ndarray): orientation (3x3)
            vel_des (np.ndarray): spatial velocity (6,)
            qfrd_bias (np.ndarray): Gravity, Coriolis and centrifugal forces from Mujuco (nv,)

        Returns:
            np.ndarray: tau Joint torques (nv,)
        """
        
        # position error
        error_pos = pos_des - pos_curr
        error_vel_pos = vel_des[:3] - vel_curr[:3]
        
        # orientation error
        # R_err = R_des * R_curr^T
        R_c = R.from_matrix(mat_curr)
        R_d = R.from_matrix(mat_des)
        error_ori = (R_d * R_c.inv()).as_rotvec()
        error_vel_ori = vel_des[3:] - vel_curr[3:]
        
        # compute cartesian wrench (force and torque in task space)
        force_pos = self.Kp_pos @ error_pos + self.Kd_pos @ error_vel_pos
        force_ori = self.Kd_ori @ error_ori + self.Kd_ori @ error_vel_ori
        
        # combine into a 6D spatial wrench [F_x, F_y, F_z, Tau_x, Tau_y, Tau_z]
        wrench = np.concatenate([force_pos, force_ori])
        
        # map to joint torques (tau = J^T * F) and add gravity compensation
        tau = J.T @ wrench + qfrd_bias
        
        return tau