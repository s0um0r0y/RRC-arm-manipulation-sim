import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def quintic_trajectory(t: float, T: float, p0:np.ndarray, pf:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Computes the position, velocity, and acceleration for a quintic polynomial trajectory.
    Ensures zero velocity and acceleration at start and end times.

    Args:
        t (float): current time 
        T (float): total duration of the trajectory
        p0 (np.ndarray): initial position
        pf (np.ndarray): final position

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: position, velocity and acceleration at time t
    """
    t = np.clip(t, 0.0, T)
    s = t/T # normalized time
    
    # quintic polynomial for smooth start/stop
    # equation : 10*s^3 - 15*s^4 + 6*s^5
    c_pos = 10*s**3 - 15*s**4 + 6*s**5
    c_vel = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    c_acc = (60 * s - 180 * s**2 + 120 * s**3) / (T**2)
    
    pos = p0 + (pf-p0) * c_pos
    vel = (pf-p0) * c_vel
    acc = (pf-p0) * c_acc
    
    return pos, vel, acc

def slerp_trajectories(t: float, T: float, R0: np.ndarray, Rf: np.ndarray) -> np.ndarray:
    pass