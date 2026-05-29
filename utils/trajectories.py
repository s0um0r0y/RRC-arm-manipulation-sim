import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def quintic_trajectory(t: float, T: float, p0:np.ndarray, pf:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pass

def slerp_trajectories(t: float, T: float, R0: np.ndarray, Rf: np.ndarray) -> np.ndarray:
    pass