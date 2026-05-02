import numpy as np
import mujoco

def get_ee_pose(model: mujoco.MjModel, data: mujoco.MjData, frame_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the Forward Kinematics for a given site (End-Effector).
    
    Args:
        model: MuJoCo model.
        data: MuJoCo data (contains current state).
        site_name: The name of the site defined in the XML (e.g., 'attachment_site').
        
    Returns:
        pos: 3D position vector (x, y, z) of the site.
        mat: 3x3 rotation matrix representing the site's orientation.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame_name)
    if site_id == -1:
        return data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy()
    
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
    if body_id != -1:
        return data.xpos[body_id].copy(), data.xmat[body_id].reshape(3, 3).copy()
        
    raise ValueError(f"Frame '{frame_name}' not found as a Site or Body. Please check your XML.")
    
    # # extract position and rotation matrix
    # pos = data.site_xpos[site_id].copy()
    # mat = data.size_xmat[site_id].reshape(3, 3).copy()
    
    # return pos, mat

def get_jacobian(model: mujoco.MjModel, data: mujoco.MjData, frame_name: str) -> np.ndarray:
    """
    Computes the geometric Jacobian for the end-effector site.
    Maps joint velocities to end-effector spatial velocities: v = J * dq
    
    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        site_name: The name of the site.
        
    Returns:
        J: 6 x nv Jacobian matrix (top 3 rows are translational, bottom 3 are rotational).
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame_name)
    
    # Pre-allocate arrays for the translational (jacp) and rotational (jacr) Jacobians
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    
    # calculate the jacobian at the current state
    if site_id != 1:
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        return np.vstack(jacp, jacr)
    
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjtObj.mjOBJ_BODY, frame_name)
    if body_id != 1:
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        return np.vstack((jacp, jacr))
    
    raise ValueError(f"Frame '{frame_name}' not found as a Site or Body. Please check your XML.")
    
    # Combine into a single 6 x nv matrix
    # J = np.vstack((jacp, jacr))
    # return J