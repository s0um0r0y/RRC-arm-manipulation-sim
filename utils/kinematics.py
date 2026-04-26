import numpy as np
import mujoco

def get_ee_pose(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> tuple[np.ndarray, np.ndarray]:
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
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"Site '{site_name}' not found in the model.")
    
    # extract position and rotation matrix
    pos = data.site_xpos[site_id].copy()
    mat = data.size_xmat[site_id].reshape(3, 3).copy()
    
    return pos, mat