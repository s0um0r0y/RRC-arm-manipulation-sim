import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from utils.kinematics import get_ee_pose

def simulate_visualize(xml_path: str, site_name: str = "attachment_site", duration: float = 5.0):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0.0
    
    ee_trajectory = []
    times = []
    
    print(f"Starting kinematic trajectory on {site_name}...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        # joint positions ( panda has 7 joints + 2 gripper joints)
        q_init = data.qpos.copy()
        
        while viewer.is_running() and (time.time() - start_time) < duration:
            t = time.time() - start_time
            step_start = time.time()
            
            # joints 0 (base), 3 (elbow) and 5 (Wrist)
    