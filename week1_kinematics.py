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
            data.qpos[0] = q_init[0] + 0.5 * np.sin(2.0 * t)
            data.qpos[3] = q_init[3] + 0.3 * np.sin(2.5 * t)
            data.qpos[5] = q_init[5] + 0.4 * np.sin(3.0 * t)
            
            # Forward kinematics
            mujoco.mj_kinematics(model, data)
            
            # extract task space (end-effector)
            pos, _ = get_ee_pose(model, data)
            ee_trajectory.append(pos)
            times.append(t)
            viewer.sync()
            
            # run at approx 60fps for 
            time_until_next_step = (1.0 / 60.0) - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
    plot_trajectory(np.array(ee_trajectory))            
    