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
    
def plot_trajectory(trajectory: np.ndarray):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    
    ax.plot(x, y, z, label='EE Path', color='b', linewidth=2)
    ax.scatter(x[0], y[0], z[0], color='g', s=100, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='r', s=100, label='End')
    
    ax.set_title("End-Effector 3D Trajectory (Forward Kinematics)")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.legend()
    plt.show()
    
