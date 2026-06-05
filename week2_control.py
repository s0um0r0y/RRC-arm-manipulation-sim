from asyncio.tasks import Task
import time
import numpy as np
import mujoco
import mujoco.viewer

from utils.kinematics import get_ee_pose, get_jacobian
from utils.trajectories import quintic_trajectory, slerp_trajectories
from controllers.pd_controller import TaskSpacePDController
from scipy.spatial.transform import Rotation as R

def run_pd_control(xml_path: str, frame_name: str = "hand"):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    model.opt.gravity[:] = [0, 0, -9.81]
    
    # Panda is a heavy robot, so we need strong stiffness (Kp) and damping (Kd)
    # Damping is typically tuned as 2 * sqrt(Kp) for a critically damped response
    Kp_p = np.diag([2000.0, 2000.0, 2000.0])
    Kd_p = np.diag([80.0, 80.0, 80.0])
    
    Kp_o = np.diag([100.0, 100.0, 100.0])
    Kd_o = np.diag([20.0, 20.0, 20.0])
    
    controller = TaskSpacePDController(Kp_p, Kd_p, Kp_o, Kd_o)
    
    # Forward step once to initialize the kinematic tree
    mujoco.mj_step(model, data)
    
    # Trajectory parameters
    duration = 5.0
    pos_start, mat_start = get_ee_pose(model, data, frame_name)
    
    # Move 20cm forward in X, 15cm left in Y, and stay at same Z
    pos_goal = pos_start + np.array([0.2, 0.15, 0.0])
    
    # rotate 45 degree around the Z-axis
    rot_offset = R.from_euler('z', 45, degrees=True).as_matrix()
    mat_goal = rot_offset @ mat_start
    
    print(f"Starting PD Control. Tracking trajectory for {duration} seconds...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        # might need to change the duration later
        while viewer.is_running() and (time.time() - start_time) < (duration + 2.0):
            t = time.time() - start_time
            step_start = time.time()
            
            # get current state
            pos_curr, mat_curr = get_ee_pose(model, data, frame_name)
            J = get_jacobian(model, data, frame_name)
            
            # end effector spatial velocity (v = J * dq)
            vel_curr = J @ data.qvel
            
            # desired state from trajectory generators
            pos_des, vel_des_pos, _ = quintic_trajectory(t, duration, pos_start, pos_goal)
            mat_des = slerp_trajectories(t, duration, mat_start, mat_goal)
            
            # combine desired velocities
            vel_des = np.zeros(6)
            vel_des[:3] = vel_des_pos
            
            # compute torques
            # pass data.qfrc_bias (gravity + Coriolis) to keep the arm from falling 
            torques = controller.compute_torques(
                J, pos_curr, mat_curr, vel_curr, 
                pos_des, mat_des, vel_des, 
                data.qfrc_bias
            )
            
            # We only apply torques to the first 7 actuators (the arm). The gripper actuators stay at 0.
            data.ctrl[:7] = torques[:7]
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # timesync with MuJoCo timstep
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)
            

if __name__ == "__main__":
    panda_xml = "third_party/mujoco_menagerie/franka_emika_panda/scene.xml"
    run_pd_control(panda_xml, frame_name="hand")