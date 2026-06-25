import time
import numpy as np
import mujoco
import mujoco.viewer

from utils.kinematics import get_ee_pose, get_jacobian
from utils.trajectories import quintic_trajectory, slerp_trajectories
from controllers.qp_controller import QPImpedanceController
from scipy.spatial.transform import Rotation as R

def run_qp_control(xml_path: str, frame_name: str = "hand"):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = [0, 0, -9.81]
    
    # Kp and Kd are typically lower for Inverse Dynamics controllers 
    # compared to direct J^T PD controllers
    Kp_p = np.diag([400.0, 400.0, 400.0])
    Kd_p = np.diag([40.0, 40.0, 40.0])
    Kp_o = np.diag([50.0, 50.0, 50.0])
    Kd_o = np.diag([10.0, 10.0, 10.0])
    
    # actuated joints (7 for the arm)
    controller = QPImpedanceController(7, Kp_p, Kd_p, Kp_o, Kd_o)
    mujoco.mj_step(model, data)
    
    duration = 5.0
    pos_start, mat_start = get_ee_pose(model, data, frame_name)
    pos_goal = pos_start + np.array([0.2, -0.2, 0.1])
    
    mat_goal = (R.from_euler('x', 30, degrees=True) * R.from_matrix(mat_start)).as_matrix()
    print("Starting QP Optimization Controller...")
    
    mass_matrix = np.zeros((model.nv, model.nv))
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            t = min(time.time() - start_time, duration)
            
            # state extraction
            pos_curr, mat_curr = get_ee_pose(model, data, frame_name)
            
            # We slice [:7] to only grab the Jacobian components for the arm, ignoring the gripper fingers
            J = get_jacobian(model, data, frame_name)[:, :7]
            vel_curr = J @ data.qvel[:7]
            
            # extract mass matrix (M) for inverse dynamics
            mujoco.mj_fullM(model, mass_matrix, data.qM)
            M_arm = mass_matrix[:7, :7]
            
            # extract gravity
            bias_arm = data.qfrc_bias[:7]
            
            # desired state
            pos_des, vel_des_pos, _ = quintic_trajectory(t, duration, pos_start, pos_goal)
            mat_des = slerp_trajectories(t, duration, mat_start, mat_goal)
            vel_des = np.zeros(6)
            vel_des[:3] = vel_des_pos
            
            torques = controller.compute_torques(
                J, M_arm, bias_arm,
                pos_curr, mat_curr, vel_curr,
                pos_des, mat_des, vel_des
            )
            
            data.ctrl[:7] = torques
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    panda_xml = "third_party/mujoco_menagerie/franka_emika_panda/scene.xml"
    run_qp_control(panda_xml)                