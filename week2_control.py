from asyncio.tasks import Task
import time
import numpy as np
import mujoco
import mujoco.viewer

from utils.kinematics import get_ee_pose, get_jacobian
from utils.trajectories import quintic_trajectory, slerp_trajectories
from controllers.pd_controller import TaskSpacePDController

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

if __name__ == "__main__":
    panda_xml = "third_party/mujoco_menagerie/franka_emika_panda/scene.xml"
    run_pd_control(panda_xml, frame_name="hand")