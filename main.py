import time
from turtle import mode
import mujoco
import mujoco.viewer

def inspect_run_panda(xml_path: str, duration: int = 10):
    try:
        print(f"Loading model from: {xml_path}")
    except Exception as e:
        print(f"Failed to load the robot model : {e}")
        
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print(f" verifying DoF: {model.nv}")
        
        print("\nActuated Joints:")
        for i in range(model.nu):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"  - Actuator {i}: {actuator_name}")
            
        print("\nDefined Sites (Often used for End-Effectors):")
        for i in range(model.nsite):
            site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            print(f"  - Site {i}: {site_name}")
            
        print("Launching MuJoCo Viewer. Press ESC or close the window to exit.")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            
            while viewer.is_running() and time.time() - start_time < duration:
                step_start = time.time()
                mujoco.mj_step(model, data)
                viewer.sync()
                
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
        print("Simulation completed.")

    except Exception as e:
        print(f"Failed to load sim: {e}")
            
if __name__ == "__main__":
    panda_xml = "third_party/mujoco_menagerie/franka_emika_panda/scene.xml"
    inspect_run_panda(panda_xml, duration=20)