import time
import numpy as np
import mujoco.viewer
from envs.panda_env import FrankaPandaEnv

def test_environment_loop():
    panda_xml = "third_party/mujoco_menagerie/franka_emika_panda/scene.xml"
    
    print("Initializing FrankaPandaEnv...")
    env = FrankaPandaEnv(xml_path=panda_xml, frame_name="hand")
    
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space Shape: {env.observation_space.shape}")
    
    obs, info = env.reset()
    print(f"Initial Observation Shape: {obs.shape}")
    print(f"Initial End-Effector Position: {obs[-3:]}")
    
    print("\nLaunching Passive Viewer for Random Rollout. Press ESC to quit...")
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        start_time = time.time()
        steps = 0
        
        while viewer.is_running():
            step_start = time.time()
            
            # Only step the environment for the first 150 steps
            if steps < 150:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                
                if steps % 30 == 0:
                    print(f"Step {steps} | EE Position: {info['ee_position']}")
                    
            viewer.sync()
                
            # Keep execution synced close to real-world control frequencies
            # Our environment control period is 10 physics steps * 0.002s = 0.02s
            elapsed = time.time() - step_start
            if elapsed < 0.02:
                time.sleep(0.02 - elapsed)
                
    print("\nEnvironment loop completed successfully! Architecture verified.")

if __name__ == "__main__":
    test_environment_loop()