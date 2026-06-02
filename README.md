## RRC manipulation task

- Week 0 – Environment Setup
1. Set up Python, PyTorch, MuJoCo, CVXPY, Stable-Baselines3
2. Run a basic MuJoCo simulation successfully
- Week 1 – Robotics & Math Basics
1. Understand robot joints, end-effectors, frames
2. Implement simple forward kinematics and visualize trajectories
- Week 2 – Control & Trajectories
1. Learn PD and impedance control concepts
2. Implement quintic position + SLERP orientation trajectory generation
- Week 3 – Simulation Environment
1. Build a Gym-style dual-arm MuJoCo environment
2. Implement observations and resets (object pose, joints, end-effector states)
- Week 4 – QP-Based Controller (Core DAVIL Component)
1. Implement CVXPY-based QP impedance controller
2. Enforce joint, torque, and end-effector constraints
3. Convert accelerations to torques and validate in simulation
- Week 5 – Reinforcement Learning (PPO)
1. Understand PPO and actor–critic training
2. Train a PPO policy to predict stiffness values (K)
3. Integrate RL policy with the QP controller

## Run instructions
```bash
uv sync

# to run the Franka setup
uv run main.py
```

![image](./assests/franka_setup.png)

## Week 1
- Forward Kinematics and trajectory pic
```
uv run week1_kinematics.py
```

![graph](./assests/graph.png)

## Week 2
- Trajectory plot from test trajectories
```
uv run week2_test_traj.py
```
![graph](./assests/trajectory_profiles.png)