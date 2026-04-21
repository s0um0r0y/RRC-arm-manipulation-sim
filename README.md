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
- Week 6 – Rewards, Stability & Logging
1. Implement DAVIL reward terms (tracking, infeasibility penalty, EMA smoothness)
2. Add logging, checkpoints, and seed control
3. Train stable policies on simple object cases
- Week 7 – Full DAVIL Training
1. Train across multiple objects and masses
2. Evaluate tracking performance over many trajectories
