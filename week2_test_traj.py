from turtle import position
import numpy as np
import matplotlib.pyplot as plt 
from utils.trajectories import quintic_trajectory

def test_quintic():
    T = 5.0 # 5 second movement
    times = np.linspace(0, T, 200)
    
    p0 = np.array([0.0, 0.0, 0.0])
    pf = np.array([0.1, 0.2, -0.05])
    
    position = []
    velocity = []
    acceleration = []
    
    for t in times:
        pos, vel, acc = quintic_trajectory(t, T, p0, pf)
        position.append(pos)
        velocity.append(vel)
        acceleration.append(acc)
        
    positions = np.array(position)
    velocities = np.array(velocity)
    accelerations = np.array(acceleration)
    
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle("Quintic Trajectory Profiles (X, Y, Z)", fontsize=16)
    
    # Position Plot
    axs[0].plot(times, positions[:, 0], 'r', label='X')
    axs[0].plot(times, positions[:, 1], 'g', label='Y')
    axs[0].plot(times, positions[:, 2], 'b', label='Z')
    axs[0].set_ylabel("Position (m)")
    axs[0].legend()
    axs[0].grid(True)
    
    # Velocity Plot
    axs[1].plot(times, velocities[:, 0], 'r')
    axs[1].plot(times, velocities[:, 1], 'g')
    axs[1].plot(times, velocities[:, 2], 'b')
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].grid(True)
    
    # Acceleration Plot
    axs[2].plot(times, accelerations[:, 0], 'r')
    axs[2].plot(times, accelerations[:, 1], 'g')
    axs[2].plot(times, accelerations[:, 2], 'b')
    axs[2].set_ylabel("Acceleration (m/s²)")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    test_quintic()