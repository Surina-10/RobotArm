# ============================================================================
# APPROACH 3: plotting collected simulation data
#
# Plots the joint positions and target-distance values from
# genesis_sim_observations.npy over all simulation steps (simulation_plot.png).
# ============================================================================

import numpy as np
import matplotlib
# Use non-interactive backend explicitly to avoid warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Load the recorded observation array
obs_data = np.load("genesis_sim_observations.npy")
print(f"Data shape loaded: {obs_data.shape}")  # (51201, 9)

# 2. Extract components
joint_positions = obs_data[:, 0:4]  # First 4 elements: Joint Angles
joint_velocities = obs_data[:, 4:8] # Next 4 elements: Joint Velocities
target_metric = obs_data[:, 8]      # 9th element: Target Distance Metric

# 3. Plot Joint Positions over Simulation Steps
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(joint_positions[:, 0], label="Joint 1")
plt.plot(joint_positions[:, 1], label="Joint 2")
plt.plot(joint_positions[:, 2], label="Joint 3")
plt.plot(joint_positions[:, 3], label="Joint 4")
plt.title("Simulation Joint Positions Over Time")
plt.xlabel("Simulation Steps")
plt.ylabel("Angle (radians)")
plt.legend()
plt.grid(True)

# 4. Plot Target Distance Metric over Simulation Steps
plt.subplot(2, 1, 2)
plt.plot(target_metric, color="purple", label="Distance to Target")
plt.title("Target Distance Metric Over Time")
plt.xlabel("Simulation Steps")
plt.ylabel("Distance Value")
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the plot directly as a PNG file
output_file = "simulation_plot.png"
plt.savefig(output_file, dpi=300)
print(f"Graph successfully saved to {output_file}!")
