import numpy as np
import torch
import genesis as gs
from stable_baselines3 import PPO

# 1. Initialize Genesis
gs.init(backend=gs.gpu)  # Use gs.cpu if running on CPU

# 2. Setup Genesis Scene
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    viewer_options=gs.options.ViewerOptions(camera_pos=(1.5, 0.0, 1.0)),
)

# 3. Add Plane and Robot Arm
plane = scene.add_entity(gs.morphs.Plane())
arm = scene.add_entity(
    gs.morphs.URDF(file="lss_arm_description/urdf/lss_arm.urdf"),
)

# Build Scene
scene.build()

num_episodes = 100
dataset = []

print(f"Running {num_episodes} randomized Genesis simulation episodes...")

for episode in range(num_episodes):
    # A. Generate a randomized target goal coordinate (X, Y, Z)
    target_goal = np.array([
        np.random.uniform(0.15, 0.35),  # X distance (m)
        np.random.uniform(-0.2, 0.2),   # Y distance (m)
        np.random.uniform(0.05, 0.25),  # Z distance (m)
    ])

    # B. Add slight random noise to initial joint positions (Domain Randomization)
    initial_joint_noise = np.random.normal(loc=0.0, scale=0.05, size=4)
    arm.set_dofs_position(initial_joint_noise)
    
    # Step simulation forward
    for step in range(50):
        scene.step()
        
        # Read current joint states from Genesis
        current_joints = arm.get_dofs_position().cpu().numpy()
        current_vels = arm.get_dofs_velocity().cpu().numpy()
        
        # Create full 9-element observation: [4 pos, 4 vel, 1 target metric]
        distance_to_target = np.linalg.norm(target_goal)
        obs = np.concatenate([current_joints, current_vels, [distance_to_target]])
        
        # Save step observation
        dataset.append(obs)

print(f"Successfully collected {len(dataset)} simulation observations!")

# 4. Train the PPO Policy
print("Training PPO model on simulation data...")

# Note: Wrap your custom Genesis environment or dataset into Gymnasium interface
# or train PPO using Stable-Baselines3 on the Genesis Env:
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=50000)

# 5. Save Model for Real Robot Deployment
model.save("lss_arm_policy.zip")
print("Saved trained model to lss_arm_policy.zip!")
