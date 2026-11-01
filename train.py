import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

import genesis as gs

class GenesisArmEnv(gym.Env):
    """Custom Environment that wraps Genesis physics engine for RL training."""
    def __init__(self):
        super().__init__()
        
        # 1. Initialize Genesis in headless mode
        gs.init(backend=gs.cpu)
        self.scene = gs.Scene(show_viewer=False)
        
        # 2. Add Entities
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.arm = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/lss_arm.urdf",
                fixed=True
            )
        )
        self.scene.build()
        
        # Identify movable DoFs
        self.n_dofs = self.arm.n_dofs
        
        # 3. Define Action and Observation Spaces
        # Action: Target position changes (deltas) for each joint
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(self.n_dofs,), dtype=np.float32
        )
        
        # Observation: Current Joint Positions + Target 3D Position (X, Y, Z)
        obs_dim = self.n_dofs + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.target_pos = np.array([0.2, 0.0, 0.2], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset arm to zero position
        self.arm.set_dofs_position(np.zeros(self.n_dofs))
        self.scene.step()
        
        # Randomize target position slightly for training robustness
        self.target_pos = np.array([
            np.random.uniform(0.15, 0.3),
            np.random.uniform(-0.15, 0.15),
            np.random.uniform(0.1, 0.3)
        ], dtype=np.float32)
        
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        current_joints = self.arm.get_dofs_position().numpy()
        return np.concatenate([current_joints, self.target_pos]).astype(np.float32)

    def step(self, action):
        # Apply action delta to current joint angles
        current_joints = self.arm.get_dofs_position().numpy()
        target_joints = current_joints + action
        
        # Step physics forward
        self.arm.control_dofs_position(target_joints)
        self.scene.step()
        
        # Get End-Effector position (last link)
        ee_pos = self.arm.get_link('lss_arm_link_4').get_pos().numpy()
        
        # Calculate Reward (Negative distance to target)
        distance = np.linalg.norm(ee_pos - self.target_pos)
        reward = -distance
        
        # Termination conditions
        terminated = bool(distance < 0.02)  # Success if within 2 cm
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}

# --- RUN TRAINING ---
if __name__ == "__main__":
    env = GenesisArmEnv()
    
    # Train using Proximal Policy Optimization (PPO)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-3)
    
    print("Training RL agent inside Genesis simulation...")
    model.learn(total_timesteps=50000)
    
    # Save the trained policy model file
    model.save("lss_arm_policy")
    print("Saved trained model to lss_arm_policy.zip!")
