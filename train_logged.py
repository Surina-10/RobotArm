"""Run train.py's GenesisArmEnv + PPO unchanged, but log every episode so a
learning curve can be plotted.

Additions over train.py (train.py itself is not modified):
  - TimeLimit of 200 steps per episode (train.py's env never truncates, so an
    episode only ends if the arm gets within 2 cm).
  - the final end-effector distance of each episode is logged to
    logs/monitor.csv alongside SB3's episode reward and length.
"""
import os
import sys
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import GenesisArmEnv  # noqa: E402

MAX_STEPS = 200
TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000


class LogDistance(gym.Wrapper):
    """Put the end-effector distance (m) into info so Monitor can record it."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info, distance=float(-reward))  # reward = -distance in train.py
        return obs, reward, terminated, truncated, info


os.makedirs("logs", exist_ok=True)
env = GenesisArmEnv()
env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_STEPS)
env = LogDistance(env)
env = Monitor(env, "logs/monitor", info_keywords=("distance",))

# Same algorithm and hyperparameters as train.py.
model = PPO("MlpPolicy", env, verbose=0, learning_rate=1e-3, seed=0)
print(f"training {TIMESTEPS} timesteps, episodes capped at {MAX_STEPS} steps", flush=True)
t0 = time.time()
model.learn(total_timesteps=TIMESTEPS)
model.save("logs/lss_arm_policy_logged")
print(f"done in {time.time() - t0:.0f}s -> logs/monitor.csv, logs/lss_arm_policy_logged.zip", flush=True)
