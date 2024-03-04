#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:37:57 2024

@author: alex
"""

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = gym.make("LunarLander-v2", render_mode=None)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=250000)

env = gym.make("LunarLander-v2", render_mode="human")
obs, info = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, trunc, info = env.step(action)
    env.render("human")