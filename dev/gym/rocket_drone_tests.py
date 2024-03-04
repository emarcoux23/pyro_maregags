#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 08:27:06 2021

@author: alex
"""

import numpy as np

from pyro.dynamic.rocket        import Rocket
from pyro.dynamic.drone         import Drone2D

from stable_baselines3 import PPO

# Non-linear model
sys = Rocket()
sys = Drone2D()

sys.x_ub = np.array([10, 10, 10, 10, 10, 10])
sys.x_lb = -sys.x_ub

sys.u_ub = np.array([100, 100])
sys.u_lb = np.array([0, 0])

env = sys.convert_to_gymnasium()

env.reset_mode = "noisy_x0"

#env.render_mode = "human"

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)

env = sys.convert_to_gymnasium()
env.render_mode = "human"
env.reset_mode = "noisy_x0"

episodes = 10
for episode in range(episodes):
    y, info = env.reset()
    terminated = False
    truncated = False

    print("\n Episode:", episode)
    while not (terminated or truncated):
        u, _states = model.predict(y, deterministic=True)
        y, r, terminated, truncated, info = env.step(u)
