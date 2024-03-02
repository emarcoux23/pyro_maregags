#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:08:03 2024

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

#################################################################
# Create a Gym Environment from a Pyro System
#################################################################


class Sys2Gym(gym.Env):

    metadata = {"render_modes": ["human"]}

    #################################################################
    def __init__(self, sys, dt=0.05, tf=10.0, t0=0.0, render_mode=None):

        self.observation_space = spaces.Box(sys.x_lb, sys.x_ub)
        self.action_space = spaces.Box(sys.u_lb, sys.u_ub)

        self.sys = sys
        self.dt = dt

        self.tf = tf  # For truncation of episodes
        self.render_mode = render_mode

        # Memory
        self.x = sys.x0
        self.u = sys.ubar
        self.t = t0

        if self.render_mode == "human":

            self.animator = self.sys.get_animator()
            self.animator.show_plus(self.x, self.u, self.t)
            plt.pause(0.001)

    #################################################################
    def _get_obs(self):

        y = self.sys.h(self.x, self.u, self.t)

        return y

    #################################################################
    def _get_info(self):

        return {"state": self.x, "action": self.u}

    #################################################################
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.x = np.random.uniform(self.sys.x_lb, self.sys.x_ub)
        self.u = self.sys.ubar
        self.t = 0.0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    #################################################################
    def step(self, u):

        u = np.clip(u, self.sys.u_lb, self.sys.u_ub)
        x = self.x
        t = self.t
        dt = self.dt

        # Derivatives
        dx = self.sys.f(x, u, t)

        # Euler integration
        x_new = x + dx * dt
        t_new = t + dt

        # Reward = negative of cost function
        r = -self.sys.cost_function.g(x, u, t)

        # Termination of episodes
        terminated = t_new > self.tf

        # Truncation of episodes if out of bounds
        truncated = not self.sys.isavalidstate(x_new)

        # Memory update
        self.x = x_new
        self.t = t_new
        self.u = u

        # Observation
        y = self._get_obs()

        # Info
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return y, r, terminated, truncated, info

    #################################################################
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    #################################################################
    def _render_frame(self):

        self.animator.show_plus_update(self.x, self.u, self.t)
        plt.pause(0.001)


"""
#################################################################
##################          Main                         ########
#################################################################
"""


if __name__ == "__main__":
    """MAIN TEST"""

    from pyro.dynamic import pendulum

    from stable_baselines3 import PPO

    sys = pendulum.SinglePendulum()

    gym_env = Sys2Gym(sys, render_mode="human")

    model = PPO("MlpPolicy", gym_env, verbose=1)
    model.learn(total_timesteps=1000)

    episodes = 10
    for episode in range(episodes):
        y, info = gym_env.reset()
        terminated = False
        truncated = False

        print("\n Episode:", episode)
        while not (terminated or truncated):
            u, _states = model.predict(y)
            y, r, terminated, truncated, info = gym_env.step(u)
            print("t=", gym_env.t, "x=", gym_env.x, "u=", gym_env.u, "r=", r)
