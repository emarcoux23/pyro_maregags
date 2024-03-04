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

        self.reset_mode = "random"
        

        # Memory
        self.x = sys.x0
        self.u = sys.ubar
        self.t = t0

        # Init
        self.render_is_initiated = False

        if self.render_mode == "human":

            self.init_render()

    #################################################################
    def reset(self, seed=None, options=None):

        if self.reset_mode == "random":

            super().reset(seed=seed)

            self.x = self.np_random.uniform(self.sys.x_lb, self.sys.x_ub)
            self.u = self.sys.ubar
            self.t = 0.0

        elif self.reset_mode == "noisy_x0":

            super().reset(seed=seed)

            self.x = self.sys.x0 + 0.1 * self.np_random.uniform(
                self.sys.x_lb, self.sys.x_ub
            )
            self.u = self.sys.ubar
            self.t = 0.0

        else:

            self.x = self.sys.x0
            self.u = self.sys.ubar
            self.t = 0.0

        # Observation
        y = self.sys.h(self.x, self.u, self.t)

        # Info
        info = {"state": self.x, "action": self.u}

        return y, info

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
        terminated = False

        # Truncation of episodes if out of bounds
        truncated = (t_new > self.tf) or (not self.sys.isavalidstate(x_new))

        # Memory update
        self.x = x_new
        self.t = t_new
        self.u = u

        # Observation
        y = self.sys.h(self.x, self.u, self.t)

        # Info
        info = {"state": self.x, "action": self.u}

        if self.render_mode == "human":
            self.render()

        return y, r, terminated, truncated, info
    
    #################################################################
    def init_render(self):

        self.render_is_initiated = True

        self.animator = self.sys.get_animator()
        self.animator.show_plus(self.x, self.u, self.t)
        plt.pause(0.001)

    #################################################################
    def render(self):

        if self.render_mode == "human":
            if not self.render_is_initiated:
                self.init_render()
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

    sys = pendulum.InvertedPendulum()

    # Physical parameters
    sys.gravity = 10.0
    sys.m1 = 1.0
    sys.l1 = 1.0
    sys.lc1 = 0.5 * sys.l1
    sys.I1 = (1.0 / 12.0) * sys.m1 * sys.l1**2

    sys.l_domain = 2 * sys.l1  # graphical domain

    # Min/max state and control inputs
    sys.x_ub = np.array([+3.0 * np.pi, +20])
    sys.x_lb = np.array([-3.0 * np.pi, -20])
    sys.u_ub = np.array([+2.0])
    sys.u_lb = np.array([-2.0])

    # Cost Function
    # The reward function is defined as: r = -(theta2 + 0.1 * theta_dt2 + 0.001 * torque2)
    sys.cost_function.xbar = np.array([0, 0])  # target
    sys.cost_function.R[0, 0] = 0.001
    sys.cost_function.Q[0, 0] = 1.0
    sys.cost_function.Q[1, 1] = 0.1

    sys.x0 = np.array([ -np.pi, 0.0])

    gym_env = Sys2Gym(sys, render_mode=None)
    gym_env.reset_mode = "noisy_x0"

    model = PPO("MlpPolicy", gym_env, verbose=1)
    model.learn(total_timesteps=100000)

    gym_env = Sys2Gym(sys, render_mode="human")

    gym_env.reset_mode = "x0"

    episodes = 3
    for episode in range(episodes):
        y, info = gym_env.reset()
        terminated = False
        truncated = False

        print("\n Episode:", episode)
        while not (terminated or truncated):
            u, _states = model.predict(y, deterministic=True)
            y, r, terminated, truncated, info = gym_env.step(u)
            # print("t=", gym_env.t, "x=", gym_env.x, "u=", gym_env.u, "r=", r)
