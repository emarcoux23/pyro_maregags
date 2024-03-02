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

from pyro.dynamic import pendulum

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# %%
# Declaration and Initialization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Our custom environment will inherit from the abstract class
# ``gymnasium.Env``. You shouldn’t forget to add the ``metadata``
# attribute to your class. There, you should specify the render-modes that
# are supported by your environment (e.g. ``"human"``, ``"rgb_array"``,
# ``"ansi"``) and the framerate at which your environment should be
# rendered. Every environment should support ``None`` as render-mode; you
# don’t need to add it in the metadata. In ``GridWorldEnv``, we will
# support the modes “rgb_array” and “human” and render at 4 FPS.
#
# The ``__init__`` method of our environment will accept the integer
# ``size``, that determines the size of the square grid. We will set up
# some variables for rendering and define ``self.observation_space`` and
# ``self.action_space``. In our case, observations should provide
# information about the location of the agent and target on the
# 2-dimensional grid. We will choose to represent observations in the form
# of dictionaries with keys ``"agent"`` and ``"target"``. An observation
# may look like ``{"agent": array([1, 0]), "target": array([0, 3])}``.
# Since we have 4 actions in our environment (“right”, “up”, “left”,
# “down”), we will use ``Discrete(4)`` as an action space. Here is the
# declaration of ``GridWorldEnv`` and the implementation of ``__init__``:


class SysEnv(gym.Env):

    def __init__(self, sys, dt=0.1, tf=10.0, render_mode=None):

        # x-y ouputs
        y_ub = np.array([+1, +1, sys.x_ub[1]])
        y_lb = np.array([-1, -1, sys.x_lb[1]])

        self.observation_space = spaces.Box(y_lb, y_ub)
        self.action_space = spaces.Box(sys.u_lb, sys.u_ub)

        self.sys = sys
        self.dt = dt

        self.tf = tf
        self.render_mode = render_mode

        # Memory
        self.x = sys.x0
        self.u = sys.ubar
        self.t = 0.0

        if self.render_mode == "human":

            self.animator = self.sys.get_animator()
            self.animator.show_plus(self.x, self.u, self.t)
            plt.pause(0.001)

    # %%
    # Constructing Observations From Environment States
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # Since we will need to compute observations both in ``reset`` and
    # ``step``, it is often convenient to have a (private) method ``_get_obs``
    # that translates the environment’s state into an observation. However,ƒ
    # this is not mandatory and you may as well compute observations in
    # ``reset`` and ``step`` separately:ƒ

    def _get_obs(self):

        theta = self.x[0]
        thetadot = self.x[1]

        y = np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

        return y

    # %%
    # We can also implement a similar method for the auxiliary information
    # that is returned by ``step`` and ``reset``. In our case, we would like
    # to provide the manhattan distance between the agent and the target:

    def _get_info(self):

        return {"state": self.x, "action": self.u}

    # %%
    # Reset
    # ~~~~~
    #
    # The ``reset`` method will be called to initiate a new episode. You may
    # assume that the ``step`` method will not be called before ``reset`` has
    # been called. Moreover, ``reset`` should be called whenever a done signal
    # has been issued. Users may pass the ``seed`` keyword to ``reset`` to
    # initialize any random number generator that is used by the environment
    # to a deterministic state. It is recommended to use the random number
    # generator ``self.np_random`` that is provided by the environment’s base
    # class, ``gymnasium.Env``. If you only use this RNG, you do not need to
    # worry much about seeding, *but you need to remember to call
    # ``super().reset(seed=seed)``* to make sure that ``gymnasium.Env``
    # correctly seeds the RNG. Once this is done, we can randomly set the
    # state of our environment. In our case, we randomly choose the agent’s
    # location and the random sample target positions, until it does not
    # coincide with the agent’s position.
    #
    # The ``reset`` method should return a tuple of the initial observation
    # and some auxiliary information. We can use the methods ``_get_obs`` and
    # ``_get_info`` that we implemented earlier for that:

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.x = np.random.uniform(np.array([-np.pi, -1]), np.array([np.pi, 1]))
        self.u = self.sys.ubar
        self.t = 0.0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # %%
    # Step
    # ~~~~
    #
    # The ``step`` method usually contains most of the logic of your
    # environment. It accepts an ``action``, computes the state of the
    # environment after applying that action and returns the 5-tuple
    # ``(observation, reward, terminated, truncated, info)``. See
    # :meth:`gymnasium.Env.step`. Once the new state of the environment has
    # been computed, we can check whether it is a terminal state and we set
    # ``done`` accordingly. Since we are using sparse binary rewards in
    # ``GridWorldEnv``, computing ``reward`` is trivial once we know
    # ``done``.To gather ``observation`` and ``info``, we can again make
    # use of ``_get_obs`` and ``_get_info``:

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

        x_new[0] = self.angle_normalize(x_new[0])

        # Sat speed --> I hate they do this in gym env
        if x_new[1] > sys.x_ub[1]:
            x_new[1] = sys.x_ub[1]
        if x_new[1] < sys.x_lb[1]:
            x_new[1] = sys.x_lb[1]

        # Cost function
        r = -self.sys.cost_function.g(x, u, t)

        terminated = t > self.tf

        truncated = not self.sys.isavalidstate(x_new)

        # Memory update
        self.x = x_new
        self.t = t_new
        self.u = u

        y = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return y, r, terminated, truncated, info

    def angle_normalize(self, x):

        return ((x + np.pi) % (2 * np.pi)) - np.pi

    # %%
    # Rendering
    # ~~~~~~~~~
    #
    # Here, we are using PyGame for rendering. A similar approach to rendering
    # is used in many environments that are included with Gymnasium and you
    # can use it as a skeleton for your own environments:

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):

        self.animator.show_plus_update(self.x, self.u, self.t)
        plt.pause(0.001)


sys = pendulum.InvertedPendulum()

# Physical parameters
sys.gravity = 10.0
sys.m1 = 1.0
sys.lc1 = 1.0

# Min/max state and control inputs
sys.x_ub = np.array([+2 * np.pi, +8])
sys.x_lb = np.array([-2 * np.pi, -8])
sys.u_ub = np.array([+2.0])
sys.u_lb = np.array([-2.0])

# Cost Function
# The reward function is defined as: r = -(theta2 + 0.1 * theta_dt2 + 0.001 * torque2)
sys.cost_function.xbar = np.array([0, 0])  # target
sys.cost_function.R[0, 0] = 0.001
sys.cost_function.Q[0, 0] = 1.0
sys.cost_function.Q[1, 1] = 0.1

gym_env = SysEnv(sys, dt=0.05, render_mode=None)

model = PPO("MlpPolicy", gym_env, verbose=1)
model.learn(total_timesteps=250000)
# model.learn(total_timesteps=1000)

gym_env = SysEnv(sys, render_mode="human")
y, info = gym_env.reset()

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
