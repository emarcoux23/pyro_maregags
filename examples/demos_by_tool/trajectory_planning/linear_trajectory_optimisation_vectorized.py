#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 08:27:06 2021

@author: alex
"""

import numpy as np

from pyro.dynamic.massspringdamper        import TwoMass
from pyro.planning.trajectoryoptimisation import DirectCollocationTrajectoryOptimisation
from pyro.analysis.costfunction          import QuadraticCostFunctionVectorized


sys = TwoMass()

sys.u_ub[0] = +2
sys.u_lb[0] = 0

cf = QuadraticCostFunctionVectorized( 4, 1)
# cf = sys.cost_function

planner = DirectCollocationTrajectoryOptimisation( sys , 0.1, 40 , cost_function = cf)

planner.x_start = np.array([0.5,0.5,0,0])
planner.x_goal  = np.array([0,0,0,0])

# planner.init_dynamic_plot()
planner.compute_optimal_trajectory()
# planner.show_solution()
planner.animate_solution()


