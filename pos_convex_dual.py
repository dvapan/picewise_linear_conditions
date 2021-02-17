#!/usr/bin/env python
""" Dual solution for picewise linear function.
Find y0 for given x0
"""
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

import numpy as np

x = np.array([-4., 0., 1.  , 2., 3., 4.])
y = np.array([ 0., 0., 0.5, 2., 4., 7.])

x0 = 3.5

A = np.hstack([x.reshape(-1,1),np.ones(len(x)).reshape(-1,1)])
A = np.matrix(A)
s = CyClpSimplex()
dim = 2

l = s.addVariable('l', dim)
y = CyLPArray(y)
s += A*l <= y

s.optimizationDirection = 'max'
s.objectiveCoefficients = np.array([x0, 1])



s.primal()

print(s.primalVariableSolution)
print(s.dualVariableSolution)

print(s.dualConstraintSolution)
print(s.primalConstraintSolution)
