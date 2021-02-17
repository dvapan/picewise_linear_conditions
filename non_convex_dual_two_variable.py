#!/usr/bin/env python3
""" Dual solution for picewise linear function.
Find y0 for given x0
"""

from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

import numpy as np

x = np.arange(0, 1, 0.1, dtype=np.double)
z = np.arange(0, 1, 0.1, dtype=np.double)
x, z = np.meshgrid(x, z)
y = x * z

x = x.flatten()
z = z.flatten()
y = y.flatten()

mask = x+z < 1

x = x[mask]
z = z[mask]
y = y[mask]

x0, z0 = 0.5, 0.5

rhs_p = np.array([x0, z0, 1], dtype=np.double)

dim_d = len(rhs_p)


A_d = np.hstack([x.reshape(-1, 1),
                 z.reshape(-1, 1),
                 np.ones(len(x)).reshape(-1, 1)])
A_d = np.matrix(A_d)

s = CyClpSimplex()

l = s.addVariable('l', dim_d)

b_d = CyLPArray(y)

s += A_d * l >= b_d


s.optimizationDirection = 'max'
s.objectiveCoefficients = rhs_p

s.primal()

print(s.primalVariableSolution)
print(s.dualVariableSolution)

print(s.dualConstraintSolution)
print(s.primalConstraintSolution)
