#!/usr/bin/env python
""" Using Simplex method for find value of pisewise function.
Find y0 for given x0

"""

from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

import numpy as np

x = np.array([-4., 0., 1.  , 2., 3., 4.])
y = np.array([ 0., 0., 0.5, 2., 4., 7.])

x0 = 4


s = CyClpSimplex()
dim = len(x)

u = s.addVariable('u', dim)


A = np.matrix([
    [-4, 0, 1, 2, 3 ,4],
    [ 1, 1, 1, 1, 1, 1]
])

b = CyLPArray([x0,1])

s += A*u == b

for i in range(dim):
    s += u[i] >= 0

s.objectiveCoefficients = y
s.primal()


print(s.primalVariableSolution)
cond = s.primalVariableSolution['u']
print(np.dot(cond, x))
print(np.dot(cond, y))
