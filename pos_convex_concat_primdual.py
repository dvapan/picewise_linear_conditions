#!/usr/bin/env python
""" Cocatination between primal and dual linear problem.
We have eqvivalent dual and primal solutions so we can concatinate.
This actions can help include piecwise conditions to linear problems.
"""

from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray


import numpy as np

x = np.array([-4., 0., 1.  , 2., 3., 4.])
y = np.array([ 0., 0., 0.5, 2., 4., 7.])

x01 = 2
x02 = 4

rhs_p = np.array([x01, 1])

dim_p = len(x)
dim_d = len(rhs_p)




A_p = np.vstack([y,
                 x,
                 np.ones(dim_p)])

A_p = np.matrix(A_p)

A_d1 = np.matrix(np.vstack([-rhs_p,np.zeros((2,2))]))

A_d = np.hstack([x.reshape(-1, 1),np.ones(dim_p).reshape(-1, 1)])
A_d = np.matrix(A_d)

b_p = CyLPArray(np.hstack([[0, x02, 1]]))
b_d = CyLPArray(y)


s= CyClpSimplex()

u = s.addVariable('u',dim_p)
l = s.addVariable('l',dim_d)

print(s.variableScale)

v = s.variables
s += A_p*u+A_d1*l == b_p

for i in range(dim_p):
    s+=u[i] >= 0

s += A_d*l <= b_d

s.objective = u[0]

s.primal()


print(s.primalVariableSolution)
print(s.dualVariableSolution)

print(s.dualConstraintSolution)
print(s.primalConstraintSolution)
