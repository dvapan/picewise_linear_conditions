#!/usr/bin/env python
""" Cocatination between primal and dual linear problem.
We have eqvivalent dual and primal solutions so we can concatinate.
This actions can help include piecwise conditions to linear problems.
"""

from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray


import numpy as np

# # y - convex function
# x = np.array([-4., 0., 1.  , 2., 3., 4.])
# y = np.array([ 0., 0., 0.5, 2., 4., 7.])

# nonconvex function y = |x|x
x = np.arange(-4,5,1,dtype=np.double)
y_pos = np.piecewise(x, [x < 0, x >= 0], [0, lambda x: abs(x)*x])
y_neg = np.piecewise(x, [x < 0, x >= 0], [lambda x: abs(x)*x, 0])

print(x)
print(y_pos)
print(y_neg)
x0 = 2

rhs_p = np.array([x0, 1],dtype=np.double)

dim_p = len(x)
dim_d = len(rhs_p)


A_p_pos = np.vstack([y_pos,
                     x,
                     np.ones(dim_p)])
A_p_neg = np.vstack([y_neg,
                     x,
                     np.ones(dim_p)])

A_p_pos = np.matrix(A_p_pos)
A_p_neg = np.matrix(A_p_neg)

A_d1 = np.matrix(np.vstack([-rhs_p, np.zeros((2,2))]))

A_d = np.hstack([x.reshape(-1, 1), np.ones(dim_p).reshape(-1, 1)])
A_d = np.matrix(A_d)

b_p = CyLPArray(np.array([0.0, x0, 1]))
b_d_pos = CyLPArray(y_pos)
b_d_neg = CyLPArray(y_neg)


s_pos = CyClpSimplex()
s_neg = CyClpSimplex()

u_pos = s_pos.addVariable('u', dim_p)
l_pos = s_pos.addVariable('l', dim_d)
u_neg = s_neg.addVariable('u', dim_p)
l_neg = s_neg.addVariable('l', dim_d)

s_pos += A_p_pos*u_pos + A_d1*l_pos == b_p
s_neg += A_p_neg*u_neg + A_d1*l_neg == b_p

for i in range(dim_p):
    s_pos += u_pos[i] >= 0
    s_neg += u_neg[i] >= 0

s_pos += A_d*l_pos <= b_d_pos
s_neg += A_d*l_neg >= b_d_neg

s_pos.objective = u_pos[0]
s_neg.objective = u_neg[0]

s_pos.primal()
s_neg.primal()


print(s_pos.primalVariableSolution)
print(s_pos.dualVariableSolution)

cond_pos = s_pos.primalVariableSolution['u']
x1_pos = np.dot(cond_pos, x)
y1_pos = np.dot(cond_pos, y_pos)

cond_neg = s_neg.primalVariableSolution['u']
x1_neg = np.dot(cond_neg, x)
y1_neg = np.dot(cond_neg, y_neg)

# print(x1_pos, x1_neg, y1_pos, y1_neg)

if x1_pos == x1_neg:
    x1 = x1_pos
else:
    x1 = x1_pos, x1_neg

y1 = y1_neg + y1_pos

print(x1,y1)
