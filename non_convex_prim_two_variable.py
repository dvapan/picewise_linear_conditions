#!/usr/bin/env python
""" Using Simplex method for find value of pisewise function
with 2 variables.
"""

from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray


import numpy as np

# nonconvex function y = x_1 * x_2
x = np.arange(0, 1, 0.1, dtype=np.double)
z = np.arange(0, 1, 0.1, dtype=np.double)
x,z = np.meshgrid(x, z)
y = x * z

x = x.flatten()
z = z.flatten()
y = y.flatten()

mask = x+z <= 1

x = x[mask]
z = z[mask]
y = y[mask]

# y_pos = np.piecewise(y,[y < 0, y>= 0], [0, lambda y: y])
# y_neg = np.piecewise(y,[y < 0, y>= 0], [lambda y: y, 0])

x0, z0 = 0.2, 0.5

rhs_p = np.array([x0,z0, 1], dtype=np.double)

dim_p = len(x)
dim_d = len(rhs_p)


A_p = np.vstack([x,
                 z,
                 np.ones(dim_p)])
# A_p_pos = np.vstack([x,
#                      z,
#                      np.ones(dim_p)])
# A_p_neg = np.vstack([y_neg,
#                      x,
#                      np.ones(dim_p)])


A_p = np.matrix(A_p)
# A_p_pos = np.matrix(A_p_pos)
# A_p_neg = np.matrix(A_p_neg)

# A_d1 = np.matrix(np.vstack([-rhs_p, np.zeros((2,2))]))

# A_d = np.hstack([x.reshape(-1, 1), np.ones(dim_p).reshape(-1, 1)])
# A_d = np.matrix(A_d)

b_p = CyLPArray(np.array([x0, z0, 1]))
# b_d_pos = CyLPArray(y_pos)
# b_d_neg = CyLPArray(y_neg)


s = CyClpSimplex()
# s_pos = CyClpSimplex()
# s_neg = CyClpSimplex()

u = s.addVariable('u', dim_p)
# u_pos = s_pos.addVariable('u', dim_p)
# l_pos = s_pos.addVariable('l', dim_d)
# u_neg = s_neg.addVariable('u', dim_p)
# l_neg = s_neg.addVariable('l', dim_d)

s += A_p*u == b_p

# s_pos += A_p_pos*u_pos + A_d1*l_pos == b_p
# s_neg += A_p_neg*u_neg + A_d1*l_neg == b_p

for i in range(dim_p):
    s += u[i] >= 0
    # s_pos += u_pos[i] >= 0
    # s_neg += u_neg[i] >= 0

# s_pos += A_d*l_pos <= b_d_pos
# s_neg += A_d*l_neg >= b_d_neg

# s_pos.objective = u_pos[0]
# s_neg.objective = u_neg[0]

s.optimizationDirection = 'min'
s.objectiveCoefficients = y

# s_pos.primal()
# s_neg.primal()

s.primal()

# print(s_pos.primalVariableSolution)
# print(s_pos.dualVariableSolution)

# cond_pos = s_pos.primalVariableSolution['u']
# x1_pos = np.dot(cond_pos, x)
# y1_pos = np.dot(cond_pos, y_pos)

# cond_neg = s_neg.primalVariableSolution['u']
# x1_neg = np.dot(cond_neg, x)
# y1_neg = np.dot(cond_neg, y_neg)

# # print(x1_pos, x1_neg, y1_pos, y1_neg)

# if x1_pos == x1_neg:
#     x1 = x1_pos
# else:
#     x1 = x1_pos, x1_neg

# y1 = y1_neg + y1_pos

# print(x1,y1)

print(s.primalVariableSolution)
cond = s.primalVariableSolution['u']
print(np.dot(cond, x), x0)
print(np.dot(cond, z), z0)
print(np.dot(cond, y), x0*z0)
