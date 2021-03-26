#!/usr/bin/env python3

from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

import numpy as np

xa,xb = 0,1
za,zb = 0,1
grid_step = 0.01
xa_ext = xa - grid_step
xb_ext = xb + grid_step
za_ext = za - grid_step
zb_ext = zb + grid_step


x = np.arange(xa_ext, xb_ext, grid_step, dtype=np.double)
z = np.arange(za_ext, zb_ext, grid_step, dtype=np.double)
x,z = np.meshgrid(x, z)
y_cvx = x**2 + x * z + z**2
y_ccv = - x**2 - z**2

x = x.flatten()
z = z.flatten()
y_cvx = y_cvx.flatten()
y_ccv = y_ccv.flatten()

mask = x+z <= 1+grid_step

xs = x[mask]
zs = z[mask]
ys_pos = y_cvx[mask]
ys_neg = y_ccv[mask]


def f(x,z,y_pos,y_neg,x_0,z_0,x_b=1,z_b=1):
    dim_p = len(x)
    rhs_p = np.array([x_0,z_0, 1], dtype=np.double)
    dim_d = len(rhs_p)

    s_pos = CyClpSimplex()
    u_pos = s_pos.addVariable('u', dim_p)
    l_pos = s_pos.addVariable('l', dim_d)

    s_neg = CyClpSimplex()
    u_neg = s_neg.addVariable('u', dim_p)
    l_neg = s_neg.addVariable('l', dim_d)


    A_p_pos = np.vstack([y_pos,
                         x,
                         z,
                         np.ones(dim_p)])
    A_p_neg = np.vstack([y_neg,
                         x,
                         z,
                         np.ones(dim_p)])

    A_p_pos = np.matrix(A_p_pos)
    A_p_neg = np.matrix(A_p_neg)

    b_p = CyLPArray(np.hstack([0,rhs_p]))
    b_d_pos = CyLPArray(y_pos)
    b_d_neg = CyLPArray(y_neg)

    A_d = np.hstack([x.reshape(-1, 1),
                     z.reshape(-1, 1),
                     np.ones(len(x)).reshape(-1, 1)])
    A_d = np.matrix(A_d)


    # A_d1 = np.matrix(np.vstack([-rhs_p,np.zeros((3,3))]))
    t = np.array([x_b,z_b, 1], dtype=np.double)
    A_d1 = np.matrix(np.vstack([-t,np.zeros((3,3))]))

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

    cond_pos = s_pos.primalVariableSolution['u']
    yr_pos = np.dot(cond_pos, y_pos)

    cond_neg = s_neg.primalVariableSolution['u']
    yr_neg = np.dot(cond_neg, y_neg)
    return yr_pos + yr_neg, x_0*z_0, s_pos.getStatusString(), s_neg.getStatusString(),s_pos.primalVariableSolution['l'], s_neg.primalVariableSolution['l']

for x in np.arange(0.1,1,0.1):
    for y in np.arange(0.1,1,0.1):
        if x+y <= 1:
            r = f(xs,zs,ys_pos,ys_neg,x,y)
            arr = [*r[-2],*r[-1]]
            print("{:2.1f} {:2.1f} :fx= {:+.4f}, {:+.4f}, \u0394fx: {:.4f}; l \u2208 [{:.2f},{:.2f}]".
                  format(x,y,r[0],x*y,abs(r[0]-x*y),min(arr),max(arr)))
