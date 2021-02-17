from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

import numpy as np

x = np.arange(0.1, 1, 0.1, dtype=np.double)
z = np.arange(0.1, 1, 0.1, dtype=np.double)
x,z = np.meshgrid(x, z)
y = 1/(x * z)

x = x.flatten()
z = z.flatten()
y = y.flatten()

mask = x+z <= 1

x = x[mask]
z = z[mask]
y = y[mask]

def plpd2d(xs,zs,ys,x_0,z_0):
    x,z,y = xs,zs,ys
    dim_p = len(x)
    rhs_p = np.array([x_0,z_0, 1], dtype=np.double)
    dim_d = len(rhs_p) 
    
    s = CyClpSimplex()
    u = s.addVariable('u', dim_p)
    l = s.addVariable('l', dim_d)


    A_p = np.vstack([y,
                     x,
                     z,
                     np.ones(dim_p)])
    
    A_p = np.matrix(A_p)
    b_p = CyLPArray([0,x_0,z_0, 1])
    
    A_d = np.hstack([x.reshape(-1, 1),
                     z.reshape(-1, 1),
                     np.ones(len(x)).reshape(-1, 1)])
    A_d = np.matrix(A_d)
    b_d = CyLPArray(y)
    
    A_d1 = np.matrix(np.vstack([-rhs_p,np.zeros((3,3))]))
    
    s += A_p*u + A_d1*l  == b_p

    s += A_d*l <= b_d
    
    for i in range(dim_p):
        s += u[i] >= 0
        
    s.optimizationDirection = 'max'
    s.objective = u[0]
    s.primal()
    cond = s.primalVariableSolution['u']
    y_res = np.dot(cond, y)
    
    return y_res
plpd2d(x,z,y,0.5,0.5)
