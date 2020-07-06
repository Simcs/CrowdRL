import numpy as np
from libc.math import sqrt
cimport numpy as np

DTYPE = np.float64

def external_states(np.ndarray d, np.ndarray p, np.ndarray r, int n_agents, int n_rays, int n_objs, double t_max):

    cdef np.ndarray state_ext = np.zeros([n_agents, n_rays], dtype=DTYPE)
    cdef np.ndarray depth_map = np.zeros([n_rays], dtype=DTYPE)
    cdef np.ndarray dij, pik
    cdef double t_min, tm, lm_2, dt, t0, t1
    
    for i in range(n_agents):
        for j in range(n_rays):
            dij = d[i][j]
            t_min = t_max
            for k in range(n_objs):
                if i == k:
                    continue
                pik = p[i][k]
                tm = np.dot(pik, dij)
                lm_2 = np.dot(pik, pik) - tm ** 2
                dt = r[k] ** 2 - lm_2
                if dt > 0:
                    dt = sqrt(dt)
                    t0 = tm - dt
                    t1 = tm + dt
                    if t0 > 0:
                        t_min = min(t_min, t0)
                    elif t1 > 0:
                        t_min = min(t_min, t1)
            depth_map[j] = t_min
        state_ext[i] = depth_map
    
    return state_ext

