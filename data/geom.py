# NAME: geom.py
# DESCRIPTION: utils about matrix manipulation

import numpy as np


def skew_matrix(v):
    zero = np.zeros_like(v[:, 0])
    
    M = np.stack([
                zero, -v[:,2], v[:,1],
                v[:,2], zero, -v[:,0],
                -v[:,1], v[:,0], zero,
                ], axis=1)
    return M


def get_episym(x1, x2, dR, dt):
    # calculate symetric distance for fundamental matrix
    num_point = len(x1)

    # Create homogeneous coordinate 
    x1 = np.concatenate([x1, np.ones((num_point, 1))], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([x2, np.ones((num_point, 1))], axis=-1).reshape(-1, 3, 1)

    # Compute fundamental matrix
    dR = dR.reshape(1,3,3)
    dt = dt.reshape(1,3)

    F = np.repeat(np.matmul(np.reshape(skew_matrix(dt), (-1, 3, 3)), dR).reshape(-1, 3, 3), num_point, axis=0)

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()
    Fx1 = np.matmul(F, x1).reshape(-1, 3)
    Ftx2 = np.matmul(F.transpose(0, 2, 1), x2).reshape(-1, 3)

    ys = x2Fx1**2 * (1.0 / (Fx1[..., 0]**2 + Fx1[..., 1]**2) + 1.0 / (Ftx2[..., 0]**2 + Ftx2[..., 1]**2))

    return ys.flatten()
