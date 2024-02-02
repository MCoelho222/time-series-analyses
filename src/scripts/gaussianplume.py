from __future__ import annotations

import math

import numpy as np


def plume3d(mass, wind, dist, height, dz, dy, alpha):

    dg = math.radians(alpha)
    mdist = dist

    x = math.cos(dg) * mdist
    y = math.sin(dg) * mdist
    z = height
    u = wind

    c = (mass * 1000)/(4 * np.pi * np.sqrt(dz * dy) * x)
    e1 = (u * y**2)/(4 * dy * mdist)
    e2 = (u * z**2)/(4 * dz * mdist)
    decay = np.e ** (- e1 - e2)

    return c * decay

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    load = 1   # kg/s
    u = 0.5       # m/s
    dy = 0.7      # m²/s
    dz = 0.7      # m²/s
    z = 31        # m
    dist = 200    # m
    alpha = 0    # graus
    cx = []
    x_axis = []
    for i in range(1, 1000, 1):
        x_axis.append(i)
        cx.append(plume3d(load, u, i, z, dz, dy, alpha))

    plt.plot(x_axis, cx)
    plt.show()
