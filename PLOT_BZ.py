import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Voronoi

def get_brillouin_zone_3d(cell):
    cell = np.asarray(cell, dtype=float)
    assert cell.shape == (3, 3)

    px, py, pz = np.tensordot(cell, np.mgrid[-1:2, -1:2, -1:2], axes=[0, 0])
    points = np.c_[px.ravel(), py.ravel(), pz.ravel()]

    vor = Voronoi(points)

    bz_facets = []
    bz_ridges = []
    bz_vertices = []

    for pid, rid in zip(vor.ridge_points, vor.ridge_vertices):
        if pid[0] == 13 or pid[1] == 13:
            bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
            bz_facets.append(vor.vertices[rid])
            bz_vertices += rid

    bz_vertices = list(set(bz_vertices))

    return vor.vertices[bz_vertices], bz_ridges, bz_facets

cell = np.array([[1.98985, -7.235,0.0],
                 [1.98985,  7.235, 0.0],
                 [0.0,     0.0,  13.676]])


v, e, f = get_brillouin_zone_3d(cell)

fig = plt.figure(figsize=(3, 3), dpi=300)
ax = fig.add_subplot(111, projection='3d')
for ridge in e:
    ax.plot(ridge[:, 0], ridge[:, 1], ridge[:, 2], color='black', lw=1.0)

ax.set_axis_off()
ax.view_init(elev=20, azim=30) 

plt.tight_layout()
plt.show()
