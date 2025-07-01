import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Voronoi
from matplotlib import rc

def get_brillouin_zone_3d(cell):
    """
    Generate the Brillouin Zone of a given cell. The BZ is the Wigner-Seitz cell
    of the reciprocal lattice, which can be constructed by Voronoi decomposition
    to the reciprocal lattice.  A Voronoi diagram is a subdivision of the space
    into the nearest neighborhoods of a given set of points. 

    https://en.wikipedia.org/wiki/Wigner%E2%80%93Seitz_cell
    https://docs.scipy.org/doc/scipy/reference/tutorial/spatial.html#voronoi-diagrams
    """

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

Acell = np.array([[1.98985, -7.235,0.0],
                 [1.98985,  7.235, 0.0],
                 [0.0,     0.0,  13.676]])
cell = np.linalg.inv(Acell).T
# cell = Acell


v, e, f = get_brillouin_zone_3d(cell)

fig = plt.figure(figsize=(3, 3), dpi=300)
ax = fig.add_subplot(111, projection='3d')

ax.set_aspect('equal')
ax.set_xlim(-0.15, 0.15)
ax.set_ylim(-0.15, 0.15)
ax.set_zlim(-0.15, 0.15)

ax.scatter(0,0,0,color='red', lw=.050,label='$\Gamma$')
ax.scatter(0,6.91085003e-02,0,color='red', lw=.050)
ax.scatter(0,6.91085003e-02,-3.65603978e-2,color='red', lw=.050)
ax.scatter(1.16134117e-01,  6.91085003e-02, -3.65603978e-02,color='red', lw=.050)
ax.scatter(1.35141104e-01, -3.73565094e-17, -3.65603978e-02,color='red', lw=.050)
ax.scatter(0,0, -3.65603978e-02,color='red', lw=.050)

ax.scatter(1.16134117e-01,  6.91085003e-02, 0,color='red', lw=.050)
ax.scatter(1.35141104e-01, -3.73565094e-17, 0,color='red', lw=.050)

for ridge in e:
    ax.plot(ridge[:, 0], ridge[:, 1], ridge[:, 2], color='black', lw=.60)
print(e)


points = [
    (0, 0, 0, r'$\Gamma$'),
    (0, 6.91085003e-02, 0, 'Y'),
    (0, 6.91085003e-02, -3.65603978e-2, 'T'),
    (1.16134117e-01, 6.91085003e-02, -3.65603978e-02, r'E$_{0}$'),
    (1.35141104e-01, -3.73565094e-17, -3.65603978e-02, 'A$_{0}$'),
    (0, 0, -3.65603978e-02, 'Z'),
    (1.16134117e-01, 6.91085003e-02, 0, r'C$_{0}$'),
    (1.35141104e-01, -3.73565094e-17, 0, r'$\sum$$_0$'),
    (0.12513711, 0.03455425, 0, 'S'),
    (0.12513711, 0.03455425, -0.0365603978, 'R'),
]

coords = {label: (x, y, z) for x, y, z, label in points}
#path = ['X', r'$\Gamma$', 'Z', 'A', r'$\Gamma$', 'Y', r'X$_{1}$', r'A$_{1}$', 'T', 'Y']
#path_coords = [coords[label] for label in path]
#xs, ys, zs = zip(*path_coords)
#ax.plot(xs, ys, zs, color='red', lw=1.0)


for x, y, z, label in points:
    ax.scatter(x, y, z, color='red', lw=0.3)
    ax.text(x+0.002, y+0.002, z+0.002, label, fontsize=6, ha='left', va='bottom',color='red')


ax.set_axis_off()  
plt.tight_layout()
plt.show()
