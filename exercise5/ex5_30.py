import numpy as np
import matplotlib.pyplot as plt

####################################################################
# Load CT
CT = np.load("volume.npy")          # shape (Ny, Nx)
Ny, Nx = CT.shape

dx = dy = 1.0

####################################################################
# Construct planes
x_plane_first = -Nx/2
y_plane_first = -Ny/2

x_planes = np.arange(x_plane_first, x_plane_first + (Nx+1)*dx, dx)
y_planes = np.arange(y_plane_first, y_plane_first + (Ny+1)*dy, dy)

####################################################################
# Ray tracing parameters

radiological_path = np.zeros(Ny)
theta = np.pi/6
Lx = Nx * dx

y_start_min = y_planes[0] - np.tan(theta) * Lx
y_start_max = y_planes[-1]

y_starts = np.arange(y_start_min, y_start_max, dy)
radiological_path = np.zeros(len(y_starts))



####################################################################
# Loop over rays (one per row)

# Ray endpoints (long enough to cross the whole image)
for ray_idx, y0 in enumerate(y_starts):

    p1 = [x_planes[0], y0]
    p2 = [x_planes[-1], y0 + np.tan(theta) * (x_planes[-1] - x_planes[0])]

    ################################################################
    # Alpha limits
    ax_first = (x_planes[0]  - p1[0]) / (p2[0] - p1[0])
    ax_last  = (x_planes[-1] - p1[0]) / (p2[0] - p1[0])

    ay_first = (y_planes[0]  - p1[1]) / (p2[1] - p1[1])
    ay_last  = (y_planes[-1] - p1[1]) / (p2[1] - p1[1])

    a_min = max(0.0, min(ax_first, ax_last), min(ay_first, ay_last))
    a_max = min(1.0, max(ax_first, ax_last), max(ay_first, ay_last))

    if a_min >= a_max: #skip rays outside the figure
        continue

    ################################################################
    # Plane indices
    i_min = 1 + (p1[0] + a_min*(p2[0]-p1[0]) - x_planes[0]) / dx
    i_max = 1 + (p1[0] + a_max*(p2[0]-p1[0]) - x_planes[0]) / dx

    j_min = 1 + (p1[1] + a_min*(p2[1]-p1[1]) - y_planes[0]) / dy
    j_max = 1 + (p1[1] + a_max*(p2[1]-p1[1]) - y_planes[0]) / dy

    i_range = np.arange(np.ceil(i_min), np.floor(i_max) + 1)
    j_range = np.arange(np.ceil(j_min), np.floor(j_max) + 1)

    ################################################################
    # Alpha intersections
    ax_range = (x_planes[0] + (i_range-1)*dx - p1[0]) / (p2[0] - p1[0])
    ay_range = (y_planes[0] + (j_range-1)*dy - p1[1]) / (p2[1] - p1[1])

    a_range = np.sort(np.concatenate((ax_range, ay_range)))
    a_range = np.concatenate(([a_min], a_range, [a_max]))

    ################################################################
    # Segment lengths
    segments = a_range[1:] - a_range[:-1]
    total_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    segments *= total_dist

    ################################################################
    # Midpoints
    a_mid = 0.5 * (a_range[1:] + a_range[:-1])

    i_voxel = np.floor(
        1 + (p1[0] + a_mid*(p2[0]-p1[0]) - x_planes[0]) / dx
    ).astype(int) -1

    j_voxel = np.floor(
        1 + (p1[1] + a_mid*(p2[1]-p1[1]) - y_planes[0]) / dy
    ).astype(int) -1
    #minus 1 to have python indices from 0 to N-1

    i_voxel = np.clip(i_voxel, 0, Nx-1) #this protect to wrong approximation in the floating point for the calculation of the voxel indices
    j_voxel = np.clip(j_voxel, 0, Ny-1)

    ################################################################
    # Radiological path
    radiological_path[ray_idx] = np.sum(
        segments * CT[j_voxel, i_voxel]
    )

####################################################################
# Plot
plt.figure(figsize=(10,6))
plt.plot(radiological_path, lw=2)
plt.xlabel("Ray ID", fontsize=16)
plt.ylabel("Radiological path", fontsize=16)
plt.title("Ray tracing at 30°", fontsize=18, weight='bold')
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig("ray_tracing_30.png", dpi=300, bbox_inches="tight")