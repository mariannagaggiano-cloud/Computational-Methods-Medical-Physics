import numpy as np
import matplotlib.pyplot as plt

####################################################################
# Load CT (radiological properties)
CT = np.load("volume.npy")      # shape (Ny, Nx)
Ny, Nx = CT.shape

dx = 1.0
dy = 1.0

####################################################################
# Construct planes
x_plane_first = -Nx/2
y_plane_first = -Ny/2

x_planes = np.arange(x_plane_first, x_plane_first + (Nx+1)*dx, dx)
y_planes = np.arange(y_plane_first, y_plane_first + (Ny+1)*dy, dy)

####################################################################
# Ray tracing parallel to x axis
radiological_path = np.zeros(Nx)

for ray_id in range(Nx):

    # Ray definition (horizontal ray)
    y_ray = y_planes[ray_id] + dy/2
    p1 = np.array([x_planes[0], y_ray])
    p2 = np.array([x_planes[-1], y_ray])

    # --- Alpha limits ---
    ax_first = (x_planes[0]  - p1[0]) / (p2[0] - p1[0])
    ax_last  = (x_planes[-1] - p1[0]) / (p2[0] - p1[0])

    a_min = max(0, min(ax_first, ax_last))
    a_max = min(1, max(ax_first, ax_last))

    # --- Plane indices ---
    i_min = 1 + (p1[0] + a_min*(p2[0]-p1[0]) - x_planes[0]) / dx
    i_max = 1 + (p1[0] + a_max*(p2[0]-p1[0]) - x_planes[0]) / dx

    i_min_corr = int(np.ceil(i_min))
    i_max_corr = int(np.floor(i_max))

    i_range = np.arange(i_min_corr, i_max_corr+1)

    # --- Alpha intersections ---
    ax_range = (x_planes[0] + (i_range-1)*dy - p1[0]) / (p2[0] - p1[0])

    a_range = np.sort(ax_range)
    a_range = np.insert(a_range, 0, a_min)
    a_range = np.append(a_range, a_max)

    # --- Segment lengths ---
    segments = (a_range[1:] - a_range[:-1])
    total_dist = np.linalg.norm(p2 - p1)
    segments = segments * total_dist

    # --- Midpoints ---
    a_mid = 0.5*(a_range[1:] + a_range[:-1])

    i_voxel = np.floor(
        1 + (p1[0] + a_mid*(p2[0]-p1[0]) - x_planes[0]) / dx
    ).astype(int) - 1

    i_voxel = np.clip(i_voxel, 0, Nx-1)

    j_voxel = ray_id

    # --- Radiological path ---
    radiological_path[ray_id] = np.sum(
        segments * CT[j_voxel, i_voxel]
    )

####################################################################
# Validation: simple column sum
validation = np.sum(CT, axis=1) * dx

####################################################################
# Plot result
plt.figure(figsize=(10,6))
plt.plot(radiological_path, label="Ray tracing")
plt.plot(validation, "--", label="Row sum (validation)")
plt.xlabel("Ray ID", fontsize=16)
plt.ylabel("Radiological path", fontsize=16)
plt.title("Ray tracing (parallel to x-axis)", fontsize=18, weight='bold')
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig("ray_tracing_x.png", dpi=300, bbox_inches="tight")