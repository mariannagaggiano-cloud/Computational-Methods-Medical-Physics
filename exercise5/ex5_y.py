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
# Ray tracing parallel to y axis
radiological_path = np.zeros(Nx)

for ray_id in range(Nx):

    # Ray definition (vertical ray)
    x_ray = x_planes[ray_id] + dx/2
    p1 = np.array([x_ray, y_planes[0]])
    p2 = np.array([x_ray, y_planes[-1]])

    # --- Alpha limits ---
    ay_first = (y_planes[0]  - p1[1]) / (p2[1] - p1[1])
    ay_last  = (y_planes[-1] - p1[1]) / (p2[1] - p1[1])

    a_min = max(0, min(ay_first, ay_last))
    a_max = min(1, max(ay_first, ay_last))

    # --- Plane indices ---
    j_min = 1 + (p1[1] + a_min*(p2[1]-p1[1]) - y_planes[0]) / dy
    j_max = 1 + (p1[1] + a_max*(p2[1]-p1[1]) - y_planes[0]) / dy

    j_min_corr = int(np.ceil(j_min))
    j_max_corr = int(np.floor(j_max))

    j_range = np.arange(j_min_corr, j_max_corr+1)

    # --- Alpha intersections ---
    ay_range = (y_planes[0] + (j_range-1)*dy - p1[1]) / (p2[1] - p1[1])

    a_range = np.sort(ay_range)
    a_range = np.insert(a_range, 0, a_min)
    a_range = np.append(a_range, a_max)

    # --- Segment lengths ---
    segments = (a_range[1:] - a_range[:-1])
    total_dist = np.linalg.norm(p2 - p1)
    segments = segments * total_dist

    # --- Midpoints ---
    a_mid = 0.5*(a_range[1:] + a_range[:-1])

    j_voxel = np.floor(
        1 + (p1[1] + a_mid*(p2[1]-p1[1]) - y_planes[0]) / dy
    ).astype(int) - 1

    j_voxel = np.clip(j_voxel, 0, Ny-1)

    i_voxel = ray_id

    # --- Radiological path ---
    radiological_path[ray_id] = np.sum(
        segments * CT[j_voxel, i_voxel]
    )

####################################################################
# Validation: simple column sum
validation = np.sum(CT, axis=0) * dy

####################################################################
# Plot result
plt.figure(figsize=(10,6))
plt.plot(radiological_path, label="Ray tracing")
plt.plot(validation, "--", label="Column sum (validation)")
plt.xlabel("Ray ID", fontsize=16)
plt.ylabel("Radiological path", fontsize=16)
plt.title("Ray tracing (parallel to y-axis)", fontsize=18, weight='bold')
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig("ray_tracing_y.png", dpi=300, bbox_inches="tight")

