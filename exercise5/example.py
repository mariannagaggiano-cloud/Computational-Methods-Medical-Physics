# Import stuff
import numpy as np


########################################################################################################################
# Constructing a 3 x 3 pixelated image
# Centered at 0 with voxel dimension 1 x 1
dx = 1
dy = 1
# Create 3 x 3 radiological properties' matrix (this is the "CT")
pixel_val = np.array([[1., 2., 1.], [1., 1., 1.], [2., 1., 2.]])
print("CT = ", pixel_val)
print("")

# X planes constructed recursively
# for the algorithm only need the first and last
Nx_planes = 4
x_plane_first = -1.5
x_planes = np.arange(x_plane_first, x_plane_first + Nx_planes*dx, dx)
print("x_planes = ", x_planes)
# x_planes = np.array([-1.5, -0.5, 0.5, 1.5]) # don't need to be build manually, here for illustration
# Y planes constructed recursively
# # for the algorithm only need the first and last
Ny_planes = 4
y_plane_first = -1.5
y_planes = np.arange(y_plane_first, y_plane_first + Ny_planes*dy, dy)
print("y_planes = ", y_planes)
print("")
# y_planes = np.array([-1.5, -0.5, 0.5, 1.5]) # don't need to be build manually, here for illustration

# Line defined by two points
p1 = np.array([-0.7, -1.5])
p2 = np.array([0.3, 1.5])


########################################################################################################################
# Calculate first and last planes alphas
# TODO: add protection for p1[0]=p2[0] and p1[1]=p2[1]
# for x
ax_first = (x_planes[0] - p1[0]) / (p2[0] - p1[0])
ax_last = (x_planes[Nx_planes-1] - p1[0]) / (p2[0] - p1[0])
# for y
ay_first = (y_planes[0] - p1[1]) / (p2[1] - p1[1])
ay_last = (y_planes[Ny_planes-1] - p1[1]) / (p2[1] - p1[1])

print("ax_first, ax_last, ay_first, ay_last")
print(ax_first, " , ", ax_last, " , ", ay_first, " , ", ay_last)
print("")

# # # Sanity check
# x_first_check = p1[0] + ax_first * (p2[0] - p1[0])
# y_first_check = p1[1] + ay_first * (p2[1] - p1[1])
#
# x_last_check = p1[0] + ax_last * (p2[0] - p1[0])
# y_last_check = p1[1] + ay_last * (p2[1] - p1[1])
#
# print(x_first_check, " , ", y_first_check, " , ", x_last_check, " , ", y_last_check)

########################################################################################################################
# Calculate alpha min/max, TODO add cases for ray travelling towards different directions
a_min = max(0, min(ax_first, ax_last), min(ay_first, ay_last))
a_max = min(1, max(ax_first, ax_last), max(ay_first, ay_last))
print("a_min = ", a_min, " , a_max = ", a_max)
print("")

########################################################################################################################
# Calculate plane indices for alpha min/max
i_min = 1 + (p1[0] + a_min*(p2[0]-p1[0]) - x_planes[0])/dx
i_max = 1 + (p1[0] + a_max*(p2[0]-p1[0]) - x_planes[0])/dx

j_min = 1 + (p1[1] + a_min*(p2[1]-p1[1]) - y_planes[0])/dy
j_max = 1 + (p1[1] + a_max*(p2[1]-p1[1]) - y_planes[0])/dy

print("i_min = ", i_min, " , i_max = ", i_max)
print("j_min = ", j_min, " , j_max = ", j_max)
print("")

########################################################################################################################
# Correct indices min/max for non-integer values

if p1[0] <= p2[0]:
    i_min_corr = np.ceil(i_min)
    i_max_corr = np.floor(i_max)
else:
    i_min_corr = np.floor(i_min)
    i_max_corr = np.ceil(i_max)

if p1[1] <= p2[1]:
    j_min_corr = np.ceil(j_min)
    j_max_corr = np.floor(j_max)
else:
    j_min_corr = np.floor(j_min)
    j_max_corr = np.ceil(j_max)

print("i_min_corr = ", i_min_corr, " , i_max_corr = ", i_max_corr)
print("j_min_corr = ", j_min_corr, " , j_max_corr = ", j_max_corr)
print("")

########################################################################################################################
# Fill in the in between indices
i_range = np.arange(i_min_corr, i_max_corr+1, 1)
j_range = np.arange(j_min_corr, j_max_corr+1, 1)

print("i_range = ", i_range, " , j_range = ", j_range)
print("")

########################################################################################################################
# calculate the alpha values for all the indices
ax_range = (x_planes[0] + (i_range - 1)*dx - p1[0]) / (p2[0] - p1[0])
ay_range = (y_planes[0] + (j_range - 1)*dy - p1[1]) / (p2[1] - p1[1])

print("ax_range = ", ax_range, " , ay_range = ", ay_range)
print("")

# x_test = p1[0] + ax_range*(p2[0] - p1[0])
# y_test = p1[1] + ay_range*(p2[1] - p1[1])
# print(x_test)
# print(y_test)

########################################################################################################################
# Calculate alphas for all intersections by merging 
# sorting alphas of each dimension
a_range = np.sort(np.concatenate((ax_range, ay_range)))


# Need to add alpha min/max if not already in the set
if a_range[0] != a_min:
    a_range = np.insert(a_range, 0, a_min)

if a_range[-1] != a_max:
    a_range = np.append(a_range, a_max)

print("a_range sorted = ", a_range)
print("")

########################################################################################################################
# Initializing intersection segments (one less than the alphas)
segments = np.zeros(np.size(a_range)-1)

# Initializing mid-point alphas (one less than the alphas)
a_mid = np.zeros(np.size(a_range)-1)

# Calculate segments based on alphas and mid-point alphas
for counter in range(np.size(a_range)-1):
    segments[counter] = a_range[counter+1] - a_range[counter]
    a_mid[counter] = (a_range[counter+1] + a_range[counter]) / 2

print("segments (alphas) = ", segments)
print("")

########################################################################################################################
# Calculate total distance between endpoints of the line
total_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

print("total_dist = ",  total_dist)
print("")

# Weight alpha based segments with total distance 
# to get real segment length
segments = total_dist*segments

print("segment lengths = ", segments)
print("")

########################################################################################################################
# Calculate pixel indices
# Need to be floored
i_voxel_index = np.floor(1 + (p1[0] + a_mid*(p2[0] - p1[0]) - x_planes[0])/dx)
j_voxel_index = np.floor(1 + (p1[1] + a_mid*(p2[1] - p1[1]) - y_planes[0])/dy)

print("i_voxel_index = ", i_voxel_index)
print("j_voxel_index = ", j_voxel_index)
print("")

########################################################################################################################
# Switch from 2D indices to 1D indices
rad_prop_index = 3*(j_voxel_index - 1) + (i_voxel_index-1)

# Switch from a 3x3 matrix to an 1D vector of the CT
pixel_vector = np.reshape(pixel_val, (1, np.size(pixel_val)))

print("Voxel flat indices = ",  rad_prop_index)
print("Pixel value = ",  pixel_vector[0][rad_prop_index.astype(int)])
print("")

# print(pixel_val[rad_prop.T])

# Multiplying each segment with the value in the respective voxel
total_rad_length = np.multiply(segments, pixel_vector[0][rad_prop_index.astype(int)])

print("Radiological path of each segment = ", total_rad_length)
print("")

total_rad_length = np.sum(total_rad_length)

print("********************************************************************************")
print("* Total radiological path = ", total_rad_length, " (should be 3.58391467...)")
