import numpy as np
import matplotlib.pyplot as plt

X=np.arange(-5,100,1)
Y=np.arange(-50,50,1)

N=1000000

all_x_paths = []
all_y_paths = []
all_angles_paths = []
angles_tot = []
all_distance_paths = []
distances_tot = []

abs_interactions=0
scat_interactions=0

count5=0
count20=0
y5=[]
y20=[]


for n in range(N):

    x=0
    y=0
    angle=0
    angle_tot=0
    distance=0
    distance_tot=0

    val=True
    step=0


    x_path=[x]
    y_path=[y]
    angles_path=[]
    distance_path=[]


    while val==True:

        step=step+1

        #step1 (new distance and new position)
        rn1=np.random.rand()
        distance=-np.log(1-rn1)
        distance_tot=distance_tot+distance

        if x<5<=x+distance*np.cos(np.radians(angle_tot))>=5:
            count5=count5+1
            y5_val=y+(5-x)*np.tan(np.radians(angle_tot))
            y5.append(y5_val)

        if x<20<=x+distance*np.cos(np.radians(angle_tot)):
            count20=count20+1
            y20_val=y+(20-x)*np.tan(np.radians(angle_tot))
            y20.append(y20_val)


        angles_path.append(angle)
        distance_path.append(distance)

        x=x+distance*np.cos(np.radians(angle_tot))
        y=y+distance*np.sin(np.radians(angle_tot))

        x_path.append(x)
        y_path.append(y)

    
        if x<min(X) or x>max(X):
            print(f"particle killed at step {step} (x={x})")
            val=False
            break

        if y<min(Y) or y>max(Y):
            print(f"particle killed at step {step} (y={y})")
            val=False
            break

        #step2 (absorption 20%, scattering 80%)   
        nr2=np.random.rand()
        if nr2<0.2:
            print(f"particle absorbed at step {step}")
            abs_interactions=abs_interactions+1
            val=False
            break

        #step3 (new angle)
        scat_interactions=scat_interactions+1
        rn3=np.random.rand()
        angle=25*np.sqrt(-np.log(1 - rn3))

        if np.random.rand()<0.5:
            angle=-angle

        angle_tot=angle_tot+angle

        

    all_x_paths.append(x_path)
    all_y_paths.append(y_path)
    all_angles_paths.append(angles_path)
    all_distance_paths.append(distance_path)
    distances_tot.append(distance_tot)
    angles_tot.append(angle_tot)


# results
if scat_interactions > 0:
    ratio = abs_interactions/scat_interactions
    print(f"ratio of absorption-scattering interactions: r={ratio}")
else:
    print("No scattering interactions occurred")

print(f"number of particle reachin the detector at x=5mm: count(5)={count5}")
print(f"number of particle reachin the detector at x=20mm: count(20)={count20}")

"""
print("total path for each particle")
print(distances_tot)
print()
print("scattering angle for each particle at each step")
print(all_angles_paths)
print()
print("x-path for each particle at each step")
print(all_x_paths)
print()
print("y-path angle for each particle at each step")
print(all_y_paths)
print()
print("path for each particle at each step")
print(all_distance_paths)
"""


###PLOT

#plot distribution total distance
plt.figure(figsize=(10, 6))
plt.hist(distances_tot, bins=20, edgecolor='black', alpha=0.7, color='skyblue', density=True, label='simulated data')
plt.axvline(np.mean(distances_tot), color='r', linestyle='--', linewidth=2, 
            label=f'Mean: {np.mean(distances_tot):.2f}')
plt.xlabel('Path lenght [mm]', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title(f'Total Path Lenght Distribution ({N} particles)', fontsize=18, fontweight='bold')
plt.legend(fontsize=16)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig("dist_total.png", dpi=300, bbox_inches='tight')
plt.show()

#plot distribution total angle
plt.figure(figsize=(10, 6))
plt.hist(angles_tot, bins=20, edgecolor='black', alpha=0.7, color='skyblue', density=True, label='simulated data')
plt.axvline(np.mean(angles_tot), color='r', linestyle='--', linewidth=2, 
            label=f'Mean: {np.mean(angles_tot):.2f}')
plt.xlabel('Scattering Angle [degrees]', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title(f'Total Scattering Angle Distribution ({N} particles)', fontsize=18, fontweight='bold')
plt.legend(fontsize=16)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig("ang_total.png", dpi=300, bbox_inches='tight')
plt.show()

#plot single step distance
all_distances_flat = []
for distance_path in all_distance_paths:
    all_distances_flat.extend(distance_path)

plt.figure(figsize=(10, 6))

plt.hist(all_distances_flat, bins=20, edgecolor='black', alpha=0.7, 
         color='skyblue', density=True, label='Simulated Data')

x_theory = np.linspace(0, max(all_distances_flat) if all_distances_flat else 10, 1000)
plt.plot(x_theory, np.exp(-x_theory), 'r-', linewidth=2, label='$f(x)=e^{-x}$')

plt.axvline(np.mean(all_distances_flat), color='g', linestyle='--', linewidth=2, 
            label=f'Mean: {np.mean(all_distances_flat):.2f}')
plt.xlabel('Path length [mm]', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title(f'Single-Step Path Length Distribution ({N} particles)', 
          fontsize=18, fontweight='bold')
plt.legend(fontsize=16)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlim(0, 10)
plt.tight_layout()
plt.savefig("dist_singlestep.png", dpi=300, bbox_inches='tight')
plt.show()

#plot angle distribution
all_angles_flat = []
for angles_path in all_angles_paths:
    # eliminate first element (no scattering)
    if len(angles_path) > 1:
        all_angles_flat.extend(angles_path[1:])

plt.figure(figsize=(10, 6))

plt.hist(all_angles_flat, bins=20, edgecolor='black', alpha=0.7, 
         color='skyblue', density=True, label='Simulated Data')

# theory curve for positive angles
x_theory_pos = np.linspace(0, 180, 1000)
f_theory_pos = x_theory_pos * np.exp(-x_theory_pos**2 / 625)/625

plt.plot(x_theory_pos, f_theory_pos, 'r-', linewidth=2, 
         label=r'$f(\theta) = \frac{C}{2} \cdot |\theta| \cdot e^{-\theta^2/a^2}$')

# theory curve for negative angles
x_theory_neg = np.linspace(-180, 0, 1000)
f_theory_neg = (-x_theory_neg) * np.exp(-x_theory_neg**2 / 625)/625


plt.plot(x_theory_neg, f_theory_neg, 'r-', linewidth=2)

# Linee verticali per media
plt.axvline(np.mean(all_angles_flat), color='g', linestyle='--', linewidth=2, 
            label=f'Mean: {np.mean(all_angles_flat):.2f}°')


plt.xlabel('Scattering Angle [degrees]', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title(f'Scattering Angle Distribution ({N} particles)', 
          fontsize=18, fontweight='bold')
plt.legend(fontsize=16)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlim(-180, 180)
plt.tight_layout()
plt.savefig("dist_angle.png", dpi=300, bbox_inches='tight')
plt.show()



from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 2D tracking
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
'''
#principal plot
for i in range(N):
    if i == 0:
        ax1.plot(all_x_paths[i], all_y_paths[i], 'b-', alpha=0.4, linewidth=0.8, 
                label='Trajectories')
        ax1.plot(all_x_paths[i][0], all_y_paths[i][0], 'go', markersize=5, 
                label='Start')
        ax1.plot(all_x_paths[i][-1], all_y_paths[i][-1], 'ro', markersize=5, 
                label='End (absorption/kill)')
    else:
        ax1.plot(all_x_paths[i], all_y_paths[i], 'b-', alpha=0.4, linewidth=0.8)
        ax1.plot(all_x_paths[i][0], all_y_paths[i][0], 'go', markersize=5)
        ax1.plot(all_x_paths[i][-1], all_y_paths[i][-1], 'ro', markersize=5)

ax1.axhline(Y.min(), color='r', linestyle='--', alpha=0.5, linewidth=2, label='Boundaries')
ax1.axhline(Y.max(), color='r', linestyle='--', alpha=0.5, linewidth=2)
ax1.axvline(X.min(), color='r', linestyle='--', alpha=0.5, linewidth=2)
ax1.axvline(X.max(), color='r', linestyle='--', alpha=0.5, linewidth=2)

ax1.set_xlabel('x [mm]', fontsize=16)
ax1.set_ylabel('y [mm]', fontsize=16)
ax1.set_title(f'Full View: {N} particle trajectories', fontsize=18, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=16, loc='best')
ax1.tick_params(axis='both', which='major', labelsize=14)

# zoom
x_zoom_min, x_zoom_max = 0, 30
y_zoom_min, y_zoom_max = -20, 20

for i in range(N):
    ax2.plot(all_x_paths[i], all_y_paths[i], 'b-', alpha=0.5, linewidth=1)
    ax2.plot(all_x_paths[i][0], all_y_paths[i][0], 'go', markersize=6)
    ax2.plot(all_x_paths[i][-1], all_y_paths[i][-1], 'ro', markersize=6)

ax2.set_xlim(x_zoom_min, x_zoom_max)
ax2.set_ylim(y_zoom_min, y_zoom_max)
ax2.set_xlabel('x [mm]', fontsize=16)
ax2.set_ylabel('y [mm]', fontsize=16)
ax2.set_title(f'Zoomed View: {N} particle trajectories', fontsize=18, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', which='major', labelsize=14)

# rectangle of the zoom
from matplotlib.patches import Rectangle
rect = Rectangle((x_zoom_min, y_zoom_min), 
                 x_zoom_max - x_zoom_min, 
                 y_zoom_max - y_zoom_min,
                 linewidth=2, edgecolor='purple', facecolor='none', 
                 linestyle='--', label='Zoom area')
ax1.add_patch(rect)
ax1.legend(fontsize=16, loc='best')

plt.tight_layout() 
plt.savefig("path.png", dpi=300, bbox_inches='tight')
plt.show()
'''
#plot distribution detector 5
if count5>0:
    plt.figure(figsize=(10, 6))
    plt.hist(y5, bins=20, edgecolor='black', alpha=0.7, color='skyblue', density=True, label='simulated data')
    plt.axvline(np.mean(y5), color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(y5):.2f}')
    plt.xlabel('Position along y-axis [mm]', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title(f'Position Distribution at x=5mm ({N} particles)', fontsize=18, fontweight='bold')
    plt.suptitle(f'{count5} particles reached the detector', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig("dec5.png", dpi=300, bbox_inches='tight')
    plt.show()

#plot distribution detector 20
if count20>0:
    plt.figure(figsize=(10, 6))
    plt.hist(y20, bins=20, edgecolor='black', alpha=0.7, color='skyblue', density=True, label='simulated data')
    plt.axvline(np.mean(y20), color='r', linestyle='--', linewidth=2, 
            label=f'Mean: {np.mean(y20):.2f}')
    plt.xlabel('Position along y-axis [mm]', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title(f'Position Distribution at x=20mm ({N} particles)', fontsize=18, fontweight='bold')
    plt.suptitle(f'{count20} particles reached the detector', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig("dec20.png", dpi=300, bbox_inches='tight')
    plt.show()