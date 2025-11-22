import numpy as np
import matplotlib.pyplot as plt

X=np.arange(-5,100,1)
Y=np.arange(-50,50,1)

N=500000


all_x_paths = []
all_y_paths = []


for n in range(N):

    x=0
    y=0
    angle=0

    
    val=True
    step=0

    x_path=[x]
    y_path=[y]


    while val==True:

        step=step+1

        #step1 (new distance and new position)
        rn1=np.random.rand()
        distance=-np.log(1-rn1)

        x=x+distance*np.cos(np.radians(angle))
        y=y+distance*np.sin(np.radians(angle))

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
            val=False
            break

        #step3 (new angle)
        rn3=np.random.rand()
        angle=25*np.sqrt(-np.log(1 - rn3))

        if np.random.rand()<0.5:
            angle=-angle

    all_x_paths.append(x_path)
    all_y_paths.append(y_path)


# trajectories plot
plt.figure(figsize=(14, 10))

for i in range(N):
    plt.plot(all_x_paths[i], all_y_paths[i], '-', alpha=0.4, linewidth=0.8)
    plt.plot(all_x_paths[i][0], all_y_paths[i][0], 'go', markersize=3)
    plt.plot(all_x_paths[i][-1], all_y_paths[i][-1], 'ro', markersize=3)

# limits of the world
plt.axhline(Y.min(), color='r', linestyle='--', alpha=0.5, linewidth=2, label='Boundaries')
plt.axhline(Y.max(), color='r', linestyle='--', alpha=0.5, linewidth=2)
plt.axvline(X.min(), color='r', linestyle='--', alpha=0.5, linewidth=2)
plt.axvline(X.max(), color='r', linestyle='--', alpha=0.5, linewidth=2)

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title(f'Monte Carlo Simulation: {N} particle trajectories', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(['Boundaries'], loc='best')

plt.tight_layout()
plt.savefig("path.png", dpi=300, bbox_inches='tight')
plt.show()