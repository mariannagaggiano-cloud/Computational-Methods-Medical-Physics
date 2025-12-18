import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('s2.txt')

plt.figure(figsize=(10, 6))



plt.scatter(data[:, 0], data[:, 1], c='black', marker='.')

plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title(f'Data Set', fontsize=18)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('plot.png', dpi=200)
print("saved:plot.png")
