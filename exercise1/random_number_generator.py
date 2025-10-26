import numpy as np
import matplotlib.pyplot as plt
print("Example of random number generator via numpy")
print("")
# limits and sample size
number = 10
minF = 0.
maxF = 1.
minI = 100
maxI = 200
print("Estimating for N = ", number, " samples")
# random numbers [0,1)
rnumber = np.random.rand(number)
print(rnumber)
# random numbers int [minI, maxI+1]
rnumberInt = np.random.randint(minI, maxI+1, number)
print(rnumberInt)
# plot uniform float number sampling
plt.figure(0)
plt.suptitle("Uniform float sampling", fontsize=16)
plt.hist(rnumber, 100, density=1, facecolor='green', alpha=0.75)
plt.show(block=True)
# plot uniform int number sampling
plt.figure(1)
plt.suptitle("Uniform int sampling", fontsize=16)
plt.hist(rnumberInt, 100, density=1, facecolor='green', alpha=0.75)
plt.show(block=True)