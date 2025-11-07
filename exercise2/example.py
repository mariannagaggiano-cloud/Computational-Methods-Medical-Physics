# Example of Monte Carlo sampling mean vs control variates
# Main function is f = exp(-x^2)
# Control variate is g = -0.6*x^2+1
# Calculate If, sigmaIf, If' and sigmaIf'
# If=0.746824132812427

import numpy as np
import matplotlib.pyplot as plt

print("Calculating f = exp(-x^2) integral from 0 to 1 with two different MC methods")
print("Sampling mean and Control variate")
print()

number = 10000  # number of random number try 10, 100, 1000, 10000
factor = 1.0
trueI = 0.746824132812427
print("Estimating for N = ", number, " trials")
print()

# random numbers [0,1]
rn = np.random.rand(number)


# function f
f = np.exp(-rn**2)

# function g
g = factor*(-0.6*rn**2+1)


# function f_prime = f-g
f_prime = f - g

# calculate integrals (simple averages in this case)
I_f = np.sum(f)/number
I_g = np.sum(g)/number
I_fprime = np.sum(f_prime)/number + factor*0.8

# calculate variances
var_f = (number/(number-1))*((np.sum(f**2)/number) - ((np.sum(f)/number)**2))
var_g = (number/(number-1))*((np.sum(g**2)/number) - ((np.sum(g)/number)**2))
cov_fg = (1/(number-1))*np.sum((f-I_f)*(g-I_g))
var_fprime = (number/(number-1))*((np.sum(f_prime**2)/number) - ((np.sum(f_prime)/number)**2))

print()
print("*******************************************************************************************")
print("***************************>>> PRINT RESULTS <<<*******************************************")
print("*******************************************************************************************")
print("For M = ", factor)
print()
print("If = ", I_f, " with SEM = ", np.sqrt(var_f/number), "(", 100*np.sqrt(var_f/number)/I_f, "%)")
print("If - true % = ", 100*(I_f - trueI)/trueI)
print()
print("Ig = ", I_g, " with SEM = ", np.sqrt(var_g/number))
print()
print("If' = ", I_fprime, " with SEM = ", np.sqrt(var_fprime/number), "(", 100*np.sqrt(var_fprime/number)/I_fprime, "%)")
print("If' - true % = ", 100*(I_fprime - trueI)/trueI)
print()
print("Var_f = ", var_f)
print("Var_g = ", var_g)
print("Var_f_prime = ", var_fprime)
print("Var_f/Var_f_prime = ", var_f/var_fprime)
print("CoVar = ", cov_fg)
print("corr factor = ", cov_fg/np.sqrt(var_f*var_g))
print("optimal M = ", cov_fg/var_g)
print("************************************>>> END <<<********************************************")
print("*******************************************************************************************")

# plot figures
x_plot = np.arange(0, 1, 0.01)
f_plot = np.exp(-x_plot**2)
g_plot = factor*(-0.6*x_plot**2+1)
f_minus_g_plot = f_plot - g_plot

plt.figure(0)
plt.suptitle("Functions", fontsize=16)
plt.plot(x_plot, f_plot, 'b')
plt.plot(x_plot, g_plot, 'g')
plt.plot(x_plot, f_minus_g_plot, 'r')
plt.xlabel("y")
plt.ylabel("x")
plt.gca().legend(('f', 'g', 'f - g'))
plt.show(block=True)
plt.savefig("example.png", dpi=300, bbox_inches='tight')
