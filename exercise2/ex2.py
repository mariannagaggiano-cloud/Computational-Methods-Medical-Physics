#3 Methods Monte Carlo Integration
#Main function is f = exp(-x^2)
# Control variate is g = exp(-x) 
# Calculate If, sigmaIf, If' and sigmaIf'
# If=0.746824132812427

import numpy as np
import matplotlib.pyplot as plt

print("Calculating f = exp(-x^2) integral from 0 to 1 with three different MC methods")
print("Sampling mean, Importance Sampling and Control variate")
print()

number_array = [10, 100, 1000, 10000]  # number of random number try 10, 100, 1000, 10000
#factor = 1.0
trueI = 0.746824132812427
Ig_value = 1-np.exp(-1)
print("Estimating for N = ", number, " trials")
print()

# random numbers [0,1]
rn = np.random.rand(number)


# calculate integrals

#sampling mean
f = np.exp(-rn**2)
I_SM = np.sum(f)/number

var_SM = (number/(number-1))*((np.sum(f**2)/number) - ((np.sum(f)/number)**2))


#importance sampling
y = rn
Ginv_y = - np.log(1-y*(1-np.exp(-1)))           #G^-1(y)
f_Ginv_y = np.exp(-Ginv_y**2)                   #f(G^-1(y))
g_Ginv_y = np.exp(-Ginv_y)/(1 - np.exp(-1))     #g(G^-1(y)) in this case g has to be normalized
I_IS = np.sum(f_Ginv_y/g_Ginv_y)/number

var_IS = (number/(number-1))*((np.sum((f_Ginv_y/g_Ginv_y)**2)/number) - (np.sum((f_Ginv_y/g_Ginv_y)/number)**2))


#control variates
g = np.exp(-rn)
f_prime= f-g
I_g = np.sum(g)/number
I_CV = np.sum(f_prime)/number + Ig_value

var_g = (number/(number-1))*((np.sum(g**2)/number) - ((np.sum(g)/number)**2))
cov_fg = (1/(number-1))*np.sum((f-I_SM)*(g-I_g))
var_CV = (number/(number-1))*((np.sum(f_prime**2)/number) - ((np.sum(f_prime)/number)**2))

#print results

print("***************************>>> PRINT RESULTS <<<*******************************************")
print()
print("If(samp meam) = ", I_SM, " with SEM = ", np.sqrt(var_f/number), "(", 100*np.sqrt(var_f/number)/I_SM, "%)")
print("If - true % = ", 100*(I_SM - trueI)/trueI)
print()
print("Ig = ", I_g, " with SEM = ", np.sqrt(var_g/number))
print()
print("If(control variates) = ", I_CV, " with SEM = ", np.sqrt(var_fprime/number), "(", 100*np.sqrt(var_fprime/number)/I_CV, "%)")
print("If' - true % = ", 100*(I_CV - trueI)/trueI)
print()
print("If(importance) = ", I_IS, " with SEM = ", np.sqrt(var_f_Ginv_y/number), "(", 100*np.sqrt(var_f_Ginv_y/number)/I_IS, "%)")
print("If' - true % = ", 100*(I_IS - trueI)/trueI)
print()

print("Var_f = ", var_f)
print("Var_g = ", var_g)
print("Var_f_prime = ", var_fprime)
print("Var_f/Var_f_prime = ", var_f/var_fprime)
print("CoVar = ", cov_fg)
print("corr factor = ", cov_fg/np.sqrt(var_f*var_g))
print("optimal M = ", cov_fg/var_g)
print("************************************>>> END <<<********************************************")