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

number_array = [10, 100, 1000, 10000, 100000, 1000000]  # number of random number try 10, 100, 1000, 10000

trueI = 0.746824132812427
Ig_value = 1-np.exp(-1)

I_SM_array=np.zeros(len(number_array))
I_IS_array=np.zeros(len(number_array))
I_CV_array=np.zeros(len(number_array))
sem_perc_SM_array=np.zeros(len(number_array))
sem_perc_IS_array=np.zeros(len(number_array))
sem_perc_CV_array=np.zeros(len(number_array))
conv_perc_SM=np.zeros(len(number_array))
conv_perc_IS=np.zeros(len(number_array))
conv_perc_CV=np.zeros(len(number_array))


for i in range(len(number_array)):

    number = number_array[i]

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
    g_Ginv_y = np.exp(-Ginv_y)/Ig_value             #g(G^-1(y)) in this case g has to be normalized
    I_IS = np.sum(f_Ginv_y/g_Ginv_y)/number

    var_IS = (number/(number-1))*((np.sum((f_Ginv_y/g_Ginv_y)**2)/number) - ((np.sum((f_Ginv_y/g_Ginv_y)/number))**2))


    #control variates
    g = np.exp(-rn)
    f_prime= f-g
    I_g = np.sum(g)/number
    I_CV = np.sum(f_prime)/number + Ig_value

    var_g = (number/(number-1))*((np.sum(g**2)/number) - ((np.sum(g)/number)**2))
    cov_fg = (1/(number-1))*np.sum((f-I_SM)*(g-I_g))
    var_CV = (number/(number-1))*((np.sum(f_prime**2)/number) - ((np.sum(f_prime)/number)**2))

    #save results

    I_SM_array[i]=I_SM
    I_IS_array[i]=I_IS
    I_CV_array[i]=I_CV
    sem_perc_SM_array[i]=100*np.sqrt(var_SM/number)/I_SM
    sem_perc_IS_array[i]=100*np.sqrt(var_IS/number)/I_IS
    sem_perc_CV_array[i]=100*np.sqrt(var_CV/number)/I_CV
    conv_perc_SM[i]=100*(I_SM - trueI)/trueI
    conv_perc_IS[i]=100*(I_IS - trueI)/trueI
    conv_perc_CV[i]=100*(I_CV - trueI)/trueI


    #print results

    print("***************************>>> PRINT RESULTS for N = ", number, " trials<<<*******************************************")
    print()
    print("Sampling Mean Method")
    print("If = ", I_SM, " with SEM = ", np.sqrt(var_SM/number), "(", sem_perc_SM_array[i], "%)")
    print("If - true % = ", conv_perc_SM[i])
    print()
    print("Control Variates Method")
    print("If = ", I_CV, " with SEM = ", np.sqrt(var_CV/number), "(", sem_perc_CV_array[i], "%)")
    print("If - true % = ", conv_perc_CV[i])
    print()
    print("Importance Sampling Method")
    print("If = ", I_IS, " with SEM = ", np.sqrt(var_IS/number), "(", sem_perc_IS_array[i], "%)")
    print("If - true % = ", conv_perc_IS[i])
    print()

    print("Var_SM/Var_CV = ", var_SM/var_CV)
    print("Var_SM/Var_IS = ", var_SM/var_IS)
    print("Var_CV/Var_IS = ", var_CV/var_IS)

    print("************************************>>> END <<<********************************************")
    print()

    
#plot figures

plt.figure(figsize=(10, 6))
plt.suptitle("Convergence to the Correct Answer", fontsize=18)
# Plot con scala logaritmica sull'asse x
plt.semilogx(number_array, abs(conv_perc_SM), 'o-', markersize=8, linewidth=2, color='blue', label=rf'Sampling Mean Method')
plt.semilogx(number_array, abs(conv_perc_IS), 'o-', markersize=8, linewidth=2, color='red', label=rf'Importance Sampling Method')
plt.semilogx(number_array, abs(conv_perc_CV), 'o-', markersize=8, linewidth=2, color='green', label=rf'Control Variates Method')
plt.xlabel('Batch size (N)', fontsize=16)
plt.ylabel('(Result - True value)%', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=16)
plt.savefig("convergence.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.suptitle("Standard Error of the Mean", fontsize=18)
# Plot con scala logaritmica sull'asse x
plt.semilogx(number_array, abs(sem_perc_SM_array), 'o-', markersize=8, linewidth=2, color='blue', label=rf'Sampling Mean Method')
plt.semilogx(number_array, abs(sem_perc_IS_array), 'o-', markersize=8, linewidth=2, color='red', label=rf'Importance Sampling Method')
plt.semilogx(number_array, abs(sem_perc_CV_array), 'o-', markersize=8, linewidth=2, color='green', label=rf'Control Variates Method')
plt.xlabel('Batch size (N)', fontsize=16)
plt.ylabel('SEM%', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=16)
plt.savefig("sem.png", dpi=300, bbox_inches='tight')
plt.close()


x_plot = np.arange(0, 1, 0.01)
f_plot = np.exp(-x_plot**2)
g_plot = np.exp(-x_plot)
f_minus_g_plot = f_plot - g_plot
ratio_f_g = f_plot/g_plot

plt.figure(figsize=(10, 6))
plt.plot(x_plot, f_plot, 'b', linewidth=2, label=rf'$f(x)=e^{'-x^2'}$' )
plt.plot(x_plot, g_plot, 'g', linewidth=2, label=rf'$f(x)=e^{'-x'}$')
plt.plot(x_plot, f_minus_g_plot, 'r', linewidth=2, label=rf'$f(x)-g(x)$')
plt.plot(x_plot, ratio_f_g, 'm', linewidth=2, label=rf'$f(x)/g(x)$')
plt.xlabel("x", fontsize=16)
plt.ylabel("y", fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show(block=True)
plt.savefig("functions.png", dpi=300, bbox_inches='tight')

