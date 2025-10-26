import numpy as np
import matplotlib.pyplot as plt
import matplotlib

batch_size = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
true_value = np.ones(len(batch_size))*0.326543231734227
n_repetitions = 10
integral_values = np.zeros((n_repetitions, len(batch_size)))
for rep in range(n_repetitions):
    for i in range(len(batch_size)):

        ##randoom number generator
        number = batch_size[i]
        minF = 0.
        maxF = 1.
        #print("Estimating for N = ", number, " samples")

        rnumber = np.random.rand(number)
    
        #print(rnumber)
        # plot uniform float number sampling
        #plt.figure(0)
        #plt.suptitle("Uniform float sampling", fontsize=16)
        #plt.hist(rnumber, 100, density=1, facecolor='green', alpha=0.75)
        #plt.show(block=True)
        #plt.savefig("uniform_sampling.png", dpi=300, bbox_inches='tight')

        ##method 1: sampling
        f_value = np.cos(1/rnumber)**2
        integral = np.mean(f_value)/(maxF-minF)
        # Salva il valore dell'integrale
        integral_values[rep, i] = integral
        #print("\nIntegral of the function cos(1/x)^2:", integral)
        #print("\n")
    
##mean value for each batch size
#axis 0 indicates mean and variance along the rows
#ddof 1 is for bessel correction
means = np.mean(integral_values, axis=0)
variance = np.var(integral_values, axis=0, ddof=1)
std_err_mean = np.sqrt(variance/n_repetitions)


##plot Std error of the mean 
plt.figure(figsize=(10, 6))
plt.suptitle("Standard Error of the Mean (Sampling Mean Method)", fontsize=16)
plt.semilogx(batch_size, abs(std_err_mean), 'o', markersize=8, label='Data')
N_theory = np.logspace(1, 7, 100)  # Da 10 a 10^7
# Normalization
C = std_err_mean[0] * np.sqrt(batch_size[0]) 
theory_curve = C / np.sqrt(N_theory)
plt.semilogx(N_theory, theory_curve, '-', linewidth=2, color='red', label=r'Theoretical Curve: $C/\sqrt{N}$')
plt.xlabel('Batch size (N)', fontsize=12)
plt.ylabel('Standard Error of the Mean', fontsize=12)
plt.grid(True, alpha=0.3, which='both')
plt.legend()
plt.savefig("std_error_sampling.png", dpi=300, bbox_inches='tight')
plt.close()

##plot Std error of the mean (log-log)
plt.figure(figsize=(10, 6))
plt.suptitle("Linear Fit Standard Error of the Mean (Sampling Mean Method)", fontsize=16)
plt.loglog(batch_size, abs(std_err_mean), 'o', markersize=8, label='Data')

## Linear Fit log-log
# log(y) = log(C) - 0.5*log(N)  
log_N = np.log10(batch_size)
log_std_err = np.log10(abs(std_err_mean))
coefficients = np.polyfit(log_N, log_std_err, 1)
slope = coefficients[0]
intercept = coefficients[1]
N_fit = np.logspace(1, 7, 100)
fit_curve = 10**intercept * N_fit**slope
theo_curve_log = 10**intercept * N_fit**(-0.5)

plt.loglog(N_fit, fit_curve, '--', linewidth=2, color='blue', label=rf'Fit $\propto N^{{{slope:.3f}}}$')
plt.loglog(N_fit, theo_curve_log, '--', linewidth=2, color='red', label=rf'Theory $\propto N^{{-0.5}}$')
plt.xlabel('Batch size (N)', fontsize=12)
plt.ylabel('Standard Error of the Mean', fontsize=12)
plt.grid(True, alpha=0.3, which='both')
plt.legend()
plt.savefig("std_error_sampling_loglog_fit.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Fit Slope: {slope:.6f}")

## Convergence (plot the convergence for he first repetition)
plt.figure(figsize=(10, 6))
plt.suptitle("Convergence to the Correct Answer (Sampling Mean Method)", fontsize=16)
# Plot con scala logaritmica sull'asse x
plt.semilogx(batch_size, abs(integral_values[0, :]-true_value), 'o-', markersize=8, linewidth=2)
plt.xlabel('Batch size (N)', fontsize=12)
plt.ylabel('Result - True value', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig("convergence_sampling.png", dpi=300, bbox_inches='tight')
plt.close()
