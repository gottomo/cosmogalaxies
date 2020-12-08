import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# ToDo: write code to estimate uncertainty on fitted parameters

# Defining functions to be fitted
def linfnc(par, xdata):
    return par[0] * xdata + par[1]

def gammafnc(par, xdata):
    return par[0] * (1 + xdata)**par[1] * np.exp(par[2]*(1 + xdata))

# Writing chisq function for general cases, taking a function as an input
def chisq(par, xdata, ydata, yerr, func):
    # Returns Chisq, with inputs of: array of parameters, arr; xdata; ydata; yerr; and function for evaluating chisq, func 
    return np.sum( ( ( ydata - func(par, xdata) ) / yerr )**2 )


def chisq_fit(xdata, ydata, yerr, func, parlength):
    # Function that does chisq fitting, for inputs of xdata, ydata, yerr, function to be fitted, and number of parameters for the function being fitted
    # Return the fitted parameters and the reduced Chisq
    par0 = np.ones(parlength)    # creates an array filled with 1 for the initial guess of the parameters
    result = optimize.minimize(chisq, par0, args=(xdata, ydata, yerr, func))   # optimises here
    print(result['message'])
    par = result['x']
    chisqrd_r = result['fun'] / (np.size(xdata) - parlength)
    return par, chisqrd_r



# ----------- Specific example of fitting alpha from Schechter function ------------

# Define data
z = np.array([0.45, 0.75, 1.25, 1.75, 2.25, 2.75])
alpha = np.array([-1.41, -1.34, -1.31, -1.51, -1.56, -1.69])
d_alpha = np.array([0.02, 0.02, 0.03, 0.03, 0.06, 0.06])

# Fit using the chisq_fit function
par, chisq_r = chisq_fit(z, alpha, d_alpha, linfnc, 2)
a, b = par
print('parameters: ' + str(par))
print('Reduced Chisq = ' + str(chisq_r))

# Plot the results
fig, ax = plt.subplots()
ax.set_xlabel('redshift z')
ax.set_ylabel(r'$\alpha$')
ax.errorbar(z, alpha, d_alpha, fmt = '.', label = 'data points', color = 'black', ecolor = 'black')
ax.plot(z, linfnc((a,b), z), label = r'$\alpha = a z + b$', color = 'black')
fig.legend()
plt.grid()
plt.show()


# ----------- Obsolete functions and codes -------------

# Writing chisq functions for individual cases (obsolete)
def chisq_gammafnc(par, xdata, ydata, yerr):
    gammafnc = par[0] * (1 + xdata)**par[1] * np.exp(par[2]*(1 + xdata))
    chisq = np.sum(((ydata - gammafnc)/yerr)**2)
    return chisq

def chisq_linfnc(par, xdata, ydata, yerr):
    linfnc = par[0] * xdata + par[1]
    chisq = np.sum(((ydata - linfnc)/yerr)**2)
    return chisq
