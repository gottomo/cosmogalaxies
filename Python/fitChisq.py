import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# ToDo: write code to estimate uncertainty on fitted parameters

# Defining functions to be fitted
def linfnc(par, xdata):
    return par[0] * xdata + par[1]

def gammafnc(par, xdata):
    return par[0] * (1 + xdata)**par[1] * np.exp(par[2]*(1 + xdata))

# least square fitting needs different ordering of argument (indp. var. first)
def linfnc_ls(xdata, par1=0, par2=0):
    return par1 * xdata + par2

def gammafnc_ls(xdata, par1=0, par2=0, par3=0):
    return par1 * (1 + xdata)**par2 * np.exp(par3*(1 + xdata))

# Writing chisq function for general cases, taking a function as an input
def chisq(par, xdata, ydata, yerr, func):
    # Returns Chisq, with inputs of: array of parameters, arr; xdata; ydata; yerr; and function for evaluating chisq, func 
    chisqrd = np.sum( np.divide(  ydata - func(par, xdata), yerr, out=np.zeros(ydata.shape, dtype=float), where=yerr!=0 )**2 )
    return chisqrd


def chisq_fit(xdata, ydata, yerr, func, parlength):
    # Function that does chisq fitting, for inputs of xdata, ydata, yerr, function to be fitted, and number of parameters for the function being fitted
    # Return the fitted parameters and the reduced Chisq
    par0 = np.ones(parlength)    # creates an array filled with 1 for the initial guess of the parameters
    result = optimize.minimize(chisq, par0, args=(xdata, ydata, yerr, func))   # optimises here
    print(result['message'])
    par = result['x']
    chisqrd_r = result['fun'] / (np.size(xdata) - parlength)
    hess_inv = result['hess_inv'] # Inverse of hessian is equal to the covariance matrix
    # print(hess_inv)
    d_par = np.array([])
    for i in range( np.size(par) ):
        d_par = np.append(d_par, np.sqrt(hess_inv[i][i])) # minimum uncertainty estimate on the parameter (only qualitative!)
    return par, d_par, chisqrd_r

def ls_fit(xdata, ydata, yerr, func):
    # Function to fit least square fitting, returning parameters and their uncertainty using covariance matrix
    # Because of how the arguments of the fitting function must be in curve_fit(), we have to write if statement each time for how many fitting parameters we're using
    d_par = np.array([])
    print(xdata.size)
    print(ydata.size)
    par, cov = optimize.curve_fit(func, xdata, ydata, sigma=yerr)
    # print(par)
    # print(cov)
    for i in range( np.size(par) ):
        d_par = np.append(d_par, np.sqrt(cov[i][i])) # minimum uncertainty estimate on the parameter (only qualitative!)
    if len(par)==2:
        chisq = np.sum( np.power( (ydata - func(xdata, par[0], par[1])) / yerr, 2) )
    elif len(par)==3:
        chisq = np.sum( np.power( (ydata - func(xdata, par[0], par[1], par[2])) / yerr, 2) )
    else:
        chisq = 0
    chisq_r = chisq / (np.size(xdata)-len(par))
    return par, d_par, chisq_r


# ----------- Specific example of fitting alpha from Schechter function ------------

# # Define data
# z = np.array([0.45, 0.75, 1.25, 1.75, 2.25, 2.75])
# alpha = np.array([-1.41, -1.34, -1.31, -1.51, -1.56, -1.69])
# d_alpha = np.array([0.02, 0.02, 0.03, 0.03, 0.06, 0.06])

# # Fit using the chisq_fit function
# par, d_par, chisq_r = ls_fit(z, alpha, d_alpha, linfnc_ls) #chisq_fit(z, alpha, d_alpha, linfnc, 2)
# a, b = par
# print('Parameters: ' + str(par))
# print('Uncertainty estimate on parameters: ' + str(d_par))
# print('Reduced Chisq = ' + str(chisq_r))

# # Plot the results
# fig, ax = plt.subplots()
# ax.set_xlabel('redshift z')
# ax.set_ylabel(r'$\alpha$')
# ax.errorbar(z, alpha, d_alpha, fmt = '.', label = 'data points', color = 'black', ecolor = 'black')
# ax.plot(z, linfnc((a,b), z), label = r'$\alpha = a z + b$', color = 'black')
# fig.legend()
# plt.grid()
# plt.show()


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
