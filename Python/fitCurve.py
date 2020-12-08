import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

def linfnc(x, a, b):
    return a*x + b

def chisq_linfnc(par1, par2, xdata, ydata, yerr):
    linfnc = par1 * xdata + par2
    chisq = np.sum(((ydata - linfnc)/yerr)**2)
    return chisq

def chisq_gammafnc(par1, par2, par3, xdata, ydata, yerr):
    gammafnc = par1 * (1 + xdata)**par2 * np.exp(par3*(1 + xdata))
    chisq = np.sum(((ydata - gammafnc)/yerr)**2)
    return chisq

# Define data
z = np.array([0.45, 0.75, 1.25, 1.75, 2.25, 2.75])
alpha = np.array([-1.41, -1.34, -1.31, -1.51, -1.56, -1.69])
d_alpha = np.array([0.02, 0.02, 0.03, 0.03, 0.06, 0.06])

# Fit data with linear function
popt, pcov = optimize.curve_fit(linfnc, z, alpha, sigma = d_alpha)
print('a = ' + str(popt[0]))
print('b = ' + str(popt[1]))

# Calculate the chi squared
chi_sqr = np.sum( (alpha - linfnc(z, popt[0], popt[1]))**2 / d_alpha**2 )
print('Chi squared: ' + str(chi_sqr))
print('Reduced chi squared: ' + str(chi_sqr/(np.size(alpha)-2)))

# Plot the results
fig, ax = plt.subplots()
ax.set_xlabel('redshift z')
ax.set_ylabel(r'$\alpha$')
ax.errorbar(z, alpha, d_alpha, fmt = '.', label = 'data points', color = 'black', ecolor = 'black')
ax.plot(z, linfnc(z, popt[0], popt[1]), label = r'$\alpha = a z + b$', color = 'black')
fig.legend()
plt.grid()
plt.show()

