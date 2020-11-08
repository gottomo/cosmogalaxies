
# Imports
import math
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


# Test LBT


c = 3 * 10**5 #km.s-1
h = 0.7
H_0 = 100 * h #km s−1 Mpc−1

D_H = (c/H_0) # Mpc
t_H = 1 / H_0

H_0 = 100 * h
t_H = 1 / H_0


def omega_de(z, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the density parameter of dark energy for different models
	# If using constant_w model, you need to also input the value of w to test models other than w=-1

	if model == 'LCDM':
		# This branch not really necessary, can be replaces with model = 'linear'
		return omega_0
	elif model == 'constant_w':
		# This branch not really necessary, could be replaces with model = 'linear'
		return omega_0 * (1 + z)**(3*(1+w_0))
	elif model == 'linear':
		return omega_0 * (1 + z)**(3*(1+w_0+w_1)) * np.exp(-3*w_1*z)



def e(z, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the parameter E(z) required for other calculations

	e = np.sqrt(omega_m * (1 + z) ** 3 + omega_k * (1 + z) **2 + omega_de(z, omega_0, model, w_0 = w_0, w_1 = w_1))
	return e



def LBTime(z, omega_m, omega_k, omega_0, model, h=0.7):
    t_H = 9.78 * h
    
    
    integral = integrate.quad(lambda Z: 1/((1+Z)*e(Z, omega_m, omega_k, omega_0, model)), 0, z) # integration holds error
    return integral[0] * t_H #* (1/100*h) # Want this in GIGAYEAR

LBTimeVec = np.vectorize(LBTime)

# Confirm that we can recreate the hogg plot for lookback time,
# and thus that the lookback time function is correct
# =============================================================================
# EDS = np.array([1, 0, 0])
# LD = np.array([0.05, 0, 0.95])
# HL = np.array([0.2, 0.8, 0])
# 
# zArray = np.linspace(0.01, 5, num = 200)
# EDS = LBTimeVec(zArray, 1, 0, 0, "LCDM")
# plt.plot(zArray, EDS, label = "einstein")
# LD = LBTimeVec(zArray, 0.05, 0.95, 0, "LCDM")
# plt.plot(zArray, LD, label = "LD")
# HL = LBTimeVec(zArray, 0.2, 0, 0.8, "LCDM")
# plt.plot(zArray, HL, label = "HL")
# plt.legend()
# plt.show()
# =============================================================================




def Phi(z, mass, OriginalPhi, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
    # This function calculates the evolution of phi iteratively, ie,
    # it uses the last known value of galaxy density, times merger rate
    # at that z, to give the next value of galaxy density.
    # As such, every value of density depends on the last.
    # So these calculations don't go to waste, they are kept in an array
    # and the whole array is returned so there is no need to repeat
    # the calculation for every value of z.
    
    
    Times = LBTimeVec(z, omega_m, omega_k, omega_0, model, h=h)
    #print(Times)
    
    Densities = np.zeros(len(z))
    
    Densities[0] = OriginalPhi
    #print(Densities)
    
    for i, time in enumerate(Times):
        if i == 0:
            continue
        R = 0
        
        if mass > 10.3:
        
            if z[i] < 1:
                R = 0.07 # per GIGAYER!
            elif z[i] < 1.5:
                R = 0.12
            elif z[i] < 2:
                R = 0.29
            elif z[i] < 2.5:
                R = 0.39
            elif z[i] < 3:
                R = 0.58
            elif z[i] < 3.5:
                R = 1.07
            elif z[i] < 4.5:
                R = 1.21
            elif z[i] < 5.5:
                R = 3.29
            elif z[i] < 6.5:
                R = 7.59
        
        else:
        
            if z[i] < 1:
                R = 0.08 # per GIGAYER!
            elif z[i] < 1.5:
                R = 0.22
            elif z[i] < 2:
                R = 0.53
            elif z[i] < 2.5:
                R = 0.97
            elif z[i] < 3:
                R = 1.26
            elif z[i] < 3.5:
                R = 2.91
            else:
                R = 14.15
        
        
        Densities[i] = Densities[i - 1] * (1 + R * (Times[i] - Times[i - 1]))
        # Does this work backwards???
    
    return Densities


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Merger rate is a function of pair fraction and merger timescale; ie.
# the number of galaxies within a given radius of one another is counted
# to find the fraction of galaxies in pairs. It is then assumed that 
# all those pairs will eventually fuse over a certain timescale.
    
# The pair fraction given here is the fit found by Duncan, Conselice at al.
# This is then conveted into a merger rate using the Snyder approximation


def Pair_Fraction(z, mass):
    a = 0
    M = 0
    
    if mass < 9.7:
        print("Galaxy mass too low, merger rate unknown")
    elif mass < 10.3:
        M = 0.024
        a = 1.8
    else:
        M = 0.032
        a = 0.8
    
    return M*(1+z)**a


def Merger_Rate_Typical(z, mass): # Does not work, because paper uses 
    # Snyder, and therefore the parameters for snyder are given only 
    # a glance. Should be functions of z. Links are given to the papers
    # that DO describe the function.

    return (Pair_Fraction(z, mass) * 0.6)/0.5
    
Merger_Rate_TypicalVec = np.vectorize(Merger_Rate_Typical)


def Merger_Rate_Snyder(z, mass): # Looks good

    return (Pair_Fraction(z,mass) * (1+z) ** 2) / 2.4

Merger_Rate_SnyderVec = np.vectorize(Merger_Rate_Snyder)


# Now check the merger rates given in the paper agree with the function
# for merger rate as a function of z:


zArray = np.linspace(0.01, 6, num = 200)
Typ = Merger_Rate_TypicalVec(zArray, 11)
Sny = Merger_Rate_SnyderVec(zArray, 11)

LargerMergerRates = np.array([[0.75, 0.07, 0.01], [1.25, 0.12, 0.02], [1.75, 0.29, 0.03], [2.25, 0.39, 0.06], [2.75, 0.58, 0.1], [3.25, 1.07, 0.25], [4, 1.21, 0.37], [5, 3.29, 1.07], [6, 7.59, 2.069]])

#plt.plot(zArray, Typ, label = "Typical")
plt.plot(zArray, Sny, label = "Snyder")
#plt.scatter(LargerMergerRates[:,0], LargerMergerRates[:,1])
plt.errorbar(LargerMergerRates[:,0],LargerMergerRates[:,1], yerr = LargerMergerRates[:,2], fmt="x", label = "Given merger rates")

#plt.yscale("Log")
plt.legend()
plt.show()


zArray = np.linspace(0.01, 4.5, num = 200)
Typ = Merger_Rate_TypicalVec(zArray, 10)
Sny = Merger_Rate_SnyderVec(zArray, 10)

SmallerMergerRates = np.array([[0.75, 0.08, 0.01], [1.25, 0.22, 0.02], [1.75, 0.53, 0.04], [2.25, 0.97, 0.09], [2.75, 1.26, 0.21]])

#plt.plot(zArray, Typ, label = "Typical")
plt.plot(zArray, Sny, label = "Snyder")
#plt.scatter(SmallerMergerRates[:,0], SmallerMergerRates[:,1])
plt.errorbar(SmallerMergerRates[:,0],SmallerMergerRates[:,1], yerr = SmallerMergerRates[:,2], fmt="x", label = "Given merger rates")

#plt.yscale("Log")
plt.legend()
plt.show()


# These aren't perfect; it might be worth fitting to find the parameters
# a & M instead of using those from the paper, but then not sure that the
# merger rates in the paper aren't derived using these parameters, and
# therefore parameters more fundamental Must check.


def Phi_func(z, mass, OriginalPhi, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
    # This function calculates an array of phi densities iteratively,
    # but uses the snyder approximation to calculate the merger rate,
    # rather than using a set of merger rates which change after a
    # certain redsift interval.
    
    Times = LBTimeVec(z, omega_m, omega_k, omega_0, model)
    #print(Times)
    
    Densities = np.zeros(len(z))
    
    Densities[0] = OriginalPhi
    #print(Densities)
    
    for i, time in enumerate(Times):
        if i == 0:
            continue
        
        Densities[i] = Densities[i - 1] * (1 + Merger_Rate_Snyder(z[i], mass) * (Times[i] - Times[i - 1]))
        # Does this work backwards???
    
    return Densities




    
def Phi_direct(z, mass, OriginalPhi, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
    # This phi calculation integrates the Snyder approximation, which lets us 
    # directly access phi for a given z, without iteratively calculating
    # every previous phi value. This just makes it neater, and more in line
    # with the style of functions already in the code, which return one
    # value and then are vectorized.
    
    t_H = 9.78 * h
    
    
    if mass < 9.7: # Depending on mass, choose the parameters of the
        # pair fraction function
        print("Galaxy mass too low, merger rate unknown")
    elif mass < 10.3:
        M = 0.024
        a = 1.8
    else:
        M = 0.032
        a = 0.8
    
    
    # integrate the Snyder approximation to find how much phi should change.
    
    integral = integrate.quad(lambda Z: ((t_H * M * (1+Z) ** (a+1)) / (2.4 * e(Z, omega_m, omega_k, omega_0, model))), 0, z) # integration holds error
    
    
    return np.exp(integral[0]) * OriginalPhi


# Plot the iterative evolution of number density, the number density evolution
# using the snyder function iteratively, and finally the integrated snyder
# to compare how they work.




plt.figure(figsize=(8,8))

Phi_directVec = np.vectorize(Phi_direct)

zArray = np.linspace(0.01, 6, num = 2000)

Density = Phi_directVec(zArray, 11, 0.002, 0.3, 0, 0.7, "linear")
plt.plot(zArray, Density, label = "Snyder integrated")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Phi_funcVec = np.vectorize(Phi_func)

zArray = np.linspace(0.01, 6, num = 2000)

Density = Phi(zArray, 11, 0.002, 0.3, 0, 0.7, "linear")

# Vertical lines show where the merger ratio changes

plt.axvline(x=1, linestyle = ":")
plt.axvline(x=1.5, linestyle = ":")
plt.axvline(x=2, linestyle = ":")
plt.axvline(x=2.5, linestyle = ":")
plt.axvline(x=3, linestyle = ":")
plt.axvline(x=3.5, linestyle = ":")
plt.axvline(x=4.5, linestyle = ":")
plt.axvline(x=5.5, linestyle = ":")


plt.plot(zArray, Density, label = "Iterative")

Density = (Phi_func(zArray, 11, 0.002, 0.3, 0, 0.7, "linear"))

plt.plot(zArray, Density, label = "Snyder approximation")

plt.title("Evolution of number density phi due to mergers, for Galaxies log(M)>10.3")

plt.yscale("Log")

plt.ylabel("Number density $\Phi$ ($Mpc^{-1}$)")
plt.xlabel("Redshift z")

plt.legend()
plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



plt.figure(figsize=(8,8))

Phi_directVec = np.vectorize(Phi_direct)

zArray = np.linspace(0.01, 4, num = 2000)

Density = Phi_directVec(zArray, 10, 0.002, 0.3, 0, 0.7, "linear")
plt.plot(zArray, Density, label = "Snyder integrated")



Phi_funcVec = np.vectorize(Phi_func)

zArray = np.linspace(0.01, 4, num = 2000)

Density = Phi(zArray, 10, 0.002, 0.3, 0, 0.7, "linear")

# Vertical lines show where the merger ratio changes

plt.axvline(x=1, linestyle = ":")
plt.axvline(x=1.5, linestyle = ":")
plt.axvline(x=2, linestyle = ":")
plt.axvline(x=2.5, linestyle = ":")
plt.axvline(x=3, linestyle = ":")
plt.axvline(x=3.5, linestyle = ":")


plt.plot(zArray, Density, label = "Iterative")

Density = (Phi_func(zArray, 10, 0.002, 0.3, 0, 0.7, "linear"))

plt.plot(zArray, Density, label = "Snyder approximation")


plt.yscale("Log")

plt.ylabel("Number density $\Phi$ ($Mpc^{-1}$)")
plt.xlabel("Redshift z")

plt.title("Evolution of number density phi due to mergers, for Galaxies 9.7<log(M)<10.3")
plt.legend()
plt.show()



















