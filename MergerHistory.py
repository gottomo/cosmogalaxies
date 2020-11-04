

# Imports
import math
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


c = 3 * 10**5 #km.s-1
h = 0.7
H_0 = 100 * h #km s−1 Mpc−1

D_H = (c/H_0) # Mpc
t_H = 1 / H_0 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# David Hogg Paper
EDS = np.array([1, 0, 0])
LD = np.array([0.05, 0, 0.95])
HL = np.array([0.2, 0.8, 0])

#Msci paper
LCDM = np.array([0.3, 0.7, 0])
EDS = np.array([1, 0, 0])
OCDM = ([0.3, 0, 0.7])

wBasic = np.array([-1, 0])
w8 = np.array([-0.8, 0])
w9 = np.array([-0.9, 0])
w11 = np.array([-1.1, 0])
w12 = np.array([-1.2, 0])

zArray = np.linspace(0.01, 5, num = 200)

# Schechter parameters
CanonicalSchechter = np.array([3.96*10**-3, 0.79*10**-3,-0.35, -1.47, 1010.66])









def Ez(z, OmeM, OmeCos, OmeK, w0, w1): #Doesn't switch equation automatically
    if OmeK > 0: # Is this distribution of equations true?
        return np.sqrt(OmeM * (1 + z)**3 + OmeK * (1 + z)**2 + OmeCos)
    elif OmeK == 0:
        return np.sqrt(OmeM * (1 + z)**3 + (1 - OmeM) * (1 + z)**(3 * (1 + w0 + w1)) * np.exp(-3 * w1 * z))




# Lookback time

def LBTime(z, OmeM, OmeCos, OmeK, w0, w1):
    integral = integrate.quad(lambda Z: 1/((1+Z)*Ez(Z, OmeM, OmeCos, OmeK, w0, w1)), 0, z) # integration holds error
    return integral[0] * t_H

LBTimeVec = np.vectorize(LBTime, excluded=['ws'])






# Density Change

def Mergers(z, OmeM, OmeCos, OmeK, w0, w1, OriginalDensity):
    Times = LBTimeVec(z, OmeM, OmeCos, OmeK, w0, w1)
    print(Times)
    
    Densities = np.zeros(len(z))
    
    Densities[0] = OriginalDensity
    print(Densities)
    
    for i, time in enumerate(Times):
        if i == 0:
            continue
        
        print(i)
        
        print(Times[i])
        print(Densities[i - 1])
        
        Densities[i] = Densities[i - 1] * (1 + (Times[i] - Times[i - 1]))
    
    return Densities
    
    
    
    
Evol = Mergers(zArray, LCDM[0], LCDM[1], LCDM[2], wBasic[0], wBasic[1], CanonicalSchechter[0])

plt.plot(zArray, Evol)
plt.show()







