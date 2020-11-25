import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate # this module does the integration
import os

from scipy.signal import find_peaks

# This code finds the comoving volume element per solid angle per redshift for different cosmological models
# Hubble constant is h = 0.7 by default
# By default the functions are written so that it returns cosmological constant (LCDM) dark energy model

one_sqr_degree = (np.pi/180)**2 

# Functinos
	
def omega_de(z, omega_0, model, **kwargs ):
	# Returns the density parameter of dark energy for different models
	# If using constant_w model, you need to also input the value of w to test models other than w=-1
	w_0 = kwargs.get('w_0') # by default gives LCDM model
	w_1 = kwargs.get('w_1')
	if model == 'LCDM':
		# This branch not really necessary, can be replaces with model = 'linear'
		return omega_0
	elif model == 'constant_w':
		# This branch not really necessary, could be replaces with model = 'linear'
		return omega_0 * (1 + z)**(3*(1+w_0))
	elif model == 'linear':
		return omega_0 * (1 + z)**(3*(1+w_0+w_1)) * np.exp(-3*w_1*z)
		# if np.isscalar(z):
			# return omega_0 * np.exp(3 * integrate.quad(integrand_linear, 0, z, args = (w_0, w_1) )[0] )
		# else:
			# return omega_0 * np.exp(3 * integrate.quad(integrand_linear, 0, val, args = (w_0, w_1) )[0] )

def e(z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the parameter E(z) required for other calculations
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	e = np.sqrt(omega_m * (1 + z) ** 3 + omega_k * (1 + z) **2 + omega_de(z, omega_0, model, w_0 = w_0, w_1 = w_1))
	return e
	
def comoving_d_los(z, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the line-of-sight comoving distance in Mpc
	# Also works in the case if z is array
	# ToDo: also include error of integration to the results?
	com_d_los = 3000 / h * integrate.quad(lambda Z: 1/e(Z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1), 0, z)[0]
	return com_d_los
	
def comoving_d(z, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the transverse comoving distance in Mpc
	# Works for all geometry of space.
	com_d_los = comoving_d_los(z, omega_m, omega_k, omega_0, model, w_0, w_1, h) # reassigning the function to a variable so that the code is shorter
	if omega_k == 0:
		return com_d_los
	elif omega_k < 0:
		com_d = 3000 / (h * np.sqrt(omega_k)) * np.sin(np.sqrt(abs(omega_k)) * com_d_los * h / 3000)
		return com_d
	elif omega_k > 0:
		com_d = 3000 / (h * np.sqrt(omega_k)) * np.sinh(np.sqrt(abs(omega_k)) * com_d_los * h / 3000)
		return com_d

def ang_diameter_d(z, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns angular diameter distance
	ang_di_d = comoving_d(z, omega_m, omega_k, omega_0, model, w_0, w_1, h) / (1 + z)
	return ang_di_d
	
def lum_d(z, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the luminosity distance
	lum_d = comoving_d(z, omega_m, omega_k, omega_0, model, w_0, w_1, h) * (1 + z)
	return lum_d

def comoving_vol_elm(z, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns comoving volume element per unit solid angle per unit redshift
	# is in unit of Mpc^3
	com_vol_elm = 3000 / h * (1 + z)**2 * ang_diameter_d(z, omega_m, omega_k, omega_0, model, w_0, w_1, h)**2 / e(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1)
	return com_vol_elm
	
def delta_com_vol_elm(z, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the difference between comoving volume element of LCDM and a chosen model
	return comoving_vol_elm(z, 0.3, 0, 0.7, 'LCDM', -1, 0, h) - comoving_vol_elm(z, omega_m, omega_k, omega_0, model, w_0 , w_1, h)
	
def magnitude_bol(L, z, omega_m, omega_k, omega_0, model, w_0=0, w_1=0, h=0.7):
	# Returns the magnitude of an object with luminosity L at redshift z for different cosmology
	# Since in this magnitude expression, the distance must be in the unit of 10 pc, the lum_d (in Mpc) is multiplied by 10^-5
	# const. wouldn't matter if we only use magnitude difference, so maybe could write different function in that case?
	const = 4.74 + np.log10(3.84e26 / (4*np.pi*10**2))
	return -2.5 * np.log10(L / (4*np.pi*lum_d(z, omega_m, omega_k, omega_0, model, w_0, w_1, h)**2 * 1e-5**2 ) )
	
def delta_mag(z, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the difference between magnitudes of LCDM and a given cosmology model for a given redshift
	return 5 * np.log10(lum_d(z, omega_m, omega_k, omega_0, model, w_0, w_1, h) / lum_d(z, 0.3, 0, 0.7, 'LCDM', -1, 0, h))

def schechter_mass(mass, break_mass, phi1, phi2, alpha1, alpha2):
	# Returns Schechter function as a function of mass
	# mass shall be given in the unit of dex (i.e. mass = log10(mass_in_solar_mass) )
	# Will be in unit of dex^-1 Mpc^-3
	mass_diff = mass - break_mass
	return np.log(10) * np.exp(- np.power(10, mass_diff)) * (phi1 * np.power(10, alpha1 * mass_diff) + phi2 * np.power(10, alpha2 * mass_diff)) * np.power(10, mass_diff)
	
def appmag_to_absmag(mag, z, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Converts apparent magnitude to absolute magnitude
	mag_abs = mag - 5 * np.log10(lum_d(z, omega_m, omega_k, omega_0, model, w_0, w_1, h) *1e6 / 10)
	return mag_abs

def mag_to_mass(mag, z, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the mass of a galaxy given its apparent magnitude and the redshift
	# First calculate the luminosity from the apparent magnitude and redshift
	dm = 5 * np.log10(lum_d(z, omega_m, omega_k, omega_0, model, w_0, w_1, h)*1e6 /10)
	lum = np.power(10, (4.74 + dm - mag) / 2.5)  # note that lum_d is in unit of Mpc
	# Then calculate and return mass by multiplying luminosity with mass-to-light ratio to get mass (note that mass is actually in log scale, so have to modify the equation as below
	#mass = np.log10(lum) + np.log10(5)
	mass = np.log10(lum) + np.log10(5)
	return mass

def mag_to_mass_notlog(mag, z, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the mass of a galaxy given its apparent magnitude and the redshift
	# First calculate the luminosity from the apparent magnitude and redshift
	dm = 5 * np.log10(lum_d(z, omega_m, omega_k, omega_0, model, w_0, w_1, h) * 1e6 / 10)
	lum = np.power(10, (4.74 + dm - mag) / 2.5)  # note that lum_d is in unit of Mpc
	# Then calculate and return mass by multiplying luminosity with mass-to-light ratio to get mass (note that mass is actually in log scale, so have to modify the equation as below
	#mass = np.log10(lum) + np.log10(5)
	mass = lum * 5
	return mass


def mag_to_massaa(mag, z, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the mass of a galaxy given its apparent magnitude and the redshift
	# First calculate the luminosity from the apparent magnitude and redshift
	mag_abs = mag - 5 * np.log10(lum_d(z, omega_m, omega_k, omega_0, model, w_0, w_1, h) * 1e6 / 10)
	lum = np.power(10, (4.74 - mag_abs) / 2.5) * 3.84e26 # note that lum_d is in unit of Mpc
	# Then calculate and return mass by multiplying luminosity with mass-to-light ratio to get mass (note that mass is actually in log scale, so have to modify the equation as below
	#mass = np.log10(lum) + np.log10(5)
	mass = 5 * lum / 1.989e30
	return np.log10(mass)
	
def mag_to_mass_old(mag, z, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the mass of a galaxy given its apparent magnitude and the redshift
	# First calculate the luminosity from the apparent magnitude and redshift
	# k is a const. for setting a reference point for magnitude, which will return the luminosity in the unit of solar luminosity
	k = -0.26
	lum = np.power(10, (k - mag) / 2.5) * lum_d(z, omega_m, omega_k, omega_0, model, w_0, w_1, h)**2 * 1e12# note that lum_d is in unit of Mpc
	# Then calculate and return mass by multiplying luminosity with mass-to-light ratio to get mass (note that mass is actually in log scale, so have to modify the equation as below
	mass = np.log10(lum) + np.log10(5)
	return mass

def galaxy_no_density(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the number density of galaxies as a function of redshift for a given magnitude limit with given cosmology model
	# In the unit of Mpc^-3
	
	mass_min = mag_to_mass(mag, z, omega_m, omega_k, omega_0, model, w_0, w_1, h)
	
	return integrate.quad(lambda mass: schechter_mass(mass, break_mass, phi1, phi2, alpha1, alpha2), mass_min, 15)[0]
	
def galaxy_number(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	
	number = galaxy_no_density(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h) * comoving_vol_elm(z, omega_m, omega_k, omega_0, model, w_0, w_1, h)
	return number


	
def delta_galaxy_number(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the difference in number density of galaxies between LCDM and a given cosmology as a function of redshift, with given mass limit
	number_lambda = galaxy_no_density(z, mag, break_mass, phi1, phi2, alpha1, alpha2, 0.3, 0, 0.7, 'LCDM', -1, 0, h) * comoving_vol_elm(z, 0.3, 0, 0.7, 'LCDM', -1, 0, h)
	number = galaxy_no_density(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h) * comoving_vol_elm(z, omega_m, omega_k, omega_0, model, w_0, w_1, h)
	return number - number_lambda
	
def delta_galaxy_number_z_old(z, z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the difference in number density of galaxies between an arbitrary z and a set z
	number_z_ref = galaxy_no_density(z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h) * comoving_vol_elm(z_ref, omega_m, omega_k, omega_0, model, w_0, w_1, h)
	number = galaxy_no_density(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h) * comoving_vol_elm(z, omega_m, omega_k, omega_0, model, w_0, w_1, h)
	return number - number_z_ref

def delta_galaxy_number_z(z, z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the difference in number density of galaxies between an arbitrary z and a set z per one square degree
	number_z_ref = galaxy_number(z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h)
	number = galaxy_number(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h)
	# print(model)
	# print("number: " + str(number))
	# print("number_z: " + str(number_z_ref))
	return number - number_z_ref


def experimental_delta_galaxy_number_rel_z(z, z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the difference in number density of galaxies between an arbitrary z and a set z
	number_z_ref = galaxy_number(z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h)
	number = galaxy_number(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h)
	# print(model)
	# print("number: " + str(number))
	# print("number_z: " + str(number_z_ref))
	# Error analysis
	d_number = np.sqrt(number)
	d_number_z_ref = np.sqrt(number_z_ref)
	rel = (number - number_z_ref) / number_z_ref
	if(number!=number_z_ref):
		d_rel = np.sqrt((d_number**2+d_number_z_ref**2)/(number-number_z_ref)**2 + (d_number_z_ref/number_z_ref)**2)
	else:
		d_rel=0
	# print(np.array([rel, d_rel]))
	return np.array([[rel, d_rel]])

def delta_galaxy_number_rel_z(z, z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the difference in number density of galaxies between an arbitrary z and a set z
	number_z_ref = galaxy_number(z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h)
	number = galaxy_number(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h)
	# print(model)
	# print("number: " + str(number))
	# print("number_z: " + str(number_z_ref))
	rel = (number - number_z_ref) / number_z_ref
	return rel

def d_delta_galaxy_number_rel_z(z, z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, sqd=1, w_0=-1, w_1=0, h=0.7):
	# Returns the uncertianty of difference in number density of galaxies between an arbitrary z and a set z
	# Not the most ideal way to do things, by calling this function alongside the delta_galaxy_number_rel_z(), we're doing twice the work
	number_z_ref = galaxy_number(z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h) * one_sqr_degree * sqd
	number = galaxy_number(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h) * one_sqr_degree * sqd
	# print(model)
	# print("number: " + str(number))
	# print("number_z: " + str(number_z_ref))
	# Error analysis
	d_number = np.sqrt(number)
	d_number_z_ref = np.sqrt(number_z_ref)
	rel = (number - number_z_ref) / number_z_ref
	if(number!=number_z_ref):
		d_rel = np.sqrt((d_number**2+d_number_z_ref**2)/(number-number_z_ref)**2 + (d_number_z_ref/number_z_ref)**2) * rel
	else:
		d_rel=0
	return d_rel


def alpha(z, a, b):
	# Returns the parameter alpha for single shcechter function as a function of z using fitting parameters alpha = a * z + b
	return a * z + b
	
#---------------- functions for galaxy mergers--------------------------------------
def LBTime(z, omega_m, omega_k, omega_0, model, h=0.7):
    t_H = 9.78 / h
    
    
    integral = integrate.quad(lambda Z: 1/((1+Z)*e(Z, omega_m, omega_k, omega_0, model)), 0, z) # integration holds error
    return integral[0] * t_H #* (1/100*h) # Want this in GIGAYEAR

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

def Merger_Rate_Snyder(z, mass): # Looks good

    return (Pair_Fraction(z,mass) * (1+z) ** 2) / 2.4

def Phi_direct(z, mass, OriginalPhi, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
    # This phi calculation integrates the Snyder approximation, which lets us 
    # directly access phi for a given z, without iteratively calculating
    # every previous phi value. This just makes it neater, and more in line
    # with the style of functions already in the code, which return one
    # value and then are vectorized.
    
    t_H = 9.78 / h
    
    
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
    
    integral = integrate.quad(lambda Z: ((t_H * M * (1+Z) ** (a+1)) / (2.4 * e(Z, omega_m, omega_k, omega_0, model, w_0=w_0, w_1=w_1, h=h))), 0, z) # integration holds error
    
    
    return np.exp(integral[0]) * OriginalPhi

Merger_Rate_SnyderVec = np.vectorize(Merger_Rate_Snyder)


# Vectorizes the functions I want to plot with matplotlib, so that function can accept np arrays
comoving_vol_elm_vec = np.vectorize(comoving_vol_elm)
delta_com_vol_elm_vec = np.vectorize(delta_com_vol_elm)
magnitude_bol_vec = np.vectorize(magnitude_bol)
schechter_mass_vec = np.vectorize(schechter_mass)
mag_to_mass_vec = np.vectorize(mag_to_mass)
galaxy_number_vec = np.vectorize(galaxy_number)
delta_galaxy_number_vec = np.vectorize(delta_galaxy_number)
lum_d_vec = np.vectorize(lum_d)
delta_mag_vec = np.vectorize(delta_mag)
delta_galaxy_number_z_vec = np.vectorize(delta_galaxy_number_z)
delta_galaxy_number_rel_z_vec = np.vectorize(delta_galaxy_number_rel_z)
appmag_to_absmag_vec = np.vectorize(appmag_to_absmag)
alpha_vec = np.vectorize(alpha)
LBTimeVec = np.vectorize(LBTime)
Phi_directVec = np.vectorize(Phi_direct)


# ------------------------------------------Set the constants-------------------------------------------
h_today = 0.7
magnitude_min = 32
z_ref = 6
z = np.linspace(0.01,z_ref,round(50*z_ref/3))

modelname = np.array(['LCDM', 'EdS', 'OCDM', 'w8', 'w9', 'w11', 'w12'])

# Set the subfloder where the text outputs will be placed
cur_path = os.path.dirname(__file__) # get the directory where the script is placed
# print("path is: " + cur_path)

# print("Constant magnitude=21, varying redshift")
# print(np.array([0.1, 1, 2, 3]))
# print(mag_to_mass_vec(21, np.array([0.1, 1, 2, 3]), 0.3, 0, 0.7, 'LCDM', h = h_today))

# print("Constant z=0.1, varying magnitude")
# print(np.array([15, 20, 25, 30]))
# print(mag_to_mass_vec(np.array([15, 20, 25, 30]), 0.1, 0.3, 0, 0.7, 'LCDM'))

# -------------------------Plotting stuff------------------------------------
# z=1

# ---------------------------Comoving volume element------------------------------
# Plots of comoving volume element per uint solid angle per unit redshift, normalized by 1/(D_H)^3
# fig1, ax1 = plt.subplots()
# ax1.set_xlabel("Redshift z")
# ax1.set_ylabel(r"Dimensionless comoving volume element $\frac{dV_C}{d\Omega dz}  \frac{1}{H_0^3}$")
# ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, 'LCDM', -1, 0, h=h_today) / (3000/h_today)**3, '-', label='LCDM') # LCDM
# ax1.plot(z, comoving_vol_elm_vec(z, 1, 0, 0, 'LCDM', -1, 0, h=h_today) / (3000/h_today)**3, '--', label='E-deS') # E-deS
# ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0.7, 0, 'LCDM', -1, 0, h=h_today) / (3000/h_today)**3, '-.', label='OCDM') # OCDM
# ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, 'constant_w', -0.8, 0, h_today) / (3000/h_today)**3, ':', label='w = -0.8')
# ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, 'constant_w', -0.9, 0, h_today) / (3000/h_today)**3, ':', label='w = -0.9')
# ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, 'constant_w', -1.1, 0, h_today) / (3000/h_today)**3, ':', label='w = -1.1')
# ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, 'constant_w', -1.2, 0, h_today) / (3000/h_today)**3, ':', label='w = -1.2')
# plt.grid()
# fig1.legend()


# fig1_1, ax1_1 = plt.subplots()
# ax1_1.set_xlabel("Redshift z")
# ax1_1.set_ylabel(r"Dimensionless comoving volume element $\frac{dV_C}{d\Omega dz}  \frac{1}{H_0^3}$")
# ax1_1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, 'LCDM', h=h_today), '-', label='LCDM') # LCDM
# ax1_1.plot(z, comoving_vol_elm_vec(z, 1, 0, 0, 'LCDM', h=h_today), '--', label='E-deS') # E-deS
# ax1_1.plot(z, comoving_vol_elm_vec(z, 0.3, 0.7, 0, 'LCDM', h=h_today), '-.', label='OCDM') # OCDM
# ax1_1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), ':', label='w = -0.8')
# ax1_1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), ':', label='w = -0.9')
# ax1_1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), ':', label='w = -1.1')
# ax1_1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), ':', label='w = -1.2')
# plt.grid()
# fig1_1.legend()

# ------------------Angular diameter distance---------------------------
# fig2, ax2 = plt.subplots()
# ax2.set_xlabel("Redshift z")
# ax2.set_ylabel(r"Angular diameter distance $D_A$ (Mpc)")
# # Plots of comoving volume element per uint solid angle per unit redshift, normalized by 1/(D_H)^3
# ax2.plot(z, ang_diameter_d(z, 0.3, 0, 0.7, h=h_today), '-', label='LCDM') # LCDM
# ax2.plot(z, ang_diameter_d(z, 1, 0, 0, h=h_today), '--', label='E-deS') # E-deS
# ax2.plot(z, ang_diameter_d(z, 0.3, 0.7, 0, h=h_today), '-.', label='OCDM') # OCDM
# ax2.plot(z, ang_diameter_d(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), ':', label='w = -0.8')
# ax2.plot(z, ang_diameter_d(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), ':', label='w = -0.9')
# ax2.plot(z, ang_diameter_d(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), ':', label='w = -1.1')
# ax2.plot(z, ang_diameter_d(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), ':', label='w = -1.2')
# # plt.plot(z, ang_diameter_d(z, 0.3, 0, 0.7, model = 'linear', w_0 = -0.9 , h=h_today) / (3000/h_today)**3, label='linear')
# fig2.legend()
# plt.grid()


# -----------------------Plot of delta_com_vol_elm--------------------------------
# fig3, ax3 = plt.subplots()
# ax3.set_title(r"Difference in comoving volume elements between LCDM and arbitrary model")
# ax3.set_xlabel("Redshift z")
# ax3.set_ylabel(r"$\Delta\frac{dV_C}{d\Omega dz}  \frac{1}{H_0^3}$")
# ax3.plot(z, delta_com_vol_elm_vec(z, 0.3, 0, 0.7, 'LCDM', h=h_today) / (3000/h_today)**3, '-', label='LCDM') # LCDM
# ax3.plot(z, delta_com_vol_elm_vec(z, 1, 0, 0, 'LCDM', h=h_today) / (3000/h_today)**3, '--', label='E-deS') # E-deS
# ax3.plot(z, delta_com_vol_elm_vec(z, 0.3, 0.7, 0, 'LCDM', h=h_today) / (3000/h_today)**3, '-.', label='OCDM') # OCDM
# ax3.plot(z, delta_com_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today) / (3000/h_today)**3, ':', label='w = -0.8')
# ax3.plot(z, delta_com_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today) / (3000/h_today)**3, ':', label='w = -0.9')
# ax3.plot(z, delta_com_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today) / (3000/h_today)**3, ':', label='w = -1.1')
# ax3.plot(z, delta_com_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today) / (3000/h_today)**3, ':', label='w = -1.2')
# fig3.legend(loc=5)
# plt.grid()

mass_dex_array = np.linspace(7,12,200)

# -------------------Plot of schechter function as a function of mass---------------------------
# This one using the equation from A. Mortlock (2015) paper, all mass in unit of dex
# fig4, ax4 = plt.subplots()
# ax4.set_xlabel(r"$M$ $(dex)$")
# ax4.set_ylabel(r"$\phi (M)$ (${dex}^{-1} {pc}^{-3}$)")
# ax4.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47))
# ax4.set_yscale('log')
# plt.grid()

#-------------------Plot of difference in no. density of galaxies btw LCDM and a given model----
#dN/dz dOmega
# fig5, ax5 = plt.subplots()
# ax5.set_ylabel(r"$\Delta \frac{dN}{dz d \Omega}$")
# ax5.set_xlabel(r"$z$")
# ax5.set_title("min. app. magnitude = " + str(magnitude_min) + ", over all sky")
# ax5.plot(z, delta_galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today), '-', label='LCDM')
# ax5.plot(z, delta_galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 1, 0, 0, 'LCDM', h=h_today), '--', label='E-deS')
# ax5.plot(z, delta_galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today), '-', label='OCDM')
# ax5.plot(z, delta_galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), ':', label='w = -0.8')
# ax5.plot(z, delta_galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), ':', label='w = -0.9')
# ax5.plot(z, delta_galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), ':', label='w = -1.1')
# ax5.plot(z, delta_galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), ':', label='w = -1.2')

# fig5.legend()
# plt.grid()

#------------------------Plot of delta dN/dz for one square degree region on the sky--------------
# fig6, ax6 = plt.subplots()
# ax6.set_ylabel(r"$\Delta \frac{dN}{dz}$")
# ax6.set_xlabel(r"$z$")
# ax6.set_title("min. app. magnitude = " + str(magnitude_min) + ", per square degree")
# ax6.plot(z, delta_galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today) * one_sqr_degree, '-', label='LCDM')
# ax6.plot(z, delta_galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 1, 0, 0, 'LCDM', h=h_today) * one_sqr_degree, '--', label='E-deS')
# ax6.plot(z, delta_galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today) * one_sqr_degree, '-', label='OCDM')
# ax6.plot(z, delta_galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today) * one_sqr_degree, ':', label='w = -0.8')
# ax6.plot(z, delta_galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today) * one_sqr_degree, ':', label='w = -0.9')
# ax6.plot(z, delta_galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today) * one_sqr_degree, ':', label='w = -1.1')
# ax6.plot(z, delta_galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today) * one_sqr_degree, ':', label='w = -1.2')

# fig6.legend()
# plt.grid()

#---------------Plot of absolute number of galaxies per sqd---------------
# fig7, ax7 = plt.subplots()
# ax7.set_ylabel(r"$\frac{dN}{dz}$")
# ax7.set_xlabel(r"$z$")
# ax7.set_title("min. app. magnitude = " + str(magnitude_min) + ", per square degree")
# ax7.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today) * one_sqr_degree, '-', label='LCDM')
# ax7.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 1, 0, 0, 'LCDM', h=h_today) * one_sqr_degree, '--', label='E-deS')
# ax7.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today) * one_sqr_degree, '-', label='OCDM')
# ax7.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today) * one_sqr_degree, ':', label='w = -0.8')
# ax7.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today) * one_sqr_degree, ':', label='w = -0.9')
# ax7.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today) * one_sqr_degree, ':', label='w = -1.1')
# ax7.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today) * one_sqr_degree, ':', label='w = -1.2')

# fig7.legend()
# plt.grid()

#------------Plot of minimum mass of galaxies observable for a given magnitude thres.-----------
# fig8, ax8 = plt.subplots()
# ax8.set_ylabel(r"Mass (dex)")
# ax8.set_xlabel(r"$z$")
# ax8.set_title('The lightest galaxy observable with the apparent magnitude threshold of ' + str(magnitude_min))
# ax8.plot(z, mag_to_mass_vec(magnitude_min, z, 0.3, 0, 0.7, 'LCDM', h=h_today), '-', label='LCDM')
# ax8.plot(z, mag_to_mass_vec(magnitude_min, z, 1, 0, 0, 'LCDM', h=h_today), '--', label='E-deS')
# ax8.plot(z, mag_to_mass_vec(magnitude_min, z, 0.3, 0.7, 0, 'LCDM', h=h_today), '-', label='OCDM')
# ax8.plot(z, mag_to_mass_vec(magnitude_min, z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), ':', label='w = -0.8')
# ax8.plot(z, mag_to_mass_vec(magnitude_min, z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), ':', label='w = -0.9')
# ax8.plot(z, mag_to_mass_vec(magnitude_min, z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), ':', label='w = -1.1')
# ax8.plot(z, mag_to_mass_vec(magnitude_min, z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), ':', label='w = -1.2')

# fig8.legend(loc = 'center right')
# plt.grid()

#------------Plot of apparent magnitude vs absolute magnitude-----------
# fig8, ax8 = plt.subplots()
# ax8.set_ylabel(r"absolute magnitude")
# ax8.set_xlabel(r"$z$")
# ax8.set_title('Abs. mag. of an object with app. mag. ' + str(magnitude_min))
# ax8.plot(z, appmag_to_absmag_vec(magnitude_min, z, 0.3, 0, 0.7, 'LCDM', h=h_today), '-', label='LCDM')
# ax8.plot(z, appmag_to_absmag_vec(magnitude_min, z, 1, 0, 0, 'LCDM', h=h_today), '--', label='E-deS')
# ax8.plot(z, appmag_to_absmag_vec(magnitude_min, z, 0.3, 0.7, 0, 'LCDM', h=h_today), '-', label='OCDM')
# ax8.plot(z, appmag_to_absmag_vec(magnitude_min, z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), ':', label='w = -0.8')
# ax8.plot(z, appmag_to_absmag_vec(magnitude_min, z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), ':', label='w = -0.9')
# ax8.plot(z, appmag_to_absmag_vec(magnitude_min, z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), ':', label='w = -1.1')
# ax8.plot(z, appmag_to_absmag_vec(magnitude_min, z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), ':', label='w = -1.2')


# fig8.legend(loc = 'center right')
# plt.grid()


# fig9, ax9 = plt.subplots()
# ax9.set_ylabel(r"Luminosity distance (Mpc)")
# ax9.set_xlabel(r"$z$")
# ax9.plot(z, lum_d_vec(z, 0.3, 0, 0.7, 'LCDM', h=h_today), '-', label='LCDM')
# ax9.plot(z, lum_d_vec(z, 1, 0, 0, 'LCDM', h=h_today), '--', label='E-deS')
# ax9.plot(z, lum_d_vec( z, 0.3, 0.7, 0, 'LCDM', h=h_today), '-', label='OCDM')
# ax9.plot(z, lum_d_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), ':', label='w = -0.8')
# ax9.plot(z, lum_d_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), ':', label='w = -0.9')
# ax9.plot(z, lum_d_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), ':', label='w = -1.1')
# ax9.plot(z, lum_d_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), ':', label='w = -1.2')

# fig9.legend(loc = 'center right')
# plt.grid()


# fig10, ax10 = plt.subplots()
# ax10.set_ylabel(r"$\Delta$ magnitude")
# ax10.set_xlabel(r"$z$")
# ax10.plot(z, delta_mag_vec(z, 0.3, 0, 0.7, 'LCDM', h=h_today), '-', label='LCDM')
# ax10.plot(z, delta_mag_vec(z, 1, 0, 0, 'LCDM', h=h_today), '--', label='E-deS')
# ax10.plot(z, delta_mag_vec( z, 0.3, 0.7, 0, 'LCDM', h=h_today), '-', label='OCDM')
# ax10.plot(z, delta_mag_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), ':', label='w = -0.8')
# ax10.plot(z, delta_mag_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), ':', label='w = -0.9')
# ax10.plot(z, delta_mag_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), ':', label='w = -1.1')
# ax10.plot(z, delta_mag_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), ':', label='w = -1.2')

# fig10.legend(loc = 'center right')
# plt.grid()

#---------------Plot of difference in number of galaxies with respect to some z---------------
# fig11, ax11 = plt.subplots()
# ax11.set_ylabel(r"Difference in $\frac{dN}{dz}$ between $z$ and $z=$" + str(z_ref))
# ax11.set_xlabel(r"$z$")
# ax11.set_title("min. app. magnitude = " + str(magnitude_min) + ", per square degree")
# ax11.plot(z, delta_galaxy_number_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today) * one_sqr_degree, '-', label='LCDM')
# ax11.plot(z, delta_galaxy_number_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 1, 0, 0, 'LCDM', h=h_today) * one_sqr_degree, '--', label='E-deS')
# ax11.plot(z, delta_galaxy_number_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today) * one_sqr_degree, '-', label='OCDM')
# ax11.plot(z, delta_galaxy_number_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today) * one_sqr_degree, ':', label='w = -0.8')
# ax11.plot(z, delta_galaxy_number_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today) * one_sqr_degree, ':', label='w = -0.9')
# ax11.plot(z, delta_galaxy_number_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today) * one_sqr_degree, ':', label='w = -1.1')
# ax11.plot(z, delta_galaxy_number_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today) * one_sqr_degree, ':', label='w = -1.2')

# fig11.legend()
# plt.grid()

# Relative difference
# fig12, ax12 = plt.subplots()
# ax12.set_ylabel(r"Relative difference of $\frac{dN}{dz}$ between $z$ and $z=$" + str(z_ref))
# ax12.set_xlabel(r"$z$")
# ax12.set_title("min. app. magnitude = " + str(magnitude_min))
# ax12.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today), '-', label='LCDM')
# ax12.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 1, 0, 0, 'LCDM', h=h_today), '--', label='E-deS')
# ax12.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today), '-', label='OCDM')
# ax12.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), ':', label='w = -0.8')
# ax12.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), ':', label='w = -0.9')
# ax12.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), ':', label='w = -1.1')
# ax12.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), ':', label='w = -1.2')

# fig12.legend()
# plt.grid()

# -----------------------Plot of single Schechter function at different z using varying alpha--------------------
# Don't need to define separate function for single Shcechter function, because all we need to do is to set the parameter phi2 of double Schechter funtion to 0 (phi2 = 0)
# mass_dex_array = np.linspace(7,12,200)
# fig13, ax13 = plt.subplots()
# ax13.set_xlabel(r"$M$ $(dex)$")
# ax13.set_ylabel(r"$\phi (M)$ (${dex}^{-1} {Mpc}^{-3}$)")
# ax13.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 10.66, 2.88e-3, 0, alpha(0, -0.093, -1.3), 0), '-', label='z=0')
# # ax13.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 10.66, 2.88e-3, 0, alpha(0.5, -0.093, -1.3), 0), label='z=0.5')
# ax13.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 10.66, 2.88e-3, 0, alpha(1, -0.093, -1.3), 0), '--', label='z=1')
# # ax13.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 10.66, 2.88e-3, 0, alpha(1.5, -0.093, -1.3), 0), label='z=1.5')
# ax13.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 10.66, 2.88e-3, 0, alpha(2, -0.093, -1.3), 0), '-.', label='z=2')
# # ax13.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 10.66, 2.88e-3, 0, alpha(2.5, -0.093, -1.3), 0), label='z=2.5')
# ax13.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 10.66, 2.88e-3, 0, alpha(3, -0.093, -1.3), 0), ':', label='z=3')
# ax13.set_yscale('log')
# fig13.legend()
# plt.grid()

# ---------------------------------Relative difference using varyig alpha------------------------------
# rel_num_LCDM = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
# rel_num_EdS = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 1, 0, 0, 'LCDM', h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today)
# rel_num_OCDM = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0.7, 0, 'LCDM', h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today)
# rel_num_w8 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today)
# rel_num_w9 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today)
# rel_num_w11 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today)
# rel_num_w12 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today)

# # Return the redshift at which maximum separation occurs
# sep = rel_num_LCDM - rel_num_w9
# max_sep = np.amax(np.abs(sep))
# z_max_sep_index = np.where(sep == max_sep)
# print("(Varying alpha) Max difference between LCDM and w=-0.9 at z = " + str(z[z_max_sep_index])) 

# # Write data to file, 1st row: z, 2nd row: rel_num_<model>
# models = np.array([rel_num_LCDM, rel_num_EdS, rel_num_OCDM, rel_num_w8, rel_num_w9, rel_num_w11, rel_num_w12])
# i=0
# # print("Rel path is: " + os.path.join(cur_path, "RelDif//RelDif_varAlpha_zref"))
# for model in models:
# 	datafile=open(os.path.join(cur_path, "RelDif//RelDif_varAlpha_zref") + str(z_ref) + "_m" + str(magnitude_min) + "_" + modelname[i] + ".txt", 'w')
# 	data = np.stack((z, model), axis=1)
# 	np.savetxt(datafile, data)
# 	i=i+1
# 	datafile.close()


# fig14, ax14 = plt.subplots()
# ax14.set_ylabel(r"Relative difference of $\frac{dN}{dz}$ between $z$ and $z=$" + str(z_ref))
# ax14.set_xlabel(r"$z$")
# ax14.set_title("min. app. magnitude = " + str(magnitude_min))
# ax14.plot(z, rel_num_LCDM, '-', label='LCDM')
# ax14.plot(z, rel_num_EdS, '--', label='E-deS')
# ax14.plot(z, rel_num_OCDM, '-', label='OCDM')
# ax14.plot(z, rel_num_w8, ':', label='w = -0.8')
# ax14.plot(z, rel_num_w9, ':', label='w = -0.9')
# ax14.plot(z, rel_num_w11, ':', label='w = -1.1')
# ax14.plot(z, rel_num_w12, ':', label='w = -1.2')

# fig14.legend()
# plt.grid()

# Relative difference using constant alpha
# fig14_1, ax14_1 = plt.subplots()
# ax14_1.set_ylabel(r"Relative difference of $\frac{dN}{dz}$ between $z$ and $z=$" + str(z_ref))
# ax14_1.set_xlabel(r"$z$")
# ax14_1.set_title("min. app. magnitude = " + str(magnitude_min))
# ax14_1.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 2.88e-3, 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today), '-', label='LCDM')
# ax14_1.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 2.88e-3, 0, alpha(0, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today), '--', label='E-deS')
# ax14_1.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 2.88e-3, 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today), '-', label='OCDM')
# ax14_1.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 2.88e-3, 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), ':', label='w = -0.8')
# ax14_1.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 2.88e-3, 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), ':', label='w = -0.9')
# ax14_1.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 2.88e-3, 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), ':', label='w = -1.1')
# ax14_1.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 2.88e-3, 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), ':', label='w = -1.2')

# fig14_1.legend()
# plt.grid()

#---------------Plot of absolute number of galaxies per sqd with varying alpha---------------
# fig15_1, ax15_1 = plt.subplots()
# ax15_1.set_ylabel(r"$\frac{dN}{dz}$")
# ax15_1.set_xlabel(r"$z$")
# ax15_1.set_title("min. app. magnitude = " + str(magnitude_min) + ", per square degree")
# ax15_1.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 2.88e-3, 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today), '-', label='LCDM')
# ax15_1.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 2.88e-3, 0, alpha(z, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today), '--', label='E-deS')
# ax15_1.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 2.88e-3, 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today), '-', label='OCDM')
# ax15_1.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 2.88e-3, 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), ':', label='w = -0.8')
# ax15_1.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 2.88e-3, 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), ':', label='w = -0.9')
# ax15_1.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 2.88e-3, 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), ':', label='w = -1.1')
# ax15_1.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 2.88e-3, 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), ':', label='w = -1.2')

# fig15_1.legend()
# plt.grid()


#---------------Plot of absolute number of galaxies per sqd with constant alpha---------------
# fig15, ax15 = plt.subplots()
# ax15.set_ylabel(r"$\frac{dN}{dz}$")
# ax15.set_xlabel(r"$z$")
# ax15.set_title("min. app. magnitude = " + str(magnitude_min) + ", per square degree")
# ax15.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 2.88e-3, 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today), '-', label='LCDM')
# ax15.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 2.88e-3, 0, alpha(0, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today), '--', label='E-deS')
# ax15.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 2.88e-3, 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today), '-', label='OCDM')
# ax15.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 2.88e-3, 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), ':', label='w = -0.8')
# ax15.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 2.88e-3, 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), ':', label='w = -0.9')
# ax15.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 2.88e-3, 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), ':', label='w = -1.1')
# ax15.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, 2.88e-3, 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), ':', label='w = -1.2')

# fig15.legend()
# plt.grid()

#------------Plot of minimum mass of galaxies observable for a given magnitude thres.-----------
# fig16, ax16 = plt.subplots()
# ax16.set_ylabel(r"Mass (dex)")
# ax16.set_xlabel(r"$z$")
# ax16.set_title('The lightest galaxy observable in LCDM \n for different apparent magnitude thresholds between 21~35')
# ax16.plot(z, mag_to_mass_vec(21, z, 0.3, 0, 0.7, 'LCDM', h=h_today), ':', label='m=21')
# ax16.plot(z, mag_to_mass_vec(25, z, 0.3, 0, 0.7, 'LCDM', h=h_today), '--', label='m=25')
# ax16.plot(z, mag_to_mass_vec(28, z, 0.3, 0, 0.7, 'LCDM', h=h_today), '-', label='m=28')
# ax16.plot(z, mag_to_mass_vec(30, z, 0.3, 0, 0.7, 'LCDM', h=h_today), '-.', label='m=30')
# ax16.plot(z, mag_to_mass_vec(35, z, 0.3, 0, 0.7, 'LCDM', h=h_today), ':', label='m=35')

# fig16.legend(loc = 'center right')
# plt.grid()

#-----------Plot of parameter phi for single Schechter function as a function of z----------------
# fig16_1, ax16_1 = plt.subplots()
# ax16_1.plot(z, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), label = "Snyder integrated")
# plt.title("Evolution of number density phi due to mergers, for Galaxies log(M)>10.3")

# plt.yscale("Log")

# plt.ylabel("Number density $\Phi$ ($Mpc^{-1}$)")
# plt.xlabel("Redshift z")

# plt.legend()


#--------------------Investigating the behaviour of relative no. of galaxy subject to galaxy mergers only--------------------

# Relative difference using varyig phi
# rel_num_LCDM = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
# rel_num_EdS = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 1, 0, 0, 'LCDM', h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today)
# rel_num_OCDM = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0.7, 0, 'LCDM', h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today)
# rel_num_w8 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today)
# rel_num_w9 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today)
# rel_num_w11 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today)
# rel_num_w12 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today)

# # Return the redshift at which maximum separation occurs
# sep = rel_num_LCDM - rel_num_w9
# max_sep = np.amax(np.abs(sep))
# z_max_sep_index = np.where(sep == max_sep)
# print("(Varying phi) Max difference between LCDM and w=-0.9 at z = " + str(z[z_max_sep_index]))

# # Write data to file, 1st row: z, 2nd row: rel_num_<model>
# models = np.array([rel_num_LCDM, rel_num_EdS, rel_num_OCDM, rel_num_w8, rel_num_w9, rel_num_w11, rel_num_w12])
# i=0
# for model in models:
# 	datafile=open(os.path.join(cur_path, "RelDif//RelDif_varPhi_zref") + str(z_ref) + "_m" + str(magnitude_min) + "_" + modelname[i] + ".txt", 'w')
# 	data = np.stack((z, model), axis=1)
# 	np.savetxt(datafile, data)
# 	i=i+1
# 	datafile.close()

# fig17, ax17 = plt.subplots()
# ax17.set_ylabel(r"Relative difference of $\frac{dN}{dz}$ between $z$ and $z=$" + str(z_ref))
# ax17.set_xlabel(r"$z$")
# ax17.set_title("RelDif//min. app. magnitude = " + str(magnitude_min))
# ax17.plot(z, rel_num_LCDM, '-', label='LCDM')
# ax17.plot(z, rel_num_EdS, '--', label='E-deS')
# ax17.plot(z, rel_num_OCDM, '-', label='OCDM')
# ax17.plot(z, rel_num_w8, ':', label='w = -0.8')
# ax17.plot(z, rel_num_w9, ':', label='w = -0.9')
# ax17.plot(z, rel_num_w11, ':', label='w = -1.1')
# ax17.plot(z, rel_num_w12, ':', label='w = -1.2')

# fig17.legend()
# plt.grid()

# Relative difference using constant phi (actually same as fig14_1, where the alpha is set constant, so in a way plotting this is unnecessary)
# rel_num_LCDM = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
# rel_num_EdS = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 1, 0, 0, 'LCDM', h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today)
# rel_num_OCDM = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0.7, 0, 'LCDM', h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today)
# rel_num_w8 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today)
# rel_num_w9 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today)
# rel_num_w11 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today)
# rel_num_w12 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today)

# # Return the redshift at which maximum separation occurs
# sep = rel_num_LCDM - rel_num_w9
# max_sep = np.amax(np.abs(sep))
# z_max_sep_index = np.where(sep == max_sep)
# print("(Const alpha & phi) Max difference between LCDM and w=-0.9 at z = " + str(z[z_max_sep_index]))

# # Write data to file, 1st row: z, 2nd row: rel_num_<model>
# models = np.array([rel_num_LCDM, rel_num_EdS, rel_num_OCDM, rel_num_w8, rel_num_w9, rel_num_w11, rel_num_w12])
# i=0
# for model in models:
# 	datafile=open(os.path.join(cur_path, "RelDif//RelDif_constPhiAlpha_zref") + str(z_ref) + "_m" + str(magnitude_min) + "_" + modelname[i] + ".txt", 'w')
# 	data = np.stack((z, model), axis=1)
# 	np.savetxt(datafile, data)
# 	i=i+1
# 	datafile.close()

# fig18, ax18 = plt.subplots()
# ax18.set_ylabel(r"Relative difference of $\frac{dN}{dz}$ between $z$ and $z=$" + str(z_ref))
# ax18.set_xlabel(r"$z$")
# ax18.set_title("min. app. magnitude = " + str(magnitude_min))
# ax18.plot(z, rel_num_LCDM, '-', label='LCDM')
# ax18.plot(z, rel_num_EdS, '--', label='E-deS')
# ax18.plot(z, rel_num_OCDM, '-', label='OCDM')
# ax18.plot(z, rel_num_w8, ':', label='w = -0.8')
# ax18.plot(z, rel_num_w9, ':', label='w = -0.9')
# ax18.plot(z, rel_num_w11, ':', label='w = -1.1')
# ax18.plot(z, rel_num_w12, ':', label='w = -1.2')

# fig18.legend()
# plt.grid()

# ----------------------Create plot of maximum separation as a function of reference redshift---------------------------------------------------
# max_sepArray = np.array([])
# z_max_sepArray = np.array([])
# zrefArray = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6])

# for zref in zrefArray:
# 	rel_num_LCDM = delta_galaxy_number_rel_z_vec(z, zref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
# 	rel_num_w9 = delta_galaxy_number_rel_z_vec(z, zref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today)
# 	sep = rel_num_LCDM - rel_num_w9
# 	max_sep = np.amax(np.abs(sep))
# 	z_max_sep_index = np.where(sep == max_sep)
# 	z_max_sep = z[z_max_sep_index]
# 	max_sepArray = np.append(max_sepArray, max_sep)
# 	z_max_sepArray = np.append(z_max_sepArray, z_max_sep)

# fig17_1, ax17_1 = plt.subplots()
# ax17_1.plot(zrefArray, max_sepArray, 'o-')
# ax17_1.set_xlabel(r"reference redshift $z_{ref}$")
# ax17_1.set_ylabel(r"maximum relative number difference")
# ax17_1.set_title("min. app. magnitude = " + str(magnitude_min))
# # ax17_1.set_yscale('log')

# fig17_2, ax17_2 = plt.subplots()
# ax17_2.plot(zrefArray, z_max_sepArray, 'o-')
# ax17_2.set_xlabel(r"reference redshift $z_{ref}$")
# ax17_2.set_ylabel(r"redshift $z$ with maximum difference")
# ax17_2.set_title("min. app. magnitude = " + str(magnitude_min))

# # -----------------Create plot of maximum separation as a function of magnitude-------------------------------------
# max_sepArray = np.array([])
# z_max_sepArray = np.array([])
# magminArray = np.array([25, 27, 28, 30, 32])

# for magmin in magminArray:
# 	rel_num_LCDM = delta_galaxy_number_rel_z_vec(z, z_ref, magmin, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
# 	rel_num_w9 = delta_galaxy_number_rel_z_vec(z, z_ref, magmin, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today)
# 	sep = rel_num_LCDM - rel_num_w9
# 	max_sep = np.amax(np.abs(sep))
# 	z_max_sep_index = np.where(sep == max_sep)
# 	z_max_sep = z[z_max_sep_index]
# 	max_sepArray = np.append(max_sepArray, max_sep)
# 	z_max_sepArray = np.append(z_max_sepArray, z_max_sep)

# fig13_1, ax13_1 = plt.subplots()
# ax13_1.plot(magminArray, max_sepArray, 'o-')
# ax13_1.set_xlabel(r"apparent magnitude threshold")
# ax13_1.set_ylabel(r"maximum relative number difference")
# ax13_1.set_title(r"$z_{ref} = $" + str(z_ref))
# ax13_1.set_yscale('log')

# fig13_2, ax13_2 = plt.subplots()
# ax13_2.plot(magminArray, z_max_sepArray, 'o-')
# ax13_2.set_xlabel(r"apparent magnitude threshold")
# ax13_2.set_ylabel(r"redshift $z$ with maximum difference")
# ax13_2.set_title(r"$z_{ref} = $" + str(z_ref))
#---------------------------------------------------------------------------------------------------------------------



#---------------Plot of absolute number of galaxies per sqd with varying phi---------------
# fig19, ax19 = plt.subplots()
# ax19.set_ylabel(r"$\frac{dN}{dz}$")
# ax19.set_xlabel(r"$z$")
# ax19.set_title("min. app. magnitude = " + str(magnitude_min) + ", per square degree")
# ax19.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today), '-', label='LCDM')
# ax19.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 1, 0, 0, 'LCDM', h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today), '--', label='E-deS')
# ax19.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0.7, 0, 'LCDM', h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today), '-', label='OCDM')
# ax19.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), ':', label='w = -0.8')
# ax19.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), ':', label='w = -0.9')
# ax19.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), ':', label='w = -1.1')
# ax19.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), ':', label='w = -1.2')

# fig19.legend()
# plt.grid()


#---------------Plot of absolute number of galaxies per sqd with constant phi---------------
# fig20, ax20 = plt.subplots()
# ax20.set_ylabel(r"$\frac{dN}{dz}$")
# ax20.set_xlabel(r"$z$")
# ax20.set_title("min. app. magnitude = " + str(magnitude_min) + ", per square degree")
# ax20.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today), '-', label='LCDM')
# ax20.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 1, 0, 0, 'LCDM', h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today), '--', label='E-deS')
# ax20.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0.7, 0, 'LCDM', h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today), '-', label='OCDM')
# ax20.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), ':', label='w = -0.8')
# ax20.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), ':', label='w = -0.9')
# ax20.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), ':', label='w = -1.1')
# ax20.plot(z, galaxy_number_vec(z, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), ':', label='w = -1.2')

# fig20.legend()
# plt.grid()

# ---------------Plots of Schechter function at different redshift for varying phi
# fig21, ax21 = plt.subplots()
# ax21.set_xlabel(r"$M$ $(dex)$")
# ax21.set_ylabel(r"$\phi (M)$ (${dex}^{-1} {Mpc}^{-3}$)")
# ax21.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), 0), '-', label='z=0')
# # ax13.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 10.66, 2.88e-3, 0, alpha(0.5, -0.093, -1.3), 0), label='z=0.5')
# ax21.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 10.66, Phi_directVec(1, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), 0), '--', label='z=1')
# # ax13.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 10.66, 2.88e-3, 0, alpha(1.5, -0.093, -1.3), 0), label='z=1.5')
# ax21.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 10.66, Phi_directVec(2, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), 0), '-.', label='z=2')
# # ax13.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 10.66, 2.88e-3, 0, alpha(2.5, -0.093, -1.3), 0), label='z=2.5')
# ax21.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 10.66, Phi_directVec(3, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), 0), ':', label='z=3')
# ax21.set_yscale('log')
# fig21.legend()
# plt.grid()

plt.show()



def experimental_delta_galaxy_number_rel_z(z, numberofdegrees, z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
	# Returns the difference in number density of galaxies between an arbitrary z and a set z
	number_z_ref = numberofdegrees * 3.0462*10** -4 * galaxy_number(z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h)
	number = numberofdegrees * 3.0462*10** -4 * galaxy_number(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h)
	# print(model)
	# print("number: " + str(number))
	# print("number_z: " + str(number_z_ref))
	# Error analysis
	d_number = np.sqrt(number)
	d_number_z_ref = np.sqrt(number_z_ref)
	rel = (number - number_z_ref) / number_z_ref
	if(number!=number_z_ref):
		d_rel = np.sqrt((d_number**2+d_number_z_ref**2)/(number-number_z_ref)**2 + (d_number_z_ref/number_z_ref)**2)
	else:
		d_rel=0
	# print(np.array([rel, d_rel]))
	return rel, d_rel

experimental_delta_galaxy_number_rel_zVec = np.vectorize(experimental_delta_galaxy_number_rel_z)

def TestError(numberofdegrees, z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
    zArray = np.linspace(0.01, z_ref, num=50)
    
    LCDM = np.array(experimental_delta_galaxy_number_rel_zVec(zArray, numberofdegrees, z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, 0.3, 0, 0.7, "LCDM", w_0=w_0, w_1=w_1, h=h))
    test = np.array(experimental_delta_galaxy_number_rel_zVec(zArray, numberofdegrees, z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0=w_0, w_1=w_1, h=h))
    
    #print(test[1])
    
    diff = np.abs(test[0] - LCDM[0])
    
    index = np.argmax(diff)
    
    return diff[index], test[1,index]
    
# =============================================================================
#     number_z_ref = numberofdegrees * 3.0462*10** -4 * galaxy_number(z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h)
#     number = numberofdegrees * 3.0462*10** -4 * galaxy_number(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h)
# 	
#     rel = (number - number_z_ref) / number_z_ref
# =============================================================================




TestErrorVec = np.vectorize(TestError)

Degrees = np.linspace(0.001, 3, num=50)

OCDM = TestErrorVec(Degrees, 3, 25, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0.7, 0, "LCDM")
EdS = TestErrorVec(Degrees, 3, 25, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 1, 0, 0, "LCDM")


plt.plot(Degrees, OCDM[0])
plt.plot(Degrees, OCDM[1])
plt.yscale("log")

plt.ylabel("Difference in rel. galaxy number, and error")#, fontsize = 20)
plt.xlabel("Number of degrees")#, fontsize = 20)
plt.title("OCDM: differece in relative number of galaxies relative to LCDM, \n and error on relative number of galaxies, at mag25")
plt.legend()

plt.show()

plt.plot(Degrees, EdS[0])
plt.plot(Degrees, EdS[1])
plt.yscale("log")

plt.ylabel("Difference in rel. galaxy number, and error")#, fontsize = 20)
plt.xlabel("Number of degrees")#, fontsize = 20)
plt.title("Einstein de Sitter: differece in relative number of galaxies relative to LCDM, \n and error on relative number of galaxies, at mag25")
plt.legend()

plt.show()



 
W09 = TestErrorVec(Degrees, 3, 25, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, "constant_w", w_0 = 0.9)
W11 = TestErrorVec(Degrees, 3, 25, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, "constant_w", w_0 = 1.1) 

plt.plot(Degrees, W09[0])
plt.plot(Degrees, W09[1])
plt.yscale("log")

plt.ylabel("Difference in rel. galaxy number, and error")#, fontsize = 20)
plt.xlabel("Number of degrees")#, fontsize = 20)
plt.title("w = -0.9: differece in relative number of galaxies relative to LCDM, \n and error on relative number of galaxies, at mag25")
plt.legend()

plt.show()

plt.plot(Degrees, W11[0])
plt.plot(Degrees, W11[1])
plt.yscale("log")

plt.ylabel("Difference in rel. galaxy number, and error")#, fontsize = 20)
plt.xlabel("Number of degrees")#, fontsize = 20)
plt.title("w = -1.1: differece in relative number of galaxies relative to LCDM, \n and error on relative number of galaxies, at mag25")
plt.legend()

plt.show()




TestErrorVec = np.vectorize(TestError)

Degrees = np.linspace(0.25, 20, num=50)

OCDM = TestErrorVec(Degrees, 3, 32, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0.7, 0, "LCDM")
EdS = TestErrorVec(Degrees, 3, 32, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 1, 0, 0, "LCDM")


plt.plot(Degrees, OCDM[0])
plt.plot(Degrees, OCDM[1])
plt.yscale("log")

plt.ylabel("Difference in rel. galaxy number, and error")#, fontsize = 20)
plt.xlabel("Number of degrees")#, fontsize = 20)
plt.title("OCDM: differece in relative number of galaxies relative to LCDM, \n and error on relative number of galaxies, at mag32")
plt.legend()

plt.show()

plt.plot(Degrees, EdS[0])
plt.plot(Degrees, EdS[1])
plt.yscale("log")

plt.ylabel("Difference in rel. galaxy number, and error")#, fontsize = 20)
plt.xlabel("Number of degrees")#, fontsize = 20)
plt.title("Einstein de Sitter: differece in relative number of galaxies relative to LCDM, \n and error on relative number of galaxies, at mag32")
plt.legend()

plt.show()



 
W09 = TestErrorVec(Degrees, 3, 32, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, "constant_w", w_0 = 0.9)
W11 = TestErrorVec(Degrees, 3, 32, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, "constant_w", w_0 = 1.1) 

plt.plot(Degrees, W09[0])
plt.plot(Degrees, W09[1])
plt.yscale("log")

plt.ylabel("Difference in rel. galaxy number, and error")#, fontsize = 20)
plt.xlabel("Number of degrees")#, fontsize = 20)
plt.title("w = -0.9: differece in relative number of galaxies relative to LCDM, \n and error on relative number of galaxies, at mag32")
plt.legend()

plt.show()

plt.plot(Degrees, W11[0])
plt.plot(Degrees, W11[1])
plt.yscale("log")

plt.ylabel("Difference in rel. galaxy number, and error")#, fontsize = 20)
plt.xlabel("Number of degrees")#, fontsize = 20)
plt.title("w = -1.1: differece in relative number of galaxies relative to LCDM, \n and error on relative number of galaxies, at mag32")
plt.legend()

plt.show()
















