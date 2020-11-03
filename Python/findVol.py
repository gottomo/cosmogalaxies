import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate # this module does the integration

# This code finds the comoving volume element per solid angle per redshift for different cosmological models
# Hubble constant is h = 0.7 by default
# By default the functions are written so that it returns cosmological constant (LCDM) dark energy model

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
	
def comoving_d_los(z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the line-of-sight comoving distance in Mpc
	# Also works in the case if z is array
	# ToDo: also include error of integration to the results?
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h')
	com_d_los = 3000 / h * integrate.quad(lambda Z: 1/e(Z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1), 0, z)[0]
	return com_d_los
	
def comoving_d(z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the transverse comoving distance in Mpc
	# Works for all geometry of space.
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h')
	com_d_los = comoving_d_los(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h) # reassigning the function to a variable so that the code is shorter
	if omega_k == 0:
		return com_d_los
	elif omega_k < 0:
		com_d = 3000 / (h * np.sqrt(omega_k)) * np.sin(np.sqrt(abs(omega_k)) * com_d_los * h / 3000)
		return com_d
	elif omega_k > 0:
		com_d = 3000 / (h * np.sqrt(omega_k)) * np.sinh(np.sqrt(abs(omega_k)) * com_d_los * h / 3000)
		return com_d

def ang_diameter_d(z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns angular diameter distance
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h')
	ang_di_d = comoving_d(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h) / (1 + z)
	return ang_di_d
	
def lum_d(z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the luminosity distance
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h')
	lum_d = comoving_d(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h) * (1 + z)
	return lum_d

def comoving_vol_elm(z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns comoving volume element per unit solid angle per unit redshift
	# is in unit of Mpc^3
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h', 0.7)
	com_vol_elm = 3000 / h * (1 + z)**2 * ang_diameter_d(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h)**2 / e(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1)
	return com_vol_elm
	
def delta_com_vol_elm(z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the difference between comoving volume element of LCDM and a chosen model
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h', 0.7)
	return comoving_vol_elm(z, 0.3, 0, 0.7, 'LCDM') - comoving_vol_elm(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1)
	
def magnitude_bol(L, z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the magnitude of an object with luminosity L at redshift z for different cosmology
	# Since in this magnitude expression, the distance must be in the unit of 10 pc, the lum_d (in Mpc) is multiplied by 10^-5
	# const. wouldn't matter if we only use magnitude difference, so maybe could write different function in that case?
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h', 0.7)
	const = 4.74 + np.log10(3.84e26 / (4*np.pi*10**2))
	return -2.5 * np.log10(L / (4*np.pi*lum_d(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h)**2 * 1e-5**2 ) )
	
def delta_mag(z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the difference between magnitudes of LCDM and a given cosmology model for a given redshift
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h', 0.7)

	return 5 * np.log10(lum_d(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h) / lum_d(z, 0.3, 0, 0.7, 'LCDM', h = h))

def schechter_mass(mass, break_mass, phi1, phi2, alpha1, alpha2):
	# Returns Schechter function as a function of mass
	# mass shall be given in the unit of dex (i.e. mass = log10(mass_in_solar_mass) )
	# Will be in unit of dex^-1 Mpc^-3
	mass_diff = mass - break_mass
	return np.log(10) * np.exp(- np.power(10, mass_diff)) * (phi1 * np.power(10, alpha1 * mass_diff) + phi2 * np.power(10, alpha2 * mass_diff)) * np.power(10, mass_diff)
	
def appmag_to_absmag(mag, z, omega_m, omega_k, omega_0, model, **kwargs):
	# Converts apparent magnitude to absolute magnitude
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h')
	mag_abs = mag - 5 * np.log10(lum_d(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h) *1e6 / 10)
	return mag_abs

def mag_to_mass(mag, z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the mass of a galaxy given its apparent magnitude and the redshift
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h')
	# First calculate the luminosity from the apparent magnitude and redshift
	dm = 5 * np.log10(lum_d(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h)*1e6/10)
	lum = np.power(10, (4.74 + dm - mag) / 2.5)  # note that lum_d is in unit of Mpc
	# Then calculate and return mass by multiplying luminosity with mass-to-light ratio to get mass (note that mass is actually in log scale, so have to modify the equation as below
	#mass = np.log10(lum) + np.log10(5)
	mass = np.log10(lum) + np.log10(5)
	return mass

def mag_to_mass_oldold(mag, z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the mass of a galaxy given its apparent magnitude and the redshift
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h')
	# First calculate the luminosity from the apparent magnitude and redshift
	mag_abs = mag - 5 * np.log10(lum_d(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h) *1e6 / 10)
	lum = np.power(10, (4.74 - mag_abs) / 2.5) * 3.84e26 # note that lum_d is in unit of Mpc
	# Then calculate and return mass by multiplying luminosity with mass-to-light ratio to get mass (note that mass is actually in log scale, so have to modify the equation as below
	#mass = np.log10(lum) + np.log10(5)
	mass = 5 * lum / 1.989e30
	return np.log10(mass)
	
def mag_to_mass_old(mag, z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the mass of a galaxy given its apparent magnitude and the redshift
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h')
	# First calculate the luminosity from the apparent magnitude and redshift
	# k is a const. for setting a reference point for magnitude, which will return the luminosity in the unit of solar luminosity
	k = -0.26
	lum = np.power(10, (k - mag) / 2.5) * lum_d(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h)**2 * 1e12# note that lum_d is in unit of Mpc
	# Then calculate and return mass by multiplying luminosity with mass-to-light ratio to get mass (note that mass is actually in log scale, so have to modify the equation as below
	mass = np.log10(lum) + np.log10(5)
	return mass

def galaxy_no_density(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the number density of galaxies as a function of redshift for a given magnitude limit with given cosmology model
	# In the unit of Mpc^-3
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h')
	
	mass_min = mag_to_mass(mag, z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h)
	
	return integrate.quad(lambda mass: schechter_mass(mass, break_mass, phi1, phi2, alpha1, alpha2), mass_min, 15)[0]
	
def galaxy_number(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, **kwargs):
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h')
	
	number = galaxy_no_density(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h) * comoving_vol_elm(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h)
	return number


	
def delta_galaxy_number(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the difference in number density of galaxies between LCDM and a given cosmology as a function of redshift, with given mass limit
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h')
	number_lambda = galaxy_no_density(z, mag, break_mass, phi1, phi2, alpha1, alpha2, 0.3, 0, 0.7, 'LCDM', h = h) * comoving_vol_elm(z, 0.3, 0, 0.7, 'LCDM', h = h)
	number = galaxy_no_density(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h) * comoving_vol_elm(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h)
	return number - number_lambda
	
def delta_galaxy_number_z(z, z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the difference in number density of galaxies between an arbitrary z and a set z
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h')
	number_z_ref = galaxy_no_density(z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h) * comoving_vol_elm(z_ref, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h)
	number = galaxy_no_density(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h) * comoving_vol_elm(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h)
	return number - number_z_ref

def delta_galaxy_number_rel_z(z, z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the difference in number density of galaxies between an arbitrary z and a set z
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h')
	number_z_ref = galaxy_no_density(z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h) * comoving_vol_elm(z_ref, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h)
	number = galaxy_no_density(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h) * comoving_vol_elm(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h)
	return (number - number_z_ref) / number_z_ref

	
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

# Set the Hubbles constant
h_today = 0.7

print("Constant magnitude=21, varying redshift")
print(np.array([0.1, 1, 2, 3]))
print(mag_to_mass_vec(21, np.array([0.1, 1, 2, 3]), 0.3, 0, 0.7, 'LCDM', h = h_today))

# print("Constant z=0.1, varying magnitude")
# print(np.array([15, 20, 25, 30]))
# print(mag_to_mass_vec(np.array([15, 20, 25, 30]), 0.1, 0.3, 0, 0.7, 'LCDM'))

# -------------------------Plotting stuff------------------------------------
z = np.linspace(0.01,5,50)

# ---------------------------Comoving volume element------------------------------
# Plots of comoving volume element per uint solid angle per unit redshift, normalized by 1/(D_H)^3
# fig1, ax1 = plt.subplots()
# ax1.set_xlabel("Redshift z")
# ax1.set_ylabel(r"Dimensionless comoving volume element $\frac{dV_C}{d\Omega dz}  \frac{1}{H_0^3}$")
# ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, 'LCDM', h=h_today) / (3000/h_today)**3, '-', label='LCDM') # LCDM
# ax1.plot(z, comoving_vol_elm_vec(z, 1, 0, 0, 'LCDM', h=h_today) / (3000/h_today)**3, '--', label='E-deS') # E-deS
# ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0.7, 0, 'LCDM', h=h_today) / (3000/h_today)**3, '-.', label='OCDM') # OCDM
# ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today) / (3000/h_today)**3, ':', label='w = -0.8')
# ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today) / (3000/h_today)**3, ':', label='w = -0.9')
# ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today) / (3000/h_today)**3, ':', label='w = -1.1')
# ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today) / (3000/h_today)**3, ':', label='w = -1.2')
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
magnitude_min = 30
one_sqr_degree = (np.pi/180)**2 
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

#------------------------Plot of dN/dz for one square degree region on the sky--------------
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

#---------------Plot of absolute number of galaxies---------------
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
z_ref = 3
fig11, ax11 = plt.subplots()
ax11.set_ylabel(r"$\frac{dN}{dz}$")
ax11.set_xlabel(r"$z$")
ax11.set_title("min. app. magnitude = " + str(magnitude_min) + ", per square degree")
ax11.plot(z, delta_galaxy_number_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today) * one_sqr_degree, '-', label='LCDM')
ax11.plot(z, delta_galaxy_number_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 1, 0, 0, 'LCDM', h=h_today) * one_sqr_degree, '--', label='E-deS')
ax11.plot(z, delta_galaxy_number_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today) * one_sqr_degree, '-', label='OCDM')
ax11.plot(z, delta_galaxy_number_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today) * one_sqr_degree, ':', label='w = -0.8')
ax11.plot(z, delta_galaxy_number_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today) * one_sqr_degree, ':', label='w = -0.9')
ax11.plot(z, delta_galaxy_number_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today) * one_sqr_degree, ':', label='w = -1.1')
ax11.plot(z, delta_galaxy_number_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today) * one_sqr_degree, ':', label='w = -1.2')

fig11.legend()
plt.grid()

# Relative difference
fig12, ax12 = plt.subplots()
ax12.set_ylabel(r"$\frac{dN}{dz}$")
ax12.set_xlabel(r"$z$")
ax12.set_title("min. app. magnitude = " + str(magnitude_min) + ", per square degree")
ax12.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today) * one_sqr_degree, '-', label='LCDM')
ax12.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 1, 0, 0, 'LCDM', h=h_today) * one_sqr_degree, '--', label='E-deS')
ax12.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today) * one_sqr_degree, '-', label='OCDM')
ax12.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today) * one_sqr_degree, ':', label='w = -0.8')
ax12.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today) * one_sqr_degree, ':', label='w = -0.9')
ax12.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today) * one_sqr_degree, ':', label='w = -1.1')
ax12.plot(z, delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today) * one_sqr_degree, ':', label='w = -1.2')

fig12.legend()
plt.grid()

plt.show()
