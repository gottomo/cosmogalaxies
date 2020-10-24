import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad # this module does the integration

# This code finds the comoving volume element per solid angle per redshift for different cosmological models
# Hubble constant is h = 0.7 by default
# By default the functions are written so that it returns cosmological constant (lambda) dark energy model

# Functinos
# def w(z, model = 'lambda', **kwargs):
	# # Returns the redshift (z) dependent dark energy equation of state parameters w(z) = pressure(z)/density(z)
	# # Maybe other models should also be considered where not necessarily pressure = w * density
	# # Not used anymore, because it has to be incorporated into f_integral for the scipy's quad() function to work
	# w_0 = kwargs.get('w_0', -1)
	# w_1 = kwargs.get('w_1', 0)
	# if model == 'lambda':
		# return -1
	# elif model == 'constant_w':
		# return w_0
	# elif model == 'linear':
		# return w_0 - w_1 * z
	
# def f_integral(z, model = 'lambda', **kwargs):
	# # An intermediate function.  The function 1+w/(1+z), which appears inside the integral on the exponent for the function f(z) as in omega_de(z) = omega_de_0 * f(z)
	# # There are analytical solutions to f(z) for many parametrizatinos, so this function may be unnecessary
	# w_0 = kwargs.get('w_0', -1)
	# w_1 = kwargs.get('w_1', 0)
	# #return 1 + w(z, model, w_0 = w_0, w_1 = w_1) / (1 + z)
	# if model == 'lambda':
		# return 0
	# elif model == 'constant_w':
		# return 1 + w_0 / (1 + z)
	# elif model == 'linear':
		# return 1 + w_0 - w_1 * z / (1 + z)

# def integrand_linear(z, w_0, w_1):
	# return 1 + w_0 - w_1 * z / (1 + z)

def omega_de(z, omega_0, model, **kwargs ):
	# Returns the density parameter of dark energy for different models
	# If using constant_w model, you need to also input the value of w to test models other than w=-1
	w_0 = kwargs.get('w_0') # by default gives lambda model
	w_1 = kwargs.get('w_1')
	if model == 'lambda':
		# This branch not really necessary, can be replaces with model = 'linear'
		return omega_0
	elif model == 'constant_w':
		# This branch not really necessary, could be replaces with model = 'linear'
		return omega_0 * (1 + z)**(3*(1+w_0))
	elif model == 'linear':
		return omega_0 * (1 + z)**(3*(1+w_0+w_1)) * np.exp(-3*w_1*z)
		# if np.isscalar(z):
			# return omega_0 * np.exp(3 * quad(integrand_linear, 0, z, args = (w_0, w_1) )[0] )
		# else:
			# return omega_0 * np.exp(3 * quad(integrand_linear, 0, val, args = (w_0, w_1) )[0] )

def e(z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the parameter E(z) required for other calculations
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	return np.sqrt(omega_m * (1 + z) ** 3 + omega_k * (1 + z) **2 + omega_de(z, omega_0, model, w_0 = w_0, w_1 = w_1))
	
# def inv_e(z, omega_m, omega_k, omega_0, model = 'lambda', **kwargs):
	# # Inverse of e, 1/e to be used in the integral for calculating line-of-sight comoving distance
	# # There may be a better way to do, but scipy's quad function only seems to accept python functions, not equations by itself
	# w_0 = kwargs.get('w_0', -1)
	# w_1 = kwargs.get('w_1', 0)
	# return 1 / np.sqrt(omega_m * (1 + z) ** 3 + omega_k * (1 + z) **2 + omega_de(z, omega_0, model, w_0 = w_0, w_1 = w_1))

def comoving_d_los(z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the line-of-sight comoving distance in Mpc
	# Also works in the case if z is array
	# ToDo: also include error of integration to the results?
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h')
	return 3000 / h * quad(lambda Z: 1/e(Z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1), 0, z)[0]
	
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
		return 3000 / (h * np.sqrt(omega_k)) * np.sin(np.sqrt(abs(omega_k)) * com_d_los * h / 3000)
	elif omega_k > 0:
		return 3000 / (h * np.sqrt(omega_k)) * np.sinh(np.sqrt(abs(omega_k)) * com_d_los * h / 3000)

def ang_diameter_d(z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns angular diameter distance
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h')
	return comoving_d(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h) / (1 + z)
	
def lum_d(z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the luminosity distance
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h')
	return comoving_d(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h) * (1 + z)

def comoving_vol_elm(z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns comoving volume element per unit solid angle per unit redshift
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h', 0.7)
	return 3000 / h * (1 + z)**2 * ang_diameter_d(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h)**2 / e(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1)
	
def delta_com_vol_elm(z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the difference between comoving volume element of LCDM and a chosen model
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h', 0.7)
	return comoving_vol_elm(z, 0.3, 0, 0.7, 'lambda') - comoving_vol_elm(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1)
	
def magnitude_bol(L, z, omega_m, omega_k, omega_0, model, **kwargs):
	# Returns the magnitude of an object with luminosity L at redshift z for different cosmology
	# Since in this magnitude expression, the distance must be in the unit of 10 pc, the lum_d (in Mpc) is multiplied by 10^-5
	# const. wouldn't matter if we only use magnitude difference, so maybe could write different function in that case?
	w_0 = kwargs.get('w_0')
	w_1 = kwargs.get('w_1')
	h = kwargs.get('h', 0.7)
	const = 4.74 + np.log10(3.84e26 / (4*np.pi*10**2))
	return -2.5 * np.log10(L / (4*np.pi*lum_d(z, omega_m, omega_k, omega_0, model, w_0 = w_0, w_1 = w_1, h = h)**2 * 1e-5**2 ) )

# Vectorize the functions you want to plot with matplotlib, so that function can accept np arrays
comoving_vol_elm_vec = np.vectorize(comoving_vol_elm)
delta_com_vol_elm_vec = np.vectorize(delta_com_vol_elm)
magnitude_bol_vec = np.vectorize(magnitude_bol)

# Set the Hubbles constant
h_today = 0.7

# Plotting stuff
z = np.linspace(0.01,3,100)

fig1, ax1 = plt.subplots()
# Plots of comoving volume element per uint solid angle per unit redshift, normalized by 1/(D_H)^3
ax1.set_xlabel("Redshift z")
ax1.set_ylabel(r"Dimensionless comoving volume element $\frac{dV_C}{d\Omega dz}  \frac{1}{H_0^3}$")
ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, 'lambda', h=h_today) / (3000/h_today)**3, '-', label='LCDM') # LCDM
ax1.plot(z, comoving_vol_elm_vec(z, 1, 0, 0, 'lambda', h=h_today) / (3000/h_today)**3, '--', label='E-deS') # E-deS
ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0.7, 0, 'lambda', h=h_today) / (3000/h_today)**3, '-.', label='OCDM') # OCDM
ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today) / (3000/h_today)**3, ':', label='w = -0.8')
ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today) / (3000/h_today)**3, ':', label='w = -0.9')
ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today) / (3000/h_today)**3, ':', label='w = -1.1')
ax1.plot(z, comoving_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today) / (3000/h_today)**3, ':', label='w = -1.2')
# ax1.plot(z, comoving_vol_elm(z, 0.3, 0, 0.7, model = 'linear', w_0 = -0.9 , h=h_today) / (3000/h_today)**3, label='linear')
# plt.plot(z, comoving_vol_elm(z, 0.3, 0, 0.7, h=h_today), '-', label='LCDM') # LCDM
# plt.plot(z, comoving_vol_elm(z, 1, 0, 0, h=h_today), '--', label='E-deS') # E-deS
# plt.plot(z, comoving_vol_elm(z, 0.3, 0.7, 0, h=h_today), '-.', label='OCDM') # OCDM
plt.grid()
fig1.legend()

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


# Plot of delta_com_vol_elm
fig3, ax3 = plt.subplots()
ax3.set_title(r"Difference in comoving volume elements between LCDM and arbitrary model")
ax3.set_xlabel("Redshift z")
ax3.set_ylabel(r"$\Delta\frac{dV_C}{d\Omega dz}  \frac{1}{H_0^3}$")
ax3.plot(z, delta_com_vol_elm_vec(z, 0.3, 0, 0.7, 'lambda', h=h_today) / (3000/h_today)**3, '-', label='LCDM') # LCDM
ax3.plot(z, delta_com_vol_elm_vec(z, 1, 0, 0, 'lambda', h=h_today) / (3000/h_today)**3, '--', label='E-deS') # E-deS
ax3.plot(z, delta_com_vol_elm_vec(z, 0.3, 0.7, 0, 'lambda', h=h_today) / (3000/h_today)**3, '-.', label='OCDM') # OCDM
ax3.plot(z, delta_com_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today) / (3000/h_today)**3, ':', label='w = -0.8')
ax3.plot(z, delta_com_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today) / (3000/h_today)**3, ':', label='w = -0.9')
ax3.plot(z, delta_com_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today) / (3000/h_today)**3, ':', label='w = -1.1')
ax3.plot(z, delta_com_vol_elm_vec(z, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today) / (3000/h_today)**3, ':', label='w = -1.2')
fig3.legend(loc=5)
plt.grid()


plt.show()
