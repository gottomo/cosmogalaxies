import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate # this module does the integration
import os
import sys

# This enables the findVol.py to be imported from the directory one above of this file
cur_path = os.path.dirname(__file__) # get the directory where the script is placed
module_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
sys.path.insert(1,module_path)
import findVol

# Vectorize the functions
comoving_vol_elm_vec = np.vectorize(findVol.comoving_vol_elm)
delta_com_vol_elm_vec = np.vectorize(findVol.delta_com_vol_elm)
magnitude_bol_vec = np.vectorize(findVol.magnitude_bol)
schechter_mass_vec = np.vectorize(findVol.schechter_mass)
mag_to_mass_vec = np.vectorize(findVol.mag_to_mass)
galaxy_number_vec = np.vectorize(findVol.galaxy_number)
delta_galaxy_number_vec = np.vectorize(findVol.delta_galaxy_number)
lum_d_vec = np.vectorize(findVol.lum_d)
delta_mag_vec = np.vectorize(findVol.delta_mag)
delta_galaxy_number_z_vec = np.vectorize(findVol.delta_galaxy_number_z)
delta_galaxy_number_rel_z_vec = np.vectorize(findVol.delta_galaxy_number_rel_z)
d_delta_galaxy_number_rel_z_vec = np.vectorize(findVol.d_delta_galaxy_number_rel_z)
appmag_to_absmag_vec = np.vectorize(findVol.appmag_to_absmag)
alpha_vec = np.vectorize(findVol.alpha)
LBTimeVec = np.vectorize(findVol.LBTime)
Phi_directVec = np.vectorize(findVol.Phi_direct)
alpha = findVol.alpha

# Constants and Settings
h_today = 0.67
magnitude_min = 26
one_sqr_degree = (np.pi/180)**2 
z_ref = 3
z = np.linspace(0.01,z_ref,50*z_ref/3)
modelname = np.array(['LCDM', 'EdS', 'OCDM', 'w8', 'w9', 'w11', 'w12'])

# Set the subfloder where the figure outputs will be placed
save_path = cur_path + '/figures/wErr/reldif'


# ---------------------------------Relative difference using varyig alpha------------------------------
rel_num_LCDM = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
rel_num_EdS = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 1, 0, 0, 'LCDM', h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today)
rel_num_OCDM = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0.7, 0, 'LCDM', h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today)
rel_num_w8 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today)
rel_num_w9 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today)
rel_num_w11 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today)
rel_num_w12 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today)

d_rel_num_LCDM = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
d_rel_num_EdS = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 1, 0, 0, 'LCDM', h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today)
d_rel_num_OCDM = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0.7, 0, 'LCDM', h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today)
d_rel_num_w8 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today)
d_rel_num_w9 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today)
d_rel_num_w11 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today)
d_rel_num_w12 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today)

fig1, ax1 = plt.subplots()
ax1.set_ylabel(r"Relative difference of $\frac{dN}{dz}$ between $z$ and $z=$" + str(z_ref))
ax1.set_xlabel(r"$z$")
ax1.set_title("min. app. magnitude = " + str(magnitude_min))
ax1.errorbar(z, rel_num_LCDM, yerr=d_rel_num_LCDM, fmt='-', label='LCDM')
ax1.errorbar(z, rel_num_EdS, yerr=d_rel_num_EdS, fmt='--', label='E-deS')
ax1.errorbar(z, rel_num_OCDM, yerr=d_rel_num_OCDM, fmt='-', label='OCDM')
ax1.errorbar(z, rel_num_w8, yerr=d_rel_num_w8, fmt=':', label='w = -0.8')
ax1.errorbar(z, rel_num_w9, yerr=d_rel_num_w9, fmt=':', label='w = -0.9')
ax1.errorbar(z, rel_num_w11, yerr=d_rel_num_w11, fmt=':', label='w = -1.1')
ax1.errorbar(z, rel_num_w12, yerr=d_rel_num_w12, fmt=':', label='w = -1.2')

fig1.legend()
plt.grid()
plt.tight_layout()


# ------------------------Relative difference using varyig phi--------------------------------------
rel_num_LCDM = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
rel_num_EdS = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 1, 0, 0, 'LCDM', h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today)
rel_num_OCDM = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0.7, 0, 'LCDM', h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today)
rel_num_w8 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today)
rel_num_w9 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today)
rel_num_w11 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today)
rel_num_w12 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today)

d_rel_num_LCDM = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
d_rel_num_EdS = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 1, 0, 0, 'LCDM', h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today)
d_rel_num_OCDM = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0.7, 0, 'LCDM', h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today)
d_rel_num_w8 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today)
d_rel_num_w9 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today)
d_rel_num_w11 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today)
d_rel_num_w12 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today)

fig2, ax2 = plt.subplots()
ax2.set_ylabel(r"Relative difference of $\frac{dN}{dz}$ between $z$ and $z=$" + str(z_ref))
ax2.set_xlabel(r"$z$")
ax2.set_title("RelDif//min. app. magnitude = " + str(magnitude_min))
ax2.errorbar(z, rel_num_LCDM, yerr=d_rel_num_LCDM, fmt='-', label='LCDM')
ax2.errorbar(z, rel_num_EdS, yerr=d_rel_num_EdS, fmt='--', label='E-deS')
ax2.errorbar(z, rel_num_OCDM, yerr=d_rel_num_OCDM, fmt='-', label='OCDM')
ax2.errorbar(z, rel_num_w8, yerr=d_rel_num_w8, fmt=':', label='w = -0.8')
ax2.errorbar(z, rel_num_w9, yerr=d_rel_num_w9, fmt=':', label='w = -0.9')
ax2.errorbar(z, rel_num_w11, yerr=d_rel_num_w11, fmt=':', label='w = -1.1')
ax2.errorbar(z, rel_num_w12, yerr=d_rel_num_w12, fmt=':', label='w = -1.2')

fig2.legend()
plt.grid()
plt.tight_layout()


# ------------------------Relative difference using constant phi and alpha-----------------------------------------------------
rel_num_LCDM = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
rel_num_EdS = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 1, 0, 0, 'LCDM', h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today)
rel_num_OCDM = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0.7, 0, 'LCDM', h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today)
rel_num_w8 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today)
rel_num_w9 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today)
rel_num_w11 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today)
rel_num_w12 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today)

d_rel_num_LCDM = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
d_rel_num_EdS = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 1, 0, 0, 'LCDM', h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today)
d_rel_num_OCDM = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0.7, 0, 'LCDM', h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today)
d_rel_num_w8 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today)
d_rel_num_w9 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today)
d_rel_num_w11 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today)
d_rel_num_w12 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today)

fig3, ax3 = plt.subplots()
ax3.set_ylabel(r"Relative difference of $\frac{dN}{dz}$ between $z$ and $z=$" + str(z_ref))
ax3.set_xlabel(r"$z$")
ax3.set_title("min. app. magnitude = " + str(magnitude_min))
ax3.errorbar(z, rel_num_LCDM, yerr=d_rel_num_LCDM, fmt='-', label='LCDM')
ax3.errorbar(z, rel_num_EdS, yerr=d_rel_num_EdS, fmt='--', label='E-deS')
ax3.errorbar(z, rel_num_OCDM, yerr=d_rel_num_OCDM, fmt='-', label='OCDM')
ax3.errorbar(z, rel_num_w8, yerr=d_rel_num_w8, fmt=':', label='w = -0.8')
ax3.errorbar(z, rel_num_w9, yerr=d_rel_num_w9, fmt=':', label='w = -0.9')
ax3.errorbar(z, rel_num_w11, yerr=d_rel_num_w11, fmt=':', label='w = -1.1')
ax3.errorbar(z, rel_num_w12, yerr=d_rel_num_w12, fmt=':', label='w = -1.2')

fig3.legend()
plt.grid()
plt.tight_layout()


# Save figures
fig1.savefig(save_path + '_varalpha' + '_h' + str(h_today) + '_zref' + str(z_ref) + '_minM' + str(magnitude_min) + '_wErr' + '.png')
fig2.savefig(save_path + '_varphi' + '_h' + str(h_today) + '_zref' + str(z_ref) + '_minM' + str(magnitude_min) + '_wErr' + '.png')
fig3.savefig(save_path + '_const' + '_h' + str(h_today) + '_zref' + str(z_ref) + '_minM' + str(magnitude_min) + '_wErr' + '.png')