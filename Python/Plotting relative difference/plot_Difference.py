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
appmag_to_absmag_vec = np.vectorize(findVol.appmag_to_absmag)
alpha_vec = np.vectorize(findVol.alpha)
LBTimeVec = np.vectorize(findVol.LBTime)
Phi_directVec = np.vectorize(findVol.Phi_direct)
alpha = findVol.alpha

# Constants and Settings
h_today = 0.74
magnitude_min = 32
one_sqr_degree = (np.pi/180)**2 
z_ref = 6
z = np.linspace(0.01,z_ref,50*z_ref/3)

modelname = np.array(['LCDM', 'EdS', 'OCDM', 'w8', 'w9', 'w11', 'w12'])

# Set the subfloder where the figure outputs will be placed
save_path = cur_path + '/figures/reldif'


# ----------------------Create plot of maximum separation as a function of reference redshift---------------------------------------------------
max_sepArray = np.array([])
z_max_sepArray = np.array([])
zrefArray = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6])

for zref in zrefArray:
	rel_num_LCDM = delta_galaxy_number_rel_z_vec(z, zref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
	rel_num_w9 = delta_galaxy_number_rel_z_vec(z, zref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today)
	sep = rel_num_LCDM - rel_num_w9
	max_sep = np.amax(np.abs(sep))
	z_max_sep_index = np.where(sep == max_sep)
	z_max_sep = z[z_max_sep_index]
	max_sepArray = np.append(max_sepArray, max_sep)
	z_max_sepArray = np.append(z_max_sepArray, z_max_sep)

fig, axs = plt.subplots(2,2)
axs[0,0].plot(zrefArray, max_sepArray, 'o-')
axs[0,0].set_xlabel(r"reference redshift $z_{ref}$")
axs[0,0].set_ylabel(r"max relative number difference")
axs[0,0].set_title("min. app. magnitude = " + str(magnitude_min))
# axs[0,0].set_yscale('log')

axs[1,0].plot(zrefArray, z_max_sepArray, 'o-')
axs[1,0].set_xlabel(r"reference redshift $z_{ref}$")
axs[1,0].set_ylabel(r"redshift $z$")
axs[1,0].set_title("min. app. magnitude = " + str(magnitude_min))

# # -----------------Create plot of maximum separation as a function of magnitude-------------------------------------
max_sepArray = np.array([])
z_max_sepArray = np.array([])
magminArray = np.array([25, 27, 28, 30, 32])

for magmin in magminArray:
	rel_num_LCDM = delta_galaxy_number_rel_z_vec(z, z_ref, magmin, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
	rel_num_w9 = delta_galaxy_number_rel_z_vec(z, z_ref, magmin, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today)
	sep = rel_num_LCDM - rel_num_w9
	max_sep = np.amax(np.abs(sep))
	z_max_sep_index = np.where(sep == max_sep)
	z_max_sep = z[z_max_sep_index]
	max_sepArray = np.append(max_sepArray, max_sep)
	z_max_sepArray = np.append(z_max_sepArray, z_max_sep)

axs[0,1].plot(magminArray, max_sepArray, 'o-')
axs[0,1].set_xlabel(r"apparent magnitude threshold")
axs[0,1].set_ylabel(r"max relative number difference")
axs[0,1].set_title(r"$z_{ref} = $" + str(z_ref))
axs[0,1].set_yscale('log')

axs[1,1].plot(magminArray, z_max_sepArray, 'o-')
axs[1,1].set_xlabel(r"apparent magnitude threshold")
axs[1,1].set_ylabel(r"redshift $z$")
axs[1,1].set_title(r"$z_{ref} = $" + str(z_ref))
#---------------------------------------------------------------------------------------------------------------------

# Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()

plt.tight_layout()
fig.set_size_inches(8, 6)
fig.savefig(save_path + '_dif' + '_h' + str(h_today) + '_zref' + str(z_ref) + '_minM' + str(magnitude_min) + '.png')