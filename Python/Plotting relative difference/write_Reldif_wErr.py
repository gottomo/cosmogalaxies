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
h_today = 0.7
magnitude_min = 26
one_sqr_degree = (np.pi/180)**2 
z_ref = 3
z = np.linspace(0.01,z_ref,50*z_ref/3)

modelname = np.array(['LCDM', 'EdS', 'OCDM', 'w8', 'w9', 'w11', 'w12'])

# Set the subfloder where the text outputs will be placed
cur_path = os.path.dirname(__file__) # get the directory where the script is placed


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

# Return the redshift at which maximum separation occurs
sep = rel_num_LCDM - rel_num_w9
max_sep = np.amax(np.abs(sep))
z_max_sep_index = np.where(sep == max_sep)
print("(Const alpha & phi) Max difference between LCDM and w=-0.9 at z = " + str(z[z_max_sep_index]))

# Write data to file, 1st row: z, 2nd row: rel_num_<model>
models = np.array([rel_num_LCDM, rel_num_EdS, rel_num_OCDM, rel_num_w8, rel_num_w9, rel_num_w11, rel_num_w12])
errors = np.array([d_rel_num_LCDM, d_rel_num_EdS, d_rel_num_OCDM, d_rel_num_w8, d_rel_num_w9, d_rel_num_w11, d_rel_num_w12])
i=0
for model in models:
	datafile=open(os.path.join(cur_path, "data_wErr//RelDif_constPhiAlpha_zref") + str(z_ref) + "_m" + str(magnitude_min) + "_" + modelname[i] + ".txt", 'w')
	data = np.stack((z, model, errors[i]), axis=1)
	np.savetxt(datafile, data)
	i=i+1
	datafile.close()



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
# 	datafile=open(os.path.join(cur_path, "data_wErr//RelDif_varAlpha_zref") + str(z_ref) + "_m" + str(magnitude_min) + "_" + modelname[i] + ".txt", 'w')
# 	data = np.stack((z, model), axis=1)
# 	np.savetxt(datafile, data)
# 	i=i+1
# 	datafile.close()



# ------------------------Relative difference using varyig phi--------------------------------------
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
# 	datafile=open(os.path.join(cur_path, "data_wErr//RelDif_varPhi_zref") + str(z_ref) + "_m" + str(magnitude_min) + "_" + modelname[i] + ".txt", 'w')
# 	data = np.stack((z, model), axis=1)
# 	np.savetxt(datafile, data)
# 	i=i+1
# 	datafile.close()
