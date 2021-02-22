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
magnitude_min = 28
one_sqr_degree = (np.pi/180)**2 
z_ref = 3
z = np.linspace(0.01,z_ref,round(50*z_ref/3))
modelname = np.array(['LCDM', 'EdS', 'OCDM', 'w8', 'w9', 'w11', 'w12'])
# sqds = np.array([1,10,20,50]) # square degree on the sky obsesrved
sqds = np.array([0.05,0.1,0.2,0.3,0.5,0.7,1])
# sqds = np.array([1,3,5,10])

# Set the subfloder where the figure outputs will be placed
save_path = cur_path + '/figures/hcomp/reldif'




# ------------------------Relative difference using constant phi and alpha, comparing different hubble constants-----------------------------------------------------
# Values of Hubble parameters we are comparing
h1 = 0.67
h2 = 0.7
h3 = 0.73

# Initialize arrays for storing standard deviations between models with different Hubble parameters h
sd_h1h2 = np.array([])
sd_h2h3 = np.array([])
sd_h1h3 = np.array([])

for sqd in sqds:
    rel_num_LCDM_h1 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h1), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h1)
    rel_num_LCDM_h2 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h2), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h2)
    rel_num_LCDM_h3 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h3), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h3)

    d_rel_num_LCDM_h1 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h1), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', sqd, h=h1)
    d_rel_num_LCDM_h2 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h2), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', sqd, h=h2)
    d_rel_num_LCDM_h3 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h3), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', sqd, h=h3)

    # Find where the max separation is, then find the standard deviation for that separation between different h
    sep = rel_num_LCDM_h1 - rel_num_LCDM_h2
    max_sep = np.amax(np.abs(sep))
    max_index = np.where(sep == max_sep)
    sigma = np.sqrt(d_rel_num_LCDM_h1[max_index]**2 + d_rel_num_LCDM_h2[max_index]**2)
    sd_h1h2 = np.append(sd_h1h2, max_sep / sigma)

    sep = rel_num_LCDM_h2 - rel_num_LCDM_h3
    max_sep = np.amax(np.abs(sep))
    max_index = np.where(sep == max_sep)
    sigma = np.sqrt(d_rel_num_LCDM_h2[max_index]**2 + d_rel_num_LCDM_h3[max_index]**2)
    sd_h2h3 = np.append(sd_h2h3, max_sep / sigma)

    sep = rel_num_LCDM_h1 - rel_num_LCDM_h3
    max_sep = np.amax(np.abs(sep))
    max_index = np.where(sep == max_sep)
    sigma = np.sqrt(d_rel_num_LCDM_h1[max_index]**2 + d_rel_num_LCDM_h3[max_index]**2)
    sd_h1h3 = np.append(sd_h1h3, max_sep / sigma)


    fig1, ax1 = plt.subplots()
    ax1.set_ylabel(r"Relative difference of $\frac{dN}{dz}$ between $z$ and $z=$" + str(z_ref))
    ax1.set_xlabel(r"$z$")
    # ax1.set_title("min. app. magnitude = " + str(magnitude_min))
    ax1.plot(z, rel_num_LCDM_h1, '-.', label=r'$h=$' + str(h1))
    ax1.plot(z, rel_num_LCDM_h2, '-', label=r'$h=$' + str(h2))
    ax1.plot(z, rel_num_LCDM_h3, ':', label=r'$h=$' + str(h3))

    ax1.fill_between(z, rel_num_LCDM_h1 - d_rel_num_LCDM_h1, rel_num_LCDM_h1 + d_rel_num_LCDM_h1, alpha=0.2)
    ax1.fill_between(z, rel_num_LCDM_h2 - d_rel_num_LCDM_h2, rel_num_LCDM_h2 + d_rel_num_LCDM_h2, alpha=0.2)
    ax1.fill_between(z, rel_num_LCDM_h3 - d_rel_num_LCDM_h3, rel_num_LCDM_h3 + d_rel_num_LCDM_h3, alpha=0.2)

    ax1.legend()
    plt.grid()
    # plt.tight_layout()
    fig1.savefig(save_path + '_hcomp' + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_const' + '_zref' + str(z_ref) + '_wErr' + '.png')
    fig1.savefig(save_path + '_hcomp' + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_const' + '_zref' + str(z_ref) + '_wErr' + '.pdf')


# # Plot the standard deviation as a function of sky area observed
# fig2, ax2 = plt.subplots()
# ax2.set_ylabel(r'Statistical significance ($\sigma$)')
# ax2.set_xlabel(r'Observed area (square degree)')
# ax2.plot(sqds, sd_h1h2, '--', label = 'between h = ' + str(h1) + ' and ' + str(h2))
# ax2.plot(sqds, sd_h2h3, '-.', label = 'between h = ' + str(h2) + ' and ' + str(h3))
# ax2.plot(sqds, sd_h1h3, ':', label = 'between h = ' + str(h1) + ' and ' + str(h3))

# ax2.legend()
# plt.grid()
# # plt.tight_layout()
# fig2.savefig(save_path + '_sd' + '_hcomp' + '_const' + '_zref' + str(z_ref) + '_minM' + str(magnitude_min) + '_sqd' + str(sqds[0]) + 'to' + str(sqds[np.size(sqds)-1]) + '_wErr' + '.png')







# ------------------------Relative difference using varying phi and alpha, comparing different hubble constants-----------------------------------------------------
# Values of Hubble parameters we are comparing
h1 = 0.67
h2 = 0.7
h3 = 0.73

save_path = cur_path + '/figures/hcomp_cmplt/reldif'
report_save_path = os.path.abspath(os.path.join(module_path, '..')) + '/PlotsForReports/reldif'
# print(report_save_path)


# Initialize arrays for storing standard deviations between models with different Hubble parameters h
sd_h1h2 = np.array([])
sd_h2h3 = np.array([])
sd_h1h3 = np.array([])

for sqd in sqds:
    rel_num_LCDM_h1 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 11.12, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h1), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h1)
    rel_num_LCDM_h2 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 11.12, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h2), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h2)
    rel_num_LCDM_h3 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 11.12, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h3), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h3)

    d_rel_num_LCDM_h1 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 11.12, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h1), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', sqd, h=h1)
    d_rel_num_LCDM_h2 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 11.12, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h2), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', sqd, h=h2)
    d_rel_num_LCDM_h3 = d_delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 11.12, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h3), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', sqd, h=h3)

    # Find where the max separation is, then find the standard deviation for that separation between different h
    sep = rel_num_LCDM_h1 - rel_num_LCDM_h2
    max_sep = np.amax(np.abs(sep))
    max_index = np.where(sep == max_sep)
    sigma = np.sqrt(d_rel_num_LCDM_h1[max_index]**2 + d_rel_num_LCDM_h2[max_index]**2)
    sd_h1h2 = np.append(sd_h1h2, max_sep / sigma)

    sep = rel_num_LCDM_h2 - rel_num_LCDM_h3
    max_sep = np.amax(np.abs(sep))
    max_index = np.where(sep == max_sep)
    sigma = np.sqrt(d_rel_num_LCDM_h2[max_index]**2 + d_rel_num_LCDM_h3[max_index]**2)
    sd_h2h3 = np.append(sd_h2h3, max_sep / sigma)

    sep = rel_num_LCDM_h1 - rel_num_LCDM_h3
    max_sep = np.amax(np.abs(sep))
    max_index = np.where(sep == max_sep)
    sigma = np.sqrt(d_rel_num_LCDM_h1[max_index]**2 + d_rel_num_LCDM_h3[max_index]**2)
    sd_h1h3 = np.append(sd_h1h3, max_sep / sigma)


    fig1, ax1 = plt.subplots()
    ax1.set_ylabel(r"$\Delta_{rel}\frac{dN}{dz}$ between $z$ and $z=$" + str(z_ref), fontsize=13 )
    ax1.set_xlabel(r"$Redshift (z)$", fontsize=13)
    # ax1.set_title("min. app. magnitude = " + str(magnitude_min))
    ax1.plot(z, rel_num_LCDM_h1, '-.', label=r'$h=$' + str(h1))
    ax1.plot(z, rel_num_LCDM_h2, '-', label=r'$h=$' + str(h2))
    ax1.plot(z, rel_num_LCDM_h3, ':', label=r'$h=$' + str(h3))

    ax1.fill_between(z, rel_num_LCDM_h1 - d_rel_num_LCDM_h1, rel_num_LCDM_h1 + d_rel_num_LCDM_h1, alpha=0.2)
    ax1.fill_between(z, rel_num_LCDM_h2 - d_rel_num_LCDM_h2, rel_num_LCDM_h2 + d_rel_num_LCDM_h2, alpha=0.2)
    ax1.fill_between(z, rel_num_LCDM_h3 - d_rel_num_LCDM_h3, rel_num_LCDM_h3 + d_rel_num_LCDM_h3, alpha=0.2)
    
    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)
    ax1.legend()
    plt.tight_layout()
    fig1.savefig(save_path + '_hcomp' + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_var' + '_zref' + str(z_ref) + '_wErr' + '.png')
    fig1.savefig(report_save_path + '_hcomp' + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_var' + '_zref' + str(z_ref) + '_wErr' + '.pdf', format='pdf')


# Plot the standard deviation as a function of sky area observed
fig2, ax2 = plt.subplots()
ax2.set_ylabel(r'Statistical significance ($\sigma$)', fontsize=13)
ax2.set_xlabel(r'Observed area (square degree)', fontsize=13)
ax2.plot(sqds, sd_h1h2, '--', label = 'between h = ' + str(h1) + ' and ' + str(h2))
ax2.plot(sqds, sd_h2h3, '-.', label = 'between h = ' + str(h2) + ' and ' + str(h3))
ax2.plot(sqds, sd_h1h3, ':', label = 'between h = ' + str(h1) + ' and ' + str(h3))

ax2.legend()
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.tight_layout()
fig2.savefig(save_path + '_sd' + '_hcomp' + '_var' + '_zref' + str(z_ref) + '_minM' + str(magnitude_min) + '_sqd' + str(sqds[0]) + 'to' + str(sqds[np.size(sqds)-1]) + '_wErr' + '.png')
fig2.savefig(report_save_path + '_sd' + '_hcomp' + '_var' + '_zref' + str(z_ref) + '_minM' + str(magnitude_min) + '_sqd' + str(sqds[0]) + 'to' + str(sqds[np.size(sqds)-1]) + '_wErr' + '.pdf', format='pdf')