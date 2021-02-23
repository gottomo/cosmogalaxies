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
delta_galaxy_number_rel_z_masslim_vec = np.vectorize(findVol.delta_galaxy_number_rel_z)
d_delta_galaxy_number_rel_z_masslim_vec = np.vectorize(findVol.d_delta_galaxy_number_rel_z)
appmag_to_absmag_vec = np.vectorize(findVol.appmag_to_absmag)
alpha_vec = np.vectorize(findVol.alpha)
LBTimeVec = np.vectorize(findVol.LBTime)
Phi_directVec = np.vectorize(findVol.Phi_direct)
alpha = findVol.alpha
mass_limit_rel_to_LCDM_vec = np.vectorize(findVol.mass_limit_rel_to_LCDM)
delta_galaxy_number_rel_z_masslim_vec = np.vectorize(findVol.delta_galaxy_number_rel_z_masslim)
d_delta_galaxy_number_rel_z_masslim_vec = np.vectorize(findVol.d_delta_galaxy_number_rel_z_masslim)


# Constants and Settings
h_today = 0.7
mass_min = 10
one_sqr_degree = (np.pi/180)**2 
z_ref = 3
z = np.linspace(0.01,z_ref,round(50*z_ref/3))
modelname = np.array(['LCDM', 'EdS', 'OCDM', 'w8', 'w9', 'w11', 'w12'])
# sqds = np.array([1,10,20,50]) # square degree on the sky obsesrved
# sqds = np.array([0.05,0.1,0.2,0.3,0.5,0.7,1])
sqds = np.array([1,3,5,10])

# Set the subfloder where the figure outputs will be placed
save_path = cur_path + '/figures/wcomp_masslim/reldif'


# Plotting how the mass limit looks like at different redshift
masslim_LCDM = mass_limit_rel_to_LCDM_vec(z, mass_min, 0.3, 0, 0.7, "LCDM", h=h_today)
masslim_w8 = mass_limit_rel_to_LCDM_vec(z, mass_min, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today)
masslim_w9 = mass_limit_rel_to_LCDM_vec(z, mass_min, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today)
masslim_w11 = mass_limit_rel_to_LCDM_vec(z, mass_min, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today)
masslim_w12 = mass_limit_rel_to_LCDM_vec(z, mass_min, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today)

fig0, ax0 = plt.subplots()
ax0.set_ylabel("Mass limit", fontsize=13 )
ax0.set_xlabel(r"$Redshift (z)$", fontsize=13)
# ax1.set_title("min. app. magnitude = " + str(mass_min))
ax0.plot(z, masslim_LCDM, '-', label='LCDM')
ax0.plot(z, masslim_w8, ':', label='w = -0.8')
ax0.plot(z, masslim_w9, ':', label='w = -0.9')
ax0.plot(z, masslim_w11, '-.', label='w = -1.1')
ax0.plot(z, masslim_w12, '-.', label='w = -1.2')

plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
ax0.legend()
plt.tight_layout()
fig0.savefig(cur_path + '/figures/wcomp_cmplt_masslim/masslim' + '_wcomp' + '_minMass' + str(mass_min)  + '.png')



# ------------------------Relative difference using constant phi and alpha, comparing different w-----------------------------------------------------

# Initialize arrays for storing standard deviations between models with different Hubble parameters h
# sd_w1w8 = np.array([])
# sd_w1w9 = np.array([])
# sd_w1w11 = np.array([])
# sd_w1w12 = np.array([])

# for sqd in sqds:
#     rel_num_LCDM = delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
#     rel_num_w8 = delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', -0.8, h=h_today)
#     rel_num_w9 = delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', -0.9, h=h_today)
#     rel_num_w11 = delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', -1.1, h=h_today)
#     rel_num_w12 = delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', -1.2, h=h_today)


#     d_rel_num_LCDM = d_delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', sqd, h=h_today)
#     d_rel_num_w8 = d_delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', sqd, -0.8, h=h_today)
#     d_rel_num_w9 = d_delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', sqd, -0.9, h=h_today)
#     d_rel_num_w11 = d_delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', sqd, -1.1, h=h_today)
#     d_rel_num_w12 = d_delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 10.66, Phi_directVec(0, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), 0, alpha(0, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', sqd, -1.2, h=h_today)

#     # Find where the max separation is, then find the standard deviation for that separation between different h
#     sep = np.abs(rel_num_LCDM - rel_num_w8)
#     max_sep = np.amax(np.abs(sep))
#     max_index = np.where(sep == max_sep)
#     sigma = np.sqrt(d_rel_num_LCDM[max_index]**2 + d_rel_num_w8[max_index]**2)
#     sd_w1w8 = np.append(sd_w1w8, max_sep / sigma)

#     sep = np.abs(rel_num_LCDM - rel_num_w9)
#     max_sep = np.amax(sep)
#     max_index = np.where(sep == max_sep)
#     sigma = np.sqrt(d_rel_num_LCDM[max_index]**2 + d_rel_num_w9[max_index]**2)
#     sd_w1w9 = np.append(sd_w1w9, max_sep / sigma)

#     sep = np.abs(rel_num_LCDM - rel_num_w11)
#     max_sep = np.amax(sep)
#     max_index = np.where(sep == max_sep)
#     sigma = np.sqrt(d_rel_num_LCDM[max_index]**2 + d_rel_num_w11[max_index]**2)
#     sd_w1w11 = np.append(sd_w1w11, max_sep / sigma)

#     sep = np.abs(rel_num_LCDM - d_rel_num_w12)
#     max_sep = np.amax(sep)
#     max_index = np.where(sep == max_sep)
#     sigma = np.sqrt(d_rel_num_LCDM[max_index]**2 + d_rel_num_w12[max_index]**2)
#     sd_w1w12 = np.append(sd_w1w12, max_sep / sigma)


#     fig1, ax1 = plt.subplots()
#     ax1.set_ylabel(r"$\Delta_{rel}\frac{dN}{dz}$ between $z$ and $z=$" + str(z_ref), fontsize=13 )
#     ax1.set_xlabel(r"$Redshift (z)$", fontsize=13)
#     # ax1.set_title("min. app. magnitude = " + str(mass_min))
#     ax1.errorbar(z, rel_num_LCDM, yerr=d_rel_num_LCDM, fmt='-', label='LCDM')
#     ax1.errorbar(z, rel_num_w8, yerr=d_rel_num_w8, fmt=':', label='w = -0.8')
#     ax1.errorbar(z, rel_num_w9, yerr=d_rel_num_w9, fmt=':', label='w = -0.9')
#     ax1.errorbar(z, rel_num_w11, yerr=d_rel_num_w11, fmt='-.', label='w = -1.1')
#     ax1.errorbar(z, rel_num_w12, yerr=d_rel_num_w12, fmt='-.', label='w = -1.2')
    
#     plt.rc('xtick', labelsize=11)
#     plt.rc('ytick', labelsize=11)
#     ax1.legend()
#     plt.tight_layout()
#     fig1.savefig(save_path + '_wcomp' + '_minMass' + str(mass_min) + '_sqd' + str(sqd) + '_const' + '_zref' + str(z_ref) + '_wErr' + '.png')

# # Plot the standard deviation as a function of sky area observed
# fig2, ax2 = plt.subplots()
# ax2.set_ylabel(r'Statistical significance ($\sigma$)', fontsize=13)
# ax2.set_xlabel(r'Observed area (square degree)', fontsize=13)
# ax2.plot(sqds, sd_w1w8, '--', label = 'between w=1 and 0.8')
# ax2.plot(sqds, sd_w1w9, '-.', label = 'between w=1 and 0.9')
# ax2.plot(sqds, sd_w1w11, ':', label = 'between w=1 and 1.1')
# ax2.plot(sqds, sd_w1w12, '-', label = 'between w=1 and 1.2')

# ax2.legend()
# plt.rc('xtick', labelsize=11)
# plt.rc('ytick', labelsize=11)
# plt.tight_layout()
# fig2.savefig(save_path + '_sd' + '_wcomp' + '_const' + '_zref' + str(z_ref) + '_minMass' + str(mass_min) + '_sqd' + str(sqds[0]) + 'to' + str(sqds[np.size(sqds)-1]) + '_wErr' + '.png')





# ------------------------Relative difference using varying phi and alpha, comparing different w-----------------------------------------------------
save_path = cur_path + '/figures/wcomp_cmplt_masslim/reldif'
report_save_path = os.path.abspath(os.path.join(module_path, '..')) + '/PlotsForReports/reldif'

# Initialize arrays for storing standard deviations between models with different Hubble parameters h
sd_w1w8 = np.array([])
sd_w1w9 = np.array([])
sd_w1w11 = np.array([])
sd_w1w12 = np.array([])

for sqd in sqds:
    rel_num_LCDM = delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 11.12, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
    rel_num_w8 = delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 11.12, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', -0.8, h=h_today)
    rel_num_w9 = delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 11.12, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', -0.9, h=h_today)
    rel_num_w11 = delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 11.12, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', -1.1, h=h_today)
    rel_num_w12 = delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 11.12, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', -1.2, h=h_today)


    d_rel_num_LCDM = d_delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 11.12, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', sqd, h=h_today)
    d_rel_num_w8 = d_delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 11.12, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', sqd, -0.8, h=h_today)
    d_rel_num_w9 = d_delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 11.12, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', sqd, -0.9, h=h_today)
    d_rel_num_w11 = d_delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 11.12, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', sqd, -1.1, h=h_today)
    d_rel_num_w12 = d_delta_galaxy_number_rel_z_masslim_vec(z, z_ref, mass_min, 11.12, Phi_directVec(z, 11, 2.88e-3, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), 0, alpha(z, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', sqd, -1.2, h=h_today)

    # Find where the max separation is, then find the standard deviation for that separation between different h
    print("sqd: " + str(sqd))
    sep = np.abs(rel_num_LCDM - rel_num_w8)
    max_sep = np.amax(np.abs(sep))
    max_index = np.where(sep == max_sep)
    print("w1w8: " + str(z[max_index]))
    sigma = np.sqrt(d_rel_num_LCDM[max_index]**2 + d_rel_num_w8[max_index]**2)
    sd_w1w8 = np.append(sd_w1w8, max_sep / sigma)

    sep = np.abs(rel_num_LCDM - rel_num_w9)
    max_sep = np.amax(sep)
    max_index = np.where(sep == max_sep)
    print("w1w9: " + str(z[max_index]))
    sigma = np.sqrt(d_rel_num_LCDM[max_index]**2 + d_rel_num_w9[max_index]**2)
    sd_w1w9 = np.append(sd_w1w9, max_sep / sigma)

    sep = np.abs(rel_num_LCDM - rel_num_w11)
    max_sep = np.amax(sep)
    max_index = np.where(sep == max_sep)
    print("w1w11: " + str(z[max_index]))
    sigma = np.sqrt(d_rel_num_LCDM[max_index]**2 + d_rel_num_w11[max_index]**2)
    sd_w1w11 = np.append(sd_w1w11, max_sep / sigma)

    sep = np.abs(rel_num_LCDM - d_rel_num_w12)
    max_sep = np.amax(sep)
    max_index = np.where(sep == max_sep)
    print("w1w12: " + str(z[max_index]))
    sigma = np.sqrt(d_rel_num_LCDM[max_index]**2 + d_rel_num_w12[max_index]**2)
    sd_w1w12 = np.append(sd_w1w12, max_sep / sigma)


    fig1, ax1 = plt.subplots()
    ax1.set_ylabel(r"$\Delta_{rel}\frac{dN}{dz}$ between $z$ and $z=$" + str(z_ref), fontsize=13 )
    ax1.set_xlabel(r"$Redshift (z)$", fontsize=13)
    # ax1.set_title("min. app. magnitude = " + str(mass_min))
    ax1.plot(z, rel_num_LCDM, '-', label='LCDM')
    ax1.plot(z, rel_num_w8, ':', label='w = -0.8')
    ax1.plot(z, rel_num_w9, ':', label='w = -0.9')
    ax1.plot(z, rel_num_w11, '-.', label='w = -1.1')
    ax1.plot(z, rel_num_w12, '-.', label='w = -1.2')
    
    ax1.fill_between(z, rel_num_LCDM - d_rel_num_LCDM, rel_num_LCDM + d_rel_num_LCDM, alpha=0.2)
    ax1.fill_between(z, rel_num_w8 - d_rel_num_w8, rel_num_w8 + d_rel_num_w8, alpha=0.2)
    ax1.fill_between(z, rel_num_w9 - d_rel_num_w9, rel_num_w9 + d_rel_num_w9, alpha=0.2)
    ax1.fill_between(z, rel_num_w11 - d_rel_num_w11, rel_num_w11 + d_rel_num_w11, alpha=0.2)
    ax1.fill_between(z, rel_num_w12 - d_rel_num_w12, rel_num_w12 + d_rel_num_w12, alpha=0.2)

    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)
    ax1.legend()
    plt.tight_layout()
    fig1.savefig(save_path + '_wcomp' + '_minMass' + str(mass_min) + '_sqd' + str(sqd) + '_var' + '_zref' + str(z_ref) + '_wErr' + '.png')
    # fig1.savefig(report_save_path + '_wcomp' + '_minMass' + str(mass_min) + '_sqd' + str(sqd) + '_var' + '_zref' + str(z_ref) + '_wErr' + '.pdf', format='pdf')

# Plot the standard deviation as a function of sky area observed
fig2, ax2 = plt.subplots()
ax2.set_ylabel(r'Statistical significance ($\sigma$)', fontsize=13)
ax2.set_xlabel(r'Observed area (square degree)', fontsize=13)
ax2.plot(sqds, sd_w1w8, '--', label = 'between w=-1 and -0.8')
ax2.plot(sqds, sd_w1w9, '-.', label = 'between w=-1 and -0.9')
ax2.plot(sqds, sd_w1w11, ':', label = 'between w=-1 and -1.1')
# ax2.plot(sqds, sd_w1w12, '-', label = 'between w=1 and 1.2')

ax2.legend()
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.tight_layout()
fig2.savefig(save_path + '_sd' + '_wcomp' + '_var' + '_zref' + str(z_ref) + '_minMass' + str(mass_min) + '_sqd' + str(sqds[0]) + 'to' + str(sqds[np.size(sqds)-1]) + '_wErr' + '.png')
# fig2.savefig(report_save_path + '_sd' + '_wcomp' + '_var' + '_zref' + str(z_ref) + '_minMass' + str(mass_min) + '_sqd' + str(sqds[0]) + 'to' + str(sqds[np.size(sqds)-1]) + '_wErr' + '.pdf', format='pdf')
