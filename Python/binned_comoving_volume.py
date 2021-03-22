import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate # this module does the integration
import scipy.interpolate as interp
import os
import findVol
cur_path = os.path.dirname(__file__) # get the directory where the script is placed
galaxy_no_density = findVol.galaxy_no_density

save_path = cur_path + '/figures/'



# def comoving_vol(z_begin, z_end, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
# 	# Integrate the comoving volume element between redshift z_begin and z_end
# 	return integrate(lambda Z: comoving_vol_elm(Z, omega_m, omega_k, omega_0, model, w_0, w_1, h), z_begin, z_end)


def comoving_vol(z, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7, bin_size=0):
	# Integrate the comoving volume element between the redshifts given in the array z
    # Note: NO NEED to VECTORIZE this function!
    i=0
    #print(isinstance(z, np.float))
    if (isinstance(z, np.ndarray)):
        volume = np.array([])
        for zs in z:
            if(i+1<z.size):
                volume = np.append(volume, integrate.quad(lambda Z: findVol.comoving_vol_elm(Z, omega_m, omega_k, omega_0, model, w_0, w_1, h), zs, z[i+1])[0])
            i=i+1
    else:
        volume = integrate.quad(lambda Z: findVol.comoving_vol_elm(Z, omega_m, omega_k, omega_0, model, w_0, w_1, h), z, z + bin_size)[0]
    # for zs in z:
    #     if(i+1<z.size):
    #         volume = np.append(volume, integrate.quad(lambda Z: findVol.comoving_vol_elm(Z, omega_m, omega_k, omega_0, model, w_0, w_1, h), zs, z[i+1])[0])
    #     i=i+1

    return volume

def proper_volume(z, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7, bin_size=0):
	# Integrate the comoving volume element between the redshifts given in the array z
    # Note: NO NEED to VECTORIZE this function!
    i=0
    #print(isinstance(z, np.float))
    if (isinstance(z, np.ndarray)):
        volume = np.array([])
        for zs in z:
            if(i+1<z.size):
                volume = np.append(volume, integrate.quad(lambda Z: findVol.comoving_vol_elm(Z, omega_m, omega_k, omega_0, model, w_0, w_1, h) * 1/(4*np.pi*findVol.ang_diameter_d(Z,omega_m, omega_k, omega_0, model, w_0, w_1, h)**2), zs, z[i+1])[0])
            i=i+1
    else:
        volume = integrate.quad(lambda Z: findVol.comoving_vol_elm(Z, omega_m, omega_k, omega_0, model, w_0, w_1, h), z, z + bin_size)[0]
    # for zs in z:
    #     if(i+1<z.size):
    #         volume = np.append(volume, integrate.quad(lambda Z: findVol.comoving_vol_elm(Z, omega_m, omega_k, omega_0, model, w_0, w_1, h), zs, z[i+1])[0])
    #     i=i+1

    return volume


def galaxy_number_bin(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7, bin_size=0):
    # Finding comoving volume between the redshifts given in the array z, find the number of galaxies in each bins
    z_bin = np.array([])
    volume = np.array([])
    i=0
    for zs in z:
        if(i+1<z.size):
            volume = np.append(volume, integrate.quad(lambda Z: findVol.comoving_vol_elm(Z, omega_m, omega_k, omega_0, model, w_0, w_1, h), zs, z[i+1])[0])
            z_bin = np.append(z_bin, (zs + z[i+1])/2)
        i=i+1
    num_array = volume * galaxy_no_density_vec(z_bin, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h)
    return num_array
    
def delta_galaxy_number_rel_z_comb(z, z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7, sqd=1):
    # Returns the difference in number density of galaxies between an arbitrary z and a set z
    z_bin = np.array([])
    volume = np.array([])
    i=0
    for zs in z:
        if(i+1<z.size):
            volume = np.append(volume, integrate.quad(lambda Z: findVol.comoving_vol_elm(Z, omega_m, omega_k, omega_0, model, w_0, w_1, h), zs, z[i+1])[0])
            z_bin = np.append(z_bin, (zs + z[i+1])/2)
        i=i+1
    number = volume * galaxy_no_density_vec(z_bin, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h) * one_sqr_degree * sqd
    number_z_ref = number[-1]
    # print(model)
    # print("number: " + str(number))
    # print("number_z: " + str(number_z_ref))
    rel = (number - number_z_ref) / number_z_ref
    d_number = np.sqrt(number)
    d_number_z_ref = np.sqrt(number_z_ref)
    rel = (number - number_z_ref) / number_z_ref
    # if(number!=number_z_ref):
    #     d_rel = np.sqrt((d_number**2+d_number_z_ref**2)/(number-number_z_ref)**2 + (d_number_z_ref/number_z_ref)**2) * rel
    # else:
    #     d_rel=0
    with np.errstate(divide='ignore', invalid='ignore'):
        d_rel = np.sqrt((d_number**2+d_number_z_ref**2)/(number-number_z_ref)**2 + (d_number_z_ref/number_z_ref)**2) * rel
    d_rel[-1] = 0
    return rel, d_rel

def delta_galaxy_number_rel_z_comb_masslim(z, z_ref, mass, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7, sqd=1):
    # Returns the difference in number density of galaxies between an arbitrary z and a set z
    z_bin = np.array([])
    volume = np.array([])
    i=0
    for zs in z:
        if(i+1<z.size):
            volume = np.append(volume, integrate.quad(lambda Z: findVol.comoving_vol_elm(Z, omega_m, omega_k, omega_0, model, w_0, w_1, h), zs, z[i+1])[0])
            z_bin = np.append(z_bin, (zs + z[i+1])/2)
        i=i+1
    number = volume * galaxy_no_density_masslim_vec(z_bin, mass, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h) * one_sqr_degree * sqd
    number_z_ref = number[-1]
    # print(model)
    # print("number: " + str(number))
    # print("number_z: " + str(number_z_ref))
    rel = (number - number_z_ref) / number_z_ref
    d_number = np.sqrt(number)
    d_number_z_ref = np.sqrt(number_z_ref)
    rel = (number - number_z_ref) / number_z_ref
    # if(number!=number_z_ref):
    #     d_rel = np.sqrt((d_number**2+d_number_z_ref**2)/(number-number_z_ref)**2 + (d_number_z_ref/number_z_ref)**2) * rel
    # else:
    #     d_rel=0
    with np.errstate(divide='ignore', invalid='ignore'):
        d_rel = np.sqrt((d_number**2+d_number_z_ref**2)/(number-number_z_ref)**2 + (d_number_z_ref/number_z_ref)**2) * rel
    d_rel[-1] = 0
    return rel, d_rel

def delta_galaxy_density_rel_z_comb_masslim(z, z_ref, mass, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7, sqd=1):
    # Returns the difference in number density of galaxies between an arbitrary z and a set z
    z_bin = np.array([])
    volume = np.array([])
    i=0
    for zs in z:
        if(i+1<z.size):
            volume = np.append(volume, integrate.quad(lambda Z: findVol.comoving_vol_elm(Z, omega_m, omega_k, omega_0, model, w_0, w_1, h) * 1/(4*np.pi*findVol.ang_diameter_d(Z,omega_m, omega_k, omega_0, model, w_0, w_1, h)**2), zs, z[i+1])[0])
            z_bin = np.append(z_bin, (zs + z[i+1])/2)
        i=i+1
    number = volume * galaxy_no_density_masslim_vec(z_bin, mass, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h) * one_sqr_degree * sqd
    number_z_ref = number[-1]
    # print(model)
    # print("number: " + str(number))
    # print("number_z: " + str(number_z_ref))
    rel = (number - number_z_ref) / number_z_ref
    d_number = np.sqrt(number)
    d_number_z_ref = np.sqrt(number_z_ref)
    rel = (number - number_z_ref) / number_z_ref
    # if(number!=number_z_ref):
    #     d_rel = np.sqrt((d_number**2+d_number_z_ref**2)/(number-number_z_ref)**2 + (d_number_z_ref/number_z_ref)**2) * rel
    # else:
    #     d_rel=0
    with np.errstate(divide='ignore', invalid='ignore'):
        d_rel = np.sqrt((d_number**2+d_number_z_ref**2)/(number-number_z_ref)**2 + (d_number_z_ref/number_z_ref)**2) * rel
    d_rel[-1] = 0
    return rel, d_rel

def galaxy_number_bin_masslim(z, mass, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0=-1, w_1=0, h=0.7):
    # Finding comoving volume between the redshifts given in the array z, find the number of galaxies in each bins
    # Galaxy number density accounting for the angle 
    z_bin = np.array([])
    volume = np.array([])
    i=0
    for zs in z:
        if(i+1<z.size):
            volume = np.append(volume, integrate.quad(lambda Z: findVol.comoving_vol_elm(Z, omega_m, omega_k, omega_0, model, w_0, w_1, h) * 1/(4*np.pi*findVol.ang_diameter_d(Z,omega_m, omega_k, omega_0, model, w_0, w_1, h)**2), zs, z[i+1])[0])
            z_bin = np.append(z_bin, (zs + z[i+1])/2)
        i=i+1
    num_array = volume * galaxy_no_density_masslim_vec(z_bin, mass, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h) 
    return num_array


# Vectorize the functions
comoving_vol_elm_vec = np.vectorize(findVol.comoving_vol_elm)
delta_com_vol_elm_vec = np.vectorize(findVol.delta_com_vol_elm)
magnitude_bol_vec = np.vectorize(findVol.magnitude_bol)
schechter_mass_vec = np.vectorize(findVol.schechter_mass)
mag_to_mass_vec = np.vectorize(findVol.mag_to_mass)
galaxy_no_density_vec = np.vectorize(findVol.galaxy_no_density)
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
ang_diameter_d_vec = np.vectorize(findVol.ang_diameter_d)
mass_limit_rel_to_LCDM_vec = np.vectorize(findVol.mass_limit_rel_to_LCDM)
galaxy_no_density_masslim_vec = np.vectorize(findVol.galaxy_no_density_masslim)
galaxy_number_masslim_vec = np.vectorize(findVol.galaxy_number_masslim)
delta_galaxy_number_rel_z_masslim_vec = np.vectorize(findVol.delta_galaxy_number_rel_z_masslim)
d_delta_galaxy_number_rel_z_masslim_vec = np.vectorize(findVol.d_delta_galaxy_number_rel_z_masslim)
alpha = findVol.alpha

comoving_vol_vec = np.vectorize(comoving_vol)
galaxy_number_bin_vec = np.vectorize(galaxy_number_bin)
delta_galaxy_number_rel_z_comb_vec = np.vectorize(delta_galaxy_number_rel_z_comb)

delta_galaxy_number_rel_z_comb_masslim_vec = np.vectorize(delta_galaxy_number_rel_z_comb_masslim)

h_today = 0.7
magnitude_min = 26
one_sqr_degree = (np.pi/180)**2 
z_ref = 3
bin_size = 0.25
z_array = np.linspace(0.01,z_ref,round(z_ref/bin_size))
sqd = 1

# Create bins* of redshift from an array of redshifts
# * Here a "bin" means the average value of adjucent redshifts, which would be placed in the centre of a bin
z_bin = np.array([])
i=0
for zs in z_array:
    if(i+1<z_array.size):
        z_bin = np.append(z_bin, (zs + z_array[i+1])/2)
    i=i+1

# Plot comoving volume
# fig1_1, ax1_1 = plt.subplots()
# ax1_1.set_xlabel(r"Redshift $z$", fontsize=13)
# ax1_1.set_ylabel(r"Comoving volume ${V_C}$ per unit solid angle (Mpc$^3$)", fontsize=13)
# ax1_1.plot(z_bin, comoving_vol(z_array, 0.3, 0, 0.7, 'LCDM', h=h_today), '.', label='LCDM') # LCDM
# ax1_1.plot(z_bin, comoving_vol(z_array, 1, 0, 0, 'LCDM', h=h_today), '--', label='E-deS') # E-deS
# ax1_1.plot(z_bin, comoving_vol(z_array, 0.3, 0.7, 0, 'LCDM', h=h_today), '-.', label='OCDM') # OCDM
# ax1_1.plot(z_bin, comoving_vol(z_array, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), linestyle=(0, (1,1)), label='w = -0.8')
# ax1_1.plot(z_bin, comoving_vol(z_array, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), linestyle=(0, (1,2)), label='w = -0.9')
# ax1_1.plot(z_bin, comoving_vol(z_array, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), linestyle=(0, (1,3)), label='w = -1.1')
# ax1_1.plot(z_bin, comoving_vol(z_array, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), linestyle=(0, (1,4)), label='w = -1.2')
# plt.rc('xtick', labelsize=11)
# plt.rc('ytick', labelsize=11)
# ax1_1.legend()
# plt.tight_layout()
# fig1_1.savefig(save_path + 'comoving_vol' + '_binsize' + str(bin_size) + '_zref' + str(z_ref) + '.png')
# # fig1_1.savefig(save_path + 'comoving_vol' + '_binsize' + str(bin_size) + '_zref' + str(z_ref) + '.pdf')
# plt.show()
# plt.clf()

# Plot comoving volume * 1/4*pi*d_A
# fig1_1, ax1_1 = plt.subplots()
# ax1_1.set_xlabel(r"Redshift $z$", fontsize=13)
# ax1_1.set_ylabel(r"Proper volume ${V_C}$ (Mpc$^3$)", fontsize=13)
# ax1_1.plot(z_bin, proper_volume(z_array, 0.3, 0, 0.7, 'LCDM', h=h_today), '-', marker='.', label='LCDM') # LCDM
# ax1_1.plot(z_bin, proper_volume(z_array, 1, 0, 0, 'LCDM', h=h_today), '--', marker='.', label='E-deS') # E-deS
# ax1_1.plot(z_bin, proper_volume(z_array, 0.3, 0.7, 0, 'LCDM', h=h_today), '-.', marker='.', label='OCDM') # OCDM
# ax1_1.plot(z_bin, proper_volume(z_array, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.8, h=h_today), marker='.', linestyle=(0, (1,1)), label='w = -0.8')
# ax1_1.plot(z_bin, proper_volume(z_array, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today), marker='.', linestyle=(0, (1,2)), label='w = -0.9')
# ax1_1.plot(z_bin, proper_volume(z_array, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today), marker='.', linestyle=(0, (1,3)), label='w = -1.1')
# ax1_1.plot(z_bin, proper_volume(z_array, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today), marker='.', linestyle=(0, (1,4)), label='w = -1.2')
# plt.rc('xtick', labelsize=11)
# plt.rc('ytick', labelsize=11)
# ax1_1.legend()
# plt.tight_layout()
# fig1_1.savefig(save_path + 'proper_vol' + '_binsize' + str(bin_size) + '_zref' + str(z_ref) + '.png')
# # fig1_1.savefig(save_path + 'comoving_vol' + '_binsize' + str(bin_size) + '_zref' + str(z_ref) + '.pdf')
# plt.show()
# plt.clf()


# ------------- Galaxy number --------------------
# Uses double schechter function
# fig7, ax7 = plt.subplots()
# ax7.set_ylabel(r"Galaxy numbers $N$")
# ax7.set_xlabel(r"$z$")
# # ax7.set_title("min. app. magnitude = " + str(magnitude_min) + ", per square degree")
# #print(galaxy_number_bin(z_array, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today))
# ax7.plot(z_bin, galaxy_number_bin(z_array, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today) * one_sqr_degree, '.', label='LCDM')
# ax7.plot(z_bin, galaxy_number_bin(z_array, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 1, 0, 0, 'LCDM', h=h_today) * one_sqr_degree, '--', label='E-deS')
# ax7.plot(z_bin, galaxy_number_bin(z_array, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today) * one_sqr_degree, '-', label='OCDM')
# ax7.plot(z_bin, galaxy_number_bin(z_array, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, 'constant_w', w_0 = -0.8, h=h_today) * one_sqr_degree, ':', label='w = -0.8')
# ax7.plot(z_bin, galaxy_number_bin(z_array, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -0.9, h=h_today) * one_sqr_degree, ':', label='w = -0.9')
# ax7.plot(z_bin, galaxy_number_bin(z_array, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.1, h=h_today) * one_sqr_degree, ':', label='w = -1.1')
# ax7.plot(z_bin, galaxy_number_bin(z_array, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, model = 'constant_w', w_0 = -1.2, h=h_today) * one_sqr_degree, ':', label='w = -1.2')

# plt.rc('xtick', labelsize=11)
# plt.rc('ytick', labelsize=11)
# plt.tight_layout()
# ax7.legend()
# fig7.savefig(save_path + 'true_galaxy_number' + '_binsize' + str(bin_size) + '_zref' + str(z_ref) + '_mag' + str(magnitude_min) + '.png')
# # fig7.savefig(save_path + 'true_galaxy_number' + '_binsize' + str(bin_size) + '_zref' + str(z_ref) + '_mag' + str(magnitude_min) + '.pdf')
# plt.show()
# plt.clf()

# ------------- Relative number --------------------
# Uses double schechter function
# fig1, ax1 = plt.subplots()
# ax1.set_ylabel(r"Galaxy numbers $N$")
# ax1.set_xlabel(r"$z$")
# rel_num_LCDM, d_rel_num_LCDM = delta_galaxy_number_rel_z_comb(z_array, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today, sqd = sqd)
# rel_num_w8, d_rel_num_w8 = delta_galaxy_number_rel_z_comb(z_array, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7,  'constant_w', -0.8, h=h_today, sqd = sqd)
# rel_num_w9, d_rel_num_w9 = delta_galaxy_number_rel_z_comb(z_array, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7,  'constant_w', -0.9, h=h_today, sqd = sqd)
# rel_num_w11, d_rel_num_w11 = delta_galaxy_number_rel_z_comb(z_array, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, 'constant_w', -1.1, h=h_today, sqd = sqd)
# rel_num_w12, d_rel_num_w12 = delta_galaxy_number_rel_z_comb(z_array, z_ref, magnitude_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, 'constant_w', -1.2, h=h_today, sqd = sqd)

# ax1.plot(z_bin, rel_num_LCDM, '.', marker='.', label='LCDM')
# ax1.plot(z_bin, rel_num_w8, ':', marker='.',label='w = -0.8')
# ax1.plot(z_bin, rel_num_w9, ':', marker='.', label='w = -0.9')
# ax1.plot(z_bin, rel_num_w11, '-.', marker='.',label='w = -1.1')
# ax1.plot(z_bin, rel_num_w12, '-.', marker='.',label='w = -1.2')
    
# ax1.fill_between(z_bin, rel_num_LCDM - d_rel_num_LCDM, rel_num_LCDM + d_rel_num_LCDM, alpha=0.2)
# ax1.fill_between(z_bin, rel_num_w8 - d_rel_num_w8, rel_num_w8 + d_rel_num_w8, alpha=0.2)
# ax1.fill_between(z_bin, rel_num_w9 - d_rel_num_w9, rel_num_w9 + d_rel_num_w9, alpha=0.2)
# ax1.fill_between(z_bin, rel_num_w11 - d_rel_num_w11, rel_num_w11 + d_rel_num_w11, alpha=0.2)
# ax1.fill_between(z_bin, rel_num_w12 - d_rel_num_w12, rel_num_w12 + d_rel_num_w12, alpha=0.2)

# plt.rc('xtick', labelsize=11)
# plt.rc('ytick', labelsize=11)
# ax1.legend()
# plt.tight_layout()
# fig1.savefig(save_path + 'reldif' + '_binsize' + str(bin_size) + '_wcomp' + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_var' + '_zref' + str(z_ref) + '_wErr' + '.png')
# plt.show()
# fig1.savefig(report_save_path + '_wcomp' + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_var' + '_zref' + str(z_ref) + '_wErr' + '.pdf', format='pdf')


# def afunction(array):
#     print(isinstance(array, np.ndarray))
#     return 0

# afunction_vec = np.vectorize(afunction)

# afunction_vec(z_array)


# ------------- Relative number with fixed mass limit --------------------
mass_min = 9.5
# Uses double schechter function
# fig1, ax1 = plt.subplots()
# ax1.set_ylabel(r"Galaxy numbers $N$")
# ax1.set_xlabel(r"$z$")
# rel_num_LCDM, d_rel_num_LCDM = delta_galaxy_number_rel_z_comb_masslim(z_array, z_ref, mass_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today, sqd = sqd)
# rel_num_w8, d_rel_num_w8 = delta_galaxy_number_rel_z_comb_masslim(z_array, z_ref, mass_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7,  'constant_w', -0.8, h=h_today, sqd = sqd)
# rel_num_w9, d_rel_num_w9 = delta_galaxy_number_rel_z_comb_masslim(z_array, z_ref, mass_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7,  'constant_w', -0.9, h=h_today, sqd = sqd)
# rel_num_w11, d_rel_num_w11 = delta_galaxy_number_rel_z_comb_masslim(z_array, z_ref, mass_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, 'constant_w', -1.1, h=h_today, sqd = sqd)
# rel_num_w12, d_rel_num_w12 = delta_galaxy_number_rel_z_comb_masslim(z_array, z_ref, mass_min, 10.66, 3.96e-3, 0.79e-3, -0.35, -1.47, 0.3, 0, 0.7, 'constant_w', -1.2, h=h_today, sqd = sqd)

# ax1.plot(z_bin, rel_num_LCDM, '.', marker='.', label='LCDM')
# ax1.plot(z_bin, rel_num_w8, ':', marker='.',label='w = -0.8')
# ax1.plot(z_bin, rel_num_w9, ':', marker='.', label='w = -0.9')
# ax1.plot(z_bin, rel_num_w11, '-.', marker='.',label='w = -1.1')
# ax1.plot(z_bin, rel_num_w12, '-.', marker='.',label='w = -1.2')
    
# ax1.fill_between(z_bin, rel_num_LCDM - d_rel_num_LCDM, rel_num_LCDM + d_rel_num_LCDM, alpha=0.2)
# ax1.fill_between(z_bin, rel_num_w8 - d_rel_num_w8, rel_num_w8 + d_rel_num_w8, alpha=0.2)
# ax1.fill_between(z_bin, rel_num_w9 - d_rel_num_w9, rel_num_w9 + d_rel_num_w9, alpha=0.2)
# ax1.fill_between(z_bin, rel_num_w11 - d_rel_num_w11, rel_num_w11 + d_rel_num_w11, alpha=0.2)
# ax1.fill_between(z_bin, rel_num_w12 - d_rel_num_w12, rel_num_w12 + d_rel_num_w12, alpha=0.2)

# plt.rc('xtick', labelsize=11)
# plt.rc('ytick', labelsize=11)
# ax1.legend()
# plt.tight_layout()
# fig1.savefig(save_path + 'reldif' + '_binsize' + str(bin_size) + '_wcomp' + '_minMass' + str(mass_min) + '_sqd' + str(sqd) + '_var' + '_zref' + str(z_ref) + '_wErr' + '.png')
# plt.show()



# Uses single schechter function with varying alpha
# fig1, ax1 = plt.subplots()
# ax1.set_ylabel(r"Galaxy numbers $N$")
# ax1.set_xlabel(r"$z$")
# rel_num_LCDM, d_rel_num_LCDM = delta_galaxy_number_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today, sqd = sqd)
# rel_num_w8, d_rel_num_w8 = delta_galaxy_number_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', -0.8, h=h_today, sqd = sqd)
# rel_num_w9, d_rel_num_w9 = delta_galaxy_number_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', -0.9, h=h_today, sqd = sqd)
# rel_num_w11, d_rel_num_w11 = delta_galaxy_number_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', -1.1, h=h_today, sqd = sqd)
# rel_num_w12, d_rel_num_w12 = delta_galaxy_number_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', -1.2, h=h_today, sqd = sqd)

# ax1.plot(z_bin, rel_num_LCDM, '.', marker='.', label='LCDM')
# ax1.plot(z_bin, rel_num_w8, ':', marker='.',label='w = -0.8')
# ax1.plot(z_bin, rel_num_w9, ':', marker='.', label='w = -0.9')
# ax1.plot(z_bin, rel_num_w11, '-.', marker='.',label='w = -1.1')
# ax1.plot(z_bin, rel_num_w12, '-.', marker='.',label='w = -1.2')
    
# ax1.fill_between(z_bin, rel_num_LCDM - d_rel_num_LCDM, rel_num_LCDM + d_rel_num_LCDM, alpha=0.2)
# ax1.fill_between(z_bin, rel_num_w8 - d_rel_num_w8, rel_num_w8 + d_rel_num_w8, alpha=0.2)
# ax1.fill_between(z_bin, rel_num_w9 - d_rel_num_w9, rel_num_w9 + d_rel_num_w9, alpha=0.2)
# ax1.fill_between(z_bin, rel_num_w11 - d_rel_num_w11, rel_num_w11 + d_rel_num_w11, alpha=0.2)
# ax1.fill_between(z_bin, rel_num_w12 - d_rel_num_w12, rel_num_w12 + d_rel_num_w12, alpha=0.2)

# plt.rc('xtick', labelsize=11)
# plt.rc('ytick', labelsize=11)
# ax1.legend()
# plt.tight_layout()
# fig1.savefig(save_path + '_old' + 'reldif' + '_binsize' + str(bin_size) + '_var' + '_minMass' + str(mass_min) + '_sqd' + str(sqd) + '_var' + '_zref' + str(z_ref) + '_wErr' + '.png')
# plt.show()


# ----------------- Relative galaxy density -----------------
# Uses single schechter function with varying alpha
# fig1, ax1 = plt.subplots()
# ax1.set_ylabel(r"Relative galaxy number density (1 / redshift Mpc$^2$)")
# ax1.set_xlabel(r"$z$")
# rel_num_LCDM, d_rel_num_LCDM = delta_galaxy_density_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today, sqd = sqd)
# rel_num_w8, d_rel_num_w8 = delta_galaxy_density_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', -0.8, h=h_today, sqd = sqd)
# rel_num_w9, d_rel_num_w9 = delta_galaxy_density_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', -0.9, h=h_today, sqd = sqd)
# rel_num_w11, d_rel_num_w11 = delta_galaxy_density_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', -1.1, h=h_today, sqd = sqd)
# rel_num_w12, d_rel_num_w12 = delta_galaxy_density_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', -1.2, h=h_today, sqd = sqd)
# rel_num_EdS, d_rel_num_EdS = delta_galaxy_density_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 1, 0, 0, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today, sqd = sqd)
# rel_num_OCDM, d_rel_num_OCDM = delta_galaxy_density_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0.7, 0, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today, sqd = sqd)


# ax1.plot(z_bin, rel_num_LCDM, '-', marker='.', label='LCDM')
# ax1.plot(z_bin, rel_num_EdS, '--', marker='.', label='EdS')
# ax1.plot(z_bin, rel_num_OCDM, '-.', marker='.', label='OCDM')
# ax1.plot(z_bin, rel_num_w8, ':', marker='.',label='w = -0.8')
# ax1.plot(z_bin, rel_num_w9, ':', marker='.', label='w = -0.9')
# ax1.plot(z_bin, rel_num_w11, '-.', marker='.',label='w = -1.1')
# ax1.plot(z_bin, rel_num_w12, '-.', marker='.',label='w = -1.2')

    
# # ax1.fill_between(z_bin, rel_num_LCDM - d_rel_num_LCDM, rel_num_LCDM + d_rel_num_LCDM, alpha=0.2)
# # ax1.fill_between(z_bin, rel_num_w8 - d_rel_num_w8, rel_num_w8 + d_rel_num_w8, alpha=0.2)
# # ax1.fill_between(z_bin, rel_num_w9 - d_rel_num_w9, rel_num_w9 + d_rel_num_w9, alpha=0.2)
# # ax1.fill_between(z_bin, rel_num_w11 - d_rel_num_w11, rel_num_w11 + d_rel_num_w11, alpha=0.2)
# # ax1.fill_between(z_bin, rel_num_w12 - d_rel_num_w12, rel_num_w12 + d_rel_num_w12, alpha=0.2)

# plt.rc('xtick', labelsize=11)
# plt.rc('ytick', labelsize=11)
# ax1.legend()
# plt.tight_layout()
# fig1.savefig(save_path  + 'reldif' + 'density' + '_binsize' + str(bin_size) + '_var' + '_minMass' + str(mass_min) + '_sqd' + str(sqd) + '_var' + '_zref' + str(z_ref) + '_wErr' + '.png')
# plt.show()


# ------------ Galaxy number density from purely integrating the Schechter function -----------
# fig1_2, ax1_2 = plt.subplots()
# ax1_2.set_xlabel(r"Redshift $z$", fontsize=13)
# ax1_2.set_ylabel(r"Proper galaxy number density $\frac{dN}{dV_C}$ (Mpc$^{-3}$)", fontsize=13)
# num_LCDM = galaxy_no_density_masslim_vec(z_array, mass_min, 11.12, Phi_directVec(z_array, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_array, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
# num_w8 = galaxy_no_density_masslim_vec(z_array, mass_min, 11.12, Phi_directVec(z_array, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_array, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', -0.8, h=h_today)
# num_w9 = galaxy_no_density_masslim_vec(z_array, mass_min, 11.12, Phi_directVec(z_array, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_array, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', -0.9, h=h_today)
# num_w11 = galaxy_no_density_masslim_vec(z_array, mass_min, 11.12, Phi_directVec(z_array, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_array, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', -1.1, h=h_today)
# num_w12 = galaxy_no_density_masslim_vec(z_array, mass_min, 11.12, Phi_directVec(z_array, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_array, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', -1.2, h=h_today)
# num_EdS = galaxy_no_density_masslim_vec(z_array, mass_min, 11.12, Phi_directVec(z_array, 11, 2.88e-3, 1, 0, 0, "LCDM", h=h_today), 0, alpha(z_array, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today)
# num_OCDM = galaxy_no_density_masslim_vec(z_array, mass_min, 11.12, Phi_directVec(z_array, 11, 2.88e-3, 0.3, 0.7, 0, "LCDM", h=h_today), 0, alpha(z_array, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today)

# ax1_2.plot(z_array, num_LCDM, '-', marker='.', label='LCDM')
# ax1_2.plot(z_array, num_EdS, '--', marker='.', label='EdS')
# ax1_2.plot(z_array, num_OCDM, '-.', marker='.', label='OCDM')
# ax1_2.plot(z_array, num_w8, ':', marker='.',label='w = -0.8')
# ax1_2.plot(z_array, num_w9, ':', marker='.', label='w = -0.9')
# ax1_2.plot(z_array, num_w11, '-.', marker='.',label='w = -1.1')
# ax1_2.plot(z_array, num_w12, '-.', marker='.',label='w = -1.2')

# plt.rc('xtick', labelsize=11)
# plt.rc('ytick', labelsize=11)
# ax1_2.legend()
# plt.tight_layout()
# fig1_2.savefig(save_path + 'schechter_density' + '_binsize' + str(bin_size) + 'minMass' + str(mass_min) + '.png')
# # fig1_1.savefig(save_path + 'comoving_vol' + '_binsize' + str(bin_size) + '_zref' + str(z_ref) + '.pdf')
# plt.show()
# plt.clf()


# ------------------------- Galaxy number density using bins and fixed mass limit -------------- 
# galaxy_number_bin_masslim

# fig3, ax3 = plt.subplots()
# ax3.set_xlabel(r"Redshift $z$", fontsize=13)
# ax3.set_ylabel(r"Galaxy number density $\frac{dN}{dV}$ (z$^{-1}$ Mpc$^{-2}$)", fontsize=13)
# num_LCDM = galaxy_number_bin_masslim(z_array, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
# num_w8 = galaxy_number_bin_masslim(z_array, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', -0.8, h=h_today)
# num_w9 = galaxy_number_bin_masslim(z_array, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', -0.9, h=h_today)
# num_w11 = galaxy_number_bin_masslim(z_array, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', -1.1, h=h_today)
# num_w12 = galaxy_number_bin_masslim(z_array, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', -1.2, h=h_today)
# num_EdS = galaxy_number_bin_masslim(z_array, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 1, 0, 0, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today)
# num_OCDM = galaxy_number_bin_masslim(z_array, mass_min, 11.12, Phi_directVec(z_bin, 11, 2.88e-3, 0.3, 0.7, 0, "LCDM", h=h_today), 0, alpha(z_bin, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today)

# ax3.plot(z_bin, num_LCDM, '-', marker='.', label='LCDM')
# ax3.plot(z_bin, num_EdS, '--', marker='.', label='EdS')
# ax3.plot(z_bin, num_OCDM, '-.', marker='.', label='OCDM')
# ax3.plot(z_bin, num_w8, ':', marker='.',label='w = -0.8')
# ax3.plot(z_bin, num_w9, ':', marker='.', label='w = -0.9')
# ax3.plot(z_bin, num_w11, '-.', marker='.',label='w = -1.1')
# ax3.plot(z_bin, num_w12, '-.', marker='.',label='w = -1.2')

# plt.rc('xtick', labelsize=11)
# plt.rc('ytick', labelsize=11)
# ax3.legend()
# plt.tight_layout()
# fig3.savefig(save_path + 'galaxy_no_density' + '_binsize' + str(bin_size) + 'minMass' + str(mass_min) + '.png')
# # fig1_1.savefig(save_path + 'comoving_vol' + '_binsize' + str(bin_size) + '_zref' + str(z_ref) + '.pdf')
# plt.show()
# plt.clf()





# ----------------- Relative galaxy density, No Evolutions -----------------
# Uses single schechter function with varying alpha
mass_min = 9.5
# fig1, ax1 = plt.subplots()
# ax1.set_ylabel(r"Relative galaxy number density (1 / redshift Mpc$^2$)")
# ax1.set_xlabel(r"$z$")
# rel_num_LCDM, d_rel_num_LCDM = delta_galaxy_density_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(3, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(3, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today, sqd = sqd)
# rel_num_w8, d_rel_num_w8 = delta_galaxy_density_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(3, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(3, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', -0.8, h=h_today, sqd = sqd)
# rel_num_w9, d_rel_num_w9 = delta_galaxy_density_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(3, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(3, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', -0.9, h=h_today, sqd = sqd)
# rel_num_w11, d_rel_num_w11 = delta_galaxy_density_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(3, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(3, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', -1.1, h=h_today, sqd = sqd)
# rel_num_w12, d_rel_num_w12 = delta_galaxy_density_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(3, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(3, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', -1.2, h=h_today, sqd = sqd)
# rel_num_EdS, d_rel_num_EdS = delta_galaxy_density_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(3, 11, 2.88e-3, 1, 0, 0, "LCDM", h=h_today), 0, alpha(3, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today, sqd = sqd)
# rel_num_OCDM, d_rel_num_OCDM = delta_galaxy_density_rel_z_comb_masslim(z_array, z_ref, mass_min, 11.12, Phi_directVec(3, 11, 2.88e-3, 0.3, 0.7, 0, "LCDM", h=h_today), 0, alpha(3, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today, sqd = sqd)


# ax1.plot(z_bin, rel_num_LCDM, '-', marker='.', label='LCDM')
# ax1.plot(z_bin, rel_num_EdS, '--', marker='.', label='EdS')
# ax1.plot(z_bin, rel_num_OCDM, '-.', marker='.', label='OCDM')
# ax1.plot(z_bin, rel_num_w8, ':', marker='.',label='w = -0.8')
# ax1.plot(z_bin, rel_num_w9, ':', marker='.', label='w = -0.9')
# ax1.plot(z_bin, rel_num_w11, '-.', marker='.',label='w = -1.1')
# ax1.plot(z_bin, rel_num_w12, '-.', marker='.',label='w = -1.2')

    
# # ax1.fill_between(z_bin, rel_num_LCDM - d_rel_num_LCDM, rel_num_LCDM + d_rel_num_LCDM, alpha=0.2)
# # ax1.fill_between(z_bin, rel_num_w8 - d_rel_num_w8, rel_num_w8 + d_rel_num_w8, alpha=0.2)
# # ax1.fill_between(z_bin, rel_num_w9 - d_rel_num_w9, rel_num_w9 + d_rel_num_w9, alpha=0.2)
# # ax1.fill_between(z_bin, rel_num_w11 - d_rel_num_w11, rel_num_w11 + d_rel_num_w11, alpha=0.2)
# # ax1.fill_between(z_bin, rel_num_w12 - d_rel_num_w12, rel_num_w12 + d_rel_num_w12, alpha=0.2)

# plt.rc('xtick', labelsize=11)
# plt.rc('ytick', labelsize=11)
# ax1.legend()
# plt.tight_layout()
# fig1.savefig(save_path  + 'reldif' + 'density' + '_binsize' + str(bin_size) + '_const' + '_minMass' + str(mass_min) + '_sqd' + str(sqd) + '_var' + '_zref' + str(z_ref) + '_wErr' + '.png')
# plt.show()


# ------------ Galaxy number density from purely integrating the Schechter function, without any galaxy formation -----------
fig1_2, ax1_2 = plt.subplots()
ax1_2.set_xlabel(r"Redshift $z$", fontsize=13)
ax1_2.set_ylabel(r"Proper galaxy number density $\frac{dN}{dV_C}$ (Mpc$^{-3}$)", fontsize=13)
num_LCDM = galaxy_no_density_masslim_vec(z_array, mass_min, 11.12, Phi_directVec(3, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(3, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h_today)
num_w8 = galaxy_no_density_masslim_vec(z_array, mass_min, 11.12, Phi_directVec(3, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(3, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', -0.8, h=h_today)
num_w9 = galaxy_no_density_masslim_vec(z_array, mass_min, 11.12, Phi_directVec(3, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(3, -0.093, -1.3), -1.47, 0.3, 0, 0.7,  'constant_w', -0.9, h=h_today)
num_w11 = galaxy_no_density_masslim_vec(z_array, mass_min, 11.12, Phi_directVec(3, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(3, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', -1.1, h=h_today)
num_w12 = galaxy_no_density_masslim_vec(z_array, mass_min, 11.12, Phi_directVec(3, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h_today), 0, alpha(3, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'constant_w', -1.2, h=h_today)
num_EdS = galaxy_no_density_masslim_vec(z_array, mass_min, 11.12, Phi_directVec(3, 11, 2.88e-3, 1, 0, 0, "LCDM", h=h_today), 0, alpha(3, -0.093, -1.3), -1.47, 1, 0, 0, 'LCDM', h=h_today)
num_OCDM = galaxy_no_density_masslim_vec(z_array, mass_min, 11.12, Phi_directVec(3, 11, 2.88e-3, 0.3, 0.7, 0, "LCDM", h=h_today), 0, alpha(3, -0.093, -1.3), -1.47, 0.3, 0.7, 0, 'LCDM', h=h_today)

ax1_2.plot(z_array, num_LCDM, '-', marker='.', label='LCDM')
ax1_2.plot(z_array, num_EdS, '--', marker='.', label='EdS')
ax1_2.plot(z_array, num_OCDM, '-.', marker='.', label='OCDM')
ax1_2.plot(z_array, num_w8, ':', marker='.',label='w = -0.8')
ax1_2.plot(z_array, num_w9, ':', marker='.', label='w = -0.9')
ax1_2.plot(z_array, num_w11, '-.', marker='.',label='w = -1.1')
ax1_2.plot(z_array, num_w12, '-.', marker='.',label='w = -1.2')

plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
ax1_2.legend()
plt.tight_layout()
fig1_2.savefig(save_path + 'schechter_density' + '_const' + '_binsize' + str(bin_size) + 'minMass' + str(mass_min) + '.png')
# fig1_1.savefig(save_path + 'comoving_vol' + '_binsize' + str(bin_size) + '_zref' + str(z_ref) + '.pdf')
plt.show()
plt.clf()
