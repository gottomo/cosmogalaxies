import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate # this module does the integration
import scipy.interpolate as interp
import os
import findVol
cur_path = os.path.dirname(__file__) # get the directory where the script is placed

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
alpha = findVol.alpha

def schechter_mass_iter(z, mass, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model):
    # Schechter mass function with merger also considered
    # This function iterates backward
    mass_diff = mass - break_mass
    # z = np.flip(z)
    dz = z[0]-z[1] # find the step size for the array of z
    
    phi0 = np.log(10) * np.exp(- np.power(10, mass_diff)) * (phi1 * np.power(10, alpha1 * mass_diff) + phi2 * np.power(10, alpha2 * mass_diff)) * np.power(10, mass_diff)
    phi_merger_total = 1
    phiarray = np.array([phi0])
    for zs in z:
        # calculate merger rate from pair fraction
        tau_p = 2.4 * (1 + zs)**(-2) # in unit of Gyr
        pair_frac = 0.032 * (1 + zs)**(0.844) # the fitted pair fraction for M > 10.3 from the paper 1903.12188
        mergrate = pair_frac / tau_p
# 
        dt = LBTimeVec(zs, omega_m, omega_k, omega_0, model, h=0.7) - LBTimeVec(zs + dz, omega_m, omega_k, omega_0, model, h=0.7)
        phi_merger_ratio = 1 + mergrate * dt
        phi = phi0 * phi_merger_ratio * phi_merger_total
        phi_merger_total = phi_merger_total * phi_merger_ratio
        phiarray = np.append(phiarray, [phi], axis=0)
    # phiarray = np.flip(phiarray, 0)
    return phiarray

def schechter_mass_iter_forward(z, mass, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model):
    # Schechter mass function with merger also considered
    # This function iterates forward
    mass_diff = mass - break_mass
    z = np.flip(z)
    dz = z[0]-z[1] # find the step size for the array of z
    
    phi0 = np.log(10) * np.exp(- np.power(10, mass_diff)) * (phi1 * np.power(10, alpha1 * mass_diff) + phi2 * np.power(10, alpha2 * mass_diff)) * np.power(10, mass_diff)
    phi_merger_total = 1
    phiarray = np.array([phi0])
    for zs in z:
        # calculate merger rate from pair fraction
        tau_p = 2.4 * (1 + zs)**(-2) # in unit of Gyr
        pair_frac = 0.032 * (1 + zs)**(0.844) # the fitted pair fraction for M > 10.3 from the paper 1903.12188
        mergrate = pair_frac / tau_p
# 
        dt = LBTimeVec(zs, omega_m, omega_k, omega_0, model, h=0.7) - LBTimeVec(zs - dz, omega_m, omega_k, omega_0, model, h=0.7)
        phi_merger_ratio = 1 - mergrate * dt
        phi = phi0 * phi_merger_ratio * phi_merger_total
        phi_merger_total = phi_merger_total * phi_merger_ratio
        phiarray = np.append(phiarray, [phi], axis=0)
    phiarray = np.flip(phiarray, 0)
    return phiarray

def delta_galaxy_number_rel_z_iter(z, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model):
    # Schechter mass function with merger also considered
    mass_diff = mass - break_mass
    z = np.flip(z)
    dz = z[0]-z[1] # find the step size for the array of z
    
    phi0 = np.log(10) * np.exp(- np.power(10, mass_diff)) * (phi1 * np.power(10, alpha1 * mass_diff) + phi2 * np.power(10, alpha2 * mass_diff)) * np.power(10, mass_diff)
    phi_merger_total = 1
    phiarray = np.array([phi0])
    for zs in z:
        # calculate merger rate from pair fraction
        tau_p = 2.4 * (1 + zs)**(-2) # in unit of Gyr
        pair_frac = 0.032 * (1 + zs)**(0.844) # the fitted pair fraction for M > 10.3 from the paper 1903.12188
        mergrate = pair_frac / tau_p
        # iterate phi
        dt = LBTimeVec(zs, omega_m, omega_k, omega_0, model, h=0.7) - LBTimeVec(zs - dz, omega_m, omega_k, omega_0, model, h=0.7)
        phi_merger_ratio = 1 - mergrate * dt
        phi = phi0 * phi_merger_ratio * phi_merger_total
        phi_merger_total = phi_merger_total * phi_merger_ratio
        # integrate phi to find galaxy number density
        integrate.quad(lambda mass: schechter_mass(mass, break_mass, phi1, phi2, alpha1, alpha2), mass_min, 15)[0]
    phiarray = np.flip(phiarray, 0)
    return phiarray

def schechter_mass_ana(z, z_ref, mass, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model):
    # Calculate schechter function with mergre using analytical method
    # !!! Not correct! In the derivation I wrongly assumed mergre rate is constant in time 
    mass_diff = mass - break_mass
    phi0 = np.log(10) * np.exp(- np.power(10, mass_diff)) * (phi1 * np.power(10, alpha1 * mass_diff) + phi2 * np.power(10, alpha2 * mass_diff)) * np.power(10, mass_diff)

    # calculate merger rate and the time from pair fraction
    tau_p = 2.4 * (1 + z)**(-2) # in unit of Gyr
    pair_frac = 0.032 * (1 + z)**(0.844) # the fitted pair fraction for M > 10.3 from the paper 1903.12188
    mergrate = pair_frac / tau_p
    t = LBTimeVec(z, omega_m, omega_k, omega_0, model, h=0.7)
    print(t)
    # calculate merger rate and the time at reference redshift z_ref
    tau_p_ref = 2.4 * (1 + z_ref)**(-2) # in unit of Gyr
    pair_frac_ref = 0.032 * (1 + z_ref)**(0.844) # the fitted pair fraction for M > 10.3 from the paper 1903.12188
    mergrate_ref = pair_frac_ref / tau_p_ref
    t_ref = LBTimeVec(z_ref, omega_m, omega_k, omega_0, model, h=0.7)
    print(-mergrate_ref * t_ref + mergrate * t)
    phi = phi0 * np.exp(-mergrate_ref * t_ref + mergrate * t)

    return phi



def delta_galaxy_number_rel_z_iter_test(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, sqd=1, w_0=-1, w_1=0, h=0.7):
    # Function that returns an array of delta galaxy number for the input of array of z
    # We consider all galaxies of different masses as merging like a galaxy with mass > 10.3 dex
    z_ref = z[np.size(z)-1]
    print(z_ref)
    number_z_ref = galaxy_number_vec(z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h) * one_sqr_degree * sqd
    number = number_z_ref
    
    z = np.flip(z)
    dz = z[0]-z[1] # find the step size for the array of z
    relarray = np.array([])
    phi_merger_total = 0
    numarray = np.array([])
    for zs in z:
        dt = LBTimeVec(zs, omega_m, omega_k, omega_0, model, h=0.7) - LBTimeVec(zs - dz, omega_m, omega_k, omega_0, model, h=0.7)
        if zs > 2.5:
            dphi_merger = - 0.58 * dt
        elif zs < 2.5 and zs > 2.0:
            dphi_merger = - 0.39 * dt
        elif zs < 2.0 and zs> 1.5:
            dphi_merger = - 0.29 * dt
        elif zs < 1.5 and zs > 1.0:
            dphi_merger = - 0.12 * dt
        elif zs < 1.0 and zs > 0.5:
            dphi_merger = - 0.07 * dt
        elif zs < 0.5 and zs > 0:
            dphi_merger = 0
        
        number = galaxy_number_vec(zs, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h) * one_sqr_degree * sqd # We have to calculate numer each time, since this depends on magnitude limit
        number = number + phi_merger_total # the number calculated above does not include mergers from previous steps, so phi_merger_ total so far is substracted (or added because phi_merger is -ve)
        phi_merger = dphi_merger * number # dphi_merger is per galaxy, so to get total number of merged galaxy in time dt, multiply with the number of galaxy 
        # phi_merger = 0
        number = number + phi_merger
        numarray = np.append(numarray, number)
        phi_merger_total = phi_merger_total + phi_merger
        rel = (number - number_z_ref) / number_z_ref
        relarray = np.append(relarray, rel)
        # print(number)
        # print('phi_merger: %.5f' % (phi_merger))
        # print(number)
        # t=t+dt

    # print('total time since redshift %d: %.5f' % (z_ref, t))
    relarray = np.flip(relarray)
    numarray = np.flip(numarray)
    return numarray

def delta_galaxy_number_rel_z_iter_interp(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, sqd=1, w_0=-1, w_1=0, h=0.7):
    # Function that returns an array of delta galaxy number for the input of array of z
    # We consider all galaxies of different masses as merging like a galaxy with mass > 10.3 dex
    z_ref = z[np.size(z)-1]
    print(z_ref)
    number_z_ref = galaxy_number_vec(z_ref, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h) * one_sqr_degree * sqd
    number = number_z_ref
    
    z = np.flip(z)
    dz = z[0]-z[1] # find the step size for the array of z
    relarray = np.array([])
    no_merger_total = 0
    numarray = np.array([])
    no_densityarray = np.array([])
    for zs in z:
        # calculate merger rate from pair fraction
        tau_p = 2.4 * (1 + zs)**(-2) # in unit of Gyr
        pair_frac = 0.032 * (1 + zs)**(0.844) # the fitted pair fraction for M > 10.3 from the paper 1903.12188
        mergrate = pair_frac / tau_p

        dt = LBTimeVec(zs, omega_m, omega_k, omega_0, model, w_0, w_1, h) - LBTimeVec(zs - dz, omega_m, omega_k, omega_0, model, w_0, w_1, h)
        no_merger_pergal = - mergrate * dt # the number of mergre per galaxy during time dt per comoving volume
        no_density = galaxy_no_density_vec(zs, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, w_0, w_1, h) * one_sqr_degree * sqd # We have to calculate numer each time, since this depends on magnitude limit
        no_density = no_density + no_merger_total # the number calculated above does not include mergers from previous steps, so phi_merger_ total so far is substracted (or added because phi_merger is -ve)
        no_merger = no_merger_pergal * no_density # dphi_merger is per galaxy, so to get total number of merged galaxy in time dt, multiply with the number of galaxy 
        # phi_merger = 0
        no_density = no_density + no_merger
        no_densityarray = np.append(no_densityarray, no_density)
        number = no_density * comoving_vol_elm_vec(zs, omega_m, omega_k, omega_0, model, w_0, w_1, h)
        numarray = np.append(numarray, number)
        no_merger_total = no_merger_total + no_merger
        rel = (number - number_z_ref) / number_z_ref
        relarray = np.append(relarray, rel)
        # print(number)
        # print('phi_merger: %.5f' % (phi_merger))
        # print(number)
        # t=t+dt

    # print('total time since redshift %d: %.5f' % (z_ref, t))
    relarray = np.flip(relarray)
    numarray = np.flip(numarray)
    return no_densityarray, numarray, relarray

def ztoindex(z, z_array):
    index = np.where(z_array == z)[0][0]
    return index

# ---------- test stuff here-------------
h07 = 0.7
magnitude_min = 28
one_sqr_degree = (np.pi/180)**2 
z_ref = 3
z = np.linspace(0.01,z_ref,50*z_ref/3)

no_denitys, nums, rels = delta_galaxy_number_rel_z_iter_interp(z, magnitude_min, 11.12, Phi_directVec(z_ref, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h07), 0, alpha(z_ref, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h07)
# print(rels)
rels2 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 11.12, Phi_directVec(z_ref, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h07), 0, alpha(z_ref, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h07)
# print(rels2)
nums2 = galaxy_number_vec(z, magnitude_min, 11.12, Phi_directVec(z_ref, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h07), 0, alpha(z_ref, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h07) * one_sqr_degree 
no_denitys2 = galaxy_no_density_vec(z, magnitude_min, 11.12, Phi_directVec(z_ref, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h07), 0, alpha(z_ref, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h07)

# fig, ax = plt.subplots()
# ax.plot(z, rels, '--', label='with merger')
# ax.plot(z, rels2, '-.', label='without merger')
# ax.set_xlabel('Redshift z')
# ax.set_ylabel('Relative number of galaxies per 1 sqr degree')
# ax.legend()
# plt.tight_layout()
# plt.show()

# fig1, ax1 = plt.subplots()
# ax1.plot(z, nums, '--', label='with merger')
# ax1.plot(z, nums2, '-.', label='without merger')
# ax1.set_xlabel('Redshift z')
# ax1.set_ylabel('Number of galaxies per 1 sqr degree')
# ax1.legend()
# plt.tight_layout()
# plt.show()

# fig2, ax2 = plt.subplots()
# ax2.plot(z, no_denitys, '--', label='with merger')
# ax2.plot(z, no_denitys2, '--', label='without merger')
# ax2.set_xlabel('Redshift z')
# ax2.set_ylabel('Number of galaxies per Mpc^3')
# ax2.set_yscale('log')
# ax2.legend()
# plt.tight_layout()
# plt.show()

# ------- Plot of Schechter function at different z, produced with iterative method ------
save_path = cur_path + '/figures/merger/'

mass_dex_array = np.linspace(7,12,200)
# schechter_array = schechter_mass_iter(z, mass_dex_array, 11.12, 9.33e-5, 0, alpha(z_ref, -0.093, -1.3), 0, 0.3, 0, 0.7, 'LCDM') # for forward iteration
schechter_array = schechter_mass_iter(z, mass_dex_array, 11.12, 6.30e-4, 0, alpha(z_ref, -0.093, -1.3), 0, 0.3, 0, 0.7, 'LCDM') # for backward iteration
# print(schechter_array[1,:])
fig2, ax2 = plt.subplots()
ax2.set_xlabel(r"$M$ $(dex)$")
ax2.set_ylabel(r"$\phi (M)$ (${dex}^{-1} {Mpc}^{-3}$)")
ax2.plot(mass_dex_array, schechter_array[49,:], 'k-', label='z= %.2f' % (z[49]), alpha=0.9)
ax2.plot(mass_dex_array, schechter_array[34,:], 'k-', label='z= %.2f' % (z[34]), alpha=0.7)
ax2.plot(mass_dex_array, schechter_array[19,:], 'k-', label='z= %.2f' % (z[19]), alpha=0.4)
ax2.plot(mass_dex_array, schechter_array[4,:], 'k-', label='z= %.2f' % (z[4]), alpha=0.2)
ax2.set_yscale('log')
ax2.legend()
plt.tight_layout()
fig2.savefig(save_path + 'schechter_merger_iter.pdf')
fig2.savefig(save_path + 'schechter_merger_iter.png')
plt.show()

fig3, ax3 = plt.subplots()

ax3.set_xlabel(r"$M$ $(dex)$")
ax3.set_ylabel(r"$\phi (M)$ (${dex}^{-1} {Mpc}^{-3}$)")
ax3.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 11.12, Phi_directVec(z[49], 11, 6.30e-4, 0.3, 0, 0.7, "LCDM", h=h07), 0, alpha(z_ref, -0.093, -1.3), 0), 'k-', label='z= %.2f' % (z[49]), alpha=0.9)
ax3.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 11.12, Phi_directVec(z[34], 11, 6.30e-4, 0.3, 0, 0.7, "LCDM", h=h07), 0, alpha(z_ref, -0.093, -1.3), 0), 'k-', label='z= %.2f' % (z[34]), alpha=0.7)
ax3.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 11.12, Phi_directVec(z[19], 11, 6.30e-4, 0.3, 0, 0.7, "LCDM", h=h07), 0, alpha(z_ref, -0.093, -1.3), 0), 'k-', label='z= %.2f' % (z[19]), alpha=0.4)
ax3.plot(mass_dex_array, schechter_mass_vec(mass_dex_array, 11.12, Phi_directVec(z[4], 11, 6.30e-4, 0.3, 0, 0.7, "LCDM", h=h07), 0, alpha(z_ref, -0.093, -1.3), 0), 'k-', label='z= %.2f' % (z[4]), alpha=0.2)

ax3.set_yscale('log')
ax3.legend()
plt.tight_layout()
fig3.savefig(save_path + 'schechter_merger_direct.pdf')
fig3.savefig(save_path + 'schechter_merger_direct.png')
plt.show()


# ---------------------------- Junkyard ----------------------------------------------

# Tried interpolating merger rate, but figured it is better to just used the fitted pair fraction, and from that calculate the merger rate, as in paper 1903.12188
# mergratearray = np.array([0.58, 0.39, 0.29, 0.12, 0.07])
# zarray = np.array([2.75, 2.25, 1.75, 1.25, 0.75])
# mergrate = interp.interp1d(zarray, mergratearray)
# z_ref = 2.75
# z = np.linspace(0.75,z_ref,50*z_ref/3)
# plt.plot(z, mergrate(z))
# plt.show()

# ------- Plot of Schechter function at different z, produced with analytical method ------

# schechter_1 = schechter_mass_ana(z[49], z_ref, mass_dex_array, 10.66, 2.88e-3, 0, alpha(z_ref, -0.093, -1.3), 0, 0.3, 0, 0.7, 'LCDM')
# schechter_2 = schechter_mass_ana(z[34], z_ref, mass_dex_array, 10.66, 2.88e-3, 0, alpha(z_ref, -0.093, -1.3), 0, 0.3, 0, 0.7, 'LCDM')
# schechter_3 = schechter_mass_ana(z[19], z_ref, mass_dex_array, 10.66, 2.88e-3, 0, alpha(z_ref, -0.093, -1.3), 0, 0.3, 0, 0.7, 'LCDM')
# schechter_4 = schechter_mass_ana(z[4], z_ref, mass_dex_array, 10.66, 2.88e-3, 0, alpha(z_ref, -0.093, -1.3), 0, 0.3, 0, 0.7, 'LCDM')

# fig3, ax3 = plt.subplots()
# ax3.set_xlabel(r"$M$ $(dex)$")
# ax3.set_ylabel(r"$\phi (M)$ (${dex}^{-1} {Mpc}^{-3}$)")
# ax3.plot(mass_dex_array, schechter_1, 'k-', label='z= %.2f' % (z[49]), alpha=0.9)
# ax3.plot(mass_dex_array, schechter_2, 'k-', label='z= %.2f' % (z[34]), alpha=0.7)
# ax3.plot(mass_dex_array, schechter_3, 'k-', label='z= %.2f' % (z[19]), alpha=0.4)
# ax3.plot(mass_dex_array, schechter_4, 'k-', label='z= %.2f' % (z[4]), alpha=0.2)
# ax3.set_yscale('log')
# ax3.legend()
# plt.tight_layout()
# fig3.savefig(save_path + 'schechter_merger_ana.pdf')
# fig3.savefig(save_path + 'schechter_merger_ana.png')
# plt.show()