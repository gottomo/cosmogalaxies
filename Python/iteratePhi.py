import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate # this module does the integration
import os
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

def delta_galaxy_number_rel_z_iter(z, mag, break_mass, phi1, phi2, alpha1, alpha2, omega_m, omega_k, omega_0, model, sqd=1, w_0=-1, w_1=0, h=0.7):
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


# ---------- test stuff here-------------
h07 = 0.7
magnitude_min = 28
one_sqr_degree = (np.pi/180)**2 
z_ref = 3
z = np.linspace(0.01,z_ref,50*z_ref/3)

rels = delta_galaxy_number_rel_z_iter(z, magnitude_min, 11.12, Phi_directVec(z_ref, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h07), 0, alpha(z_ref, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h07)
# print(rels)
# rels2 = delta_galaxy_number_rel_z_vec(z, z_ref, magnitude_min, 11.12, Phi_directVec(z_ref, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h07), 0, alpha(z_ref, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h07)
# print(rels2)
rels2 = galaxy_number_vec(z, magnitude_min, 11.12, Phi_directVec(z_ref, 11, 2.88e-3, 0.3, 0, 0.7, "LCDM", h=h07), 0, alpha(z_ref, -0.093, -1.3), -1.47, 0.3, 0, 0.7, 'LCDM', h=h07) * one_sqr_degree 

fig, ax = plt.subplots()
ax.plot(z, rels, '--', label='with merger')
ax.plot(z, rels2, '-.', label='without merger')
ax.set_xlabel('Redshift z')
ax.set_ylabel('Number of galaxies per 1 sqr degree')
ax.legend()
plt.tight_layout()
plt.show()