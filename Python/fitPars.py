import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import os
import findVol
from fitChisq import chisq_fit, ls_fit, gammafnc, gammafnc_ls

# path settings for saving files and stuff
cur_path = os.path.dirname(__file__) # get the directory where the script is placed
module_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
save_path = cur_path + '/figures/parfit/parfit'
report_save_path = os.path.abspath(os.path.join(module_path, '.')) + '/PlotsForReports/parfit'

# Vectorize functions
delta_galaxy_number_rel_z_comb = np.vectorize(findVol.delta_galaxy_number_rel_z_comb)
Phi_directVec = np.vectorize(findVol.Phi_direct)
alpha_vec = np.vectorize(findVol.alpha)

# Define Setting of magnitude and reference redshift
magnitude_min = 26
z_ref = 3
sqd = 1
isforward = 0 # 1 for forward, 0 for backward

# Limit the range of fitting here
z_min = 0.5
z_max = 2.5

# Parameters for Schechter function
# set phi2 = alpha2 = 0 for single Shcechter function
break_mass = 11.12 # average of breakmass from Mortlock (2014)
phi1_backward = 6.31e-4 # from Mortlock (2014) erratum, at z=0.3~0.5
phi1_forward = 9.33e-5 # from same as above, at z=2.5~3.0
phi2 = 0 
alpha1_backward = -1.41 # from same as above, at z=0.3~0.5
alpha1_forward = -1.69 # from same as above, at z=2.5~3.0
alpha2 = 0

d_break_mass = 0.02
d_phi1_backward = 5e-5
d_phi1_forward = 1.5e-5
d_alpha1_backward = 0.02
d_alpha1_forward = 0.06

# Parameters for varying alpha
m_alpha = -0.093
b_alpha = -1.3

d_m_alpha = 0.054
d_b_alpha = 0.06

# Parameters for varying phi (Phi_direct)
masslim = 11

# Default values for standard cosmology
omega_m = 0.3
omega_k = 0
omega_0 = 0.7
w_0 = -1
w_1 = 0 
h_today = 0.7

z_array = np.linspace(0.01,z_ref,round(50*z_ref/3))
h_array = np.linspace(0.60, 0.80, 10)
om_array = np.linspace(0.0, 0.5, 10)
w_array = np.linspace(-0.5, -1.5, 10)

if isforward:
    phi1 = phi1_forward
    alpha1 = alpha1_forward

    d_phi1 = d_phi1_forward
    d_alpha1 = d_alpha1_forward
else:
    phi1 = phi1_backward
    alpha1 = alpha1_backward

    d_phi1 = d_phi1_backward
    d_alpha1 = d_alpha1_backward

# Find the index for the limit of range of z
index_min = np.min( np.where(z_array  > z_min) )
index_max = np.max( np.where(z_array < z_max) )

# Produce fits for h
m = np.array([])
a = np.array([])
b = np.array([])
d_m = np.array([])
d_a = np.array([])
d_b = np.array([])

for h in h_array:
    z_array = np.linspace(0.01,z_ref,round(50*z_ref/3))
    rel, d_rel = delta_galaxy_number_rel_z_comb(z_array, z_ref, magnitude_min, break_mass, phi1, phi2, alpha1, alpha2, omega_0, omega_k, omega_0, 'LCDM', w_0, w_1, h, sqd)
    # Limit the range of z for fitting
    z_array = z_array[index_min:index_max]
    rel = rel[index_min:index_max]
    d_rel = d_rel[index_min:index_max]
    # par, d_par, chisq_r = chisq_fit(z_array, rel, d_rel, gammafnc, 3) # Fitting using chi-sq minimization
    par, d_par, chisq_r = ls_fit(z_array, rel, d_rel, gammafnc_ls)  # Fiting using non-linear least square fitting
    # note that gamma function has the form a * (1+z)**m * exp(b*(1+z))
    print('Reduced chi squared: %f' % (chisq_r))
    a = np.append(a, par[0])
    m = np.append(m, par[1])
    b = np.append(b, par[2])
    d_a = np.append(d_a, d_par[0])
    d_m = np.append(d_m, d_par[1])
    d_b = np.append(d_b, d_par[2])

plt.plot(h_array, a)
plt.fill_between(h_array, a-d_a, a+d_a, alpha=0.2)
plt.xlabel(r'Hubble parameter $h$', fontsize=13)
plt.ylabel(r'Fitting parameter $a$', fontsize=13)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.tight_layout()
plt.savefig(save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_a_vs_h' + '.png')
plt.savefig(report_save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_a_vs_h' + '.pdf')
plt.clf()

plt.plot(h_array, b)
plt.fill_between(h_array, b-d_b, b+d_b, alpha=0.2)
plt.xlabel(r'Hubble parameter $h$', fontsize=13)
plt.ylabel(r'Fitting parameter $b$', fontsize=13)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.tight_layout()
plt.savefig(save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_b_vs_h' + '.png')
plt.savefig(report_save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_b_vs_h' + '.pdf')
plt.clf()

plt.plot(h_array, m)
plt.fill_between(h_array, m-d_m, m+d_m, alpha=0.2)
plt.xlabel(r'Hubble parameter $h$', fontsize=13)
plt.ylabel(r'Fitting parameter $b$', fontsize=13)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.tight_layout()
plt.savefig(save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_m_vs_h' + '.png')
plt.savefig(report_save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_m_vs_h' + '.pdf')
plt.clf()

# Produce fits for w
m = np.array([])
a = np.array([])
b = np.array([])
d_m = np.array([])
d_a = np.array([])
d_b = np.array([])

for w in w_array:
    z_array = np.linspace(0.01,z_ref,round(50*z_ref/3))
    rel, d_rel = delta_galaxy_number_rel_z_comb(z_array, z_ref, magnitude_min, break_mass, phi1, phi2, alpha1, alpha2, omega_0, omega_k, omega_0, 'LCDM', w, w_1, h_today, sqd)
    # Limit the range of z for fitting
    z_array = z_array[index_min:index_max]
    rel = rel[index_min:index_max]
    d_rel = d_rel[index_min:index_max]
    # par, d_par, chisq_r = chisq_fit(z_array, rel, d_rel, gammafnc, 3) # Fitting using chi-sq minimization
    par, d_par, chisq_r = ls_fit(z_array, rel, d_rel, gammafnc_ls)  # Fiting using non-linear least square fitting
    # note that gamma function has the form a * (1+z)**m * exp(b*(1+z))
    print('Reduced chi squared: %f' % (chisq_r))
    a = np.append(a, par[0])
    m = np.append(m, par[1])
    b = np.append(b, par[2])
    d_a = np.append(d_a, d_par[0])
    d_m = np.append(d_m, d_par[1])
    d_b = np.append(d_b, d_par[2])

plt.plot(w_array, a)
plt.fill_between(w_array, a-d_a, a+d_a, alpha=0.2)
plt.xlabel(r'$w$', fontsize=13)
plt.ylabel(r'Fitting parameter $a$', fontsize=13)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.tight_layout()
plt.savefig(save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_a_vs_w' + '.png')
plt.savefig(report_save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_a_vs_w' + '.pdf')
plt.clf()

plt.plot(w_array, b)
plt.fill_between(w_array, b-d_b, b+d_b, alpha=0.2)
plt.xlabel(r'$w$', fontsize=13)
plt.ylabel(r'Fitting parameter $b$', fontsize=13)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.tight_layout()
plt.savefig(save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_b_vs_w' + '.png')
plt.savefig(report_save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_b_vs_w' + '.pdf')
plt.clf()

plt.plot(w_array, m)
plt.fill_between(w_array, m-d_m, m+d_m, alpha=0.2)
plt.xlabel(r'$w$', fontsize=13)
plt.ylabel(r'Fitting parameter $b$', fontsize=13)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.tight_layout()
plt.savefig(save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_m_vs_w' + '.png')
plt.savefig(report_save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_m_vs_w' + '.pdf')
plt.clf()

# Produce fits for om
m = np.array([])
a = np.array([])
b = np.array([])
d_m = np.array([])
d_a = np.array([])
d_b = np.array([])

for om in om_array:
    z_array = np.linspace(0.01,z_ref,round(50*z_ref/3))
    rel, d_rel = delta_galaxy_number_rel_z_comb(z_array, z_ref, magnitude_min, break_mass, phi1, phi2, alpha1, alpha2, om, omega_k, omega_0, 'LCDM', w_0, w_1, h_today, sqd)
    # Limit the range of z for fitting
    z_array = z_array[index_min:index_max]
    rel = rel[index_min:index_max]
    d_rel = d_rel[index_min:index_max]
    # par, d_par, chisq_r = chisq_fit(z_array, rel, d_rel, gammafnc, 3) # Fitting using chi-sq minimization
    par, d_par, chisq_r = ls_fit(z_array, rel, d_rel, gammafnc_ls)  # Fiting using non-linear least square fitting
    # note that gamma function has the form a * (1+z)**m * exp(b*(1+z))
    print('Reduced chi squared: %f' % (chisq_r))
    a = np.append(a, par[0])
    m = np.append(m, par[1])
    b = np.append(b, par[2])
    d_a = np.append(d_a, d_par[0])
    d_m = np.append(d_m, d_par[1])
    d_b = np.append(d_b, d_par[2])

plt.plot(om_array, a)
plt.fill_between(om_array, a-d_a, a+d_a, alpha=0.2)
plt.xlabel(r'$\Omega_m$', fontsize=13)
plt.ylabel(r'Fitting parameter $a$', fontsize=13)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.tight_layout()
plt.savefig(save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_a_vs_om' + '.png')
plt.savefig(report_save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_a_vs_om' + '.pdf')
plt.clf()

plt.plot(om_array, b)
plt.fill_between(om_array, b-d_b, b+d_b, alpha=0.2)
plt.xlabel(r'$\Omega_m$', fontsize=13)
plt.ylabel(r'Fitting parameter $b$', fontsize=13)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.tight_layout()
plt.savefig(save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_b_vs_om' + '.png')
plt.savefig(report_save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_b_vs_om' + '.pdf')
plt.clf()

plt.plot(om_array, m)
plt.fill_between(om_array, m-d_m, m+d_m, alpha=0.2)
plt.xlabel(r'$\Omega_m$', fontsize=13)
plt.ylabel(r'Fitting parameter $b$', fontsize=13)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.tight_layout()
plt.savefig(save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_m_vs_om' + '.png')
plt.savefig(report_save_path + '_minM' + str(magnitude_min) + '_sqd' + str(sqd) + '_zref' + str(z_ref) + '_m_vs_om' + '.pdf')
plt.clf()