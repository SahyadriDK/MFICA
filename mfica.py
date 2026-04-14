#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:46:17 2026

@author: sk322
"""

# NOTE: MOST OF THIS CODE HAS BEEN TRANSLATED VERBATIM FROM
# PAUL HEWETT'S VARIOUS MF-ICA MATLAB WRAPPERS, WITH ORIGINAL CODE
# BY OLE WINTHER (2002, 2005)


# Import all base packages
import numpy as np
import matplotlib.pyplot as plt
import ica_base

# Load spectrum data. This can be spectra or components.

def load_data(loc):
    """
    

    Parameters
    ----------
    loc : string
        Location of the data file. Assumes data in a single space format.

    Returns
    -------
    wl : array
        Numpy array containing wavelengths.
    spec_inp : array
        Numpy vector containing spectra. Each row is an object. Each column is a flux value.

    """
    raw_dat = np.loadtxt(loc) # Rows should be objects, columns are fluxes
    # Assuming that wavelength array is first row. Can change otherwise.
    wl = raw_dat[0, :] # Wavelength array for spectra. In AA. Not in log.
    spec_inp = raw_dat[1:, :] # All the spectra as a matrix.
    return wl, spec_inp

    

# Set up the parameters and load data needed to run MFICA

par = {} # Use an empty dictionary for this

par['sources'] = 3 # The number of components to be derived
par['optimizer'] = 'aem' # Only optimizer for the time being. Used in previous work
par['solver'] = 'ec' # Only solver for time being. Used in previous papers.

# Load fixed components and data
wl, fix_comp = load_data(COMP_LOC) # Load fixed components. Comment out if not used
n_fix_comps = fix_comp.shape[0] # Number of fixed components. Comment out if not used.

_, dat = load_data(DATA_LOC) # Load the data
dat_shape = dat.shape # shape of the data array

# Set up the fixed components
par['S_init'] = np.zeros(par['sources'], dat_shape[1]) # Initial source matrix of dimension NCOMPS x NPIXELS
# Comment line below if not using fixed components
par['S_init'][:n_fix_comps, :] = fix_comp # The first set of n_fix_comps are fixed.
par['m'] = par['S_init']
par['Sconst'] = np.array([1 if i < n_fix_comps else 0 for i in range(par['sources'])]) # Indicates to the code which components are held constant or fixed

# Parameters for solving MF-ICA equations
#par['Aprior'] = 'positive'
#par['Aprior'] = 'constant'

#par['Sprior'] = 'combi'
par['Sprior'] = 'exponential';
#par['Sprior']='heavy_tail';

# Possible ICA methods: {'positive','neg_kurtosis','pos_kurtosis','fa','ppca'}
par['method'] = 'positive' # Used for positive components
#par['method'] = 'neg_kurtosis'
#par['method'] = 'fa' ;
#par['method'] = 'pos_kurtosis' ;

par['Sigmaprior'] = 'isotropic' # Noise variance
par['S_tol'] = 1e-12 # Error tolerance for E-step
par['S_max_ite'] = 100 # maximum number of iterations E-step                 
par['tol'] = 1e-5 # error tolerance M-step for bfgs and conjgrad                                  % maximum number of iterations M-step
par['max_ite'] = 10000 # maximum number of iterations M-step
par['draw'] = 1 # plot run time informationtest_weight.dat
par['D'] = dat_shape[0]
par['N'] = dat_shape[1]
par['M'] = par['sources']


#=========================================================================#
#========== RUN MAIN MF-ICA Python Code ==================================#
#=========================================================================#

S, A, loglikelihood, sigma = ica_base.icaMF_fixComp(dat, par) # The main MFICA code


# ========================================================================#
#=========== Save components and make basic plots ========================#
#=========================================================================#

np.savetxt("ICA_weights.dat", A) # Save the weights
np.savetxt("ICA_comps.dat", S) # Save components

# Visualise basic components
plt.figure()
for i in range(S.shape[0]):
    plt.subplot(3, 1, i+1)
    plt.plot(wl, S[i])
    plt.xlabel(r"Wavelength ($\AA$)")
    plt.ylabel("Flux (arbitrary units)")
plt.show()

