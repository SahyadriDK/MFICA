#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 21:48:08 2026

@author: sk322
"""

import pandas as pd
import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt
import copy
from copy import deepcopy
from lmfit import minimize, Parameters, Minimizer
import glob
from astropy.io import fits
import astropy.units as u
from dust_extinction.parameter_averages import O94, CCM89
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from dust_attenuation.averages import C00
import astropy.units as u
from scipy.integrate import simpson
from spectres import spectres
from matplotlib.ticker import MultipleLocator
import json


#===========================================================================#
#============= Spectrum loading functions (optional) =======================#
#===========================================================================#

# NOTE: THIS CODE ASSUMES THAT A WAVELENGTH ARRAY AND FLUX ARRAY (AND \
# OPTIONALLY THE REDSHIFT) HAVE BEEN PROVIDED BY YOU THE USER.



#===========================================================================#
#============== MFICA Decomposition class ==================================#
#===========================================================================#

# Define a class that takes already loaded spectra, interpolates it to the \
# wavelength grid of the MFICA components.

# Assumes that only the raw spectra have been provided. The raw spectra must \
# must be in units of erg/s/cm^2/AA, with wavleengths in angstrom only.

class mfica:
    
    # initialise basic parameters of class
    def __init__(self, wl, flux, err, z=None, gal=True, qso=True, interpolate=False, 
                 mask=None, sky_mask=False):
        """
        Initialisation of MFICA object class for decomposition of spectra.
        Must provide wavelength, flux and error from spectra.
        Need to assume that raw spectra and redshifts are given.

        Parameters
        ----------
        wl : array
            Wavelength array or list.
        flux : array
            Array or list with flux values at each wavelength bin.
        err : array
            Array or list with errors on individual flux values.
        z   : float, optional
            Redshift of object. The default is None. If no value given, assumes
            that spectra provided are already in rest-frame units.
        gal : bool, optional
            Condition to set use of galaxy components. The default is True.
        qso : bool, optional
            Condition to set use of quasar components. The default is True.
        interpolate: bool, optional
            Condition to determine if need interpolation of spectra. The default is False.
        mask: array, optional
            Nested array of rest-frame wavelengths to be masked. For example, to mask Hbeta
            and OIII 5007, you can provide the following array - [[4850, 4880], [4995, 5020]]
            In each nested list, the first element is the start and the second element is the
            ending wavelength. Everything in between these two is masked.
        sky_mask: bool, optional
            Condition to determine if telluric skylines need to be masked.

        Raises
        ------
        Exception
            Exception is raised if both components are set to False.

        Returns
        -------
        None.

        """
        self.wl = wl
        self.flux = flux
        self.err = err
        self.gspec = None
        self.qspec = None
        self.model_gal = None
        self.model_qso = None
        self.model = None
        self.fit_final = None
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.w1_err = None
        self.w2_err = None
        self.w3_err = None
        self.cov = None
        self.probs = None
        self._use_gal_comps = gal
        self._use_qso_comps = qso
        self.flux_old = None
        self.err_old = None
        self.wl_old = None
        self.interpolate = interpolate
        self.z = z
        self.feature_mask = mask
        self.sky_mask = sky_mask
        
        # Must have atleast one of the components set to True
        if (not self._use_gal_comps) and (not self._use_qso_comps):
            raise Exception("Atleast one of the components needs to be set to True.")
        
        
    # Load the components
    def load_comps(self):
        """
        Method that loads galaxy ("gspec") and quasar ("qspec") components.
        Gspec and qspec are stored as attributes.

        Returns
        -------
        None.

        """
        if self._use_qso_comps:
            self.qspec = np.loadtxt("Components/QS2399_10c_noPSB_240403.comp")
        if self._use_gal_comps:
            self.gspec = np.loadtxt("Components/GS_Sahyadri_S7_240610.comp")
        self.comp_wl = np.loadtxt("Components/wav33005200.dat")
        
        
    # Interpolate the spectra to component resolution (or change it to be vice versa?)
    def interp_flux(self):
        """
        Methods that interpolates spectra to the rest-frame 3300-5200\AA range (OPTIONAL).
        Used only if interpolate keyword is set to True.
        Overrides previous wavelength, flux and error attributes.

        Returns
        -------
        None.

        """
        
        # Save the fluxes into copies before use
        self.flux_old = deepcopy(self.flux)
        self.err_old = deepcopy(self.err)
        self.wl_old = deepcopy(self.wl)
        #  Use SPECTRES. Replace with dedicated code in future
        self.flux, self.err = spectres(self.comp_wl, self.wl, self.flux, self.err, verbose=False)
        self.wl = deepcopy(self.comp_wl)
        
    
    # Shift spectra to rest-frame, while conserving flux in log space.
    def rz_corr(self):
        """
        Method that shifts a spectrum from the observed frame to its rest-frame
        wavelength range. Uses the redshift attribute provided when initialising
        the MFICA object.
        Overrides previous wavelength, flux and eror attributes.

        Returns
        -------
        None.

        """
        
        # Obtaint the log wavelengths in rest-frame and observed frame
        log_wave_obs = np.log10(self.wl)
        log_wave_rest = log_wave_obs - (np.round((np.log10(1+self.z)*1e4))/1e4) # Calculated using 1+z factor
        
        # Check if the provided redshifts give rest-frame wavelengths that are outside threshold
        if (np.max(10.0**log_wave_rest) < 5200-1e-3) or (np.min(10.0**log_wave_rest[0]) > 3300+1e-3):
            raise Exception("After shifting to rest-frame, the spectrum is outside the 3300-5200 $\AA$ range. Try another object.")
        
        # Calculate the rest frame flux and error
        flux_rest = self.flux * ((10.0**(log_wave_obs))/(10.0**(log_wave_rest)))
        err_rest = self.err * ((10.0**(log_wave_obs))/(10.0**(log_wave_rest))) # Errors also scaled up
        
        # Make assignments and delete extra junk
        self.wl = 10.0**log_wave_rest
        self.flux = flux_rest
        self.err = err_rest
        del log_wave_rest, log_wave_obs, flux_rest, err_rest # Delete extra variables.
        
        
    # Function to mask out some specifically bad telluric emission lines.
    def mask_skylines(self):
        
        # Define a nested list of wavelengths to be masked
        skylines = [[5574,5590],[5885.9,5901.9],[4276,4282],[6297,6305],[6364,6368]]
        # The inner list are ranges of wavelengths that need to be masked.
        upper = [x[1] for x in skylines]
        lower = [x[0] for x in skylines]
        # Mask between upper and lower
        mask = []
        for i in range(len(upper)):
            index = np.where((self.wl < float(upper[i])) & (self.wl >= float(lower[i])))[0] # From Hin's code
            mask.extend(index)
        
        # Set the errors to infinity. For clarity, fluxes are untouched.
        self.err[mask] = 9.99e99 # Just a very large number
        del mask, skylines, upper, lower
        
        
    # Function to mask any rest-frame features during MFICA fitting.
    def mask_pixels(self):
        """
        Generic function to mask a range of pixels during fitting. The mask 
        of features needs to be of the format [[start, finish], ...], where
        each element of the nested list is the lower and upper wavelength limit of
        the feature being masked. The provided wavelengths should correspond to
        REST-FRAME features. Eg: To mask OIII 5007, provide [[4990, 5010]]

        Returns
        -------
        None.

        """
        
        # Find upper and lower wavlengths given in mask
        upper = [x[1] for x in self.feature_mask]
        lower = [x[0] for x in self.feature_mask]
        # Mask between upper and lower
        mask = []
        for i in range(len(upper)):
            index = np.where((self.wl < float(upper[i])) & (self.wl >= float(lower[i])))[0] # From Hin's code
            mask.extend(index)
    
        # Set errors to infinitely large numbers based on the provided boolean mask
        # Pixels to be masked set to True, all else False
        self.err[mask] = 9.99e99
        del mask
        
        
    # Function to normalise the spectra by its mean
    def mn_norm(self):
        
        # Calculate the mean in the full spectrum range
        self.mean = np.mean(self.flux)
        self.flux = self.flux / self.mean
        self.err = self.err / self.mean
        
    
    # Use the model parameters to generate host galaxy reconstruction
    def gen_gal(self, params):
        """
        Method that uses the best-fitting parameters from any stage of
        fitting returned by LMFit to construct a galaxy spectrum reconstruction.

        Parameters
        ----------
        params : obj
            Parameters stored from MFICA fitting using LMFit.
            For this method, primarily the galaxy component weights

        Returns
        -------
        model : array
            The best-fit MFICA galaxy reconstruction.

        """
        
        model = (params['GAL1']*self.gspec[:,0]
        + params['GAL2']*self.gspec[:,1]
        + params['GAL3']*self.gspec[:,2]
        + params['GAL4']*self.gspec[:,3]
        + params['GAL5']*self.gspec[:,4]
        + params['GAL6']*self.gspec[:,5]
        + params['GAL7']*self.gspec[:,6])
        return model
        
    
    # Use the model parameters to generate quasar reconstruction.
    def gen_qso(self, params):
        """
        Method that uses the best-fitting parameters from any stage of
        fitting returned by LMFit to construct a quasar spectrum reconstruction.

        Parameters
        ----------
        params : obj
            Parameters stored from MFICA fitting using LMFit.
            For this method, the quasar component weights.

        Returns
        -------
        model : array
            The best-fit MFICA quasar reconstruction.

        """
        
        model = (params['QSO1']*self.qspec[:,0]
        + params['QSO2']*self.qspec[:,1]
        + params['QSO3']*self.qspec[:,2]
        + params['QSO4']*self.qspec[:,3]
        + params['QSO5']*self.qspec[:,4]
        + params['QSO6']*self.qspec[:,5]
        + params['QSO7']*self.qspec[:,6]
        + params['QSOcorr1']*self.qspec[:,7]
        + params['QSOcorr2']*self.qspec[:,8]
        + params['QSOcorr3']*self.qspec[:,9])
        return model
        
    
    # Define the model that is being fit
    def ICArecon(self, params):
        """
        Master function that generates the best-fit linear reconstruction of a spectrum
        from MFICA fitting. 
        The model fit to spectra is a linear combination of the assigned components
        and their weights, determined by "params".
        Depending on keyword arguments used in initialisation, the reconstruction
        uses both galaxy and quasar components, or one of the two.

        Parameters
        ----------
        params : obj
            Parameters returned by LMFit during MFICA fitting.

        Returns
        -------
        model : array
            Best-fitting MFICA model spectrum from LMFit least-squares minimisation.

        """
        
        if (self._use_qso_comps) and (not self._use_gal_comps):
            model = self.gen_qso(params)
            return model

        elif (not self._use_qso_comps) and (self._use_gal_comps):
            model = self.gen_gal(params)
            return model
    
        elif (self._use_qso_comps) and (self._use_gal_comps):
            # NEED TO UNDERSTAND WHY THE ADDITION THING DOESN'T WORK
            #model = self.gen_gal(params) + self.gen_qso(params)
            model = (params['GAL1']*self.gspec[:,0]
            + params['GAL2']*self.gspec[:,1]
            + params['GAL3']*self.gspec[:,2]
            + params['GAL4']*self.gspec[:,3]
            + params['GAL5']*self.gspec[:,4]
            + params['GAL6']*self.gspec[:,5]
            + params['GAL7']*self.gspec[:,6]
            + params['QSO1']*self.qspec[:,0]
            + params['QSO2']*self.qspec[:,1]
            + params['QSO3']*self.qspec[:,2]
            + params['QSO4']*self.qspec[:,3]
            + params['QSO5']*self.qspec[:,4]
            + params['QSO6']*self.qspec[:,5]
            + params['QSO7']*self.qspec[:,6]
            + params['QSOcorr1']*self.qspec[:,7]
            + params['QSOcorr2']*self.qspec[:,8]
            + params['QSOcorr3']*self.qspec[:,9])
            return model
        
    
    # Function for calculating model scaled residuals -> objective function for least squares minimisation
    def ICAfit(self, params, data, uncertainty):
        """
        Objective function (normalised residuals) for least-squares fitting with LMFit.
        Equivalent to chisquared minimisation.

        Parameters
        ----------
        params : obj
            Parameters that are used in LMFit fitting.
        data : array
            Array of flux values.
        uncertainty : array
            Array of error values.

        Returns
        -------
        array
            Normalised residuals for a model, given the data. That is, objective
            function  to be minised.

        """

        model = self.ICArecon(params)
        return (data-model) / uncertainty
    
    
    # Function that intialises parameters
    def init_params(self):
        """
        Method that constructs the LMFit "dictionary" of parameters, initial
        values and bounds used during fitting. 
        Parameters used depends on keywords used during initialisation.

        Returns
        -------
        params_initial : obj
            An LMFit "dictionary" of parameters and their initial values to
            be used during fitting.

        """
        
        params_initial = Parameters()
        if self._use_qso_comps:
            params_initial.add('QSO1', value=0.1, min=0.0, vary=True)
            params_initial.add('QSO2', value=0.1, min=0.0, vary=True)
            params_initial.add('QSO3', value=0.1, min=0.0, vary=True)
            params_initial.add('QSO4', value=0.1, min=0.0, vary=True)
            params_initial.add('QSO5', value=0.1, min=0.0, vary=True)
            params_initial.add('QSO6', value=0.1, min=0.0, vary=True)
            params_initial.add('QSO7', value=0.1, min=0.0, vary=True)
            params_initial.add('QSOcorr1', value=0.0, vary=False)
            params_initial.add('QSOcorr2', value=0.0, vary=False)
            params_initial.add('QSOcorr3', value=0.0, vary=False)
         
        elif not self._use_qso_comps:
            params_initial.add('QSO1', value=0.0, min=0.0, vary=False)
            params_initial.add('QSO2', value=0.0, min=0.0, vary=False)
            params_initial.add('QSO3', value=0.0, min=0.0, vary=False)
            params_initial.add('QSO4', value=0.0, min=0.0, vary=False)
            params_initial.add('QSO5', value=0.0, min=0.0, vary=False)
            params_initial.add('QSO6', value=0.0, min=0.0, vary=False)
            params_initial.add('QSO7', value=0.0, min=0.0, vary=False)
            params_initial.add('QSOcorr1', value=0.0,vary=False)
            params_initial.add('QSOcorr2', value=0.0,vary=False)
            params_initial.add('QSOcorr3', value=0.0,vary=False)
        
        if self._use_gal_comps:
            params_initial.add('GAL1', value=0.1, min=0.0, vary=True)  # red
            params_initial.add('GAL2', value=0.1, min=0.0, vary=True)  # PSB
            params_initial.add('GAL3', value=0.1, min=0.0, vary=True)
            params_initial.add('GAL4', value=0.1, min=0.0, vary=True)  # Emission line 1
            params_initial.add('GAL5', value=0.1, min=0.0, vary=True)  # Emission line 2
            params_initial.add('GAL6', value=0.1, min=0.0, vary=True)  # Emission line 3
            params_initial.add('GAL7', value=0.0, vary=False)
            
        elif not self._use_gal_comps:
            params_initial.add('GAL1', value=0.0, min=0.0, vary=False)  # red
            params_initial.add('GAL2', value=0.0, min=0.0, vary=False)  # PSB
            params_initial.add('GAL3', value=0.0, min=0.0, vary=False)
            params_initial.add('GAL4', value=0.0, min=0.0, vary=False)  # Emission line 1
            params_initial.add('GAL5', value=0.0, min=0.0, vary=False)  # Emission line 2
            params_initial.add('GAL6', value=0.0, min=0.0, vary=False)  # Emission line 3
            params_initial.add('GAL7', value=0.0, vary=False)
            
        return params_initial
    
    
    # First fitting iteration
    def first_fit(self):
        """
        Method that initialises parameters and runs first LMFit fitting step.

        Returns
        -------
        None.

        """
    
        params_initial = self.init_params()
        self.fit_init = minimize(self.ICAfit, params_initial, args=(self.flux, self.err))
        
    
    # Update parameters to include corrective components
    def second_params(self):
        """
        Method that updates parameters to include corrective components.

        Returns
        -------
        params : obj
            Parameters used in LMFit fitting.

        """
    
        # Need to improve fit. So change the value of correction parameters to vary within 3sigma fractional difference
        params = deepcopy(self.fit_init.params)
        
        if self._use_gal_comps:
            params['GAL7'].vary = True
        
        # QSO components
        if (self._use_qso_comps):
            
            recon_qso = self.gen_qso(params)
            qso_cont_level = np.mean(recon_qso)
            
            if (qso_cont_level > 1e-4):
            
                params['QSOcorr1'].max =  0.0291*qso_cont_level # 3 sigma limits from input spectra to MFICA
                params['QSOcorr1'].min =  -0.0291*qso_cont_level # 3 sigma limits from input spectra to MFICA
                params['QSOcorr1'].vary = True
        
                params['QSOcorr2'].max =  0.026697*qso_cont_level # 3 sigma limits from input spectra to MFICA
                params['QSOcorr2'].min =  -0.026697*qso_cont_level # 3 sigma limits from input spectra to MFICA
                params['QSOcorr2'].vary = True
        
                params['QSOcorr3'].max =  0.02512*qso_cont_level # 3 sigma limits from input spectra to MFICA
                params['QSOcorr3'].min =  -0.02512*qso_cont_level # 3 sigma limits from input spectra to MFICA
                params['QSOcorr3'].vary = True
                
        return params
    
    
    # Second fitting iteration
    def second_fit(self):
        """
        Method that runs second stage of LMFit fitting with corrective components enabled. 

        Returns
        -------
        None.

        """
    
        params = self.second_params()
        self.fit_second = minimize(self.ICAfit, params, args=(self.flux, self.err))
    
    
    # Final iteration, with redundant components switched off for error calculation.
    def final_params(self):
        
        # Set insignificant parameters to zero and switch off.
        params_new = deepcopy(self.fit_second.params)
        
        for key in params_new.keys():
            if np.abs(params_new[key].value) < 0.0001:
                params_new[key].value = 0.0
                params_new[key].vary = False

        # Switch off OB component if no AF is used.
        if self._use_gal_comps:
            if params_new['GAL2'].value == 0.0:
                params_new['GAL3'].value = 0.0
                params_new['GAL3'].vary = False

        return params_new
    
    
    # Final fitting
    def final_fit(self):
    
        params_final = self.final_params()
        mini = Minimizer(self.ICAfit, params_final, fcn_args=(self.flux, self.err))
        self.fit_final = mini.minimize()
        #model = self.ICArecon(fit_final.params) # The total/composite spectra
        
        
    # Define a function to calculate all useful quantities from the fitting.
    def useful_quant(self):
        
        # Calcuate the host fraction at 4200, 5100 and over the full spectrum.
        self.fhost_4200 = np.mean(self.model_gal[(self.wl > 4180) & (self.wl < 4220)] /
                                  self.model[(self.wl > 4180) & (self.wl < 4220)])
        self.fhost_5100 = np.mean(self.model_gal[(self.wl > 5080) & (self.wl < 5120)] /
                                  self.model[(self.wl > 5080) & (self.wl < 5120)])
        self.fhost_full = np.mean(self.model_gal / self.model)
        
        # Calculate L5100 (TO-DO)
    
    
    # Organise into a final fitting function that does all of the stuff
    def fit_comps(self):
        
        # Load components
        self.load_comps()
        
        # Apply skyline masking in observed frame
        if self.sky_mask:
            self.mask_skylines()
        
        # Shift spectra to rest-frame
        if self.z is not None:
            self.rz_corr()
        
        # Interpolate the spectra
        if self.interpolate:
            self.interp_flux()
            
        # Normalise the spectra
        self.mn_norm()
            
        # Mask rest-frame features prior to fitting
        if self.feature_mask is not None:
            self.mask_pixels()
        
        # Fit the spectra
        self.first_fit()
        self.second_fit()
        self.final_fit()
        
        # Assign to class variables
        self.model = self.ICArecon(self.fit_final.params)
        if self._use_qso_comps:
            self.model_qso = self.gen_qso(self.fit_final.params)
        if self._use_gal_comps:
            self.model_gal = self.gen_gal(self.fit_final.params)
            
        # Calculate useful quantities.
        self.useful_quant()
        
        
    # Calculate the Jacobian matrix
    def calc_jacobian(self):
    
        norm = np.mean(((self.fit_final.params['GAL1'].value * self.gspec[:,0]) 
                        +  (self.fit_final.params['GAL2'].value * self.gspec[:,1]) 
                        +  (self.fit_final.params['GAL3'].value * self.gspec[:,2])))\
            
        gal1_mean = np.mean(self.gspec[:,0])
        gal2_mean = np.mean(self.gspec[:,1])
        gal3_mean = np.mean(self.gspec[:,2])
        
        #Calculate the fractions
        f1 = self.fit_final.params['GAL1'].value * gal1_mean / norm
        f2 = self.fit_final.params['GAL2'].value * gal2_mean / norm
        f3 = self.fit_final.params['GAL3'].value * gal3_mean / norm
        
        # assemble the individual elements of the jacobian
        # diagonal elements
        j11 = (gal1_mean/norm) * (1-f1)
        j22 = (gal2_mean/norm) * (1-f2)
        j33 = (gal3_mean/norm) * (1-f3)
        
        #Off-diagonal elements
        j12 = -gal2_mean * f1 / norm
        j13 = -gal3_mean * f1 / norm
        j21 = -gal1_mean * f2 / norm
        j23 = -gal3_mean * f2 / norm
        j31 = -gal1_mean * f3 / norm
        j32 = -gal2_mean * f3 / norm
        
        # Assemble the matrix
        j_matrix = np.array([[j11, j12, j13], 
                             [j21, j22, j23],
                             [j31, j32, j33]])
        
        del gal1_mean, gal2_mean, gal3_mean
        return j_matrix
    
    
    # Calculate the covariance matrix
    def get_cov_matrix(self):
        
        # NOTE: This is currently setup only to be used for galaxy component fractions.
        
        # Load Jacobian matrix
        j_matrix = self.calc_jacobian()
        
        # Initialise and scale the covariance matrix
        cov = np.zeros((3,3))
        
        # Assigne values to covariance matrix, making sure None values are 0
        param_names = ['GAL1', 'GAL2', 'GAL3']
        for i, p_row in enumerate(param_names):
            
            # Prior to calculation, check if covariance matrix exists
            if self.fit_final.covar is None:
                break
            
            for j, p_col in enumerate(param_names):
                
                try:
                    # Find the position of these parameters in the cov matrix
                    idx_i = self.fit_final.var_names.index(p_row)
                    idx_j = self.fit_final.var_names.index(p_col)

                    # Set value of covariance matrix to the corresponding value
                    cov[i, j] = self.fit_final.covar[idx_i, idx_j]
                    
                except (ValueError, IndexError):
                    continue

        # Scale the covariance matrix using the jacobian
        cov = j_matrix @ cov @ j_matrix.T
        self.cov = cov
        
    
    # Return the component fractions
    def get_comp_frac(self):
        
        # Check that galaxy components are ON.
        if not self._use_gal_comps:
            raise Exception("Galaxy component fractions cannot be determined if galaxy components switched off!")
        
        # Calculating useful scaling factors
        norm = np.mean((self.fit_final.params['GAL1'].value * self.gspec[:,0]) 
                       + (self.fit_final.params['GAL2'].value * self.gspec[:,1]) 
                       + (self.fit_final.params['GAL3'].value * self.gspec[:,2]))
        gal1_mean = np.mean(self.gspec[:,0])
        gal2_mean = np.mean(self.gspec[:,1])
        gal3_mean = np.mean(self.gspec[:,2])

        # Obtain component fractions
        w1 = self.fit_final.params['GAL1'].value * gal1_mean / norm
        w2 = self.fit_final.params['GAL2'].value * gal2_mean / norm
        w3 = self.fit_final.params['GAL3'].value * gal3_mean / norm
        
        # Obtain the covariance matrix
        self.get_cov_matrix()
        
        # # Obtain the error bars from the covariance matrix
        idx1 = 0; idx2 = 1; idx3 = 2
        w1_err = np.sqrt(self.cov[idx1, idx1])
        w2_err = np.sqrt(self.cov[idx2, idx2])
        w3_err = np.sqrt(self.cov[idx3, idx3])
        del idx1, idx2, idx3
        
        # Store to class variables
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w1_err = w1_err
        self.w2_err = w2_err
        self.w3_err = w3_err
        
        # Also output for use
        output = {}
        output["f_k"], output["f_k_err"] = self.w1, self.w1_err
        output["f_af"], output["f_af_err"] = self.w2, self.w2_err
        output["f_ob"], output["f_ob_err"] = self.w3, self.w3_err
        return output
        
        
    # Determine the host galaxy type
    def galaxy_type(self):
        
        self.get_comp_frac()
        
        # Check that galaxy components are ON.
        if not self._use_gal_comps:
            raise Exception("Galaxy type cannot be determined if galaxy components switched off!")
        
        self.get_cov_matrix()

        # Set-up a short monte-carlo simulation of 50 iterations
        nIters = 10_000
        rng = np.random.default_rng()

        # Create 50 random re-sampling of the weights.
        rnd = rng.multivariate_normal([self.w1, self.w2], self.cov[:2,:2], size=nIters)
        w1_new = rnd[:, 0] # 10,000 different w1 values
        w2_new = rnd[:, 1] # 10,000 different w2 values
        del rnd

        # Create masks to select the different objects
        psb_select = (w2_new > -0.875*w1_new + 0.885) & (w2_new > 0.8 * w1_new - 0.53)
        sb_select = (w2_new > 0.95*w1_new-0.25) & (w2_new < -0.875*w1_new + 0.885)
        sf_select = (w2_new < 0.95*w1_new-0.25) & (w2_new < -0.875*w1_new + 0.885) & (w1_new < 0.845)
        red_select =  (w1_new > 0.94) & (w2_new < 0.8*w1_new-0.72)
        gv_select = (w1_new > 0.845) & (w2_new < 0.8*w1_new-0.53) & ((w1_new < 0.94) | (w2_new > 0.8 * w1_new - 0.72))

        # Since the masks are boolean, can just add up and take mean
        psb_count = np.mean(psb_select)
        sb_count = np.mean(sb_select)
        sf_count = np.mean(sf_select)
        gv_count = np.mean(gv_select)
        red_count = np.mean(red_select)

        # Store as a dictionary for use later
        probs = {"SB":sb_count, "SF":sf_count, "PSB":psb_count, "GV":gv_count, "Red":red_count}
        self.probs = probs

        return probs
    
    
        
    #================== PLOTTING FUNCTIONS ==================================
        
    # Plot the component fractions
    def plot_comp_frac(self, savefig=False, savename='comp_frac.png'):
        
        # Check that the galaxy components are actually switched ON
        if not self._use_gal_comps:
            raise Exception("Galaxy components need to be switched on to plot component fractions!")
        
        # Get component fractions
        _ = self.get_comp_frac()
        
        w1_gal = np.load("Components/w1_gal_5200_alt.npy")
        w2_gal = np.load("Components/w2_gal_5200_alt.npy")
        w3_gal = np.load("Components/w3_gal_5200_alt.npy")
        
        fig = plt.figure(figsize=(12,3.5), dpi=200)
        ax = fig.add_subplot(131)
        ax.scatter(w1_gal, w2_gal, s=0.3, c='lightgrey', alpha=0.3)
        ax.errorbar(x=self.w1, y=self.w2, xerr=self.w1_err, yerr=self.w2_err, marker='o', 
                    ecolor='black', elinewidth=1, markeredgecolor='black',
                    markersize=4, color='tab:orange', capsize=0.5)

        #Demarcations for boundaries for 5200 AA
        x = np.linspace(0.845, 1.2, 1000)
        y = 0.8*x-0.53
        x1 = np.linspace(0.0, 0.845, 1000)
        #y1 = -0.83*x1 + 0.84
        y1 = -0.875*x1 + 0.885
        x2 = np.linspace(0, 0.62, 1000)
        y2 = 0.95*x2-0.25
        x3 = np.linspace(0.845, 0.9, 1000)
        y3 = -1.45*x3 + 1.36
        x4 = np.linspace(0.94, 1.2, 1000)
        y4 = 0.8*x4-0.72

        # Plot all the boundaries
        ax.plot(x,y, color='black', lw=1.5, linestyle='--')
        ax.plot(x1, y1, color='black', lw=1.5, linestyle='--')
        ax.plot(x2, y2, color='black', lw=1.5, linestyle='--')
        ax.plot(x4, y4, color='black', lw=1.5, linestyle='--')
        ax.vlines(x=0.845, ymin=-0.05, ymax=0.1458, color='black', lw=1.5, linestyle='--')
        ax.vlines(x=0.94, ymin=-0.05,ymax=0.0335, color='black', lw=1.5, linestyle='--')
        ax.set_xlim([0.08, 1.05])
        ax.set_ylim([-0.05, 0.5])

        ax.yaxis.set_minor_locator(MultipleLocator(0.02))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='in')
        ax.tick_params(axis='y', which='both', left=True, right=True, direction='in')
        ax.tick_params('both', length=8, width=1, which='major')
        ax.tick_params('both', length=4, width=1, which='minor')
        ax.set_xlabel("$f_{K}$")
        ax.set_ylabel("$f_{AF}$")
        
        ax = fig.add_subplot(132)
        ax.scatter(w3_gal, w2_gal, s=0.3, c='lightgrey', alpha=0.3)
        ax.errorbar(self.w3, self.w2, xerr=self.w3_err, yerr=self.w2_err, 
                    marker='o', ecolor='black', elinewidth=1, markersize=4,
                    color='tab:orange', markeredgecolor='black', capsize=0.5)
        ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='in')
        ax.tick_params(axis='y', which='both', left=True, right=True, direction='in')
        ax.tick_params('both', length=8, width=1, which='major')
        ax.tick_params('both', length=4, width=1, which='minor')
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_xlabel("$f_{OB}$")
        ax.set_ylabel("$f_{AF}$")  
        
        ax = fig.add_subplot(133)
        ax.scatter(w1_gal, w3_gal, s=0.3, c='lightgrey', alpha=0.3)
        ax.errorbar(self.w1, self.w3, xerr=self.w1_err, yerr=self.w3_err, 
                    marker='o', markersize=4, ecolor='black', markeredgecolor='black',
                    elinewidth=1, color='tab:orange', capsize=0.5)
        ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='in')
        ax.tick_params(axis='y', which='both', left=True, right=True, direction='in')
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.tick_params('both', length=8, width=1, which='major')
        ax.tick_params('both', length=4, width=1, which='minor')
        ax.set_xlabel("$f_{K}$")
        ax.set_ylabel("$f_{OB}$")
        
        fig.tight_layout()
        plt.show()
        
        if savefig:
            fig.savefig(savename)
                    

        
    # Plot the reconstructions
    def plot_model(self, savefig=False, savename='mfica_model.png'):
        
        # Get component fractions
        # _ = self.get_comp_frac()
        
        # Set some neat plotting attributes
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 10,
            'lines.linewidth': 1.5,
            'lines.markersize': 6,
            'axes.linewidth': 2.4,  # Bold borders
            'xtick.major.size': 7,  # Major tick size
            'ytick.major.size': 7,  # Major tick size
            'xtick.minor.size': 4,  # Minor tick size
            'ytick.minor.size': 4,  # Minor tick size
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'xtick.minor.width': 1,
            'ytick.minor.width': 1,
            'errorbar.capsize': 3,  # Length of the error bar caps
        })
        
        # Make plots
        fig = plt.figure(figsize=(10,5), dpi=200)
        gs = fig.add_gridspec(6, 10)

        ax = fig.add_subplot(gs[:4,:])
        ax.set_ylabel(r'Flux (arbitrary units)')
        ax.plot(self.wl, self.flux, drawstyle='steps-mid', label='Spectrum', linewidth=0.3)
        ax.plot(self.wl, self.model, label='Total Reconstruction', color='tab:orange', lw=1.5)
        if self._use_gal_comps:
            ax.plot(self.wl, self.model_gal, label='Galaxy Reconstruction', color='tab:green', lw=1.5)
        if self._use_qso_comps:
            ax.plot(self.wl, self.model_qso, label='QSO Reconstruction', color='tab:red', lw=1.5)
        ax.xaxis.set_minor_locator(MultipleLocator(25))
        ax.xaxis.set_major_locator(MultipleLocator(250))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='in')
        ax.tick_params(axis='y', which='both', left=True, right=True, direction='in')
        ax.tick_params('both', length=7, width=1, which='major')
        ax.tick_params('both', length=3, width=1, which='minor')
        ax.legend()

        ax2 = fig.add_subplot(gs[4:,:])
        ax2.set_xlabel('Rest frame Wavelength ($\AA$)')
        ax2.set_ylabel('Residual')
        ax2.plot(self.wl, self.flux - self.model, drawstyle='steps-mid', linewidth=0.3, color='black')
        ax2.xaxis.set_minor_locator(MultipleLocator(25))
        ax2.xaxis.set_major_locator(MultipleLocator(250))
        ax2.tick_params(axis='x', which='both', bottom=True, top=True, direction='in')
        ax2.tick_params(axis='y', which='both', left=True, right=True, direction='in')
        ax2.tick_params('both', length=7, width=1, which='major')
        ax2.tick_params('both', length=3, width=1, which='minor')
        ax2.grid(True, linestyle='--')

        fig.tight_layout()
        plt.show()
        
        if savefig:
            fig.savefig(savename)
    
    
# =========================================================================#
# ============== CODE TESTING =============================================#
# =========================================================================#

if __name__ == '__main__':
    
    # Load data
    with open("obj1.json", "r") as f:
        d = json.load(f)
    
    # Load object
    obj = mfica(np.array(d['wl']), np.array(d['flux']), np.array(d['err']))
    obj.load_comps()
    
    # Fit the spectrum
    obj.fit_comps()
    obj.get_comp_frac()
    
    # Plot spectrum
    obj.plot_model()
    obj.plot_comp_frac(savefig=True)
    #print(obj.fit_final.covar)