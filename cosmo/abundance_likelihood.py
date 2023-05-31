import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter1d
import cosmo

N_MC = 2**13

def lnlike(cosmology, M_lim, z_lim, HMF, catalog, survey_area, lnM_err=0.):
    lnM_min, lnM_max = np.log(M_lim)
    # Multiplicative mass uncertainty (systematic bias)
    M_arr = HMF['M_arr'] * cosmology['mass_bias']
    # Survey area (steradians)
    dNdlnM = HMF['dNdlnM'] * survey_area
    # Convolve with constant mass error
    if lnM_err>0.:
        Nbin = lnM_err/np.log(M_arr[1]/M_arr[0])
        dNdlnM = gaussian_filter1d(dNdlnM, Nbin, axis=1, mode='constant')
    # Total number above cuts
    dNdlnM_interp = RectBivariateSpline(HMF['z_arr'], np.log(M_arr), dNdlnM)
    N_total = dNdlnM_interp.integral(z_lim[0], z_lim[1], lnM_min, lnM_max)

    # Halo contributions
    if np.all(catalog['lnM_err']==0.):
        idx = ((catalog['z']>=z_lim[0])&(catalog['z']<=z_lim[1])&(catalog['M']>=M_lim[0])&(catalog['M']<=M_lim[1])).nonzero()[0]
        halo_contrib = np.log(dNdlnM_interp(catalog['z'][idx], np.log(catalog['M'][idx]), grid=False))
        lnlike = np.sum(halo_contrib) - N_total
    # Monte-Carlo approach
    else:
        # Set up
        rng = np.random.default_rng(int(N_total))
        # Monte-Carlo realizations [halo, MC]
        # lnMC_mass = rng.normal(np.log(catalog['M'])[:,None], catalog['lnM_err'][:,None], size=(len(catalog), N_MC))

        # Mass arrays [32, Nhalo]
        lnM_arr_halo = np.linspace(np.log(catalog['M'])-5*catalog['lnM_err'], np.log(catalog['M'])+3*catalog['lnM_err'],  32)
        P_lnM = (np.exp(-.5*(lnM_arr_halo-np.log(catalog['M']))**2/catalog['lnM_err']**2) / catalog['lnM_err'] / np.sqrt(2*np.pi) * np.exp(lnM_arr_halo)
                 * dNdlnM_interp(catalog['z'], lnM_arr_halo, grid=False))
        P_lnM_cumu = cumulative_trapezoid(P_lnM, lnM_arr_halo, axis=0)
        P_lnM_cumu/= P_lnM_cumu[-1,:]
        P_lnM_cumu = np.concatenate((np.zeros(len(catalog))[None,:], P_lnM_cumu), axis=0)
        lnMC_mass = np.array([np.interp(rng.random(N_MC), P_lnM_cumu[:,i], lnM_arr_halo[:,i]) for i in range(len(catalog))])
        lnhalo_prob = np.log(dNdlnM_interp(catalog['z'][:,None], lnMC_mass, grid=False))
        # Set halo contribution to zero outside the survey cuts
        idx = (((catalog['z']<z_lim[0])|(catalog['z']>z_lim[1]))[:,None]|(lnMC_mass<lnM_min)|(lnMC_mass>lnM_max)).nonzero()
        lnhalo_prob[idx] = 0.
        contrib_lnlike_MC = np.sum(lnhalo_prob, axis=0)
        max_ = np.amax(contrib_lnlike_MC)
        contrib_lnlike = np.log(np.sum(np.exp(contrib_lnlike_MC-max_))) + max_
        lnlike = contrib_lnlike - N_total
 

        # for n in range(N_MC):
        #     # Draw mass realization for each halo
        #     lnMC_mass = rng.normal(np.log(catalog['M']), catalog['lnM_err'])
        #     # Halo contribution is zero outside the survey cuts
        #     halo_contrib = -np.inf * np.ones(len(catalog))
        #     idx = ((catalog['z']>=z_lim[0])&(catalog['z']<=z_lim[1])&(lnMC_mass>=lnM_min)&(lnMC_mass<=lnM_max)).nonzero()[0]
        #     halo_contrib[idx] = np.log(np.array([dNdlnM_interp(catalog['z'][i], lnMC_mass[i])
        #                                          for i in idx])) 
    
        #     lnlike_MC[n] = np.sum(halo_contrib) - N_total
        # max_lnlike_MC = np.amax(lnlike_MC)
        # lnlike = np.log(np.sum(np.exp(lnlike_MC-max_lnlike_MC))) + max_lnlike_MC - np.log(N_MC)
    return lnlike

