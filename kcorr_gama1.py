# LF routines with on-the-fly K-corrections

from array import array
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpmath
import numpy as np
from numpy.polynomial import Polynomial
import os
import pdb
import pickle
import scipy.special

# from astLib import astSED
from astropy.modeling import models, fitting
from astropy import table
from astropy.table import Table, join
import healpy as hp
import kcorrect
from kcorrect.kcorrect import Kcorrect
# from sherpa.data import Data1D
# from sherpa.utils.err import EstErr
# from sherpa.fit import Fit
# from sherpa.optmethods import LevMar, NelderMead
# from sherpa.stats import Chi2
# from sherpa.estmethods import Confidence
# from sherpa.plot import IntervalProjection, RegionProjection


# import gal_sample as gs
# from schec import SchecMag
# import util

# Global parameters
lf_data = os.environ['LF_DATA']
mag_label = r'$^{0.1}M_r - 5 \log_{10} h$'
ms_label = r'$\log_{10}\ ({\cal M}_*/{\cal M}_\odot h^{-2})$'
lf_label = r'$\phi(M)\ [h^3\ {\rm Mpc}^{-3}\ {\rm mag}^{-1}]$'
metadata_conflicts = 'silent'  # Alternatives are 'warn', 'error'
# Constants
ln10 = math.log(10)

# Ticks point inwards on all axes
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True


def kcorr_init(responses_in=['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0'], 
               responses_out=['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0'],
               responses_map=['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0']):
    """Initialise the K-corrections, specifying band to be fit,
    bands to be output, and how they are mapped."""

    kc = Kcorrect(responses=responses_in,
                  responses_out=responses_out,
                  responses_map=responses_map)
    return kc


def gamaI_pan(zmin=0.001, zmax=0.5, z0=0, mlim=19.3, bands_in='UGRIZYJHK',
              bands_out='FNugrizYJHK', rband=4, pdeg=4,
              responses_in=['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0',
                            'ukirt_wfcam_Y', 'ukirt_wfcam_J', 'ukirt_wfcam_H', 'ukirt_wfcam_K'], 
              responses_out=['galex_FUV', 'galex_NUV',
                             'sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0',
                             'ukirt_wfcam_Y', 'ukirt_wfcam_J', 'ukirt_wfcam_H', 'ukirt_wfcam_K']):
    """GAMA-I panchromatic LFs, ideally match Driver+2012."""

    print('kcorrect', kcorrect.__version__)
    kc = Kcorrect(responses=responses_in)
    tc = Table.read('TilingCatv16.fits')
    print(len(tc), 'objects in tiling cat')
    tc = tc[(tc['SURVEY_CLASS'] >= 4) * (tc['NQ'] >= 3)]
    print(len(tc), 'with SC >=4, NQ >= 3')
   
    df = Table.read('DistancesFramesv06.fits')
    df = df[(df['Z_TONRY'] > zmin) * (df['Z_TONRY'] < zmax)]
    
    t = join(tc, df, keys='CATAID', metadata_conflicts=metadata_conflicts)
    print(len(t), 'with z in ', zmin, zmax)
   
    ap = Table.read('ApMatchedCatv02.fits')
    t = join(t, ap, keys='CATAID', metadata_conflicts=metadata_conflicts)
    t = t[t['MAG_AUTO_R'] < mlim]
    print(len(t), 'with r_Kron < ', mlim)

    ngal = len(t)
    nband_in = len(responses_in)
    nband_out = len(responses_out)
    redshift = t['Z_TONRY']

    flux = np.zeros((ngal, nband_in))
    flux_err, ivar = np.zeros((ngal, nband_in)), np.zeros((ngal, nband_in))
    i = 0
    for band in bands_in:
        flux[:, i] = t[f'FLUX_AUTO_{band}']
        flux_err[:, i] = t[f'FLUXERR_AUTO_{band}']
        ivar[:, i] = flux_err[:, i]**-2
        i += 1

    # For missing bands, set flux and ivar both to zero
    # fix = (flux > 1e10) + (flux < -900) + (flux_err <= 0)
    fix = (flux_err <= 0)
    flux[fix] = 0
    ivar[fix] = 0
    nfix = len(flux[fix])
    print('Fixed ', len(flux[fix]), 'missing fluxes')

    # Fit SED coeffs
    coeffs = kc.fit_coeffs(redshift, flux, ivar)

    # For galaxies that couldn't be fit, use average SED of galaxies close in redshift and r-z colour
    ztol = 0.1
    rz = t['FLUX_AUTO_R']/t['FLUX_AUTO_Z']
    bad = np.nonzero(coeffs.sum(axis=-1) == 0)[0]
    nbad = len(bad)
    if nbad > 0:
        print('Replacing', nbad, 'bad fits with mean')
        plt.clf()
        ax = plt.subplot(111)
        plt.xlabel('Band')
        plt.ylabel('Flux')
        for ibad in bad:
            close = np.nonzero((abs(redshift - redshift[ibad]) < ztol) *
                             (0.9 < rz[ibad]/rz) * (rz[ibad]/rz < 1.1))[0]
            # flux_mean = flux[close, :].mean(axis=-1)
            # ivar_mean = ivar[close, :].sum(axis=-1)
            flux_mean = flux[close, :].mean(axis=0)
            ivar_mean = ivar[close, :].sum(axis=0)
            coeffs[ibad, :] = kc.fit_coeffs(redshift[ibad], flux_mean, ivar_mean)
            color = next(ax._get_lines.prop_cycler)['color']
            plt.errorbar(range(nband_in), flux[ibad, :], yerr=flux_err[ibad, :], color=color)
            plt.plot(range(nband_in), flux_mean, color=color)
        plt.show()
    

    # Calculate and plot the k-corrections
    kc = Kcorrect(responses=responses_out)
    k = kc.kcorrect(redshift=redshift, coeffs=coeffs, band_shift=z0)
    fig, axes = plt.subplots(3, 4, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    for iband in range(nband_out):
        ax = axes.flatten()[iband]
        ax.scatter(redshift, k[:, iband], s=0.1)
        ax.text(0.5, 0.8, bands_out[iband], transform=ax.transAxes)
    axes[2, 2].set_xlabel('Redshift')
    axes[1, 0].set_ylabel('K-correction')
    plt.show()

    # Compare r-band K-corrections using mean and median coeffs
    nz = 100
    redshifts = np.linspace(zmin, zmax, nz)
    mean_coeffs = np.mean(coeffs, axis=0)
    median_coeffs = np.median(coeffs, axis=0)
    kmedian = [np.median([x[0] for x in coeffs]), np.median([x[1] for x in coeffs]), np.median([x[2] for x in coeffs]), np.median([x[3] for x in coeffs]), np.median([x[4] for x in coeffs])]
    print('mean coeffs:', mean_coeffs)
    print('median coeffs:', median_coeffs)
    print('Adrien median):', kmedian)

    plt.clf()
    plt.scatter(redshift, k[:, rband], s=0.1)

    kz = kc.kcorrect(redshift=redshifts,
                     coeffs=np.broadcast_to(mean_coeffs, (nz, 5)),
                     band_shift=z0)[:, rband]
    plt.plot(redshifts, kz, label='Mean coeffs')
    kz = kc.kcorrect(redshift=redshifts,
                     coeffs=np.broadcast_to(median_coeffs, (nz, 5)),
                     band_shift=z0)[:, rband]
    plt.plot(redshifts, kz, label='Median coeffs')
    
    plt.legend()
    plt.xlabel('Redshift')
    plt.ylabel('K_r(z)')
    plt.show()
    # pdb.set_trace()
    
    # Polynomial fits to reconstructed r-band K-correction K_r(z)
    # We fit K + 2.5 log10(1+z0) to z-z0 with constant coefficient set at zero,
    # and then set coef[0] = -2.5 log10(1+z0), so that resulting fits pass
    # through (z0, -2.5 log10(1+z0))
    pcoeffs = np.zeros((ngal, pdeg+1))
    pcoeffs[:, 0] = -2.5*np.log10(1+z0)
    nplot = 10

    plt.clf()
    ax = plt.subplot(111)
    plt.xlabel('Redshift')
    plt.ylabel('K_r(z)')
    deg = np.arange(1, pdeg+1)
    for igal in range(ngal):
        kz = kc.kcorrect(redshift=redshifts,
                         coeffs=np.broadcast_to(coeffs[igal, :], (nz, 5)),
                         band_shift=z0)
        pc = Polynomial.fit(redshifts-z0, kz[:, rband] + 2.5*np.log10(1+z0),
                            deg=deg, domain=[zmin, zmax], window=[zmin, zmax])
        pcoeffs[igal, 1:] = pc.coef[1:]
        if (igal < nplot):
            fit = pc(redshifts-z0) - 2.5*np.log10(1+z0)
            color = next(ax._get_lines.prop_cycler)['color']
            plt.scatter(redshifts, kz[:, rband], s=1, color=color)
            plt.plot(redshifts, fit, '-', color=color)
    # outtbl = Table([intbl['RAcen'], intbl['Deccen'], redshift, k, pcoeffs], names=('RA', 'DEC', 'z', 'Kcorr', 'pcoeffs'))
    # outtbl.write(outfile, overwrite=True)
    plt.show()
