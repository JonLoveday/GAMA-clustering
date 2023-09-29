# Multi-band LFs for lambdar catalogue

# from array import array
import copy
import math
import matplotlib as mpl
import matplotlib.lines as mlines
# import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker
# import mpmath
import numpy as np
rng = np.random.default_rng()
import numpy.ma as ma
import os
import pdb
import pickle
import scipy.optimize
from scipy import stats

# from astLib import astSED
from astropy.io import ascii
#from astropy.modeling import models, fitting
from astropy import table
from astropy.table import Table, join, hstack
# import healpy as hp
import illustris as il
import pydftools as df
from pydftools.plotting import dfplot, mfplot
# import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
# from sherpa.estmethods import Confidence
from sherpa.models.basic import Const1D, Gauss1D
from sherpa.plot import IntervalProjection, RegionProjection

import gal_sample as gs
import lf
from schec import (LogNormal, SchecMag, SchecMass, SchecMagSq, SchecMassSq,
                   SaundersMag, SaundersMass, SchecMagGen, SchecMassGen)
import util

# Global parameters
gama_data = os.environ['GAMA_DATA']
HOME = os.environ['HOME']
g3cgal = gama_data + 'g3cv9/G3CGalv08.fits'
g3cfof = gama_data + 'g3cv9/G3CFoFGroupv09.fits'
g3cmockfof = gama_data + 'g3cv6/G3CMockFoFGroupv06.fits'
g3cmockhalo = gama_data + 'g3cv6/G3CMockHaloGroupv06.fits'
g3cmockgal = gama_data + 'g3cv6/G3CMockGalv06.fits'
lf_data = os.environ['LF_DATA']
plot_dir = '/Users/loveday/Documents/tex/papers/gama/groupLF/'
kctemp = gama_data + 'kcorr_dmu/v5/kcorr_auto_z{}_vecv05.fits'

mag_label = r'$^{0.0}M_r - 5 \log_{10} h$'
ms_label = r'$\log_{10}\ ({\cal M}_*/{\cal M}_\odot h^{-2})$'

lf_label = r'$\phi(M)\ [h^3\ {\rm Mpc}^{-3}\ {\rm mag}^{-1}]$'
smf_label = r'$\phi({\cal M}_*)\ [h^3\ {\rm Mpc}^{-3}\ {\rm dex}^{-1}]$'
clf_label = r'$\phi_C(M)\ [{\rm group}^{-1}\ {\rm mag}^{-1}]$'
csmf_label = r'$\phi_C({\cal M}_*)\ [{\rm group}^{-1}\ {\rm dex}^{-1}]$'

bands = ['FUV', 'NUV', 'u', 'g', 'r', 'i', 'z', 'x', 'y', 'j', 'h', 'k',
         'w1', 'w2', 'w3', 'w4']
nband = len(bands)
solid_angle = 180*(math.pi/180)**2

# Constants
ln10 = math.log(10)
fwhm2sigma = (8*math.log(2))**-0.5
ngal_grouped = 25079
ngal_grouped_mock = 209926

metadata_conflicts = 'silent'  # Alternatives are 'warn', 'error'

# Ticks point inwards on all axes
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['mathtext.fontset'] = 'dejavusans'

np.seterr(all='warn')


def rel_errs(infile='Lambdar_kcorr_z00.fits'):
    """Relative errors in different passbands."""

    bands = ['FUV', 'NUV', 'U', 'G', 'R', 'I', 'Z', 'X', 'Y', 'J', 'H', 'K',
             'W1', 'W2', 'W3', 'W4', 'P100', 'P160', 'S250', 'S350', 'S500']
    t = Table.read(infile)
    for band in bands:
        f = t[f'{band}_FLUX']
        ferr = t[f'{band}_FLUXERR']
        sel = np.isfinite(f) * np.isfinite(ferr) * (f > 0) * (ferr > 0)
        rerr = ferr[sel]/f[sel]
        print(band, np.mean(rerr), np.median(rerr), np.std(rerr))
#         print(rerr)
        
        plt.clf()
        plt.hist(rerr, bins=np.linspace(0.0, 2.0, 100))
        plt.xlabel(f'{band} relative flux error')
        plt.show()


def maggies_rec_check(infile='Lambdar_kcorr_z00.fits', gals=np.arange(10)):
    """Compare reconstructed and observed fluxes."""

    # Wavelengths in nm from https://www.aavso.org/filters for FUV,NUV,ugrizYJHKW1-4
    wave = [152.8, 227.1, 354.3, 477.0, 623.1, 762.5, 909.7, 909.7, 1004, 1200, 1600, 2200, 3400, 4600, 12000, 22000]

    t = Table.read(infile)
    for igal in gals:
        plt.clf()
        plt.errorbar(wave[:-3], t['MAGGIES'][igal, :-3],
                     yerr=t['MAGGIES_IVAR'][igal, :-3]**-0.5)
        plt.plot(wave[:-3], t['MAGGIES_REC'][igal, :-3])
        plt.semilogy(base=10)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Maggies')
        plt.show()


def kcorr_check(infile='Lambdar_kcorr_z00.fits', ngal=10):
    """Plot some k-corrections to check they look sensible."""

    t = Table.read(infile)
    samp = rng.integers(low=0, high=len(t), size=ngal)
    ts = t[samp]

    plot_range = [0, 0.5, -5, 5]
    zp = np.linspace(0, 0.5, 50)
    plt.clf()
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, num=1)
    fig.set_size_inches((12, 8))
    fig.subplots_adjust(left=0.14, bottom=0.07, hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, 'Redshift', ha='center', va='center')
    fig.text(0.06, 0.5, 'K(z)', ha='center', va='center', rotation='vertical')
    for iband in range(nband):
        ax = axes.flat[iband]
        ax.axis(plot_range)
        ax.text(0.1, 0.9, bands[iband], transform=ax.transAxes)
        ax.scatter(t['Z_TONRY'], t['KCORR'][:, iband], s=0.01)
        for igal in range(ngal):
            kp = np.polynomial.polynomial.polyval(
                zp, ts['KCOEFFS'][igal, :, iband])
            ax.plot(zp, kp)
    plt.show()


def Vmax_calc(infile='Lambdar_kcorr_z00.fits', zmin=0.002, zmax=0.5):
    """Colour-magnitude diagram for each band."""

    cosmo = util.CosmoLookup(H0=100, omega_l=0.3, zlimits=(zmin, zmax), P=0)
    dmin = cosmo.dm(zmin)

    def vis_calc(Mrp, M, z, kcoeffs):
        """Add redshift visibility limits for sample defined by conditions."""

        def app_mag(M, z, kcoeffs):
            """Return apparent magnitude at given redshift."""
            kc = np.polynomial.polynomial.polyval(z, kcoeffs)
            return M + cosmo.dist_mod(z) + kc

        def rp_lim(z, igal):
            """r_petro < rplim"""
            return rplim - app_mag(Mrp[igal], z, kcoeffs[igal, :, iref])
        
        def mag_lim(z, igal):
            """m[iband] < mlim"""
            return mlim - app_mag(M[igal], z, kcoeffs[igal, :, iband])
        
        def z_upper(cond, igal):
            """Upper redshift limit from given condition."""
            if (cond(zmax, igal) > 0):
                zhi = zmax
            else:
                try:
                    zhi = scipy.optimize.brentq(
                        cond, z[igal], zmax, args=igal, xtol=1e-5, rtol=1e-5)
                except ValueError:
                    zhi = z[igal]
            return zhi

        V, Vmax = np.zeros(nsel), np.zeros(nsel)
        for igal in range(nsel):
            zhi = [z_upper(cond, igal) for cond in (rp_lim, mag_lim)]
            zhi = min(zhi)
            V[igal] = solid_angle/3*(cosmo.dm(z[igal])**3 - dmin**3)
            Vmax[igal] = solid_angle/3*(cosmo.dm(zhi)**3 - dmin**3)
        return V, Vmax
    
    if zmax > 0.4:
        mbright = [20, 20, 19.5, 18.5, 18, 18, 17.5, 17, 17, 17, 17, 17,
                   17, 17, 15.5, 14.5]
    if zmax < 0.1:
        mbright = [20.5, 20, 20, 19.5, 19, 19, 18.5, 18.5,
                   18.5, 18.5, 18.5, 18.5, 18.5, 18.5, 17.5, 15.5]
    nband = len(bands)
    iref = 4  # reference band
    colour_names = [rf'$({bands[iband]} - r_p)$' for iband in range(nband)]
    for iband in range(4, nband):
        colour_names[iband] = rf'$(r_p - {bands[iband]})$'
    plot_range = np.array([12, 25, -1, 6])
    rplim = 19.8
    t = Table.read(infile)
    t = t[(t['Z_TONRY'] >= zmin) * (t['Z_TONRY'] < zmax)]
    # maggies = t['MAGGIES'].T
    mags = np.ma.fix_invalid(8.9 - 2.5*np.ma.log10(t['MAGGIES'].T))
#    print(mags[:, 1], type(mags[:, 1]), mags[:, 1].count())
    r_p = t['R_PETRO']
    colours = np.ma.fix_invalid(mags - r_p)
    colours[4:, :] *= -1
    plt.clf()
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, num=1)
    fig.set_size_inches((12, 8))
    fig.subplots_adjust(left=0.14, bottom=0.07, hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, 'Mag', ha='center', va='center')
    fig.text(0.06, 0.5, 'Colour', ha='center', va='center', rotation='vertical')
    for iband in range(nband):
        mag = mags[iband, :]
        bright = ~mag.mask * (mag < mbright[iband])
        clr = colours[iband, :]
        ngal = mag.count()
        cmedian = np.ma.median(clr[bright])
        # if iband == 8:
        #     pdb.set_trace()
        if iband <= 4:
            mlim = cmedian + rplim
        else:
            mlim = rplim - cmedian
        print(cmedian, mlim)

        ax = axes.flat[iband]
        ax.axis(plot_range)
        ax.scatter(mag, clr, s=0.01)
        sel = ~mag.mask * (mag < mlim) * (r_p < rplim)
#         ax.scatter(mag[sel], clr[sel], s=0.01)
        nsel = len(mag[sel])
        print(f'{bands[iband]}: {nsel} out of {ngal} galaxies')

        # Absolute magnitudes and Vmax for selected galaxies
        cataid = t['CATAID'][sel]
        z = t['Z_TONRY'][sel]
        kcoeffs = t['KCOEFFS'][sel, :, :]
        Mrp = r_p[sel] - cosmo.dist_mod(z) - t['KCORR'][sel, iref]
        M = mag[sel] - cosmo.dist_mod(z) - t['KCORR'][sel, iband]
        Merr = 2.5*np.log10(1 + t['MAGGIES_IVAR'][sel, iband]**-0.5 / t['MAGGIES'][sel, iband])
        V, Vmax = vis_calc(Mrp, M, z, kcoeffs)

        tout = Table([cataid, M, Merr, z, cosmo.dm(z), V, Vmax],
                     names=('CATAID', 'Mag', 'Mag_err', 'z', 'r', 'V', 'Vmax'),
                     meta={'band': bands[iband], 'ngal': ngal, 'nsel': nsel,
                           'zmin': zmin, 'zmax':zmax})
        tout.write(f'Vmax_{bands[iband]}.fits', overwrite=True)
        
        ax.plot(plot_range[:2], [cmedian, cmedian])
        ax.plot([mbright[iband], mbright[iband]], [plot_range[2], plot_range[3]])
        if iband < 4:
            ax.plot(plot_range[:2], plot_range[:2] - rplim)
            ax.plot([mlim, mlim], [cmedian, plot_range[3]])
        else:
            ax.plot(plot_range[:2], rplim - plot_range[:2])
            ax.plot([mlim, mlim], [cmedian, plot_range[2]])
        ax.text(0.1, 0.9, f'{colour_names[iband]} vs {bands[iband]}',
                transform=ax.transAxes)
    plt.show()


def df_test():
    """Test pydftools."""
    n = 1000
    seed = 1234
    sigma = 0.5
    model = df.model.Schechter()
    p_true = model.p0

    data, selection, model, other = df.mockdata(
        n=n, seed=seed, sigma=sigma, model=model, verbose=True)
    survey = df.DFFit(data=data, selection=selection, model=model)
    print(survey.fit.p_best)
    fig = df.plotting.plotcov([survey], p_true=p_true, figsize=1.3)
    plt.show()
    fig, ax = df.mfplot(survey, xlim=(1e7,2e12), ylim=(1e-4,2), p_true=p_true,
                        bin_xmin=7.5, bin_xmax=12)
    plt.show()
#    display(Markdown(survey.fit_summary(format_for_notebook=True)))


def lf_all(infile='Vmax_z006/Vmax_r.fits', p0=(-2, -20.9, -1.25),
           xlim=[-24, -12], ylim=[-1, 6]):
    """LF for all galaxies."""

    t = Table.read(infile)
    data = df.Data(x=t['Mag'], x_err=t['Mag_err'])
    selection = df.selection.SelectionVeffPoints(veff=t['Vmax'], xval=t['Mag'])
    model = df.model.Schechter(p0)
    survey = df.DFFit(data=data, selection=selection, grid_dx=0.01, model=model)
    print(survey.fit.p_best)
    dfplot(survey, xlim=xlim, ylim=ylim, show_bias_correction=False)
    plt.show()



def lf_bin(infile='Vmax_z006/Vmax_r.fits', bins=np.linspace(-23, -12, 23)):
    """Binned LF for all galaxies."""

    t = Table.read(infile)
    tm = Table.read('../VisualMorphologyv03.fits')
    t = join(t, tm, keys='CATAID', join_type='left',
             metadata_conflicts=metadata_conflicts)
    tc = t['HUBBLE_TYPE_CODE']
    code = {'all': (0, 90), 'E': (1, 2), 'LBS': (2, 3), 'S0-Sa': (11, 13),
            'Sab-Scd': (13, 15), 'Sd-Irr': (15, 16)}
    phi = lf.LF(None, None, bins)
    phi.Mbin = bins[:-1] + 0.5*np.diff(bins)
    phi.comp_min = bins[0]
    phi.comp_max = bins[-1]
    plt.clf()
    ax = plt.subplot(111)
    for morph in ('all', 'E', 'LBS', 'S0-Sa', 'Sab-Scd', 'Sd-Irr'):
        if morph == 'all':
            sel = np.ones(len(t), dtype=np.bool)
        else:
            sel = (code[morph][0] <= tc) * (tc < code[morph][1])

        phi.ngal, edges = np.histogram(t['Mag'][sel], bins)
        phi.phi, edges = np.histogram(t['Mag'][sel], bins,
                                      weights=1.0/t['Vmax'][sel])
        phi.phi /= np.diff(bins)
        phi.phi_err = phi.phi/np.sqrt(phi.ngal)
    
        fn = SchecMag()
        fn.alpha = -1
        fn.Mstar = -21
        fn.lgps = -2
        phi.fn_fit(fn)
        print(fn)
        phi.plot(ax=ax, label=morph)
    plt.semilogy(base=10)
    plt.legend()
    plt.ylim(1e-5, 0.5)
    plt.xlabel('Mag')
    plt.ylabel('phi')
    plt.show()
    
#     plt.clf()
# #     plt.hist(t['Mag'], bins, weights=1.0/t['Vmax'])
#     plt.errorbar(phi.Mbin, phi.phi, phi.phi_err)
#     plt.semilogy(base=10)
