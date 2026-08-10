# Cloud-in-cell LF, where weight of a given galaxy is allocated
# proprtionately to two closest bins.
# See https://ned.ipac.caltech.edu/level5/Sept19/Springel/paper.pdf"""

from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from kcorrect.kcorrect import Kcorrect
import math
import matplotlib.pyplot as plt
import numpy as np
import pdb
import scipy.optimize
import util

responses = [# 'galex_FUV', 'galex_NUV',
            'sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0',
            'vista_z', 'vista_y', 'vista_j', 'vista_h', 'vista_k',
            'wise_w1', 'wise_w2']
# responses = ['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0']


def zdm(dmod, coeffs, zRange, Q=0, z0=0):
    """Calculate redshift z corresponding to distance modulus dmod, solves
    dmod = m - M = DM(z) + K(z) - Q(z-z0),
    ie. including k-correction and luminosity evolution Q.
    z is constrained to lie in range zRange."""

    if dmodk(zRange[0], coeffs, Q, z0) - dmod > 0:
        return zRange[0]
    if dmodk(zRange[1], coeffs, Q, z0) - dmod < 0:
        return zRange[1]
    z = scipy.optimize.brentq(lambda z: dmodk(z, coeffs, Q, z0) - dmod,
                              zRange[0], zRange[1], xtol=1e-5, rtol=1e-5)
    return z

def dmodk(z, coeffs, Q, z0):
    """Returns the K- and e-corrected distance modulus 
    DM(z) + k(z) - e(z)."""
    dm =  cosmo.distmod(z).value + kc.kcorrect(redshift=z, coeffs=coeffs, band_shift=z0)[kband] - Q*(z-z0)
    return dm

def gkv(infile='/Users/loveday/Data/gama/DR4/gkvScienceCatv02.fits',
        Mbins=np.linspace(-26, -10, 33),
        zrange=[0.002, 0.1], z0=0, Q=0, mlim=19.65):
    """CiC LF for GAMA-KiDS-VIKING (GKV) catalogues."""
    
    # fnames = ['FUVt', 'NUVt', 'ut', 'gt', 'rt', 'it',
    #           'Zt', 'Yt', 'Jt', 'Ht', 'Kt', 'W1t', 'W2t']
    fnames = ['ut', 'gt', 'rt', 'it',
              'Zt', 'Yt', 'Jt', 'Ht', 'Kt', 'W1t', 'W2t']

    kc = Kcorrect(responses=responses)
    kband = 2
    nband = len(responses)

    intbl = Table.read(infile)
    sel = (intbl['NQ'] > 2) * (intbl['Z'] > zrange[0]) * (intbl['Z'] < zrange[1]) * (intbl['SC'] >= 7)
    intbl = intbl[sel]
    ngal = len(intbl)
    redshift = intbl['Z']

    flux = np.zeros((ngal, nband))
    flux_err, ivar = np.zeros((ngal, nband)), np.zeros((ngal, nband))
    i = 0
    for fname in fnames:
        flux[:, i] = intbl[f'flux_{fname}']
        flux_err[:, i] = intbl[f'flux_err_{fname}']
        ivar[:, i] = flux_err[:, i]**-2
        i += 1

    # For missing bands, set flux and ivar both to zero
    fix = (flux > 1e10) + (flux < -900) + (flux_err <= 0)
    flux[fix] = 0
    ivar[fix] = 0
    nfix = len(flux[fix])
    print('Fixed ', len(flux[fix]), 'missing fluxes')

    # Fit SED coeffs
    coeffs = kc.fit_coeffs(redshift, flux, ivar)

    # For galaxies that couldn't be fit, use average SED of galaxies close in redshift and r-z colour
    ztol = 0.1
    rz = intbl['flux_rt']/intbl['flux_Zt']
    bad = np.nonzero(coeffs.sum(axis=-1) == 0)[0]
    nbad = len(bad)
    if nbad > 0:
        print('Replacing', nbad, 'bad fits with mean')
        for ibad in bad:
            close = np.nonzero((abs(redshift - redshift[ibad]) < ztol) *
                             (0.9 < rz[ibad]/rz) * (rz[ibad]/rz < 1.1))[0]
            flux_mean = flux[close, :].mean(axis=0)
            ivar_mean = ivar[close, :].sum(axis=0)
            coeffs[ibad, :] = kc.fit_coeffs(redshift[ibad], flux_mean, ivar_mean)
    

    # Calculate and plot the k-corrections
    k = kc.kcorrect(redshift=redshift, coeffs=coeffs, band_shift=z0)
    fig, axes = plt.subplots(3, 5, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    for iband in range(nband):
        ax = axes.flatten()[iband]
        ax.scatter(redshift, k[:, iband], s=0.1)
        ax.text(0.5, 0.8, fnames[iband], transform=ax.transAxes)
    axes[2, 2].set_xlabel('Redshift')
    axes[1, 0].set_ylabel('K-correction')
    plt.show()

    mapp = (8.9 - 2.5*np.log10(intbl['flux_rt'])).value
    mabs = mapp - cosmo.distmod(redshift).value - k[:, kband]
    zlim = [zdm(mlim - mabs[i], coeffs[i, :], zrange, Q, z0)
            for i in range(ngal)]
    Vmax = cosmo.comoving_volume(zlim).value
    wt = 1/Vmax
    bin_width = Mbins[1] - Mbins[0]
    Mcen = Mbins[:-1] + 0.5*bin_width
    nbins = len(Mcen)

    fig, axes = plt.subplots(2, 1, sharex=True, num=1, height_ratios=[1, 4])
    fig.set_size_inches(5, 5)
    fig.subplots_adjust(hspace=0, wspace=0)

    # Standard histogram LF
    N, edges = np.histogram(mabs, Mbins)
    phi, edges = np.histogram(mabs, Mbins, weights=wt)
    err = phi/N**0.5
    axes[0].stairs(N, Mbins)
    axes[0].semilogy()
    axes[1].errorbar(Mcen, phi, err)

    # CiC LF
    sel = (Mbins[0] <= mabs) * (mabs < Mbins[-1])
    mabs, wt = mabs[sel], wt[sel]
    hist, phi = np.zeros(nbins), np.zeros(nbins)
    pf = (mabs-Mbins[0])/bin_width - 0.5
    p = np.floor(pf).astype(int)
    ok = (p >= 0) * (p < nbins-1)
    pstar = pf[ok] - p[ok]
    np.add.at(hist, p[ok], (1-pstar))
    np.add.at(hist, p[ok]+1, pstar)
    np.add.at(phi, p[ok], (1-pstar)*wt[ok])
    np.add.at(phi, p[ok]+1, pstar*wt[ok])
    first = (p < 0)
    hist[0] += len(wt[first])
    phi[0] += np.sum(wt[first])
    last = (p >= nbins-1)
    hist[nbins-1] += len(wt[last])
    phi[nbins-1] += np.sum(wt[last])
    err = phi/hist**0.5

    axes[1].errorbar(Mcen, phi, err)
    axes[1].semilogy()
    axes[1].set_xlabel('Mag')
    axes[0].set_ylabel('N')
    axes[1].set_ylabel(r'$\phi$')
    plt.show()


def schec(M, Mstar, alpha, phistar):
    """Schechter function."""
    L = 10**(0.4*(Mstar - M))
    ans = 0.4 * math.log(10) * phistar * L**(alpha+1) * np.exp(-L)
    return ans


def cic_lf(mabs, wt, Mbins):
    """Cloud-in-cell LF, where weight of a given galaxy is allocated
    proprtionately to two closest bins;
    see https://ned.ipac.caltech.edu/level5/Sept19/Springel/paper.pdf.
    
    Inputs:
    mabs: array of absolute magnitudes
    wt: array of 1/Vmax (and any other) weights
    Mbins: array of bin edges
    
    Returns: hist, phi, err, Mmean, Mmean_wt
    hist: fractional number of galaxies in each bin
    phi: the LF
    err: error
    """

    nbins = len(Mbins) - 1
    bin_width = Mbins[1] - Mbins[0]
    sel = (Mbins[0] <= mabs) * (mabs < Mbins[-1])
    mabs, wt = mabs[sel], wt[sel]
    hist, phi = np.zeros(nbins), np.zeros(nbins)
    pf = (mabs-Mbins[0])/bin_width - 0.5
    p = np.floor(pf).astype(int)
    ok = (p >= 0) * (p < nbins-1)
    pstar = pf[ok] - p[ok]
    np.add.at(hist, p[ok], (1-pstar))
    np.add.at(hist, p[ok]+1, pstar)
    np.add.at(phi, p[ok], (1-pstar)*wt[ok])
    np.add.at(phi, p[ok]+1, pstar*wt[ok])
    first = (p < 0)
    hist[0] += len(wt[first])
    phi[0] += np.sum(wt[first])
    last = (p >= nbins-1)
    hist[nbins-1] += len(wt[last])
    phi[nbins-1] += np.sum(wt[last])
    err = phi/hist**0.5
    return hist, phi, err

def mean_vals(array, bins):
    """Return mean values when array is histogrammed into bins."""
    nbins = len(bins)-1
    mean = np.zeros(nbins)
    idxs = np.digitize(array, bins) - 1
    for i in range(nbins):
        sel = (idxs == i)
        mean[i] = np.mean(array[sel])
    return mean


def sim_test(infile='jswml_sim.fits'):
    """Compare CiC and historam LFs on simulated data 
    (jswml.simcat with no k-corrections or density fluctuations)."""

    intbl = Table.read(infile)
    ngal = len(intbl)
    mlim = 19.8
    zrange = [0.002, 0.65]
    Mbins=np.linspace(-24, -12, 25)

    # Interpolate distance modulus        
    cosmo = FlatLambdaCDM(H0=100, Om0=0.3, Tcmb0=2.725)
    zbins = np.linspace(*zrange, 1000)
    dmbins = cosmo.distmod(zbins).value 

    mapp = intbl['r_petro']
    redshift = intbl['z_tonry']
    area = intbl.meta['AREA']
    alpha = intbl.meta['ALPHA']
    Mstar = intbl.meta['MSTAR']
    phistar = intbl.meta['PHISTAR']
    mabs = mapp - cosmo.distmod(redshift).value
    zlim = np.interp(mlim - mabs, dmbins, zbins)
    np.clip(zlim, *zrange)
    sky_frac = area*math.pi/(4*180.0**2)
    print('sky_frac', sky_frac)
    Vmax =  sky_frac * cosmo.comoving_volume(zlim).value
    bin_width = Mbins[1] - Mbins[0]
    wt = 1/Vmax/bin_width
    Mcen = Mbins[:-1] + 0.5*bin_width
    nbins = len(Mcen)
    schec_fit_cen = schec(Mcen, Mstar, alpha, phistar)
    phi_pred = util.lf_pred(Mbins, schec, (Mstar, alpha, phistar))
    cic_pred = util.cic_lf_pred(Mbins, schec, (Mstar, alpha, phistar))
    fig, axes = plt.subplots(2, 1, sharex=True, num=1, height_ratios=[1, 4])
    fig.set_size_inches(5, 5)
    fig.subplots_adjust(hspace=0, wspace=0)

    # Standard histogram LF
    N, edges = np.histogram(mabs, Mbins)
    phi, edges = np.histogram(mabs, Mbins, weights=wt)
    Mmean = mean_vals(mabs, Mbins)
    err = phi/N**0.5
    axes[0].stairs(N, Mbins)
    axes[0].semilogy()
    # axes[1].errorbar(Mcen, phi/schec_fit_cen, err/schec_fit_cen, label='Hist cen')
    # schec_fit = schec(Mmean, Mstar, alpha, phistar)
    # axes[1].errorbar(Mcen, phi/schec_fit, err/schec_fit, label='Hist mean')
    axes[1].errorbar(Mcen, phi/phi_pred, err/phi_pred, label='Hist pred')
    # print(phi, err)

    # CiC LF
    hist, phi, err = util.cic_lf(mabs, wt, Mbins)
    print(N, N.sum())
    print(hist, hist.sum())
    print(Mcen)
    print(Mmean)
    axes[0].stairs(hist, Mbins)
    # axes[1].errorbar(Mcen, phi/schec_fit_cen, err/schec_fit_cen, label='CiC cen')
    # axes[1].errorbar(Mcen, phi/schec_fit, err/schec_fit, label='CiC mean')
    axes[1].errorbar(Mcen, phi/cic_pred, err/cic_pred, label='CiC pred')

    axes[1].axhline(y=1.0, color='k', linestyle=':')
    axes[1].set_ylim(0, 2)
    axes[1].legend()
    axes[1].set_xlabel('Mag')
    axes[0].set_ylabel('N')
    axes[1].set_ylabel(r'$\phi$/Schechter')
    plt.show()

def sim_test_multi(infile='sim_test_{}.fits', nsim=10, fmt='-'):
    """Compae CiC and historam LFs on multiple simulated data 
    (jswml.simcat with no k-corrections or density fluctuations)."""

    mlim = 19.8
    zrange = [0.002, 0.65]
    Mbins=np.linspace(-24, -12, 25)
    bin_width = Mbins[1] - Mbins[0]
    Mcen = Mbins[:-1] + 0.5*bin_width
    nbins = len(Mcen)
    capsize = 2

    N_arr = np.zeros((len(Mcen), nsim))
    phi_hist_arr = np.zeros((len(Mcen), nsim))
    phi_cic_arr = np.zeros((len(Mcen), nsim))
    Mabs_all = np.empty(1)

    # Interpolate distance modulus        
    cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
    zbins = np.linspace(*zrange, 1000)
    dmbins = cosmo.distmod(zbins).value 

    # plt.plot(zbins, dmbins, '-')
    # plt.xlabel('Redshift')
    # plt.ylabel('Distance modulus')
    # plt.show()

    for isim in range(10):
        print(isim)
        intbl = Table.read(infile.format(isim))
        ngal = len(intbl)

        mapp = intbl['r_petro']
        redshift = intbl['z_tonry']
        area = intbl.meta['AREA']
        alpha = intbl.meta['ALPHA']
        Mstar = intbl.meta['MSTAR']
        phistar = intbl.meta['PHISTAR']
        mabs = mapp - cosmo.distmod(redshift).value
        zlim = np.interp(mlim - mabs, dmbins, zbins)
        np.clip(zlim, *zrange)
        sky_frac = area*math.pi/(4*180.0**2)
        Vmax =  sky_frac * cosmo.comoving_volume(zlim).value
        wt = 1/Vmax/bin_width
        Mabs_all = np.hstack((Mabs_all, mabs))
        if isim == 0:
            phi_pred = util.lf_pred(Mbins, schec, (Mstar, alpha, phistar))
            cic_pred = util.cic_lf_pred(Mbins, schec, (Mstar, alpha, phistar))

        # Standard histogram LF
        N, edges = np.histogram(mabs, Mbins)
        phi, edges = np.histogram(mabs, Mbins, weights=wt)
        N_arr[:, isim] = N
        phi_hist_arr[:, isim] = phi

        # CiC LF
        hist, phi, err = util.cic_lf(mabs, wt, Mbins)
        phi_cic_arr[:, isim] = phi

    schec_fit_cen = schec(Mcen, Mstar, alpha, phistar)
    Mmean = mean_vals(Mabs_all, Mbins)
    print(Mcen)
    print(Mmean)
    schec_fit_mean = schec(Mmean, Mstar, alpha, phistar)

    fig, axes = plt.subplots(2, 1, sharex=True, num=1, height_ratios=[1, 4])
    fig.set_size_inches(5, 5)
    fig.subplots_adjust(hspace=0, wspace=0)

    axes[0].stairs(N_arr.sum(axis=1), Mbins)
    axes[0].semilogy()

    phi = np.mean(phi_hist_arr, axis=1)
    err = np.std(phi_hist_arr, axis=1)
    # axes[1].errorbar(Mcen, phi / schec_fit_cen, err / schec_fit_cen, label='hist cen')
    axes[1].errorbar(Mcen, phi / schec_fit_mean, err / schec_fit_mean, fmt=fmt, capsize=capsize, label='hist mean')
    axes[1].errorbar(Mcen, phi / phi_pred, err /phi_pred, fmt=fmt, capsize=capsize, label='hist pred')

    phi = np.mean(phi_cic_arr, axis=1)
    err = np.std(phi_cic_arr, axis=1)
    # axes[1].errorbar(Mcen, phi / schec_fit_cen, err / schec_fit_cen, label='CiC cen')
    axes[1].errorbar(Mcen, phi / schec_fit_mean, err / schec_fit_mean, fmt=fmt, capsize=capsize, label='CiC mean')
    axes[1].errorbar(Mcen, phi / cic_pred, err /cic_pred, fmt=fmt, capsize=capsize, label='CiC pred')
 
    axes[1].axhline(y=1.0, color='k', linestyle=':')
    axes[1].set_ylim(0, 2)
    axes[1].legend()
    axes[1].set_xlabel('Mag')
    axes[0].set_ylabel('N')
    axes[1].set_ylabel(r'$\phi$ / Schechter')
    plt.show()


def hist_tests(nran=1000000, bins=np.linspace(0, 1, 21)):
    """Simple histogram tests for various distributions over [0, 1]."""

    bin_width = bins[1] - bins[0]
    cen = bins[:-1] + 0.5*bin_width
    norm = len(cen)/nran
    nbins = len(cen)
    k = 5

    def const(x):
        return 1

    def linfun(x):
        return 2*x

    def quadfun(x):
        return 3*x**2

    def expfun(x):
        return k*np.exp(-k*x)

    def plot(fun):
        fig, axes = plt.subplots(2, 1, sharex=True, num=1, height_ratios=[1, 4])
        fig.set_size_inches(5, 5)
        fig.subplots_adjust(hspace=0, wspace=0)
        hist, edges = np.histogram(x, bins)
        err = hist**0.5
        mean = mean_vals(x, bins)
        Npred = np.zeros(nbins)
        for i in range(nbins):
            quad, err = scipy.integrate.quad(fun, bins[i], bins[i+1])
            Npred[i] = nran*quad
        print('Centres', cen)
        print('Mean   ', mean)
        print('Hist   ', hist)
        print('Npred   ', Npred)

        axes[0].stairs(hist, bins)
        # axes[0].semilogy()
        axes[1].errorbar(cen, norm*hist/fun(cen), norm*err/fun(cen), label='Centre')
        axes[1].errorbar(cen, norm*hist/fun(mean), norm*err/fun(mean), label='Mean')
        axes[1].errorbar(cen, hist/Npred, err/Npred, label='Npred')
        axes[1].axhline(y=1.0, color='k', linestyle=':')

        plt.legend()
        plt.show()

    # Constant distribution p(x) = 1
    x = np.random.rand(nran)
    plot(const)

    # Linear distribution p(x) = 2x
    x = (np.random.rand(nran))**0.5
    mean_pred = (bins[1:]**3 - bins[:-1]**3)*2/(bins[1:]**2 - bins[:-1]**2)/3
    print('mean pred', mean_pred)
    plot(linfun)

    # Quadratic distribution p(x) = 3x**2
    x = (np.random.rand(nran))**(1/3)
    plot(quadfun)

    # Exponential distribution p(x) = k exp(-kx)
    x = -np.log(1 - np.random.rand(nran))/k
    plot(expfun)
