# Cloud-in-cell LF, where weight of a given galaxy is allocated
# proprtionately to two closest bins.
# See https://ned.ipac.caltech.edu/level5/Sept19/Springel/paper.pdf"""

from astropy.cosmology import WMAP9 as cosmo
from astropy.table import Table
from kcorrect.kcorrect import Kcorrect
import matplotlib.pyplot as plt
import numpy as np
import pdb
import scipy.optimize

responses = ['galex_FUV', 'galex_NUV',
            'sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0',
            'vista_z', 'vista_y', 'vista_j', 'vista_h', 'vista_k',
            'wise_w1', 'wise_w2']

kc = Kcorrect(responses)
kband = 4

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
    
    fnames = ['FUVt', 'NUVt', 'ut', 'gt', 'rt', 'it',
              'Zt', 'Yt', 'Jt', 'Ht', 'Kt', 'W1t', 'W2t']
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
    hist = np.zeros(nbins)
    pf = (mabs-Mbins[0])/bin_width - 0.5
    p = np.floor(pf).astype(int)
    ok = (p >= 0) * (p < nbins-1)
    pstar = pf[ok] - p[ok]
    np.add.at(hist, p[ok], (1-pstar)*wt[ok])
    np.add.at(hist, p[ok]+1, pstar*wt[ok])
    first = (p < 0)
    hist[0] += np.sum(wt[first])
    last = (p >= nbins-1)
    hist[nbins-1] += np.sum(wt[last])
    axes[1].stairs(hist, Mbins)

    axes[1].semilogy()
    axes[1].set_xlabel('Mag')
    axes[0].set_ylabel('N')
    axes[1].set_ylabel(r'$\phi$')
    plt.show()

# gkv()