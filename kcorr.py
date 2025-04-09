from astropy.table import Table
from kcorrect.kcorrect import Kcorrect
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
import pdb
import util

def kfit(responses, cataid, redshift, flux, flux_err, refband, refclr,
         z0, pdeg, zrange, outfile):
    """Fit K-correction SED and polynomial coeffs."""

    nband = len(responses)
    ngal = len(redshift)
    ncoeff = 5

    # For missing bands, set flux and ivar both to zero
    # fix = (flux > 1e10) + (flux < -900) + (flux_err <= 0)
    ivar = flux_err**-2
    fix = (flux_err <= 0)
    flux[fix] = 0
    ivar[fix] = 0
    nfix = len(flux[fix])
    print('Fixed ', len(flux[fix]), 'missing fluxes')

    # Fit SED coeffs
    coeffs = np.zeros((ngal, ncoeff))
    kc = Kcorrect(responses=responses)
    for i, r in enumerate(redshift):
        try:
            coeffs[i, :] = kc.fit_coeffs(redshift[i], flux[i, :], ivar[i, :])
        except RuntimeError:
            pass

    # For galaxies that couldn't be fit, use average SED of galaxies close in redshift and ref colour
    ztol = 0.1
    clr = flux[:, refclr[0]]/flux[:, refclr[1]]
    bad = np.nonzero(coeffs.sum(axis=-1) == 0)[0]
    good = (coeffs.sum(axis=-1) > 0)
    nbad = len(bad)
    if nbad > 0:
        print('Replacing', nbad, 'bad fits with mean')
        for ibad in bad:
            close = np.nonzero((abs(redshift[good] - redshift[ibad]) < ztol) *
                            (0.9 < clr[ibad]/clr[good]) * (clr[ibad]/clr[good] < 1.1))[0]
            if len(close) > 0:
                coeffs[ibad, :] = np.mean(coeffs[close, :], axis=0)
            else:
                coeffs[ibad, :] = np.mean(coeffs[good, :], axis=0)

    # Calculate and plot the k-corrections
    k = kc.kcorrect(redshift=redshift, coeffs=coeffs, band_shift=z0)
    nx, ny = util.two_factors(nband)
    fig, axes = plt.subplots(nx, ny, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    for iband in range(nband):
        ax = axes.flatten()[iband]
        ax.scatter(redshift, k[:, iband], s=0.1)
        ax.text(0.5, 0.8, responses[iband], transform=ax.transAxes)
    fig.text(0.5, 0.01, 'Redshift', ha='center', va='center')
    fig.text(0.01, 0.5, 'K-correction', ha='center', va='center',
             rotation='vertical')
    plt.show()

    # Polynomial fits to reconstructed reference-band K-correction K_r(z)
    # We fit K + 2.5 log10(1+z0) to z-z0 with constant coefficient set at zero,
    # and then set coef[0] = -2.5 log10(1+z0), so that resulting fits pass
    # through (z0, -2.5 log10(1+z0))
    nz = 100
    redshifts = np.linspace(*zrange, nz)
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
        pc = Polynomial.fit(redshifts-z0, kz[:, refband] + 2.5*np.log10(1+z0),
                            deg=deg, domain=zrange, window=zrange)
        pcoeffs[igal, 1:] = pc.coef[1:]
        if (igal < nplot):
            fit = pc(redshifts-z0) - 2.5*np.log10(1+z0)
#             color = next(ax._get_lines.prop_cycler)['color']
            # plt.scatter(redshifts, kz[:, refband], s=1, color=color)
            # plt.plot(redshifts, fit, '-', color=color)
            plt.scatter(redshifts, kz[:, refband], s=1)
            plt.plot(redshifts, fit, '-')

    outtbl = Table([cataid, redshift, k, coeffs, pcoeffs],
                   names=('CATAID', 'Z', 'Kcorr', 'kcoeffs', 'pcoeffs'))
    outtbl.meta = {'RESPONSES': responses}
    outtbl.write(outfile, overwrite=True)
    plt.show()




def kcorr_gkv(infile='gkvScienceCatv02.fits', outfile='kcorr.fits', nband=5,
          zrange=[0, 1], z0=0, pdeg=4):
    """K-corrections for GAMA-KiDS-VIKING (GKV) catalogues."""

    if nband == 13:
        responses = ['galex_FUV', 'galex_NUV',
                     'sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0',
                     'vista_z', 'vista_y', 'vista_j', 'vista_h', 'vista_k',
                     'wise_w1', 'wise_w2']
        fnames = ['FUVt', 'NUVt', 'ut', 'gt', 'rt', 'it',
                  'Zt', 'Yt', 'Jt', 'Ht', 'Kt', 'W1t', 'W2t']
        refband = 4
        refclr = [4, 6]
    else:
        responses = ['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'vista_z']
        fnames = ['ut', 'gt', 'rt', 'it', 'Zt']
        refband = 2
        refclr = [2, 4]

    tbl = Table.read(infile)
    sel = ((tbl['SC'] >= 7) * (tbl['NQ'] > 2) *
           (tbl['Z'] > zrange[0]) * (tbl['Z'] < zrange[1]))
    tbl = tbl[sel]
    ngal = len(tbl)
    cataid = tbl['CATAID']
    redshift = tbl['Z']

    flux, flux_err = np.zeros((ngal, nband)), np.zeros((ngal, nband))
    i = 0
    for fname in fnames:
        flux[:, i] = tbl[f'flux_{fname}']
        flux_err[:, i] = tbl[f'flux_err_{fname}']
        i += 1

    kfit(responses, cataid, redshift, flux, flux_err, refband, refclr,
         z0, pdeg, zrange, outfile)


def par_to_dat(infile, outfile):
    """Convert response function files from .par to .dat format.
    Assumes that .par files 5 lines describing structure before data starts
    on line 6."""

    dat = np.loadtxt(infile, skiprows=5, usecols=(1, 2))
    np.savetxt(outfile, dat, fmt=(('| %7.1f', '   %8.6f')), delimiter=' | ',
               newline=' |\n', header='| lambda  |    pass    ', comments='')
    
def plot_resp(infile):
    dat = np.loadtxt(infile, skiprows=1, delimiter='|', usecols=(1, 2))
    plt.plot(dat[:,0], dat[:,1])
    plt.show()
