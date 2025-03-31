from astropy.table import Table
from kcorrect.kcorrect import Kcorrect
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
import pdb

# responses = ['galex_FUV', 'galex_NUV',
#              'vst_u', 'vst_g', 'vst_r', 'vst_i',
#              'vista_z', 'vista_y', 'vista_j', 'vista_h', 'vista_k',
#              'wise_w1', 'wise_w2']

def kcorr_gkv(infile='gkvScienceCatv02.fits', nband=13,
              outfile='kcorr_test.fits',
              zrange=[0, 1], z0=0, pdeg=4, ntest=5000):
    """K-corrections for GAMA-KiDS-VIKING (GKV) catalogues."""

    if nband == 13:
        responses = ['galex_FUV', 'galex_NUV',
                     'sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0',
                     'vista_z', 'vista_y', 'vista_j', 'vista_h', 'vista_k',
                     'wise_w1', 'wise_w2']
        fnames = ['FUVt', 'NUVt', 'ut', 'gt', 'rt', 'it',
                  'Zt', 'Yt', 'Jt', 'Ht', 'Kt', 'W1t', 'W2t']
        rband = 4
    else:
        responses = ['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'vista_z']
        fnames = ['ut', 'gt', 'rt', 'it', 'Zt']
        rband = 2

    kc = Kcorrect(responses)

    intbl = Table.read(infile)
    sel = (intbl['NQ'] > 2) * (intbl['Z'] > zrange[0]) * (intbl['Z'] < zrange[1])
    intbl = intbl[sel]
    if ntest:
        intbl = intbl[:ntest]
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
    # if len(flux[fix]) > 0:
    #     pdb.set_trace()
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
            plt.errorbar(range(13), flux_mean, yerr=ivar_mean**-0.5, color=color)
            plt.plot(range(13), flux_mean, color=color)
        plt.show()
    

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

    # Compare r-band K-corrections using mean and median coeffs
    nz = 100
    redshifts = np.linspace(*zrange, nz)
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
    pdb.set_trace()
    
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
                            deg=deg, domain=zrange, window=zrange)
        pcoeffs[igal, 1:] = pc.coef[1:]
        if (igal < nplot):
            fit = pc(redshifts-z0) - 2.5*np.log10(1+z0)
            color = next(ax._get_lines.prop_cycler)['color']
            plt.scatter(redshifts, kz[:, rband], s=1, color=color)
            plt.plot(redshifts, fit, '-', color=color)
    outtbl = Table([intbl['RAcen'], intbl['Deccen'], redshift, k, pcoeffs], names=('RA', 'DEC', 'z', 'Kcorr', 'pcoeffs'))
    outtbl.write(outfile, overwrite=True)
    plt.show()


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