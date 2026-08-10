from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, join
import kcorrect
import kcorrect.template
from kcorrect.kcorrect import Kcorrect
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
import os
import pdb
import util

metadata_conflicts = 'silent'  # Alternatives are 'warn', 'error'

def plot_templates():
    filename = os.path.join(kcorrect.KCORRECT_DIR, 'data',
                            'templates', 'kcorrect-default-v4.fits')
    templates = kcorrect.template.Template(filename=filename)
    print(templates.restframe_wave)
    plt.clf()
    for itemp in range(5):
        plt.plot(templates.restframe_wave, templates.restframe_flux[itemp, :])
    plt.semilogy()
    plt.xlabel('Wavelenth [A]')
    plt.ylabel('Flux')
    plt.show()


def plot_response(resp_name):
    filename = os.path.join(kcorrect.KCORRECT_DIR, 'data',
                            'responses', resp_name)
    data = np.loadtxt(skiprows=1, delimeter='|')
    plt.plot(data[1, :], data[2, :])
    plt.clf()
    plt.xlabel('Wavelenth [A]')
    plt.ylabel('Response')
    plt.show()


def kfit(responses, id, redshift, flux, flux_err, refband, refclr,
         z0, pdeg, zrange, outfile, id_col='CATAID'):
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
    kc = Kcorrect(responses=responses)
    # coeffs = kc.fit_coeffs(redshift, flux, ivar)
    coeffs = np.zeros((ngal, ncoeff))
    for i, r in enumerate(redshift):
        try:
            coeffs[i, :] = kc.fit_coeffs(redshift[i], flux[i, :], ivar[i, :])
        except RuntimeError:
            print('RuntimeError i =', i)

    # For galaxies that couldn't be fit (all coeffs zero),
    # use average SED of galaxies close in redshift and ref colour
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

    outtbl = Table([id, redshift, k, coeffs, pcoeffs],
                   names=(id_col, 'Z', 'Kcorr', 'kcoeffs', 'pcoeffs'))
    outtbl.meta = {'RESPONSES': responses, 'z0': z0, 'refband': refband}
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


def kcorr_devils(z_infile='D10MasterRedshifts.fits', p_infile='D10ProFoundPhotometry.fits', outfile='kcorr.fits', nband=9,
                 zrange=[0.0, 2], z0=0, pdeg=4):
    """K-corrections for DEVILS catalogues."""

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
        responses = ['capak_cfht_megaprime_sagem_u', 'subaru_suprimecam_g', 'subaru_suprimecam_r', 'subaru_suprimecam_i', 'subaru_suprimecam_z',
                     'vista_y', 'vista_j', 'vista_h', 'vista_k']
        fnames = ['u', 'g', 'r', 'i', 'z', 'Y', 'J', 'H', 'Ks']
        refband = 5
        refclr = [2, 4]

    ztbl = Table.read(z_infile)
    print(len(ztbl), 'galaxies read')
    sel = ((ztbl['zBest'] > zrange[0]) * (ztbl['zBest'] < zrange[1]))
    ztbl = ztbl[sel]
    ngal = len(ztbl)
    print(ngal, f'galaxies in redshift range {zrange}')
    ptbl = Table.read(p_infile)
    tbl = join(ztbl, ptbl, keys='UID')
    print(len(ztbl), f'galaxies after joining with phot table')

    id_col = 'UID'
    id = tbl[id_col]
    redshift = np.array(tbl['zBest'])
    # pdb.set_trace()

    flux, flux_err = np.zeros((ngal, nband)), np.zeros((ngal, nband))
    i = 0
    for fname in fnames:
        flux[:, i] = tbl[f'flux_{fname}']
        flux_err[:, i] = tbl[f'flux_err_{fname}']
        i += 1

    kfit(responses, id, redshift, flux, flux_err, refband, refclr,
         z0, pdeg, zrange, outfile, id_col)


def kcorr_shark(infile='waves_wide_gals.parquet',
                outfile='waves_wide_kcorr.fits', nband=5,
                zrange=[0, 1], z0=0, pdeg=4):
    """K-corrections for Shark mock catalogues."""

    if nband == 13:
        responses = ['galex_FUV', 'galex_NUV',
                     'sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0',
                     'vista_z', 'vista_y', 'vista_j', 'vista_h', 'vista_k',
                     'wise_w1', 'wise_w2']
        fnames = ['FUV_GALEX', 'NUV_GALEX', 'u_VST', 'g_VST', 'r_VST', 'i_VST',
                  'Z_VISTA', 'Y_VISTA', 'J_VISTA', 'H_VISTA', 'K_VISTA',
                  'W1_WISE', 'W2_WISE']
        refband = 6
        refclr = [4, 6]
    else:
        responses = ['vst_u', 'vst_g', 'vst_r', 'vst_i', 'vista_z']
        fnames = ['u_VST', 'g_VST', 'r_VST', 'i_VST', 'Z_VISTA']
        refband = 4
        refclr = [3, 4]

    tbl = Table.read(infile)
    sel = (tbl['zobs'] > zrange[0]) * (tbl['zobs'] < zrange[1])
    tbl = tbl[sel]
    ngal = len(tbl)
    cataid = tbl['id_galaxy_sky']
    redshift = tbl['zobs']

    flux, flux_err = np.zeros((ngal, nband)), np.zeros((ngal, nband))
    i = 0
    for fname in fnames:
        mag = tbl[f'total_ap_dust_{fname}']
        good = mag > 0
        flux[good, i] = 10**(0.4*(8.9-mag[good]))
        flux_err[good, i] = 0.05*flux[good, i]
        i += 1

    kfit(responses, cataid, redshift, flux, flux_err, refband, refclr,
         z0, pdeg, zrange, outfile)

def shark_comp(infile='waves_wide_gals.parquet',
                kfile='waves_wide_kcorr.fits'):
    '''Compare SHark absolute magnitues with those predicted from K-correct.'''

    cosmo = FlatLambdaCDM(H0=67.51, Om0=0.3121)
    t1 = Table.read(infile)
    t2 = Table.read(kfile)
    t = join(t1, t2, keys_left='id_galaxy_sky', keys_right='CATAID')
    sel = t['total_ap_dust_Z_VISTA'] > 0
    t = t[sel]
    z_abs_pred = t['total_ap_dust_Z_VISTA'] - cosmo.distmod(t['zobs']).value - t['Kcorr'][:, 4]
    z_abs = t['total_ab_dust_Z_VISTA']
    plt.clf()
    plt.scatter(z_abs, z_abs_pred - z_abs, s=0.1, c=t['zobs'])
    plt.colorbar(label='Redshift')
    plt.xlabel('Shark Z_abs')
    plt.ylabel('Shark Z_app - DM - k')
    plt.show()


def kcorr_gII(infile='TilingCatv46.fits', outfile='gamaII_kcorrz01.fits',
          zrange=[0, 1], z0=0.1, pdeg=4):
    """K-corrections for GAMA-II catalogues."""

    nband = 5
    responses = ['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0']
    fnames = 'ugriz'
    refband = 2
    refclr = [2, 4]

    tbl = Table.read(infile)
    t = Table.read('ApMatchedCatv06.fits')
    t.keep_columns(['CATAID',
                    'FLUX_AUTO_u', 'FLUX_AUTO_g', 'FLUX_AUTO_r', 'FLUX_AUTO_i', 'FLUX_AUTO_z',
                    'FLUXERR_AUTO_u', 'FLUXERR_AUTO_g', 'FLUXERR_AUTO_r',
                    'FLUXERR_AUTO_i', 'FLUXERR_AUTO_z'])
    tbl = join(tbl, t, keys='CATAID', metadata_conflicts=metadata_conflicts)
    t = Table.read('GalacticExtinctionv03.fits')
    t.remove_columns(['RA', 'DEC'])
    tbl = join(tbl, t, keys='CATAID', metadata_conflicts=metadata_conflicts)
    t = Table.read('DistancesFramesv14.fits')
    t.remove_columns(['RA', 'DEC', 'NQ'])
    tbl = join(tbl, t, keys='CATAID', metadata_conflicts=metadata_conflicts)

    sel = ((tbl['SURVEY_CLASS'] > 3) * (tbl['NQ'] >= 3) *
            (tbl['Z_TONRY'] >= zrange[0]) * (tbl['Z_TONRY'] < zrange[1]))
    tbl = tbl[sel]
    ngal = len(tbl)
    cataid = tbl['CATAID']
    redshift = tbl['Z_TONRY']
    flux, flux_err = np.zeros((ngal, nband)), np.zeros((ngal, nband))

    # Extinction corrections
    i = 0
    for fname in fnames:
        flux[:, i] = tbl[f'FLUX_AUTO_{fname}'] * 10**(0.4 * tbl[f'A_{fname}'])
        flux_err[:, i] = tbl[f'FLUXERR_AUTO_{fname}']
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
