# Classes and functions to support galaxy target selection and
# selection function

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import scipy.optimize
import scipy.stats
from sklearn.neighbors import NearestNeighbors

from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy import table
from astropy.table import Table, join
from kcorrect.kcorrect import Kcorrect

#from pymoc import MOC
#import pymoc.util.catalog
#import pymoc.util.plot

import util

# Global parameters
gama_data = os.environ['GAMA_DATA']
tcfile = gama_data + 'TilingCatv46.fits'
kctemp = gama_data + 'kcorr_dmu/v5/kcorr_auto_z{:02d}_vecv05.fits'
bnfile = gama_data + 'BrightNeighbours.fits'
ext_file = gama_data + 'GalacticExtinctionv03.fits'
sersic_file = gama_data + 'SersicCatSDSSv09.fits'
g3cfof = gama_data + 'g3cv9/G3CFoFGroupv09.fits'
g3cmockhalo = gama_data + 'g3cv6/G3CMockHaloGroupv06.fits'
g3cgal = gama_data + 'g3cv9/G3CGalv08.fits'
g3cmockfof = gama_data + 'g3cv6/G3CMockFoFGroupv06.fits'
g3cmockgal = gama_data + 'g3cv6/G3CMockGalv06.fits'
g3csimgrp = gama_data + 'grp_sim/sim_group.fits'
g3csimgal = gama_data + 'grp_sim/sim_gal.fits'
smfile = gama_data + 'StellarMassesLambdarv20.fits'

wmax = 5.0  # max incompleteness weighting

# Jacknife regions are 4 deg segments starting at given RA
njack = 9
ra_jack = (129, 133, 137, 174, 178, 182, 211.5, 215.5, 219.5)

# Imaging completeness from Blanton et al 2005, ApJ, 631, 208, Table 1
# Modified to remove decline at bright end and to prevent negative
# completeness values at faint end
sb_tab = (18, 19, 19.46, 19.79, 20.11, 20.44, 20.76, 21.09, 21.41,
          21.74, 22.06, 22.39, 22.71, 23.04, 23.36, 23.69, 24.01,
          24.34, 26.00)
comp_tab = (1.0, 1.0, 0.99, 0.97, 0.98, 0.98, 0.98, 0.97, 0.96, 0.96,
            0.97, 0.94, 0.86, 0.84, 0.76, 0.63, 0.44, 0.33, 0.01)

metadata_conflicts = 'silent'  # Alternatives are 'warn', 'error'

# k-correction coeffs for GAMA group mocks, see Robotham+2011, eqn (8)
mock_pcoeff = (0.2085, 1.0226, 0.5237, 3.5902, 2.3843)


class Kcorr():
    """Fit and evaluate K-corrections."""

    def __init__(self, resp_in, resp_out, bands_out, redshift, flux, flux_err, z0=0,
                 pord=4, clr_def=[3, 4], n_neighbors=6):
        self.z0 = z0
        ivar = flux_err**-2

        # For missing bands, set flux and ivar both to zero
        # fix = (flux > 1e10) + (flux < -900) + (flux_err <= 0)
        fix = (flux_err <= 0)
        flux[fix] = 0
        ivar[fix] = 0
        nfix = len(flux[fix])
        print('Fixed ', len(flux[fix]), 'missing fluxes')

        # Fit SED coeffs
        kc = Kcorrect(responses=resp_in)
        coeffs = kc.fit_coeffs(redshift, flux, ivar)

        # For galaxies that couldn't be fit, use average SED of galaxies
        # close in redshift and colour
        clr = flux[:, clr_def[0]]/flux[:, clr_def[1]]
        badidxs = np.nonzero(coeffs.sum(axis=-1) == 0)[0]
        nbad = len(badidxs)
        if nbad > 0:
            print('Replacing', nbad, 'bad fits with mean')
            knn = NearestNeighbors(n_neighbors=n_neighbors)
            X = np.concatenate([redshift[:, None], clr[:, None]], axis=1)
            knn.fit(X)
            plt.clf()
            ax = plt.subplot(111)
            plt.xlabel('Band')
            plt.ylabel('Flux')
            close = knn.kneighbors(X[badidxs], return_distance=False)
            for ibad, bad in zip(range(len(badidxs)), badidxs):
                # np.nonzero((abs(redshift - redshift[ibad]) < ztol) *
                #                     (clr_tol[0] < clr[ibad]/clr) *
                #                     (clr[ibad]/clr < clr_tol[0]))[0]
                flux_mean = flux[close[ibad][1:], :].mean(axis=0)
                ivar_mean = ivar[close[ibad][1:], :].sum(axis=0)
                coeffs[bad, :] = kc.fit_coeffs(redshift[bad], flux_mean, ivar_mean)
                color = next(ax._get_lines.prop_cycler)['color']
                plt.errorbar(range(len(resp_in)), flux[ibad, :], yerr=flux_err[ibad, :], color=color)
                plt.plot(range(len(resp_in)), flux_mean, color=color)
            plt.show()

        mean_coeffs = np.mean(coeffs, axis=0)
        median_coeffs = np.median(coeffs, axis=0)

        # Calculate and plot the k-corrections
        nzp = 50
        kc = Kcorrect(responses=resp_out)
        k = kc.kcorrect(redshift=redshift, coeffs=coeffs, band_shift=z0)
        nrow, ncol = util.two_factors(len(resp_out), landscape=True)
        zp = np.linspace(np.min(redshift), np.max(redshift), nzp)
        fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0, wspace=0)
        for iband in range(len(resp_out)):
            ax = axes.flatten()[iband]
            ax.scatter(redshift, k[:, iband], s=0.1)
            kc_mean = kc.kcorrect(redshift=zp, coeffs=np.broadcast_to(mean_coeffs, (nzp, 5)), band_shift=z0)[:, iband]
            ax.plot(zp, kc_mean, color='r')
            kc_median = kc.kcorrect(redshift=zp, coeffs=np.broadcast_to(median_coeffs, (nzp, 5)), band_shift=z0)[:, iband]
            ax.plot(zp, kc_median, color='g')
            ax.text(0.5, 0.8, bands_out[iband], transform=ax.transAxes)
        axes[2, 2].set_xlabel('Redshift')
        axes[1, 0].set_ylabel('K-correction')
        plt.show()

        self.kc = kc
        self.kcorr = k
        self.coeffs = coeffs
        self.mean_coeffs = mean_coeffs
        self.median_coeffs = median_coeffs

    def __call__(self, z, igal=-1):
        if igal >= 0:
            return self.kc.kcorrect(redshift=z, coeffs=self.coeffs[igal, :],
                                   band_shift=self.z0)
        else:
            return self.kc.kcorrect(redshift=z, coeffs=self.median_coeffs,
                                   band_shift=z0)

class Ecorr():
    """Luminosity e-correction."""

    def __init__(self, z0=0, Q=0, ev_model='z'):
        self.Q = Q
        self.z0 = z0
        self.ev_model = ev_model

    def __call__(self, z, Q=None):
        if Q is None:
            Q = self.Q
        if self.ev_model == 'z':
            return self.Q*(z - self.z0)
        if self.ev_model == 'z1z':
            return self.Q*z/(1+z)


#def kcorr(z, kcoeff):
#    """K-correction from polynomial fit."""
#    return np.polynomial.polynomial.polyval(z - kz0, kcoeff)
#
#
#def ecorr(z, Q):
#    """e-correction."""
#    if ev_model == 'none':
#        try:
#            return np.zeros(len(z))
#        except TypeError:
#            return 0.0
#    if ev_model == 'z':
#        return Q*(z - ez0)
#    if ev_model == 'z1z':
#        return Q*z/(1+z)


class Magnitude():
    """Attributes and methods for galaxy magnitudes,
    including individual k- and e-corrections."""

    def __init__(self, app, z, kcoeff, cosmo, kcorr, ecorr, band='r'):
        self.app = app
        self.z = z
        self.cosmo = cosmo
        self.kcoeff = kcoeff
        self.kcorr = kcorr
        self.ecorr = ecorr
        self.band = band
        self.abs = app - cosmo.dist_mod(z) - kcorr(z, kcoeff) + ecorr(z)

    def app_calc(self, z):
        """Return apparent magnitude galaxy would have at redshift z."""
        return (self.abs + self.cosmo.dist_mod(z) + self.kcorr(z, self.kcoeff) -
                self.ecorr(z))


class SurfaceBrightness():
    """Attributes and methods for galaxy surface brightness,
    including individual k- and e-corrections."""

    def __init__(self, app, z, kcoeff, Q=0, band='r'):
        self.app = app
        self.z = z
        self.kcoeff = kcoeff
        self.Q = Q
        self.band = band
        self.abs = app - 10*np.log10(1 + z) - kcorr(z, kcoeff) + ecorr(z, Q)

    def app_calc(self, z):
        """Return apparent surface brightness galaxy would have at redshift z."""
        return (self.abs + 10*np.log10(1 + z) + kcorr(z, self.kcoeff) -
                ecorr(z, self.Q))


class GalSample():
    """Attributes and methods for a galaxy sample.
    Attributes are stored as an astropy table."""

    def __init__(self, H0=100, omega_l=0.75, Q=1, P=1, ez0=0,
                 mlimits=(0, 19.8), zlimits=(0.002, 0.65)):
        self.cosmo = util.CosmoLookup(H0, omega_l, zlimits, P=P)
        self.mlimits = list(mlimits)
        self.zlimits = list(zlimits)
        self.vol_limited = False
        self.Q = Q
        self.P = P
        self.ecorr = Ecorr(ez0, Q)
        self.comp_min = -99
        self.comp_max = 99
        self.info = {}

    def kcorr_fix(self, coeff, chi2max=10):
        """Set any missing or bad k-corrs to median values."""

        # Fit polynomial to median K(z) for good fits
        t = self.t
        nk = t[coeff].shape[1]
    #    pdb.set_trace()
        good = np.isfinite(np.sum(t[coeff], axis=1)) * (t['CHI2'] < chi2max)
        zbin = np.linspace(self.zlimits[0], self.zlimits[1], 50) - self.kcorr.z0
        k_array = np.polynomial.polynomial.polyval(
            zbin, t[coeff][good].transpose())
        k_median = np.median(k_array, axis=0)
        self.kmean = np.polynomial.polynomial.polyfit(zbin, k_median, nk-1)

        # Set any missing or bad k-corrs to median values
        bad = np.logical_not(good)
        nbad = len(t[bad])
        if nbad > 0:
            t[coeff][bad] = self.kmean
            print(nbad, 'missing/bad k-corrections replaced with mean')

    def abs_calc(self):
        """Calculate and save absolute mags."""
        self.absmag = self.appmag - self.cosmo.dist_mod(seld.z) - self.kcorr.kcorr + ecorr(z)

    def app_calc(self, z, igal):
        """Return apparent magnitude galaxy igal would have at redshift z."""
        return (self.absmag[igal, :] + self.cosmo.dist_mod(z) + self.kcorr(z, igal) -
                self.ecorr(z))

    def abs_mags(self, magname):
        """Return absolute magnitudes corresponding to magname."""
        mags = np.array([self.t[magname][i].abs for i in range(len(self.t))])
        try:
            return mags[self.use]
        except AttributeError:
            return mags

    def read_gama(self, kref=0.1, chi2max=10, nq_min=3):
        """Reads table of basic GAMA data from tiling cat & kcorr DMU."""

        global cosmo, kz0, njack, ra_jack
        njack = 9
        ra_jack = (129, 133, 137, 174, 178, 182, 211.5, 215.5, 219.5)

#        # GAMA selection limits
#        def sel_mag_lo(z, galdat):
#            """r_petro > self.mlimits[0]."""
#            return galdat['r_petro'].app_calc(z) - self.mlimits[0]
#
#        def sel_mag_hi(z, galdat):
#            """r_petro < self.mlimits[1]."""
#            return self.mlimits[1] - galdat['r_petro'].app_calc(z)
#
        tc_table = Table.read(tcfile)
        kcfile = kctemp.format(int(10*kref))
        kc_table = Table.read(kcfile)
#        omega_l = kc_table.meta['OMEGA_L']
        kz0 = kc_table.meta['Z0']
        self.kcorr = Kcorr(kz0)
        self.area = kc_table.meta['AREA'] * (math.pi/180.0)**2
#        cosmo = util.CosmoLookup(H0, omega_l, self.zlimits, P=self.P)
        t = join(tc_table, kc_table, keys='CATAID',
                 metadata_conflicts=metadata_conflicts)

        # Select reliable, main-sample galaxies in given redshift range
        sel = ((t['SURVEY_CLASS'] > 3) * (t['NQ_1'] >= nq_min) *
               (t['Z_TONRY'] >= self.zlimits[0]) *
               (t['Z_TONRY'] < self.zlimits[1]))
        t = t[sel]
        t.rename_column('Z_TONRY', 'z')
        r_petro = [Magnitude(t['R_PETRO'][i], t['z'][i], t['PCOEFF_R'][i],
                             self.cosmo, self.kcorr, self.ecorr, band='r')
                   for i in range(len(t))]

        # Copy required columns to new table
        self.t = t['CATAID', 'RA', 'DEC', 'z', 'KCORR_R', 'PCOEFF_R', 'CHI2']
        self.t['r_petro'] = r_petro
        self.kcorr_fix('PCOEFF_R')
        self.assign_jackknife('galaxies')

        # colour according to Loveday+ 2012 eqn 3
        self.t['r_abs'] = self.abs_mags('r_petro')
        grcut = 0.15 - 0.03*self.t['r_abs']
        gr = (t['G_MODEL'] - t['KCORR_G']) - (t['R_MODEL'] - t['KCORR_R'])
        self.t['colour'] = ['c']*len(t)
        sel = (gr < grcut)
        self.t['colour'][sel] = 'b'
        sel = (gr >= grcut)
        self.t['colour'][sel] = 'r'

        # Completeness weight
        imcomp = np.interp(t['R_SB'], sb_tab, comp_tab)
        zcomp = z_comp(t['FIBERMAG_R'])
        self.t['cweight'] = np.clip(1.0/(imcomp*zcomp), 1, wmax)
#        self.t['use'] = np.ones(len(self.t), dtype=np.bool)

    def read_lowz(self, infile, chi2max=10, nq_min=3):
        """Read LOWZ data."""

        global cosmo, kz0, njack, ra_jack
        ra_jack = (0, 15, 120, 160, 180, 200, 220, 240, 340, 360)
        njack = len(ra_jack)

        def z_comp(r_fibre):
            """Sigmoid function fit to redshift succcess given r_fibre,
            from misc.zcomp."""
            p = (22.42, 2.55, 2.24)
            return (1.0/(1 + np.exp(p[1]*(r_fibre-p[0]))))**p[2]

        # Select galaxies in given redshift range and that satisfy LOWZ
        # selection criteria
        t = Table.read(infile)
        ntot = len(t)
        g_mod = t['MODELMAG_G']
        r_mod = t['MODELMAG_R']
        i_mod = t['MODELMAG_I']
        r_cmod = t['CMODELMAGCOR_R']
        c_par = 0.7*(g_mod - r_mod) + 1.2*(r_mod - i_mod - 0.18)
        c_perp = np.fabs((r_mod - i_mod) - (g_mod - r_mod)/4.0 - 0.18)
        sel = ((t['Z'] >= self.zlimits[0]) * (t['Z'] < self.zlimits[1]) *
               (r_cmod >= self.mlimits[0]) * (r_cmod < self.mlimits[1]) *
               (r_cmod < 13.5 + c_par/0.3) * (c_perp < 0.2))
        t = t[sel]
        nsel = len(t)
        print(nsel, 'out of', ntot, 'galaxies selected')
        t.rename_column('Z', 'z')

        omega_l = t.meta['OMEGA_L']
        kz0 = t.meta['Z0']
        self.area = t.meta['AREA'] * (math.pi/180.0)**2
        try:
            self.Q = t.meta['Q']
            self.P = t.meta['P']
        except KeyError:
            pass
        cosmo = util.CosmoLookup(H0, omega_l, self.zlimits, P=self.P)

        kcorr_fix(t, 'PCOEFF_G', self.zlimits)
        kcorr_fix(t, 'PCOEFF_R', self.zlimits)
        kcorr_fix(t, 'PCOEFF_I', self.zlimits)
        g_model = [Magnitude(t['MODELMAG_G'][i], t['z'][i], t['PCOEFF_G'][i],
                             self.cosmo, self.kcorr, self.ecorr, band='g')
                   for i in range(len(t))]
        r_model = [Magnitude(t['MODELMAG_R'][i], t['z'][i], t['PCOEFF_R'][i],
                             self.cosmo, self.kcorr, self.ecorr, band='r')
                   for i in range(len(t))]
        i_model = [Magnitude(t['MODELMAG_I'][i], t['z'][i], t['PCOEFF_I'][i],
                             self.cosmo, self.kcorr, self.ecorr, band='i')
                   for i in range(len(t))]
        r_cmodel = [Magnitude(t['CMODELMAGCOR_R'][i], t['z'][i], t['PCOEFF_R'][i],
                              self.cosmo, self.kcorr, self.ecorr, band='r') 
                   for i in range(len(t))]
        # Copy required columns to new table
        self.t = t['RA', 'DEC', 'z', 'PCOEFF_G', 'PCOEFF_R', 'PCOEFF_I']
        self.t['g_model'] = g_model
        self.t['r_model'] = r_model
        self.t['i_model'] = i_model
        self.t['r_cmodel'] = r_cmodel

        self.assign_jackknife('galaxies')

        # Completeness weight
#        imcomp = np.interp(t['R_SB'], sb_tab, comp_tab)
#        zcomp = z_comp(t['FIBERMAG_R'])
#        self.t['cweight'] = np.clip(1.0/(imcomp*zcomp), 1, wmax)
        self.t['cweight'] = np.ones(len(self.t), dtype=np.bool)
#        self.t['use'] = np.ones(len(self.t), dtype=np.bool)

    def read_cmass(self, infile, chi2max=10, nq_min=3):
        """Read CMASS data."""

        global cosmo, kz0, njack, ra_jack
        ra_jack = (0, 15, 120, 160, 180, 200, 220, 240, 340, 360)
        njack = len(ra_jack)

        def z_comp(r_fibre):
            """Sigmoid function fit to redshift succcess given r_fibre,
            from misc.zcomp."""
            p = (22.42, 2.55, 2.24)
            return (1.0/(1 + np.exp(p[1]*(r_fibre-p[0]))))**p[2]

        # Select galaxies in given redshift range and that satisfy cmass
        # selection criteria
        t = Table.read(infile)
        ntot = len(t)
        g_mod = t['MODELMAG_G']
        r_mod = t['MODELMAG_R']
        i_mod = t['MODELMAG_I']
        i_cmod = t['CMODELMAGCOR_I']
        i_fib2 = t['FIBER2MAGCOR_I']
        d_perp = (r_mod - i_mod) - (g_mod - r_mod)/8
        sel = ((t['Z'] >= self.zlimits[0]) * (t['Z'] < self.zlimits[1]) *
               (i_cmod >= 17.5) * (i_cmod < 19.9) * (i_fib2 < 21.5) *
               (i_cmod < 19.86 + 1.6*(d_perp - 0.8)) * (d_perp > 0.55))

        t = t[sel]
        nsel = len(t)
        print(nsel, 'out of', ntot, 'galaxies selected')
        t.rename_column('Z', 'z')

        omega_l = t.meta['OMEGA_L']
        kz0 = t.meta['Z0']
        self.area = t.meta['AREA'] * (math.pi/180.0)**2
        try:
            self.Q = t.meta['Q']
            self.P = t.meta['P']
        except KeyError:
            pass
        cosmo = util.CosmoLookup(H0, omega_l, self.zlimits, P=self.P)

        kcorr_fix(t, 'PCOEFF_G', self.zlimits)
        kcorr_fix(t, 'PCOEFF_R', self.zlimits)
        kcorr_fix(t, 'PCOEFF_I', self.zlimits)
        g_model = [Magnitude(t['MODELMAG_G'][i], t['z'][i], t['PCOEFF_G'][i],
                             self.cosmo, self.kcorr, self.ecorr, band='g') for i in range(len(t))]
        r_model = [Magnitude(t['MODELMAG_R'][i], t['z'][i], t['PCOEFF_R'][i],
                             self.cosmo, self.kcorr, self.ecorr, band='r') for i in range(len(t))]
        i_model = [Magnitude(t['MODELMAG_I'][i], t['z'][i], t['PCOEFF_I'][i],
                             self.cosmo, self.kcorr, self.ecorr, band='i') for i in range(len(t))]
        i_cmodel = [Magnitude(t['CMODELMAGCOR_I'][i], t['z'][i], t['PCOEFF_I'][i],
                              self.cosmo, self.kcorr, self.ecorr, band='i') for i in range(len(t))]
        i_fib2 = [Magnitude(t['FIBER2MAGCOR_I'][i], t['z'][i], t['PCOEFF_I'][i],
                            self.cosmo, self.kcorr, self.ecorr, band='i') for i in range(len(t))]
        # Copy required columns to new table
        self.t = t['RA', 'DEC', 'z', 'PCOEFF_G', 'PCOEFF_R', 'PCOEFF_I']
        self.t['g_model'] = g_model
        self.t['r_model'] = r_model
        self.t['i_model'] = i_model
        self.t['i_cmodel'] = i_cmodel
        self.t['i_fib2'] = i_fib2

        # Finally calculate visibility limits and hence Vmax
        self.vis_calc((sel_cmass_mag_lo, sel_cmass_mag_hi, sel_cmass_fib_mag,
                       sel_cmass_ri, sel_cmass_mag_dperp, sel_cmass_dperp))
        self.vmax_calc()
        self.assign_jackknife('galaxies')

        # Completeness weight
#        imcomp = np.interp(t['R_SB'], sb_tab, comp_tab)
#        zcomp = z_comp(t['FIBERMAG_R'])
#        self.t['cweight'] = np.clip(1.0/(imcomp*zcomp), 1, wmax)
        self.t['cweight'] = np.ones(len(self.t), dtype=np.bool)
#        self.t['use'] = np.ones(len(self.t), dtype=np.bool)

    def area_calc(self, radius=1.0, order=6):
        """Calculate survey area from MOC map."""
        coords = SkyCoord(self.t['RA'], self.t['DEC'], unit='deg')
        m = pymoc.util.catalog.catalog_to_moc(coords, radius=radius, order=order)
        self.area = m.area
        print('area:', m.area)
        pymoc.util.plot.plot_moc(m)

    def add_sersic_index(self):
        """Add r-band Sersic index."""

        st = Table.read(sersic_file)
        st = st['CATAID', 'GALINDEX_r']
        self.t = join(self.t, st, keys='CATAID', join_type='left',
                      metadata_conflicts=metadata_conflicts)

    def add_sersic_phot(self):
        """Add Sersic photometry."""

        st = Table.read(sersic_file)
        et = Table.read(ext_file)
        st = join(st, et, keys='CATAID', metadata_conflicts=metadata_conflicts)
        st['R_SERSIC'] = st['GALMAG10RE_r'] - st['A_r']
        st['R_SB_SERSIC'] = st['GALMUEAVG_r'] - st['A_r']

        st = st['CATAID', 'R_SERSIC', 'R_SB_SERSIC']
        t = self.t
        z = t['z']
        t = join(t, st, keys='CATAID', join_type='left',
                 metadata_conflicts=metadata_conflicts)
        t['ABSMAG_R_SERSIC'] = (t['R_SERSIC'] - cosmo.dist_mod(z) -
                                t['KCORR_R'] + self.ecorr(z))
        t['R_SB_SERSIC_ABS'] = (t['R_SB_SERSIC'] - 10*np.log10(1 + z) -
                                t['KCORR_R'] + self.ecorr(z))

        # Exclude objects with suspect Sersic photometry (Loveday+2015, Sec 2)
        bt = Table.read(bnfile)
        t = join(t, bt, keys='CATAID', join_type='left',
                 metadata_conflicts=metadata_conflicts)
        ncand = len(t)
        sel = (t['objid'].mask *
               (np.fabs(t['R_PETRO'] - t['R_SERSIC']) < 2))
        self.t = t[sel]
        nclean = len(self.t)
        print(nclean, 'out of', ncand, 'targets with clean Sersic photometry')
#        pdb.set_trace()

    def read_grouped(self, galfile=g3cgal, grpfile=g3cfof,
                     kref=0.1, mass_est='lum', nmin=5,
                     edge_min=0.9, masscomp=False, find_vis_groups=False):
        """Read grouped gama, mock or simulated catalogues.
        Set mass_est='true' for true mock halo masses."""

#       See Robotham+2011 Sec 2.2 for k- and e- corrections
        if 'mock' in grpfile or 'Mock' in grpfile or 'sim' in grpfile:
            kz0 = 0.2
            self.kmean = mock_pcoeff
            self.kcorr = Kcorr(kz0, mock_pcoeff)
            obs = False
        else:
            obs = True

        # Read and select groups
        grp = Table.read(grpfile)
        ngrp_orig = len(grp)
        if mass_est == 'sim':
            self.meta = grp.meta
            self.meta['nmin'] = nmin
        if mass_est == 'true':
            grp['log_mass'] = np.log10(grp['HaloMass'])
            grp['Nfof'] = grp['Nhalo']
            key = 'HaloID'
        else:
            if mass_est == 'lum':
                grp['log_mass'] = 13.98 + 1.16*(np.log10(grp['LumB']) - 11.5)
            if mass_est == 'dyn':
                grp['log_mass'] = np.log10(grp['MassAfunc'])
            key = 'GroupID'

        sel = (np.array(grp['Nfof'] >= nmin) *
               np.array(grp['IterCenZ'] >= self.zlimits[0]) *
               np.array(grp['IterCenZ'] < self.zlimits[1]))
        if mass_est != 'sim':
            sel *= (np.array(grp['GroupEdge'] > edge_min) *
                    np.logical_not(grp['log_mass'].mask))
        grp = grp[sel]
#        grp.rename_column('IterCenRA', 'RA')
#        grp.rename_column('IterCenDEC', 'DEC')
#        grp.rename_column('IterCenZ', 'z')
        try:
            self.area = grp.meta['AREA'] * (math.pi/180.0)**2
        except KeyError:
            self.area = 180 * (math.pi/180.0)**2
        print(len(grp), 'out of ', ngrp_orig, ' groups selected')

        # Read galaxies
        # Don't additionally select on redshift, as that may
        # drop group membership below nmin
        gal = Table.read(galfile)
        try:
            gal.rename_column('Z', 'z')
            gal['GalID'] = gal['CATAID']
        except KeyError:
            pass
        if grpfile == g3cmockhalo:
            gal.rename_column('RankIterCenH', 'RankIterCen')
        if grpfile == g3cmockfof:
            gal.rename_column('RankIterCenF', 'RankIterCen')

        if obs:
            kcfile = kctemp.format(int(10*kref))
            kc_table = Table.read(kcfile)
            kz0 = kc_table.meta['Z0']
            self.kcorr = Kcorr(kz0)
            gal = join(gal, kc_table, keys='CATAID',
                       metadata_conflicts=metadata_conflicts)
            tc = Table.read(tcfile)
            tc = tc['CATAID', 'R_SB', 'FIBERMAG_R']
            gal = join(gal, tc, keys='CATAID',
                       metadata_conflicts=metadata_conflicts)
        else:
            gal['PCOEFF_R'] = np.tile(mock_pcoeff, (len(gal), 1))

        g = join(grp, gal, metadata_conflicts=metadata_conflicts)  # keys=key
        g['r_petro'] = [Magnitude(g['Rpetro'][i], g['z'][i], g['PCOEFF_R'][i],
                                  self.cosmo, self.kcorr, self.ecorr, band='r')
                        for i in range(len(g))]
        self.t = g
        if grpfile == g3cfof:
            self.stellar_mass()
            self.add_sersic_index()

        # First determine luminosity-based zhi, needed for group zhi
        self.vis_calc((self.sel_mag_lo, self.sel_mag_hi))
        self.t['zhi_lum'] = self.t['zhi']
        if masscomp:
            self.masscomp = masscomp
            self.mass_limit_sel()
            self.comp_limit_mass()
            self.vis_calc((self.sel_mass_hi, self.sel_mass_lo,
                           self.sel_mag_lo, self.sel_mag_hi))
        if 'mock' in grpfile or 'Mock' in grpfile or 'sim' in grpfile:
            self.t['jack'] = self.t['Volume']
        else:
            self.assign_jackknife('galaxies')
            self.kcorr_fix('PCOEFF_R')

        # Store array of masses of groups in which each galaxy would be visible
        if find_vis_groups:
            ngal = len(self.t)
            group_masses = []
            grp_z = grp['IterCenZ']
            for igal in range(ngal):
                sel = (self.t['zlo'][igal] <= grp_z) * (grp_z < self.t['zhi'][igal])
                if 'mock' in grpfile or 'sim' in grpfile:
                    sel *= self.t['Volume'][igal] == grp['Volume']
                group_masses.append(grp['log_mass'][sel])
            self.t['group_masses'] = group_masses

        # Calculate redshift limits
        # Group redshift limits correspond to that of nmin'th brightest galaxy
        # (or faintest if not all galaxies have all measurements)
        # Galaxy redshift limits then given by min(zlim_grp, zlim_gal)
        gg = self.t.group_by(key)
        idxs = gg.groups.indices
        grp['GalID'] = np.zeros(len(grp), dtype=int)
        grp['Rpetro'] = np.zeros(len(grp))
        grp['zhi'] = np.zeros(len(grp))
        for igrp in range(len(gg.groups)):
            ilo = idxs[igrp]
            ihi = idxs[igrp+1]
            idxsort = np.argsort(gg['Rpetro'][ilo:ihi])
            idx = min(nmin, len(idxsort)) - 1
            galid = gg['GalID'][ilo:ihi][idxsort][idx]
            grp['GalID'][igrp] = galid
            grp['Rpetro'][igrp] = gg['Rpetro'][ilo:ihi][idxsort][idx]
            grp['zhi'][igrp] = gg['zhi_lum'][ilo:ihi][idxsort][idx]
            for igal in range(ihi-ilo):
                gg['zhi'][ilo:ihi][igal] = min(
                        gg['zhi'][ilo:ihi][igal], grp['zhi'][igrp])

        # Select only galaxies where zhi > zmin to avoid Vmax=0
        # This will mess up group indices, so redefine them
        sel = gg['zhi'] > self.zlimits[0]
        self.t = gg[sel]
        self.t = self.t.group_by(key)
        self.grp = grp

        # Completeness weight
        if obs:
            imcomp = np.interp(self.t['R_SB'], sb_tab, comp_tab)
            zcomp = z_comp(self.t['FIBERMAG_R'])
            self.t['cweight'] = np.clip(1.0/(imcomp*zcomp), 1, wmax)
        else:
            self.t['cweight'] = np.ones(len(self.t))

    def read_groups(self, grpfile=g3cfof, mass_est='lum', nmin=5, edge_min=0.9):
        """Read gama, mock or simulated catalogue group centres.
        Set mass_est='true' for true mock halo masses."""

#       See Robotham+2011 Sec 2.2 for k- and e- corrections
        if 'mock' or 'sim' in grpfile:
            kz0 = 0.2
            self.kmean = mock_pcoeff
            self.kcorr = Kcorr(kz0, mock_pcoeff)

        # Read and select groups
        grp = Table.read(grpfile)
        ngrp_orig = len(grp)
        if mass_est == 'sim':
            self.meta = grp.meta
            self.meta['nmin'] = nmin
        if mass_est == 'true':
            grp['log_mass'] = np.log10(grp['HaloMass'])
            grp['Nfof'] = grp['Nhalo']
        else:
            if mass_est == 'lum':
                grp['log_mass'] = 13.98 + 1.16*(np.log10(grp['LumB']) - 11.5)
            if mass_est == 'dyn':
                grp['log_mass'] = np.log10(grp['MassAfunc'])

        sel = (np.array(grp['Nfof'] >= nmin) *
               np.array(grp['IterCenZ'] >= self.zlimits[0]) *
               np.array(grp['IterCenZ'] < self.zlimits[1]))
        if mass_est != 'sim':
            sel *= (np.array(grp['GroupEdge'] > edge_min) *
                    np.logical_not(grp['log_mass'].mask))
        grp = grp[sel]
        grp.rename_column('IterCenRA', 'RA')
        try:
            grp.rename_column('IterCenDec', 'DEC')
        except KeyError:
            grp.rename_column('IterCenDEC', 'DEC')
        grp.rename_column('IterCenZ', 'z')
        try:
            self.area = grp.meta['AREA'] * (math.pi/180.0)**2
        except KeyError:
            self.area = 180 * (math.pi/180.0)**2
        print(len(grp), 'out of ', ngrp_orig, ' groups selected')

        self.t = grp
        self.t['cweight'] = np.ones(len(self.t))
        self.assign_jackknife('groups')

    def read_gama_groups_old(self, nmin=5, edge_min=0.9):
        """Read data for GAMA group centres.  Group visibility limits and Vmax
        correspond to that of nmin'th ranked member."""

        # Read and select groups meeting selection criteria
        t = Table.read(g3cfof)
        t['log_mass'] = 13.98 + 1.16*(np.log10(t['LumB']) - 11.5)
        sel = (np.array(t['GroupEdge'] > edge_min) *
               np.logical_not(t['log_mass'].mask) *
               np.array(t['Nfof'] >= nmin) *
               np.array(t['IterCenZ'] >= self.zlimits[0]) *
               np.array(t['IterCenZ'] < self.zlimits[1]))
        grps = t[sel]
        grps.rename_column('IterCenRA', 'RA')
        grps.rename_column('IterCenDec', 'DEC')
        grps.rename_column('IterCenZ', 'z')
        grps = grps['GroupID', 'Nfof', 'RA', 'DEC', 'z', 'log_mass']

        # Obtain CATAID of nmin'th brightest member of each group
        gmem = Table.read(g3cgal)
        gmem = gmem['GroupID', 'CATAID', 'Rpetro']
        g = join(grps, gmem, keys='GroupID',
                 metadata_conflicts=metadata_conflicts)
        gg = g.group_by('GroupID')
        idxs = gg.groups.indices
        grps['CATAID'] = np.zeros(len(grps), dtype=int)
        for igrp in range(len(gg.groups)):
            ilo = idxs[igrp]
            ihi = idxs[igrp+1]
            idxsort = np.argsort(gg['Rpetro'][ilo:ihi])
            cataid = gg['CATAID'][ilo:ihi][idxsort][nmin-1]
            grps['CATAID'][igrp] = cataid

        # Left join groups with GAMA galaxy data
        gal = GalSample()
        gal.read_gama(nq_min=2)
        del gal.t['RA']
        del gal.t['DEC']
        del gal.t['z']
        self.t = join(grps, gal.t, keys='CATAID', join_type='left',
                      metadata_conflicts=metadata_conflicts)
#        self.z0 = gal.z0

        # Finally calculate visibility limits and hence Vmax
        self.vis_calc((sel_gama_mag_lo, sel_gama_mag_hi))
        self.vmax_calc()
        self.assign_jackknife('groups')

    def read_gama_mocks(self, infile=g3cmockgal):
        """Read gama galaxy mocks that come with group catalogue."""

#       See Robotham+2011 Sec 2.2 fopr k- and e- corrections
#        global cosmo, kz0, ez0
        kz0 = 0.2
        self.kmean = mock_pcoeff
        self.kcorr = Kcorr(kz0, mock_pcoeff)

        t = Table.read(infile)
        try:
            self.area = t.meta['AREA'] * (math.pi/180.0)**2
        except KeyError:
            self.area = 144 * (math.pi/180.0)**2

        # Select mock galaxies in given redshift range
        try:
            t.rename_column('Z', 'z')
        except KeyError:
            pass
        sel = (t['z'] >= self.zlimits[0]) * (t['z'] < self.zlimits[1])
        t = t[sel]
        r_petro = [Magnitude(t['Rpetro'][i], t['z'][i], mock_pcoeff,
                             self.cosmo, self.kcorr, self.ecorr, band='r')
                   for i in range(len(t))]

        # Copy required columns to new table
        self.t = t
        self.t['r_petro'] = r_petro
        self.t['PCOEFF_R'] = np.tile(mock_pcoeff, (len(t), 1))
        self.t['cweight'] = np.ones(len(self.t))
#        self.t['jack'] = self.t['Volume']
        self.assign_jackknife('galaxies')

    def read_gama_group_mocks_old(self, mass_est='lum', nmin=5, edge_min=0.9):
        """Read gama group mocks.  Set mass_est='true' for true halo masses."""

#       See Robotham+2011 Sec 2.2 for k- and e- corrections
        global cosmo, kz0, ez0
        kz0 = 0.2
        ez0 = 0
        pcoeff = (0.2085, 1.0226, 0.5237, 3.5902, 2.3843)
        self.kmean = pcoeff

        if mass_est == 'true':
            t = Table.read(g3cmockhalo)
            t['log_mass'] = np.log10(t['HaloMass'])
            t['Nfof'] = t['Nhalo']
            key = 'HaloID'
        else:
            t = Table.read(g3cmockfof)
            if mass_est == 'lum':
                t['log_mass'] = 13.98 + 1.16*(np.log10(t['LumB']) - 11.5)
            if mass_est == 'dyn':
                t['log_mass'] = np.log10(t['MassAfunc'])
            key = 'GroupID'

        sel = (np.array(t['GroupEdge'] > edge_min) *
               np.logical_not(t['log_mass'].mask) *
               np.array(t['Nfof'] >= nmin) *
               np.array(t['IterCenZ'] >= self.zlimits[0]) *
               np.array(t['IterCenZ'] < self.zlimits[1]))

        grps = t[sel]
        grps.rename_column('IterCenRA', 'RA')
        grps.rename_column('IterCenDEC', 'DEC')
        grps.rename_column('IterCenZ', 'z')
        grps = grps[key, 'Nfof', 'RA', 'DEC', 'z', 'log_mass', 'Volume']
        print(len(grps), 'out of ', len(t), ' groups selected')

        # Obtain GalID of nmin'th brightest member of each group
        gmem = Table.read(g3cmockgal)
        gmem = gmem['GalID', 'GroupID', 'HaloID', 'Rpetro']
        g = join(grps, gmem, keys=key,
                 metadata_conflicts=metadata_conflicts)
        gg = g.group_by(key)
        idxs = gg.groups.indices
        grps['GalID'] = np.zeros(len(grps), dtype=int)
        grps['Rpetro'] = np.zeros(len(grps))
        for igrp in range(len(gg.groups)):
            ilo = idxs[igrp]
            ihi = idxs[igrp+1]
            idxsort = np.argsort(gg['Rpetro'][ilo:ihi])
            galid = gg['GalID'][ilo:ihi][idxsort][nmin-1]
            grps['GalID'][igrp] = galid
            grps['Rpetro'][igrp] = gg['Rpetro'][ilo:ihi][idxsort][nmin-1]

        # Left join groups with mock galaxy data
#        gal = GalSample()
#        gal.read_gama(nq_min=2)
#        del gal.t['RA']
#        del gal.t['DEC']
#        del gal.t['z']
#        self.t = join(grps, gal.t, keys='CATAID', join_type='left',
#                      metadata_conflicts=metadata_conflicts)
#        self.z0 = gal.z0
#
#        # Finally calculate visibility limits and hence Vmax
#        self.vis_calc()
#        self.vmax_calc()
#        self.assign_jackknife()


#        gal = Table.read(g3cmockgal)
#        omega_l = 0.75
#        self.area = 144 * (math.pi/180.0)**2
#        cosmo = util.CosmoLookup(H0, omega_l, self.zlimits, P=self.P)
#        t = join(gal, grp,  # join_type='left',
#                 metadata_conflicts=metadata_conflicts)
##        pdb.set_trace()
#        # Select mock galaxies in given redshift range
#        sel = (t['Z'] >= self.zlimits[0]) * (t['Z'] < self.zlimits[1])
#        t = t[sel]
#        t.rename_column('Z', 'z')

        # Copy required columns to new table
        r_petro = [Magnitude(grps['Rpetro'][i], grps['z'][i], pcoeff,
                             self.cosmo, self.kcorr, self.ecorr, band='r') for i in range(len(grps))]
        grps['r_petro'] = r_petro
        grps['PCOEFF_R'] = np.tile(pcoeff, (len(grps), 1))
        grps['cweight'] = np.ones(len(grps))
#        grps['use'] = np.ones(len(grps), dtype=np.bool)
#        self.grps = grp
#        grps['Vmax_grp'] = np.zeros(len(grps))
        grps['jack'] = grps['Volume']
        self.t = grps

#    def read_gama_group_sim(self, nmin=5):
#        """Read simulated gama groups."""
#
##       See Robotham+2011 Sec 2.2 for k- and e- corrections
#        global cosmo, kz0, ez0
#        kz0 = 0.2
#        ez0 = 0
#        pcoeff = (0.2085, 1.0226, 0.5237, 3.5902, 2.3843)
#        self.kmean = pcoeff
#        self.area = 144 * (math.pi/180.0)**2
#
#        omega_l = 0.75
#        cosmo = util.CosmoLookup(H0, omega_l, self.zlimits, P=self.P)
#        t = Table.read(g3csimgrp)
#        key = 'GroupID'
#
#        sel = (np.array(t['Nfof'] >= nmin) *
#               np.array(t['z'] >= self.zlimits[0]) *
#               np.array(t['z'] < self.zlimits[1]))
#
#        grps = t[sel]
#        print(len(grps), 'out of ', len(t), ' groups selected')
#
#        # Obtain GalID of nmin'th brightest member of each group
#        gmem = Table.read(g3csimgal)
#        g = join(grps, gmem, keys=key,
#                 metadata_conflicts=metadata_conflicts)
#        gg = g.group_by(key)
#        idxs = gg.groups.indices
#        grps['GalID'] = np.zeros(len(grps), dtype=int)
#        grps['Rpetro'] = np.zeros(len(grps))
#        for igrp in range(len(gg.groups)):
#            ilo = idxs[igrp]
#            ihi = idxs[igrp+1]
#            idxsort = np.argsort(gg['Rpetro'][ilo:ihi])
#            galid = gg['GalID'][ilo:ihi][idxsort][nmin-1]
#            grps['GalID'][igrp] = galid
#            grps['Rpetro'][igrp] = gg['Rpetro'][ilo:ihi][idxsort][nmin-1]
#
#        # Copy required columns to new table
#        r_petro = [Magnitude(grps['Rpetro'][i], grps['z'][i], pcoeff,
#                             self.cosmo, self.kcorr, self.ecorr, band='r') for i in range(len(grps))]
#        grps['r_petro'] = r_petro
#        grps['PCOEFF_R'] = np.tile(pcoeff, (len(grps), 1))
#        grps['cweight'] = np.ones(len(grps))
##        grps['use'] = np.ones(len(grps), dtype=np.bool)
##        self.grps = grp
#        grps['Vmax_grp'] = np.zeros(len(grps))
#        grps['jack'] = grps['Volume']
#        self.t = grps
#
    def select(self, sel_dict=None):
        """Select galaxies that satisfy criteria in sel_dict."""

        t = self.t
        nin = len(t)
        self.use = np.ones(len(self.t), dtype=np.bool)
        if sel_dict:
            for key, limits in sel_dict.items():
                print(key, limits)
                self.use *= ((t[key] >= limits[0]) * (t[key] < limits[1]))
            for key, limits in sel_dict.items():
                try:
                    self.info.update({'mean_' + key: np.mean(t[key][self.use])})
                except TypeError:
                    pass
        self.info.update({'mean_z': np.mean(t['z'][self.use])})
        nsel = len(t[self.use])
        print(nsel, 'out of', nin, 'galaxies selected')

    def tsel(self):
        """Return table of selected galaxies."""
        try:
            return self.t[self.use]
        except AttributeError:
            return self.t

    # def vis_calc_gama(self):
    #     """Add redshift visibility limits for GAMA.
    #     This no longer works, use vis_calc() instead."""

    #     self.t['zlo'] = [self.zdm(self.mlimits[0] - self.t['ABSMAG_R'][i],
    #                      self.t['PCOEFF_R'][i])
    #                      for i in range(len(self.t))]
    #     self.t['zhi'] = [self.zdm(self.mlimits[1] - self.t['ABSMAG_R'][i],
    #                      self.t['PCOEFF_R'][i])
    #                      for i in range(len(self.t))]

    # def vis_calc_old(self, conditions):
    #     """Add redshift visibility limits for sample defined by conditions."""

    #     def z_lower(cond, igal):
    #         """Lower redshift limit from given condition."""
    #         z = self.t[igal]['z']
    #         zmin = self.zlimits[0]
    #         if (cond(zmin, igal) > 0):
    #             zlo = zmin
    #         else:
    #             try:
    #                 zlo = scipy.optimize.brentq(
    #                         cond, zmin, z,
    #                         args=igal, xtol=1e-5, rtol=1e-5)
    #             except ValueError:
    #                 zlo = z
    #         return zlo

    #     def z_upper(cond, igal):
    #         """Upper redshift limit from given condition."""
    #         z = self.t[igal]['z']
    #         zmax = self.zlimits[1]
    #         if (cond(zmax, igal) > 0):
    #             zhi = zmax
    #         else:
    #             try:
    #                 zhi = scipy.optimize.brentq(
    #                         cond, z, zmax,
    #                         args=igal, xtol=1e-5, rtol=1e-5)
    #             except ValueError:
    #                 zhi = z
    #         return zhi

    #     self.t['zlo'] = np.zeros(len(self.t))
    #     self.t['zhi'] = np.zeros(len(self.t))
    #     for igal in range(len(self.t)):
    #         zlo = [z_lower(cond, igal) for cond in conditions]
    #         zhi = [z_upper(cond, igal) for cond in conditions]
    #         self.t['zlo'][igal] = max(zlo)
    #         self.t['zhi'][igal] = min(zhi)

    def vis_calc(self, conditions):
        """Add redshift visibility limits for sample defined by conditions."""

        def z_lower(cond, igal):
            """Lower redshift limit from given condition."""
            z = self.t[igal]['z']
            zmin = self.zlimits[0]
            if (cond(zmin, igal) > 0):
                zlo = zmin
            else:
                try:
                    zlo = scipy.optimize.brentq(
                            cond, zmin, z,
                            args=igal, xtol=1e-5, rtol=1e-5)
                except ValueError:
                    zlo = z
            return zlo

        def z_upper(cond, igal):
            """Upper redshift limit from given condition."""
            z = self.t[igal]['z']
            zmax = self.zlimits[1]
            if (cond(zmax, igal) > 0):
                zhi = zmax
            else:
                try:
                    zhi = scipy.optimize.brentq(
                            cond, z, zmax,
                            args=igal, xtol=1e-5, rtol=1e-5)
                except ValueError:
                    zhi = z
            return zhi

        self.t['zlo'] = np.zeros(len(self.t))
        self.t['zhi'] = np.zeros(len(self.t))
        for igal in range(len(self.t)):
            zlo = [z_lower(cond, igal) for cond in conditions]
            zhi = [z_upper(cond, igal) for cond in conditions]
            self.t['zlo'][igal] = max(zlo)
            self.t['zhi'][igal] = min(zhi)

    def vmax_calc(self, denfile=gama_data+'radial_density.fits'):
        """Calculate standard and density-corrected Vmax values."""

        zmin, zmax = self.zlimits
        nz = 100
        zbins = np.linspace(zmin, zmax, nz)
        Vmax_raw = np.zeros(nz)
        Vmax_ec = np.zeros(nz)
        Vmax_dc = np.zeros(nz)
        Vmax_dec = np.zeros(nz)
        if denfile:
            den = Table.read(denfile)

        afac = self.area
        for iz in range(1, nz):
            zlo, zhi = zbins[iz-1], zbins[iz]
            V, err = scipy.integrate.quad(
                    self.cosmo.dV, zlo, zhi, epsabs=1e-3, epsrel=1e-3)
            Vmax_raw[iz] = Vmax_raw[iz-1] + V
            V, err = scipy.integrate.quad(
                    self.cosmo.vol_ev, zlo, zhi, epsabs=1e-3, epsrel=1e-3)
            Vmax_ec[iz] = Vmax_ec[iz-1] + V
            if denfile:
                V, err = scipy.integrate.quad(
                        lambda z: self.cosmo.dV(z) *
                        np.interp(z, den['zbin'], den['delta_av']),
                        zlo, zhi, epsabs=1e-3, epsrel=1e-3)
                Vmax_dc[iz] = Vmax_dc[iz-1] + V
                V, err = scipy.integrate.quad(
                        lambda z: self.cosmo.vol_ev(z) *
                        np.interp(z, den['zbin'], den['delta_av']),
                        zlo, zhi, epsabs=1e-3, epsrel=1e-3)
                Vmax_dec[iz] = Vmax_dec[iz-1] + V

        zlo = np.clip(self.t['zlo'], *self.zlimits)
        zhi = np.clip(self.t['zhi'], *self.zlimits)
        # pdb.set_trace()
        self.t['Vmax_raw'] = np.interp(zhi, zbins, afac*Vmax_raw)
        self.t['Vmax_ec'] = np.interp(zhi, zbins, afac*Vmax_ec)
        if denfile:
            self.t['Vmax_dc'] = np.interp(zhi, zbins, afac*Vmax_dc)
            self.t['Vmax_dec'] = np.interp(zhi, zbins, afac*Vmax_dec)
        else:
            self.t['Vmax_dc'] = self.t['Vmax_raw']
            self.t['Vmax_dec'] = self.t['Vmax_ec']

    def group_props(self, mass_est='lum', nmin=5, edge_min=0.9,
                    grpfile=g3cfof, galfile=g3cgal):
        """Add group properties, selecting only galaxies in reliable groups
        as specified by nmin and edge_min.
        Luminosity-based mass estimate is from Viola+2015, eqn (37)."""

        t = Table.read(grpfile)
        if mass_est == 'lum':
            t['log_mass'] = 13.98 + 1.16*(np.log10(t['LumB']) - 11.5)
        if mass_est == 'dyn':
            t['log_mass'] = np.log10(t['MassA'])
        sel = (np.array(t['GroupEdge'] > edge_min) *
               np.logical_not(t['log_mass'].mask) *
               np.array(t['Nfof'] >= nmin))
        grps = t[sel]
        print(len(grps), 'out of ', len(t), ' groups selected')
        gals = Table.read(galfile)
        joined = join(gals, grps, keys='GroupID',
                      metadata_conflicts=metadata_conflicts)
#        pdb.set_trace()
        self.t = join(self.t, joined, keys='CATAID',
                      metadata_conflicts=metadata_conflicts)
#        self.grps = grps
#        self.t['Vmax_grp'] = np.zeros(len(self.t))

    def mock_group_props(self, mass_est='lum', nmin=5, edge_min=0.9,
                         grpfile=g3cmockfof):
        """Add mock group properties, selecting only galaxies in reliable
        groups as specified by nmin and edge_min.
        Luminosity-based mass estimate is from Viola+2015, eqn (37)."""

        t = Table.read(grpfile)
        if mass_est == 'lum':
            t['log_mass'] = 13.98 + 1.16*(np.log10(t['LumB']) - 11.5)
        if mass_est == 'dyn':
            t['log_mass'] = np.log10(t['MassA'])
        if mass_est == 'sim':
            sel = np.array(t['Nfof'] >= nmin)
        else:
            sel = (np.array(t['GroupEdge'] > edge_min) *
                   np.logical_not(t['log_mass'].mask) *
                   np.array(t['Nfof'] >= nmin))
        grps = t[sel]
        grps = grps['Volume', 'GroupID', 'log_mass', 'Nfof', 'IterCenZ']
        print(len(grps), 'out of ', len(t), ' groups selected')
        self.t = join(self.t, grps, keys=('Volume', 'GroupID'),
                      metadata_conflicts=metadata_conflicts)
#        self.grps = grps
#        self.t['Vmax_grp'] = np.zeros(len(self.t))
        if mass_est == 'sim':
            self.meta = t.meta
            self.meta['nmin'] = nmin
            self.mlimits = [self.meta['MLO'], self.meta['MHI']]
            self.zlimits = [self.meta['ZLO'], self.meta['ZHI']]

    def vmax_group_old(self, mlo, mhi):
        """Sets self.t['Vmax_grp'] to number of groups in log mass range
        [mlo, mhi] that are within visibility limits of each galaxy."""
        self.t['Vmax_grp'] = np.zeros(len(self.t))
        ts = self.tsel()
        grps = table.unique(ts, keys='GroupID')
        grp_sel = grps[(mlo <= grps['log_mass']) * (grps['log_mass'] < mhi)]
        grp_z = grp_sel['IterCenZ']
        zlo = np.clip(ts['zlo'], *self.zlimits)
        zhi = np.clip(ts['zhi'], *self.zlimits)
#        pdb.set_trace()
        try:
            self.t['Vmax_grp'][self.use] = [
                    len(grp_sel[(zlo[j] <= grp_z) * (grp_z < zhi[j]) *
                                (ts['Volume'][j] == grp_sel['Volume'])])
                    for j in range(len(zlo))]
        except KeyError:
            self.t['Vmax_grp'][self.use] = [
                    len(grp_sel[(zlo[j] <= grp_z) * (grp_z < zhi[j])])
                    for j in range(len(zlo))]

    def vmax_group(self, mlo, mhi):
        """Sets self.t['Vmax_grp'] to number of groups in log mass range
        [mlo, mhi] that are within visibility limits of each galaxy."""
        ts = self.tsel()
        ngal = len(ts)
        vmg = np.zeros(ngal)
        for igal in range(ngal):
            gm = ts['group_masses'][igal]
            sel = (mlo <= gm) * (gm < mhi)
            vmg[igal] = len(gm[sel])
        self.t['Vmax_grp'] = np.zeros(len(self.t))
        self.t['Vmax_grp'][self.use] = vmg
#        pdb.set_trace()

    def group_limit(self, nmin, plot=False):
        """Limit grouped galaxies to those with a minimum membership of nmin.
        This may be applied after volume-limiting or other selection,
        and so group membership may be less than Nfof.
        """

        self.t['Nmem'] = np.zeros(len(self.t), dtype=np.int)
        t_by_group = self.t.group_by('GroupID')
        idxs = t_by_group.groups.indices
        for igrp in range(len(t_by_group.groups)):
            ilo, ihi = idxs[igrp], idxs[igrp+1]
            nmem = len(t_by_group.groups[igrp])
            t_by_group['Nmem'][ilo:ihi] = nmem
        sel = t_by_group['Nmem'] >= nmin
        self.t = t_by_group[sel]
        if plot:
            plt.clf()
            plt.scatter(t_by_group['Nfof'], t_by_group['Nmem'], s=0.1)
            plt.xlabel('Nfof')
            plt.ylabel('Nmem')
            plt.show()

    def stellar_mass(self, smf=smfile, fslim=(0.8, 10)):
        """Read stellar masses for GAMA."""

        m = Table.read(smf)
        m['logmstar'] -= 2*math.log10(self.cosmo._H0/70.0)
        if fslim:
            sel = (m['fluxscale'] >= fslim[0]) * (m['fluxscale'] < fslim[1])
            print(len(m[sel]), 'out of', len(m),
                  'galaxies with fluxscale in range', fslim)
            m = m[sel]
            m['logmstar'] += np.log10(m['fluxscale'])
        m['sm_g_r'] = m['absmag_g'] - m['absmag_r']

        # intrinsic g-i colour cut
        gicut = 0.07*m['logmstar'] - 0.03
        gi = m['gminusi_stars']
        m['gi_colour'] = ['c']*len(m)
        sel = (gi < gicut)
        m['gi_colour'][sel] = 'b'
        sel = (gi >= gicut)
        m['gi_colour'][sel] = 'r'

        m = m['CATAID', 'logmstar', 'sm_g_r', 'gminusi_stars', 'gi_colour']
        self.t = join(self.t, m, keys='CATAID',
                      metadata_conflicts=metadata_conflicts)

    def smf_comp(self, Mlim, dm=0.1, magname='r_petro', pc=95):
        """Return stellar mass limit that is pc percent complete to given
        absolute magnitude limit.  Uses full sample to maximize number of
        galaxies with M = Mlim +- dm."""

        t = self.t
        mags = np.array([t[magname][i].abs for i in range(len(t))])
        try:
            mass_lim = []
            for mag in Mlim:
                sel = (mag - dm <= mags) * (mags < mag + dm)
                if len(mags[sel]) > 0:
                    mass_lim.append(np.percentile(t['logmstar'][sel], pc))
                else:
                    mass_lim.append(0)

        except TypeError:
            sel = (Mlim - dm <= mags) * (mags < Mlim + dm)
            if len(mags[sel]) > 0:
                mass_lim = np.percentile(t['logmstar'][sel], pc)
            else:
                mass_lim = 0
#        pdb.set_trace()
        return mass_lim

    def specline_props(self, infile='GaussFitSimplev05.fits', snt=3):
        """Add SpecLineSFR DMU properties.
        Classify galaxies according to BPT: unknown, quiescent, starforming,
        composite or agn.  See Kewley et al 2006, MNRAS, 372, 961.
        OIII Balmer decrement from Lamastra+2009 eqn (1).
        Luminosities in units of 10^18 W to avoid overflow errors."""

        m = Table.read(gama_data + 'StellarMassesv19.fits')
        m = m['CATAID', 'Z_TONRY', 'absmag_g', 'absmag_r']
        t = Table.read(gama_data + 'SpecLineSFR/' + infile)
        s = join(t, m, keys='CATAID', metadata_conflicts=metadata_conflicts)

        # Select reference sample
        idx = s['IS_BEST']
        s = s[idx]
        nref = len(s)
        print(nref, 'spectra in ref sample')

        # Select spectroscopic sample
        sc = s['SURVEY_CODE']
        idx = ((sc == 1) + (sc == 5)) * (s['SN'] > snt)
        s = s[idx]
        nspec = len(s)
        print(nspec, 'spectra in spec sample')

        ha_ew = s['HA_EW']
        hb_ew = s['HB_EW']
        oiii_ew = s['OIIIR_EW']
        ha = s['HA_FLUX'] * (1 + 2.5/ha_ew)
        ha_err = s['HA_FLUX_ERR'] * (1 + 2.5/ha_ew)
        hb = s['HB_FLUX'] * (1 + 2.5/hb_ew)
        hb_err = s['HB_FLUX_ERR'] * (1 + 2.5/hb_ew)
        nii = s['NIIR_FLUX']
        nii_err = s['NIIR_FLUX_ERR']
        oiii = s['OIIIR_FLUX']
        oiii_err = s['OIIIR_FLUX_ERR']
        absmag_r = s['absmag_r']
        absmag_g = s['absmag_g']
        z = s['Z_TONRY']
        s['ha_lum'] = ((ha_ew + 2.5) * 10**(-0.4*(absmag_r - 34.10)) * 3.0 /
                       (6564.61*(1+z))**2 * (np.fmax(ha/hb, 2.86)/2.86)**2.36)
        s['oiii_lum'] = (oiii_ew * 10**(-0.4*(absmag_g - 34.10)) *
                         3.0/(5007*(1+z))**2 * (np.fmax(ha/hb, 3)/3)**2.94)
        s['sfr'] = s['ha_lum']/3.43e16

        # Default classification is unknown
        bpt_type = np.array(['u']*nspec)
        good_ha = (s['HA_NPEG'] == 0) * (ha/ha_err > snt)
        good_nii = (s['NIIR_NPEG'] == 0) * (nii/nii_err > snt)
        good_hb = (s['HB_NPEG'] == 0) * (hb/hb_err > snt)
        good_oiii = (s['OIIIR_NPEG'] == 0) * (oiii/oiii_err > snt)
        good_na = good_ha * good_nii
        good_ob = good_oiii * good_hb
        na = np.log10(nii/ha)
        ob = np.log10(oiii/hb)
        print(len(s[good_ha]), 'galaxies with good Halpha')
        print(len(s[good_na * good_ob]), 'galaxies with good BPT lines')

        # Identify AGN and composite spectra from BPT diagram
        # or individual line ratios
        idx = ((good_na * good_ob) * (ob > 0.61/(na - 0.05) + 1.3) *
               (ob < 0.61/(na - 0.47) + 1.19))
        bpt_type[idx] = 'c'
        idx = (good_na * good_ob) * (ob > 0.61/(na - 0.47) + 1.19)
        bpt_type[idx] = 'a'
        idx = good_na * (na > 0.2)
        bpt_type[idx] = 'a'
        idx = good_ob * (ob > 1.0)
        bpt_type[idx] = 'a'

        # Starforming
        idx = (bpt_type == 'u') * good_ha * (ha > 1e-18)
        bpt_type[idx] = 's'
        s['bpt_type'] = bpt_type

        # Add SpecLine data
        s = s['CATAID', 'ha_lum', 'oiii_lum', 'sfr', 'bpt_type']
        self.t = join(self.t, s, keys='CATAID', join_type='left',
                      metadata_conflicts=metadata_conflicts)

        print(len(bpt_type[bpt_type == 'a']), 'AGN')
        print(len(bpt_type[bpt_type == 'c']), 'composite')
        print(len(bpt_type[bpt_type == 's']), 'star-forming')
        print(len(bpt_type[bpt_type == 'u']), 'unclassified')
#        colour = {'a': 'r', 'c': 'g', 's': 'b', 'u': 'k'}
        show = np.array(good_na * good_ob)
        clr = np.array([{'a': 'r', 'c': 'g', 's': 'b', 'u': 'k'}[type] for
                        type in bpt_type])

        plt.clf()
        plt.hist((ha/ha_err, hb/hb_err, nii/nii_err, oiii/oiii_err),
                 range=(-1, 10))
    #    plt.hist(hb/hb_err, range=(-1, 10), alpha=0.5, label=r'H$\beta$')
    #    plt.hist(nii/nii_err, range=(-1, 10), alpha=0.5, label=r'NII')
    #    plt.hist(oiii/oiii_err, range=(-1, 10), alpha=0.5, label=r'OIII')
        plt.xlabel(r'H$\alpha$, H$\beta$, NII, OIII s/n')
        plt.ylabel(r'$N$')
    #    plt.legend()
        plt.show()

        plt.clf()
        plt.scatter(na[show], ob[show], s=0.01, c=clr[show], edgecolors='face')
        plt.axis([-2, 1, -1.2, 1.5])
        plt.xlabel(r'log([NII]/H$\alpha$)')
        plt.ylabel(r'log([OIII]/H$\beta$)')
        plt.show()

        plt.clf()
        plt.scatter(z[show], s['sfr'][show], s=0.01, c=clr[show],
                    edgecolors='face')
        plt.axis([0, 0.35, 5e-3, 500])
        plt.xlabel(r'$z$')
        plt.ylabel(r'SFR')
        plt.semilogy(base=10, nonpositive='clip')
        plt.show()

    def agn_class(self, snt=3):
        """Classify AGN a la Gordon+2017 sec 3.1."""

        gfs_file = gama_data + 'SpecLineSFR/GaussFitSimplev05.fits'
        gfc_file = gama_data + 'SpecLineSFR/GaussFitComplexv05.fits'

        # m = Table.read(gama_data + 'StellarMassesv19.fits')
        # m = m['CATAID', 'Z_TONRY', 'absmag_g', 'absmag_r']
        ss = Table.read(gfs_file)
        ss = ss['SPECID', 'CATAID', 'Z', 'SURVEY_CODE', 'IS_BEST']
        # s = join(t, m, keys='CATAID', metadata_conflicts=metadata_conflicts)

        # Select best GAMA + SDSS spectra within acceptable redshift limits
        z = ss['Z']
        sc = ss['SURVEY_CODE']
        idx = (ss['IS_BEST'] * ((sc == 1) + (sc == 5)) * (z < 0.3) *
               ((z < 0.157) + (z > 0.163)) * ((z < 0.170) + (z > 0.175)))
        ss = ss[idx]
        nref = len(ss)
        print(nref, 'selected spectra')

        sc = Table.read(gfc_file)
        s = join(ss, sc, keys='SPECID', metadata_conflicts=metadata_conflicts)
        s['agn_type'] = 0
        
        # Type 1
        sel = ((s['HA_MODSEL_EMB_EM'] > 200) * (s['HB_MODSEL_EMB_EM'] > 200) *
               (s['HA_B_FLUX'] > s['HA_FLUX']) *
               (s['HB_B_FLUX'] > s['HB_FLUX']) *
               (s['HA_B_FLUX']/s['HB_B_FLUX'] < 5) *
               (s['HA_B_FLUX']/s['HA_B_FLUX_ERR'] > snt) *
               (s['HB_B_FLUX']/s['HB_B_FLUX_ERR'] > snt) *
               (s['HA_B_NPEG'] == 0) * (s['HB_B_NPEG'] == 0))
        s['agn_type'][sel] = 1
        n1 = len(s[sel])
        print(n1, 'type 1 AGN')
        
        # Type 1.5 (intermediate)
        sel = ((s['agn_type'] < 1) * (s['HA_MODSEL_EMB_EM'] > 200) *
               (s['HA_B_FLUX']/s['HA_B_FLUX_ERR'] > snt) *
               (s['HA_B_NPEG'] == 0) * (s['OIIIR_EW'] > 3))
        s['agn_type'][sel] = 1.5
        n1 = len(s[sel])
        print(n1, 'type 1.5 AGN')
        
        # Type 2
        ha_ew = s['HA_EW']
        hb_ew = s['HB_EW']
        oiii_ew = s['OIIIR_EW']
        ha = s['HA_FLUX'] * (1 + 2.5/ha_ew)
        ha_err = s['HA_FLUX_ERR'] * (1 + 2.5/ha_ew)
        hb = s['HB_FLUX'] * (1 + 2.5/hb_ew)
        hb_err = s['HB_FLUX_ERR'] * (1 + 2.5/hb_ew)
        nii = s['NIIR_FLUX']
        nii_err = s['NIIR_FLUX_ERR']
        oiii = s['OIIIR_FLUX']
        oiii_err = s['OIIIR_FLUX_ERR']

        good_ha = (s['HA_NPEG'] == 0) * (ha/ha_err > snt)
        good_nii = (s['NIIR_NPEG'] == 0) * (nii/nii_err > snt)
        good_hb = (s['HB_NPEG'] == 0) * (hb/hb_err > snt)
        good_oiii = (s['OIIIR_NPEG'] == 0) * (oiii/oiii_err > snt)
        good_na = good_ha * good_nii
        good_ob = good_oiii * good_hb
        na = np.log10(nii/ha)
        ob = np.log10(oiii/hb)
        print(len(s[good_ha]), 'galaxies with good Halpha')
        print(len(s[good_na * good_ob]), 'galaxies with good BPT lines')

        sel = ((s['agn_type'] < 1) * (good_na * good_ob) *
               (ob > 0.61/(na - 0.47) + 1.19) * (ob > math.log10(3)))
        s['agn_type'][sel] = 2
        n1 = len(s[sel])
        print(n1, 'type 2 AGN')
        
        plt.clf()
        plt.scatter(na[good_na * good_ob], ob[good_na * good_ob], s=0.01)
        plt.axis([-2, 1, -1.2, 1.5])
        plt.xlabel(r'log([NII]/H$\alpha$)')
        plt.ylabel(r'log([OIII]/H$\beta$)')
        plt.show()

    def add_vmax(self, vmfile=gama_data+'Vmax_dmu/v02/Vmax_v02.fits'):
        """Add standard and density-corrected Vmax values.
        This version reads pre-computed Vmax values"""

        vm_table = Table.read(vmfile)
        self.t = join(self.t, vm_table, keys='CATAID',
                      metadata_conflicts=metadata_conflicts)

    def vmax_calc_old(self, denfile=gama_data+'radial_density.fits'):
        """Calculate standard and density-corrected Vmax values."""

        zmin, zmax = self.zlimits
        print(zmin, zmax)
        nz = 100
        zstep = (zmax - zmin) / nz
        den = Table()
        den['zbin'] = np.linspace(zmin, zmax, nz) + 0.5*zstep
        den['delta_av'] = np.ones(nz)
        if denfile:
            # Interpolate tabulated density fluctuations onto above z grid
            den_table = Table.read(denfile)
            den['delta_av'] = np.interp(
                    den['zbin'], den_table['zbin'], den_table['delta_av'])
        bin_edges = np.linspace(zmin, zmax, nz+1)
        zstep = bin_edges[1] - bin_edges[0]
        V_int = self.area / 3.0 * cosmo.dm(bin_edges)**3
        V = np.diff(V_int)

        # Arrays S_obs and S_vis contain volume-weighted fraction of
        # redshift bin iz in which galaxy igal lies and is visible.

        afac = self.area / 3.0
        ngal = len(self.t)
        S_obs = np.zeros((nz, ngal))
        S_vis = np.zeros((nz, ngal))

        for igal in range(ngal):
            ilo = max(0, min(nz-1, int((self.t['zlo'][igal] - zmin) / zstep)))
            ihi = max(0, min(nz-1, int((self.t['zhi'][igal] - zmin) / zstep)))
            iob = max(0, min(nz-1, int((self.t['z'][igal] - zmin) / zstep)))
            S_obs[ilo+1:iob, igal] = 1
            S_vis[ilo+1:ihi, igal] = 1
#            pdb.set_trace()
            Vp = V_int[ilo+1] - afac*cosmo.dm(self.t['zlo'][igal])**3
            S_obs[ilo, igal] = Vp/V[ilo]
            S_vis[ilo, igal] = Vp/V[ilo]
            Vp = afac*cosmo.dm(self.t['z'][igal])**3 - V_int[iob]
            S_obs[iob, igal] = Vp/V[ihi]
            Vp = afac*cosmo.dm(self.t['zhi'][igal])**3 - V_int[ihi]
            S_vis[ihi, igal] = Vp/V[ihi]

        Pz = cosmo.den_evol(den['zbin'])
        self.t['Vmax_raw'] = np.dot(V, S_vis)
        self.t['Vmax_dc'] = np.dot(den['delta_av'] * V, S_vis)
        self.t['Vmax_ec'] = np.dot(Pz * V, S_vis)
        self.t['Vmax_dec'] = np.dot(den['delta_av'] * Pz * V, S_vis)

    def ran_z_gen(self, nfac):
        """Generate random redshifts nfac times larger than gal catalogue."""

        ndupe = np.round(
                nfac * self.t['Vmax_raw']/self.t['Vmax_dec']).astype(np.int32)

        ngal = len(self.t)
        nran = np.sum(ndupe)
        zran = np.zeros(nran)
        j = 0
        for i in range(ngal):
            ndup = ndupe[i]
            zran[j:j+ndup] = util.ran_fun(
                    self.vol_ev, self.t['zlo'][i], self.t['zhi'][i], ndup)
            j += ndup
        return zran

    def comp_limit_mag(self):
        """Set completeness limits in magnitude (Loveday+2012 sec 3.3)."""
        self.comp_min = self.Mvol(self.mlimits[0], self.zlimits[1])
        self.comp_max = self.Mvol(self.mlimits[1], self.zlimits[0])
        print('Mag completeness limits:', self.comp_min, self.comp_max)

    def comp_limit_mass(self):
        """Mass completeness at minimum redshift."""
        self.comp_min = self.mass_limit(self.zlimits[0])
        self.comp_max = 99
        print('Mass completeness limit:', self.comp_min)

    def Mvol(self, mlim, zlim, kc_col='PCOEFF_R', pc=95):
        """Return absolute magnitude corresponding to given redshift that will
        give a volume-limited sample complete to pc percent."""

        # Construct array of k(zlim) for selected subsample
        t = self.tsel()
#        kcorr = np.polynomial.polynomial.polyval(zlim - kz0, t[kc_col].T)
        kcorr = self.kcorr(zlim, t[kc_col].T)

        # Required percentile of kcorr distribution
        k = np.percentile(kcorr, pc)

#        return (mlim - cosmo.dist_mod(zlim) - k + ecorr(zlim, self.Q))
        return (mlim - self.cosmo.dist_mod(zlim) - k + self.ecorr(zlim))

    def vol_limit(self, Mlim, colname='r_petro', ax=None, plot_fac=0.1,
                  Mrange=(-16, -23), zrange=(0, 0.5)):
        """Select volume-limited sample."""

        self.vol_limited = True
        mlim = self.mlimits[1]
        z = self.t['z']

        zmax = min(np.max(z), self.zlimits[1])
        if self.Mvol(mlim, zmax) - Mlim > 0:
            zlim = zmax
        else:
            zlim = scipy.optimize.brentq(
                lambda z: self.Mvol(mlim, z) - Mlim, self.zlimits[0], zmax,
                xtol=1e-5, rtol=1e-5)
        if ax:
            ax.plot((zrange[0], zlim, zlim), (Mlim, Mlim, Mrange[1]))
            show = np.random.random(len(z)) < plot_fac
            ax.scatter(z[show], self.abs_mags('r_petro')[show], 1, 'k',
                       edgecolors='none')
            ax.set_xlim(zrange)
            ax.set_ylim(Mrange)
            ax.set_xlabel(r'$z$')
            ax.set_ylabel(r'$^{0.1}M_r$')

        self.zlim = zlim
        self.zlimits[1] = min(self.zlimits[1], zlim)
        self.t = self.t[(z < zlim) * (self.abs_mags(colname) < Mlim)]

    def vol_limit_z(self, zlim, colname='r_petro'):
        """Select volume-limited sample correpsonding to redshift limit zlim."""

        self.vol_limited = True
        self.zlimits[1] = min(self.zlimits[1], zlim)
        mlim = self.mlimits[1]
        z = self.t['z']

        Mlim = self.Mvol(mlim, zlim)
        self.zlim = zlim
        self.t = self.t[(z < zlim) * (self.abs_mags(colname) < Mlim)]
        print(f'Vol limit({zlim}) = {Mlim}')

    def zdm(self, dmod, kcoeff):
        """Calculate redshift z corresponding to distance modulus dmod, solves
        dmod = m - M = DM(z) + K(z) - e(z),
        ie. including k-correction and luminosity evolution Q.
        z is constrained to lie in range self.zlimits."""

        if self.cosmo.dist_mod_ke(self.zlimits[0], kcoeff, self.kcorr, self.ecorr) - dmod > 0:
            return self.zlimits[0]
        if self.cosmo.dist_mod_ke(self.zlimits[1], kcoeff, self.kcorr, self.ecorr) - dmod < 0:
            return self.zlimits[1]
        z = scipy.optimize.brentq(lambda z:
                                  self.cosmo.dist_mod_ke(z, kcoeff, self.kcorr, self.ecorr) - dmod,
                                  self.zlimits[0], self.zlimits[1],
                                  xtol=1e-5, rtol=1e-5)
        return z

    def z_s_S(self, smod, kcoeff):
        """Calculate redshift of galaxy with SB 'distance modulus'
        smod = s - S = 10*lg(1+z) + K(z) - e(z)."""

        def smodk(z, kcoeff):
            """Returns SB 'distance modulus'10*lg(1+z) + K(z) - e(z)."""
            return 10*math.log10(1+z) + self.kcorr(z, kcoeff) - self.ecorr(z)

        if smodk(self.zlimits[0], kcoeff) - smod > 0:
            return self.zlimits[0]
        if smodk(self.zlimits[1], kcoeff) - smod < 0:
            return self.zlimits[1]
        z = scipy.optimize.brentq(lambda z: smodk(z, kcoeff) - smod,
                                  self.zlimits[0], self.zlimits[1],
                                  xtol=1e-5, rtol=1e-5)
        return z


#    def kcorr(self, z, kcoeff):
#        """K-correction from polynomial fit."""
#        return np.polynomial.polynomial.polyval(z - self.z0, kcoeff)
#
#    def ecorr(self, z):
#        """e-correction."""
#        if self.ev_model == 'none':
#            return np.zeros(len(z))
#        if self.ev_model == 'z':
#            return self.Q*(z - self.z0)
#        if self.ev_model == 'z1z':
#            return self.Q*z/(1+z)
#
#    def den_evol(self, z):
#        """Density evolution at redshift z."""
#        if self.ev_model == 'none':
#            return np.ones(len(z))
#        if self.ev_model == 'z':
#            return 10**(0.4*self.P*z)
#        if self.ev_model == 'z1z':
#            return 10**(0.4*self.P*z/(1+z))
#
#    def vol_ev(self, z):
#        """Volume element multiplied by density evolution."""
#        pz = cosmo.dV(z) * self.den_evol(z)
#        return pz


    def assign_jackknife(self, name=None):
        """Add jackknife regions to table."""
        t = self.t
        t['jack'] = np.zeros(len(t), dtype=int)
        if name:
            print(name, 'in each jackknife region:')
#        pdb.set_trace()
        for jack in range(njack):
            idx = (t['RA'] >= ra_jack[jack]) * (t['RA'] < ra_jack[jack] + 4.0)
            self.t['jack'][idx] = jack
            if name:
                print(len(self.t['jack'][idx]))

    def xi_output(self, outfile, binning, theta_max, J3_pars):
        """Output the galaxy or random data for xi.c v 2.1 in single cell."""

        ncell = 1
        ix = 0
        iy = 0
        iz = 0
        cellsize = 100.0
        tu = self.tsel()
        nobj = len(tu)
        r = self.cosmo.dm(tu['z'])
        ra = np.array(tu['RA'])
        dec = np.array(tu['DEC'])
        x = r*np.cos(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
        y = r*np.sin(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
        z = r*np.sin(np.deg2rad(dec))

        print('Writing out ', outfile)
        fout = open(outfile, 'w')
        print(self.info, file=fout)
        print(nobj, ncell, ncell, njack, cellsize,
              binning[0], binning[1], binning[2],
              binning[3], binning[4], binning[5],
              theta_max, J3_pars[0], J3_pars[1], J3_pars[2], file=fout)
        print(ix, iy, iz, nobj, file=fout)
        for i in range(nobj):
            print(x[i], y[i], z[i], tu['weight'][i], tu['den'][i],
                  tu['Vmax_out'][i], tu['jack'][i], file=fout)
        fout.close()

    def cosmo(self):
        return cosmo

    # simple mag limits
    def sel_mag_lo(self, z, igal):
        """r_petro > self.mlimits[0]."""
        return self.t['r_petro'][igal].app_calc(z) - self.mlimits[0]

    def sel_mag_hi(self, z, igal):
        """r_petro < self.mlimits[1]."""
        return self.mlimits[1] - self.t['r_petro'][igal].app_calc(z)

    # simple mass limits
    def sel_mass_lo(self, z, igal):
        """mass > mass_lim."""
        return self.t['logmstar'][igal] - self.mass_limit(z)

    def sel_mass_hi(self, z, igal):
        """No upper limit, always return 1"""
        return 1

    def mass_limit(self, z):
        """Mass completeness at given redshift.  Uses results from
        group_lf.gal_mass_z"""
        p = {'gama': [1.17442222,  29.68880365, -22.58489171],
             'sim': [0.95474723,  33.0666522,  -26.39728058]}[self.masscomp]
        a = 1/(1 + z)
        return np.polynomial.polynomial.polyval(a, p)

    def mass_limit_sel(self):
        """Apply stellar mass completeness limit."""
        Mt = self.mass_limit(self.t['z'])
        sel = self.t['logmstar'] > Mt
        self.t = self.t[sel]


# GAMA selection limits
def sel_gama_mag_lo(z, galdat):
    """r_petro > self.mlimits[0]."""
    return galdat['r_petro'].app_calc(z)


def sel_gama_mag_hi(z, galdat):
    """r_petro < self.mlimits[1]."""
    return 19.8 - galdat['r_petro'].app_calc(z)


def z_comp(r_fibre):
    """Sigmoid function fit to redshift succcess given r_fibre,
    from misc.zcomp."""
    p = (22.42, 2.55, 2.24)
    return (1.0/(1 + np.exp(p[1]*(r_fibre-p[0]))))**p[2]

# Selection limits for LOWZ
def sel_lowz_mag_lo(z, galdat):
    """r_cmod > 16."""
    return galdat['r_cmodel'].app_calc(z) - 16


def sel_lowz_mag_hi(z, galdat):
    """r_cmod < 19.6."""
    return 19.6 - galdat['r_cmodel'].app_calc(z)


def sel_lowz_cpar(z, galdat):
    """r_cmod < 13.5 + c_par/0.3, where
    c_par = 0.7(g_mod - r_mod) + 1.2(r_mod - i_mod - 0.18)."""
    r_cmod = galdat['r_cmodel'].app_calc(z)
    g_mod = galdat['g_model'].app_calc(z)
    r_mod = galdat['r_model'].app_calc(z)
    i_mod = galdat['i_model'].app_calc(z)
    c_par = 0.7*(g_mod - r_mod) + 1.2*(r_mod - i_mod - 0.18)
    return 13.5 + c_par/0.3 - r_cmod


def sel_lowz_cperp_lo(z, galdat):
    """c_perp > -0.2, where
    c_perp = (r_mod - i_mod) - (g_mod - r_mod)/4.0 - 0.18."""
    g_mod = galdat['g_model'].app_calc(z)
    r_mod = galdat['r_model'].app_calc(z)
    i_mod = galdat['i_model'].app_calc(z)
    c_perp = (r_mod - i_mod) - (g_mod - r_mod)/4.0 - 0.18
    return c_perp + 0.2


def sel_lowz_cperp_hi(z, galdat):
    """c_perp < 0.2, where
    c_perp = (r_mod - i_mod) - (g_mod - r_mod)/4.0 - 0.18."""
    g_mod = galdat['g_model'].app_calc(z)
    r_mod = galdat['r_model'].app_calc(z)
    i_mod = galdat['i_model'].app_calc(z)
    c_perp = (r_mod - i_mod) - (g_mod - r_mod)/4.0 - 0.18
    return 0.2 - c_perp


# Selection limits for CMASS

def sel_cmass_mag_lo(z, galdat):
    """i_cmod > 17.5."""
    return galdat['i_cmodel'].app_calc(z) - 17.5


def sel_cmass_mag_hi(z, galdat):
    """i_cmod < 19.9."""
    return 19.9 - galdat['i_cmodel'].app_calc(z)


def sel_cmass_fib_mag(z, galdat):
    """i_fib2 < 21.5."""
    return 21.5 - galdat['i_fib2'].app_calc(z)


def sel_cmass_ri(z, galdat):
    """r_mod - i_mod < 2."""
    return 2 - (galdat['r_cmodel'].app_calc(z) -
                galdat['i_cmodel'].app_calc(z))


def sel_cmass_mag_dperp(z, galdat):
    """i_cmod < 19.86 + 1.6*(d_perp − 0.8), where
    d_perp = (r_mod - i_mod) - (g_mod - r_mod)/8."""
    i_cmod = galdat['i_cmodel'].app_calc(z)
    g_mod = galdat['g_model'].app_calc(z)
    r_mod = galdat['r_model'].app_calc(z)
    i_mod = galdat['i_model'].app_calc(z)
    d_perp = (r_mod - i_mod) - (g_mod - r_mod)/8
    return 19.86 + 1.6*(d_perp - 0.8) - i_cmod


def sel_cmass_dperp(z, galdat):
    """d_perp > 0.55, where
    d_perp = (r_mod - i_mod) - (g_mod - r_mod)/8."""
    g_mod = galdat['g_model'].app_calc(z)
    r_mod = galdat['r_model'].app_calc(z)
    i_mod = galdat['i_model'].app_calc(z)
    d_perp = (r_mod - i_mod) - (g_mod - r_mod)/8
    return d_perp - 0.55
