# Classes and functions to support galaxy target selection and
# selection function

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import scipy.optimize
import scipy.stats

from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, join
from pymoc import MOC
import pymoc.util.catalog
import pymoc.util.plot

import util

# Global parameters
gama_data = os.environ['GAMA_DATA']
tcfile = gama_data + 'TilingCatv46.fits'
kctemp = gama_data + 'kcorr_dmu/v5/kcorr_auto_z{}_vecv05.fits'
bnfile = gama_data + 'BrightNeighbours.fits'
ext_file = gama_data + 'GalacticExtinctionv03.fits'
sersic_file = gama_data + 'SersicCatSDSSv09.fits'
g3cfof = gama_data + 'g3cv9/G3CFoFGroupv09.fits'
g3cgal = gama_data + 'g3cv9/G3CGalv08.fits'
g3cmockfof = gama_data + 'g3cv6/G3CMockFoFGroupv06.fits'
g3cmockgal = gama_data + 'g3cv6/G3CMockGalv06.fits'
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

H0, cosmo, kz0, ez0, ev_model = 100.0, 0, 0, 0, 'z'


def kcorr(z, kcoeff):
    """K-correction from polynomial fit."""
    return np.polynomial.polynomial.polyval(z - kz0, kcoeff)


def ecorr(z, Q):
    """e-correction."""
    if ev_model == 'none':
        try:
            return np.zeros(len(z))
        except TypeError:
            return 0.0
    if ev_model == 'z':
        return Q*(z - ez0)
    if ev_model == 'z1z':
        return Q*z/(1+z)


class CosmoLookup():
    """Distance and volume-element lookup tables.
    NB volume element is differential per unit solid angle."""

    def __init__(self, H0, omega_l, zlimits, P=1, nz=1000):
        cosmo = FlatLambdaCDM(H0=H0, Om0=1-omega_l)
        self._P = P
        self._H0 = H0
        self._zrange = zlimits
        self._z = np.linspace(zlimits[0], zlimits[1], nz)
        self._dm = cosmo.comoving_distance(self._z)
        self._dV = cosmo.differential_comoving_volume(self._z)
        self._dist_mod = cosmo.distmod(self._z)
        print('CosmoLookup: H0={}, Omega_l={}, P={}'.format(H0, omega_l, P))

    def dm(self, z):
        """Comoving distance."""
        return np.interp(z, self._z, self._dm)

    def dl(self, z):
        """Luminosity distance."""
        return (1+z)*np.interp(z, self._z, self._dm)

    def da(self, z):
        """Angular diameter distance."""
        return np.interp(z, self._z, self._dm)/(1+z)

    def dV(self, z):
        """Volume element per unit solid angle."""
        return np.interp(z, self._z, self._dV)

    def dist_mod(self, z):
        """Distance modulus."""
        return np.interp(z, self._z, self._dist_mod)

#    def dist_mod_ke(self, z, kcoeff):
#        """Returns the K- and e-corrected distance modulus
#        DM(z) + k(z) - e(z)."""
#        dm = self.dist_mod(z) + self.kcorr(z, kcoeff) - self.ecorr(z)
#        return dm

    def den_evol(self, z):
        """Density evolution at redshift z."""
        if ev_model == 'none':
            try:
                return np.ones(len(z))
            except TypeError:
                return 1.0
        if ev_model == 'z':
            return 10**(0.4*self._P*z)
        if ev_model == 'z1z':
            return 10**(0.4*self._P*z/(1+z))

    def vol_ev(self, z):
        """Volume element multiplied by density evolution."""
        pz = self.dV(z) * self.den_evol(z)
        return pz


class Magnitude():
    """Attributes and methods for galaxy magnitudes,
    including individual k- and e-corrections."""

    def __init__(self, app, z, kcoeff, Q=0, band='r'):
        self.app = app
        self.z = z
        self.kcoeff = kcoeff
        self.Q = Q
        self.band = band
        self.abs = app - cosmo.dist_mod(z) - kcorr(z, kcoeff) + ecorr(z, Q)

    def app_calc(self, z):
        """Return apparent magnitude galaxy would have at redshift z."""
        return (self.abs + cosmo.dist_mod(z) + kcorr(z, self.kcoeff) -
                ecorr(z, self.Q))


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

    def __init__(self, Q=1, P=1, mlimits=(0, 19.8), zlimits=(0.002, 0.65)):
        self.mlimits = mlimits
        self.zlimits = zlimits
        self.vol_limited = False
        self.Q = Q
        self.P = P

    def kcorr_fix(self, coeff, chi2max=10):
        """Set any missing or bad k-corrs to median values."""

        # Fit polynomial to median K(z) for good fits
        t = self.t
        nk = t[coeff].shape[1]
    #    pdb.set_trace()
        good = np.isfinite(np.sum(t[coeff], axis=1)) * (t['CHI2'] < chi2max)
        zbin = np.linspace(self.zlimits[0], self.zlimits[1], 50) - kz0
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

    def abs_mags(self, magname):
        """Return absolute magnitudes corresponding to magname."""
        mags = np.array([self.t[magname][i].abs for i in range(len(self.t))])
        try:
            return mags[self.use]
        except AttributeError:
            return mags

    def read_gama(self, kcfile=kctemp.format('01'), chi2max=10, nq_min=3):
        """Reads table of basic GAMA data from tiling cat & kcorr DMU."""

        global cosmo, kz0, njack, ra_jack
        njack = 9
        ra_jack = (129, 133, 137, 174, 178, 182, 211.5, 215.5, 219.5)

        def z_comp(r_fibre):
            """Sigmoid function fit to redshift succcess given r_fibre,
            from misc.zcomp."""
            p = (22.42, 2.55, 2.24)
            return (1.0/(1 + np.exp(p[1]*(r_fibre-p[0]))))**p[2]

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
        kc_table = Table.read(kcfile)
        omega_l = kc_table.meta['OMEGA_L']
        kz0 = kc_table.meta['Z0']
        self.area = kc_table.meta['AREA'] * (math.pi/180.0)**2
        cosmo = CosmoLookup(H0, omega_l, self.zlimits, P=self.P)
        t = join(tc_table, kc_table, keys='CATAID',
                 metadata_conflicts=metadata_conflicts)

        # Select reliable, main-sample galaxies in given redshift range
        sel = ((t['SURVEY_CLASS'] > 3) * (t['NQ_1'] >= nq_min) *
               (t['Z_TONRY'] >= self.zlimits[0]) *
               (t['Z_TONRY'] < self.zlimits[1]))
        t = t[sel]
        t.rename_column('Z_TONRY', 'z')
        r_petro = [Magnitude(t['R_PETRO'][i], t['z'][i], t['PCOEFF_R'][i],
                             Q=self.Q, band='r') for i in range(len(t))]

        # Copy required columns to new table
        self.t = t['CATAID', 'RA', 'DEC', 'z', 'KCORR_R', 'PCOEFF_R', 'CHI2']
        self.t['r_petro'] = r_petro
        self.kcorr_fix('PCOEFF_R')
        self.assign_jackknife()

        # colour according to Loveday+ 2012 eqn 3
        r_abs = self.abs_mags('r_petro')
        grcut = 0.15 - 0.03*r_abs
        gr = (t['G_MODEL'] - t['KCORR_G']) - (t['R_MODEL'] - t['KCORR_R'])
        self.t['colour'] = ['c']*len(t)
        sel = (gr < grcut)
        self.t['colour'][sel] = 'b'
        sel = (gr >= grcut)
        self.t['colour'][sel] = 'r'

        # Finally calculate visibility limits and hence Vmax
#        self.vis_calc((sel_mag_lo, sel_mag_hi))
#        self.vmax_calc()

#        z = self.t['z']
#        kc = self.t['KCORR_R']
#        self.t['ABSMAG_R'] = (t['R_PETRO'] - cosmo.dist_mod(z) - kc +
#                              cosmo.ecorr(z))
#        self.t['R_SB_ABS'] = (t['R_SB'] - 10*np.log10(1 + z) - kc +
#                              cosmo.ecorr(z))

#        # Fit polynomial to median K(z) for good fits
#        nk = t['PCOEFF_R'].shape[1]
#        good = np.isfinite(kc) * (t['CHI2'] < chi2max)
#        zbin = np.linspace(self.zlimits[0], self.zlimits[1], 50) - self.kz0
#        k_array = np.polynomial.polynomial.polyval(
#            zbin, t['PCOEFF_R'][good].transpose())
#        k_median = np.median(k_array, axis=0)
#        self.kmean = np.polynomial.polynomial.polyfit(zbin, k_median, nk-1)
#
#        # Set any missing or bad k-corrs to median values
#        bad = np.logical_not(good)
#        nbad = len(z[bad])
#        if nbad > 0:
#            kc[bad] = np.polynomial.polynomial.polyval(
#                z[bad] - self.kz0, self.kmean)
#            self.t['PCOEFF_R'][bad] = self.kmean
#            print(nbad, 'missing/bad k-corrections replaced with mean')

        # Completeness weight
        imcomp = np.interp(t['R_SB'], sb_tab, comp_tab)
        zcomp = z_comp(t['FIBERMAG_R'])
        self.t['cweight'] = np.clip(1.0/(imcomp*zcomp), 1, wmax)
        self.t['use'] = np.ones(len(self.t), dtype=np.bool)

    def read_gama_group_mocks(self):
        """Read gama group mocks."""

#       See Robotham+2011 Sec 2.2 fopr k- and e- corrections
        global cosmo, kz0, ez0
        kz0 = 0.2
        ez0 = 0
        pcoeff = (0.2085, 1.0226, 0.5237, 3.5902, 2.3843)
        self.kmean = pcoeff

        gal = Table.read(g3cmockgal)
        grp = Table.read(g3cmockfof)
        omega_l = 0.75
        self.area = 144 * (math.pi/180.0)**2
        cosmo = CosmoLookup(H0, omega_l, self.zlimits, P=self.P)
        grp['log_mass'] = 13.98 + 1.16*(np.log10(grp['LumBfunc']) - 11.5)
        t = join(gal, grp, join_type='left', 
                 metadata_conflicts=metadata_conflicts)
#        pdb.set_trace()
        # Select mock galaxies in given redshift range
        sel = (t['Z'] >= self.zlimits[0]) * (t['Z'] < self.zlimits[1])
        t = t[sel]
        t.rename_column('Z', 'z')
        r_petro = [Magnitude(t['Rpetro'][i], t['z'][i], pcoeff,
                             Q=self.Q, band='r') for i in range(len(t))]

        # Copy required columns to new table
        self.t = t
        self.t['r_petro'] = r_petro
        self.t['PCOEFF_R'] = np.tile(pcoeff, (len(t), 1))
        self.t['cweight'] = np.ones(len(self.t))
        self.t['use'] = np.ones(len(self.t), dtype=np.bool)

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
        cosmo = CosmoLookup(H0, omega_l, self.zlimits, P=self.P)

        kcorr_fix(t, 'PCOEFF_G', self.zlimits)
        kcorr_fix(t, 'PCOEFF_R', self.zlimits)
        kcorr_fix(t, 'PCOEFF_I', self.zlimits)
        g_model = [Magnitude(t['MODELMAG_G'][i], t['z'][i], t['PCOEFF_G'][i],
                             Q=self.Q, band='g') for i in range(len(t))]
        r_model = [Magnitude(t['MODELMAG_R'][i], t['z'][i], t['PCOEFF_R'][i],
                             Q=self.Q, band='r') for i in range(len(t))]
        i_model = [Magnitude(t['MODELMAG_I'][i], t['z'][i], t['PCOEFF_I'][i],
                             Q=self.Q, band='i') for i in range(len(t))]
        r_cmodel = [Magnitude(t['CMODELMAGCOR_R'][i], t['z'][i], t['PCOEFF_R'][i],
                              Q=self.Q, band='r') for i in range(len(t))]
        # Copy required columns to new table
        self.t = t['RA', 'DEC', 'z', 'PCOEFF_G', 'PCOEFF_R', 'PCOEFF_I']
        self.t['g_model'] = g_model
        self.t['r_model'] = r_model
        self.t['i_model'] = i_model
        self.t['r_cmodel'] = r_cmodel

        self.assign_jackknife()

        # Completeness weight
#        imcomp = np.interp(t['R_SB'], sb_tab, comp_tab)
#        zcomp = z_comp(t['FIBERMAG_R'])
#        self.t['cweight'] = np.clip(1.0/(imcomp*zcomp), 1, wmax)
        self.t['cweight'] = np.ones(len(self.t), dtype=np.bool)
        self.t['use'] = np.ones(len(self.t), dtype=np.bool)

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
               (i_cmod < 19.86 + 1.6(d_perp - 0.8)) * (d_perp > 0.55))

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
        cosmo = CosmoLookup(H0, omega_l, self.zlimits, P=self.P)

        kcorr_fix(t, 'PCOEFF_G', self.zlimits)
        kcorr_fix(t, 'PCOEFF_R', self.zlimits)
        kcorr_fix(t, 'PCOEFF_I', self.zlimits)
        g_model = [Magnitude(t['MODELMAG_G'][i], t['z'][i], t['PCOEFF_G'][i],
                             Q=self.Q, band='g') for i in range(len(t))]
        r_model = [Magnitude(t['MODELMAG_R'][i], t['z'][i], t['PCOEFF_R'][i],
                             Q=self.Q, band='r') for i in range(len(t))]
        i_model = [Magnitude(t['MODELMAG_I'][i], t['z'][i], t['PCOEFF_I'][i],
                             Q=self.Q, band='i') for i in range(len(t))]
        i_cmodel = [Magnitude(t['CMODELMAGCOR_I'][i], t['z'][i], t['PCOEFF_I'][i],
                              Q=self.Q, band='i') for i in range(len(t))]
        i_fib2 = [Magnitude(t['FIBER2MAGCOR_I'][i], t['z'][i], t['PCOEFF_I'][i],
                            Q=self.Q, band='i') for i in range(len(t))]
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
        self.assign_jackknife()

        # Completeness weight
#        imcomp = np.interp(t['R_SB'], sb_tab, comp_tab)
#        zcomp = z_comp(t['FIBERMAG_R'])
#        self.t['cweight'] = np.clip(1.0/(imcomp*zcomp), 1, wmax)
        self.t['cweight'] = np.ones(len(self.t), dtype=np.bool)
        self.t['use'] = np.ones(len(self.t), dtype=np.bool)

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

    def read_gama_groups(self, nmin=5, edge_min=0.9):
        """Read data for GAMA group centres.  Group visibility limits and Vmax
        correspond to that of nmin'th ranked member."""

        # Read and select groups meeting selection criteria
        t = Table.read(g3cfof)
        t['log_mass'] = 13.98 + 1.16*(np.log10(t['LumBfunc']) - 11.5)
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
        self.z0 = gal.z0

        # Finally calculate visibility limits and hence Vmax
        self.vis_calc()
        self.vmax_calc()
        self.assign_jackknife()

    def select(self, sel_dict=None):
        """Select galaxies that satisfy criteria in sel_dict."""

        t = self.t
        nin = len(t)
        self.use = np.ones(len(self.t), dtype=np.bool)
        if sel_dict:
            for key, limits in sel_dict.items():
                print(key, limits)
                self.use *= ((t[key] >= limits[0]) * (t[key] < limits[1]))
        nsel = len(t[self.use])
        print(nsel, 'out of', nin, 'galaxies selected')

    def tsel(self):
        """Return table of selected galaxies."""
        try:
            return self.t[self.use]
        except AttributeError:
            return self.t

    def vis_calc_gama(self):
        """Add redshift visibility limits for GAMA."""

        self.t['zlo'] = [self.zdm(self.mlimits[0] - self.t['ABSMAG_R'][i],
                         self.t['PCOEFF_R'][i])
                         for i in range(len(self.t))]
        self.t['zhi'] = [self.zdm(self.mlimits[1] - self.t['ABSMAG_R'][i],
                         self.t['PCOEFF_R'][i])
                         for i in range(len(self.t))]

    def vis_calc(self, conditions):
        """Add redshift visibility limits for sample defined by conditions."""

        def z_lower(cond, galdat):
            """Lower redshift limit from given condition."""
            zmin = self.zlimits[0]
            if (cond(zmin, galdat) > 0):
                zlo = zmin
            else:
                try:
                    zlo = scipy.optimize.brentq(
                            cond, zmin, galdat['z'],
                            args=galdat, xtol=1e-5, rtol=1e-5)
                except ValueError:
                    zlo = galdat['z']
            return zlo

        def z_upper(cond, galdat):
            """Upper redshift limit from given condition."""
            zmax = self.zlimits[1]
            if (cond(zmax, galdat) > 0):
                zhi = zmax
            else:
                try:
                    zhi = scipy.optimize.brentq(
                            cond, galdat['z'], zmax,
                            args=galdat, xtol=1e-5, rtol=1e-5)
                except ValueError:
                    zhi = galdat['z']
            return zhi

        self.t['zlo'] = np.zeros(len(self.t))
        self.t['zhi'] = np.zeros(len(self.t))
        for i in range(len(self.t)):
            galdat = self.t[i]
            zlo = [z_lower(cond, galdat) for cond in conditions]
            zhi = [z_upper(cond, galdat) for cond in conditions]
            self.t['zlo'][i] = max(zlo)
            self.t['zhi'][i] = min(zhi)

    def group_props(self):
        """Add group properties.
        Luminosity-based mass estimate is from Viola+2015, eqn (37)."""

        g = Table.read(os.environ['GAMA_DATA'] + '/g3cv9/G3CFoFGroupv09.fits')
        g['log_mass'] = 13.98 + 1.16*(np.log10(g['LumBfunc']) - 11.5)
        gals = Table.read(os.environ['GAMA_DATA'] + '/g3cv9/G3CGalv08.fits')
        joined = join(gals, g, keys='GroupID', join_type='left',
                      metadata_conflicts=metadata_conflicts)
        self.t = join(self.t, joined, keys='CATAID', join_type='left',
                      metadata_conflicts=metadata_conflicts)

    def group_limit(self, nmin):
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
        plt.clf()
        plt.scatter(t_by_group['Nfof'], t_by_group['Nmem'], s=0.1)
        plt.xlabel('Nfof')
        plt.ylabel('Nmem')
        plt.show()
        sel = t_by_group['Nmem'] >= nmin
        self.t = t_by_group[sel]

    def stellar_mass(self, fslim=(0.8, 10)):
        """Read stellar masses for GAMA."""

        m = Table.read(smfile)
        m['logmstar'] -= 2*math.log10(cosmo._H0/70.0)
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

        m = Table.read(os.environ['GAMA_DATA'] + 'StellarMassesv19.fits')
        m = m['CATAID', 'Z_TONRY', 'absmag_g', 'absmag_r']
        t = Table.read(os.environ['GAMA_DATA'] + 'SpecLineSFR/' + infile)
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
        plt.xlabel(r'$z$l')
        plt.ylabel(r'SFR')
        plt.semilogy(basey=10, nonposy='clip')
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
                    cosmo.dV, zlo, zhi, epsabs=1e-3, epsrel=1e-3)
            Vmax_raw[iz] = Vmax_raw[iz-1] + V
            V, err = scipy.integrate.quad(
                    cosmo.vol_ev, zlo, zhi, epsabs=1e-3, epsrel=1e-3)
            Vmax_ec[iz] = Vmax_ec[iz-1] + V
            if denfile:
                V, err = scipy.integrate.quad(
                        lambda z: cosmo.dV(z) *
                        np.interp(z, den['zbin'], den['delta_av']),
                        zlo, zhi, epsabs=1e-3, epsrel=1e-3)
                Vmax_dc[iz] = Vmax_dc[iz-1] + V
                V, err = scipy.integrate.quad(
                        lambda z: cosmo.vol_ev(z) *
                        np.interp(z, den['zbin'], den['delta_av']),
                        zlo, zhi, epsabs=1e-3, epsrel=1e-3)
                Vmax_dec[iz] = Vmax_dec[iz-1] + V

        self.t['Vmax_raw'] = np.interp(self.t['zhi'], zbins, afac*Vmax_raw)
        self.t['Vmax_ec'] = np.interp(self.t['zhi'], zbins, afac*Vmax_ec)
        if denfile:
            self.t['Vmax_dc'] = np.interp(self.t['zhi'], zbins, afac*Vmax_dc)
            self.t['Vmax_dec'] = np.interp(self.t['zhi'], zbins, afac*Vmax_dec)
        else:
            self.t['Vmax_dc'] = self.t['Vmax_raw']
            self.t['Vmax_dec'] = self.t['Vmax_ec']

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

    def Mvol(self, mlim, zlim, kc_col='PCOEFF_R', pc=95):
        """Return absolute magnitude corresponding to given redshift that will
        give a volume-limited sample complete to pc percent."""

        # Construct array of k(zlim) for selected subsample
        t = self.tsel()
        kcorr = np.polynomial.polynomial.polyval(zlim - kz0, t[kc_col].T)

        # Required percentile of kcorr distribution
        k = np.percentile(kcorr, pc)

        return (mlim - cosmo.dist_mod(zlim) - k + ecorr(zlim, self.Q))

    def vol_limit(self, Mlim, colname='r_petro'):
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
        self.zlim = zlim
        self.t = self.t[(z < zlim) * (self.abs_mags(colname) < Mlim)]

    def zdm(self, dmod, kcoeff):
        """Calculate redshift z corresponding to distance modulus dmod, solves
        dmod = m - M = DM(z) + K(z) - e(z),
        ie. including k-correction and luminosity evolution Q.
        z is constrained to lie in range self.zlimits."""

        if cosmo.dist_mod_ke(self.zlimits[0], kcoeff) - dmod > 0:
            return self.zlimits[0]
        if cosmo.dist_mod_ke(self.zlimits[1], kcoeff) - dmod < 0:
            return self.zlimits[1]
        z = scipy.optimize.brentq(lambda z:
                                  cosmo.dist_mod_ke(z, kcoeff) - dmod,
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

    def assign_jackknife(self, verbose=1):
        """Add jackknife regions to table."""
        t = self.t
        t['jack'] = np.zeros(len(t), dtype=int)
        if verbose:
            print('Galaxies in each jackknife region:')
#        pdb.set_trace()
        for jack in range(njack):
            idx = (t['RA'] >= ra_jack[jack]) * (t['RA'] < ra_jack[jack] + 4.0)
            self.t['jack'][idx] = jack
            if verbose:
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
        r = cosmo.dm(tu['z'])
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
#            pdb.set_trace()
            print(x[i], y[i], z[i], tu['weight'][i], tu['den'][i],
                  tu['Vmax_out'][i], tu['jack'][i], file=fout)
        fout.close()
