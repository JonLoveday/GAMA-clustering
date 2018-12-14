# LF routines utilising new gal_sample utilities

from array import array
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpmath
import numpy as np
import os
import pdb
import pickle
import scipy.special

from astLib import astSED
from astropy.modeling import models, fitting
from astropy.table import Table, join
import healpy as hp
from sherpa.data import Data1D
from sherpa.utils.err import EstErr
from sherpa.fit import Fit
from sherpa.optmethods import LevMar, NelderMead
from sherpa.stats import Chi2
from sherpa.estmethods import Confidence
from sherpa.plot import IntervalProjection, RegionProjection


import gal_sample as gs
from schec import SchecMag
import util

# Global parameters
lf_data = os.environ['LF_DATA']
mag_label = r'$^{0.1}M_r - 5 \log_{10} h$'
ms_label = r'$\log_{10}\ ({\cal M}_*/{\cal M}_\odot h^{-2})$'
lf_label = r'$\phi(M)\ [h^3\ {\rm Mpc}^{-3}\ {\rm mag}^{-1}]$'

# Constants
ln10 = math.log(10)

# Ticks point inwards on all axes
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True


# GAMA selection limits
def sel_gama_mag_lo(z, galdat):
    """r_petro > self.mlimits[0]."""
    return galdat['r_petro'].app_calc(z)


def sel_gama_mag_hi(z, galdat):
    """r_petro < self.mlimits[1]."""
    return 19.8 - galdat['r_petro'].app_calc(z)


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


def lfr(outfile='lfr.pkl', colname='r_petro', clrname='gi_colour',
        bins=np.linspace(-25, -12, 26), Mmin_fit=-24, Mmax_fit=-17,
        p0=(-1, -21, -2), zlimits=(0.002, 0.65), error='jackknife'):
    """r-band LF using density-corrected Vmax."""

    samp = gs.GalSample()
    samp.read_gama()
    samp.stellar_mass()
    samp.add_sersic_index()
    samp.vis_calc((sel_gama_mag_lo, sel_gama_mag_hi))
    samp.vmax_calc()
    lf_dict = {}
    lf = LF(samp, colname, bins, error=error)
#    lf.fn_fit(fn=lf.Schechter_mag, Mmin=Mmin_fit, Mmax=Mmax_fit, p0=p0)
    lf_dict['all'] = lf
#    lf.plot(finish=True)

    for colour in 'br':
        clr_limits = ('a', 'z')
        if (colour == 'b'):
            clr_limits = ('b', 'c')
        if (colour == 'r'):
            clr_limits = ('r', 's')
        sel_dict = {clrname: clr_limits}
        samp.select(sel_dict)
        lf = LF(samp, colname, bins, error=error, sel_dict=sel_dict)
        lf_dict[colour] = lf

    for lbl, sersic_lims in zip(['nlo', 'nhi'], [[0, 1.9], [1.9, 30]]):
        sel_dict = {'GALINDEX_r': sersic_lims}
        samp.select(sel_dict)
        lf = LF(samp, colname, bins, error=error, sel_dict=sel_dict)
#        lf.fn_fit(fn=lf.Schechter_mag, Mmin=Mmin_fit, Mmax=Mmax_fit, p0=p0)
        lf_dict[lbl] = lf

    pickle.dump(lf_dict, open(outfile, 'wb'))


def plot(infile, lf_lims=(-15, -23.5, 1e-7, 0.1), nmin=5, fn=SchecMag(),
         p0=(-1, -21, -2), Mmin_fit=-24, Mmax_fit=-17,
         plot_file='/Users/loveday/Documents/tex/papers/gama/groupLF/lf_field.pdf',
         plot_size=(6, 3)):
    """Plot LFs."""

    fn.alpha = p0[0]
    fn.Mstar = p0[1]
    fn.lgps = p0[2]

    lf_dict = pickle.load(open(infile, 'rb'))
    plt.clf()
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, num=1)
    fig.set_size_inches(plot_size)
    fig.subplots_adjust(left=0, bottom=0.0, hspace=0.0, wspace=0.0)
    plt.semilogy(basey=10, nonposy='clip')
    ax = axes[0]
    ax.axis(lf_lims)
    ax.set_xlabel(mag_label)
    ax.set_ylabel(lf_label)
    phi = lf_dict['all']
    phi.fn_fit(fn, Mmin=Mmin_fit, Mmax=Mmax_fit)
    phi.plot(ax=ax, nmin=nmin, clr='k', label='All')
    phi = lf_dict['b']
    phi.fn_fit(fn, Mmin=Mmin_fit, Mmax=Mmax_fit)
    phi.plot(ax=ax, nmin=nmin, clr='b', label='Blue')
    phi = lf_dict['r']
    phi.fn_fit(fn, Mmin=Mmin_fit, Mmax=Mmax_fit)
    phi.plot(ax=ax, nmin=nmin, clr='r', label='Red')
    ax.legend()

    ax = axes[1]
    ax.axis(lf_lims)
    ax.set_xlabel(mag_label)
    phi = lf_dict['all']
    phi.fn_fit(fn, Mmin=Mmin_fit, Mmax=Mmax_fit)
    phi.plot(ax=ax, nmin=nmin, clr='k', label='All')
    phi = lf_dict['nlo']
    phi.fn_fit(fn, Mmin=Mmin_fit, Mmax=Mmax_fit)
    phi.plot(ax=ax, nmin=nmin, clr='b', label='low-n')
    phi = lf_dict['nhi']
    phi.fn_fit(fn, Mmin=Mmin_fit, Mmax=Mmax_fit)
    phi.plot(ax=ax, nmin=nmin, clr='r', label='high-n')
    ax.legend()

    plt.draw()
    plt.savefig(plot_file, bbox_inches='tight')
    plt.show()


def fortuna(outfile='lf_fortuna.dat',
            colname='ABSMAG_R', Mmin=-25, Mmax=-12, nbin=26):
    """r-band LF for Fortuna red galaxy sample."""

    samp = gs.GalSample(zlimits=(0.002, 0.22), ev_model='none', kcorr_z0='00')
    samp.read_gama()
    samp.stellar_mass()
    samp.add_vmax()
    sel = samp.t['sm_g_r'] > 0.66
    samp.t = samp.t[sel]
    lf = LF(samp, colname)
    label = 'z < 0.22'
    lf.plot(ylim=(1e-7, 1e-2), label=label, finish=True)
    lf.write(open('lf_loz.dat', 'w'), label)
    samp = gs.GalSample(zlimits=(0.22, 0.65), ev_model='none', kcorr_z0='00')
    samp.read_gama()
    samp.stellar_mass()
    samp.add_vmax()
    sel = samp.t['sm_g_r'] > 0.66
    samp.t = samp.t[sel]
    lf = LF(samp, colname)
    label = 'z >= 0.22'
    lf.plot(ylim=(1e-7, 1e-2), label=label, finish=True)
    lf.write(open('lf_hiz.dat', 'w'), label)

#    plt.clf()
#    plt.scatter(samp.t['ABSMAG_R'], samp.t['z'], s=0.1)
#    zp = np.linspace(samp.zlimits[0], samp.zlimits[1], 50)
#    Mp = []
#    for i in range(len(zp)):
#        Mp.append(samp.Mvol(samp.mlimits[1], zp[i]))
#    plt.plot(Mp, zp, 'r')
#
#    plt.xlabel(r'$M_r$')
#    plt.ylabel(r'$z$')
#    plt.show()


def lf_lowz(infile='lowz_kcorrz00.fits', outtemp='lf_lowz_{}_{}.dat',
            colname='r_cmodel', mlimits=(16, 19.6), Mmin=-24, Mmax=-21, nbin=15):
    """r-band LF for LOWZ galaxy samples."""

    for zlimits in ((0.16, 0.36), (0.16, 0.26), (0.26, 0.36)):
        outfile = outtemp.format(*zlimits)
        samp = gs.GalSample(Q=0, P=0, mlimits=mlimits, zlimits=zlimits)
        samp.read_lowz(infile)
        samp.vis_calc((sel_lowz_mag_lo, sel_lowz_mag_hi, sel_lowz_cpar,
                       sel_lowz_cperp_lo, sel_lowz_cperp_hi))
        samp.vmax_calc(denfile=None)

        Mr = samp.abs_mags(colname)
        plt.clf()
        plt.scatter(samp.t['z'], Mr, c=samp.t['Vmax_raw'], s=0.01)
        plt.xlabel('Redshift')
        plt.ylabel('Mr')
        cbar = plt.colorbar()
        cbar.set_label('Vmax')
        plt.show()

        t = Table.read(infile)
        schecp = None
        try:
            schecp = (t.meta['ALPHA'], t.meta['MSTAR'], t.meta['PHISTAR'])
        except KeyError:
            pass
        lf = LF(samp, colname, Mmin=Mmin, Mmax=Mmax, nbin=nbin)
        lf.schec_fit()
        print('alpha = {:5.2f}+-{:5.2f}, M* = {:5.2f}+-{:5.2f}, logphi* = {:5.2f}+-{:5.2f}'.format(
                lf.alpha, lf.alpha_err, lf.Mstar, lf.Mstar_err, lf.lpstar, lf.lpstar_err))
        if schecp:
            print('Comparison alpha = {:5.2f}, M* = {:5.2f}, logphi* = {:5.2f}'.format(
                    schecp[0], schecp[1], math.log10(schecp[2])))
        label = '{} < z < {}'.format(*zlimits)
        lf.plot(ylim=(1e-7, 1e-2), schecp=schecp, label=label, finish=True)
        lf.write(open(outfile, 'w'), label)


def lf_cmass(infile='cmass_kcorrz00.fits', outtemp='lf_cmass_{}_{}.dat',
colname='r_cmodel', mlimits=(17.5, 19.9), Mmin=-24, Mmax=-21, nbin=15):
    """r-band LF for LOWZ galaxy samples."""

    for zlimits in ((0.16, 0.36), (0.16, 0.26), (0.26, 0.36)):
        outfile = outtemp.format(*zlimits)
        samp = gs.GalSample(Q=0, P=0, mlimits=mlimits, zlimits=zlimits)
        samp.read_lowz(infile)
        samp.vis_calc((sel_lowz_mag_lo, sel_lowz_mag_hi, sel_lowz_cpar,
                       sel_lowz_cperp_lo, sel_lowz_cperp_hi))
        samp.vmax_calc(denfile=None)

        Mr = samp.abs_mags(colname)
        plt.clf()
        plt.scatter(samp.t['z'], Mr, c=samp.t['Vmax_raw'], s=0.01)
        plt.xlabel('Redshift')
        plt.ylabel('Mr')
        cbar = plt.colorbar()
        cbar.set_label('Vmax')
        plt.show()

        t = Table.read(infile)
        schecp = None
        try:
            schecp = (t.meta['ALPHA'], t.meta['MSTAR'], t.meta['PHISTAR'])
        except KeyError:
            pass
        lf = LF(samp, colname, Mmin=Mmin, Mmax=Mmax, nbin=nbin)
        lf.schec_fit()
        print('alpha = {:5.2f}+-{:5.2f}, M* = {:5.2f}+-{:5.2f}, logphi* = {:5.2f}+-{:5.2f}'.format(
                lf.alpha, lf.alpha_err, lf.Mstar, lf.Mstar_err, lf.lpstar, lf.lpstar_err))
        if schecp:
            print('Comparison alpha = {:5.2f}, M* = {:5.2f}, logphi* = {:5.2f}'.format(
                    schecp[0], schecp[1], math.log10(schecp[2])))
        label = '{} < z < {}'.format(*zlimits)
        lf.plot(ylim=(1e-7, 1e-2), schecp=schecp, label=label, finish=True)
        lf.write(open(outfile, 'w'), label)


def absmag_lowz(infile='lowz_kcorrz00.fits', outfile='lowz_abs.fits',
                colname='r_cmodel', zlimits=(0.16, 0.36), mlimits=(16, 19.6)):
    """Output k-corrected absolute magnitudes for LOWZ."""

    samp = gs.GalSample(Q=0, P=0, mlimits=mlimits, zlimits=zlimits)
    samp.read_lowz(infile)
    M_model_r = samp.abs_mags('r_model')
    M_cmodel_r = samp.abs_mags('r_cmodel')
    kcorr = [gs.kcorr(samp.t['z'][i],
                      samp.t['r_cmodel'][i].kcoeff) for i in range(len(M_model_r))]
    t = Table([samp.t['RA'], samp.t['DEC'], samp.t['z'], kcorr,
               M_model_r, M_cmodel_r],
              names=('RA', 'DEC', 'z', 'KCORR_R',
                     'ABS_MODELMAG_R', 'ABS_CMODELMAG_R'))
    t.write(outfile, format='fits', overwrite=True)


def smf(outfile='smf.dat',
        colname='logmstar', Mmin=6, Mmax=12, nbin=24, zmin=0.002, zmax=0.65,
        zlims=(0.002, 0.1, 0.2, 0.3)):
    """Stellar mass function using density-corrected Vmax."""

    samp = gs.GalSample(Q=0, P=0)
    samp.read_gama()
    samp.stellar_mass()
    samp.vis_calc((sel_gama_mag_lo, sel_gama_mag_hi))
    samp.vmax_calc()
    lf = LF(samp, colname, Mmin=Mmin, Mmax=Mmax, nbin=nbin)
    lf.plot(finish=True)
    lf_dict = {'all': lf}
    for iz in range(3):
        zlo, zhi = zlims[iz], zlims[iz+1]
        samp.zlimits = (zlo, zhi)
        samp.vmax_calc()
        sel_dict = {'z': (zlo, zhi)}
        samp.select(sel_dict)
        lf = LF(samp, colname, Mmin=Mmin, Mmax=Mmax, nbin=nbin,
                error='jackknife', sel_dict=sel_dict)
        lf.comp_limit_mass(samp, zlo)
        Mkey = 'z{}'.format(iz)
        lf_dict[Mkey] = lf
    pickle.dump(lf_dict, open(outfile, 'wb'))


def blf_test(outfile='blf.dat',
             cols=('ABSMAG_R', 'logmstar'), arange=((-25, -12), (6, 12)),
             bins=(13, 12), zmin=0.002, zmax=0.65, clean_photom=1, use_wt=1):
    """Mr-stellar mass bivariate function using density-corrected Vmax."""

    samp = gs.GalSample()
    samp.read_gama()
    samp.stellar_mass()
    samp.add_vmax()
    lf = LF2(samp, cols, bins, arange)
    lf.plot(finish=True)


def bbd_petro(outfile='bbd_petro.dat',
              cols=('ABSMAG_R', 'R_SB_ABS'), arange=((-25, -12), (16, 26)),
              bins=(26, 20), zmin=0.002, zmax=0.65, clean_photom=1, use_wt=1):
    """Petrosian BBD using density-corrected Vmax."""

    samp = gs.GalSample()
    samp.read_gama()
    samp.add_vmax()
    lf = LF2(samp, cols, bins, arange)
    lf.plot(chol_fit=True, finish=True)


def bbd_sersic(outfile='bbd_sersic.dat',
               cols=('ABSMAG_R_SERSIC', 'R_SB_SERSIC_ABS'),
               arange=((-25, -12), (16, 26)),
               bins=(26, 20), zmin=0.002, zmax=0.65, use_wt=1):
    """Petrosian BBD using density-corrected Vmax."""

    samp = gs.GalSample()
    samp.read_gama()
    samp.add_sersic()
    samp.add_vmax()
    lf = LF2(samp, cols, bins, arange)
    lf.plot(chol_fit=True, finish=True)


def plot_samples(samp, selcol, bins, label_template, lfcol='r_petro',
                 Mmin=-25, Mmax=-14, Mmin_fit=-25, Mmax_fit=-14, nbin=22,
                 error='jackknife', outfile=None):
    """Plot LF for sub-samples selected by column selcol in given bins."""

    plot_size = (6, 8)
    sa_left = 0.18
    sa_bot = 0.08
    plt.clf()
    npanel = len(bins) - 1
    nrow, ncol = util.two_factors(npanel)
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
    fig.set_size_inches(plot_size)
    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
    fig.text(0.55, 0.0, r'$M_r$', ha='center', va='center')
    fig.text(0.06, 0.5, r'$\phi(M)$', ha='center', va='center',
             rotation='vertical')
    plt.semilogy(basey=10, nonposy='clip')
    lf_list = []
    label_list = []
    if outfile:
        f = open(outfile, 'w')
    for i in range(npanel):
        sel_dict = {selcol: (bins[i], bins[i+1])}
        label = label_template.format(bins[i], bins[i+1])
        samp.select(sel_dict)
        norm = len(samp.t)/len(samp.tsel())
        lf = LF(samp, lfcol, Mmin=Mmin, Mmax=Mmax, nbin=nbin, norm=norm,
                error=error)
        if outfile:
            lf.write(f, label)
        lf.schec_fit(Mmin=Mmin_fit, Mmax=Mmax_fit)
        print('alpha={:5.2f}+-{:5.2f}, M*={:5.2f}+-{:5.2f}, chi2/nu = {:5.2f}/{:2d}'.format(
                lf.alpha, lf.alpha_err, lf.Mstar, lf.Mstar_err, lf.chi2, lf.ndof))
        ax = axes.flat[i]
        lf.plot(ax=ax, label=label)
        ax.text(0.1, 0.9, label, transform=ax.transAxes)
        ax.text(0.1, 0.8, r'$\alpha={:5.2f}\pm{:5.2f}, M^*={:5.2f}\pm{:5.2f}, \chi^2, \nu = {:5.2f}/{:2d}$'.format(
                lf.alpha, lf.alpha_err, lf.Mstar, lf.Mstar_err, lf.chi2, lf.ndof),
                transform=ax.transAxes)
        lf_list.append(lf)
        label_list.append(label)
        if i==0:
            schecp = (lf.alpha, lf.Mstar, lf.lpstar)
        else:
            lf.schec_plot(ax, schecp, ls='--')
    if outfile:
        f.close()
    plt.ylim(1e-7, 1)
#    plt.legend()
    plt.show()

    plt.clf()
    ax = plt.gca()
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$M^*$')
    for i in range(len(bins)-1):
        label = label_template.format(bins[i], bins[i+1])
        lf_list[i].like_cont(ax=ax, label=label)
#    plt.axis((-2, 0, -21.5, -19.5))
    plt.legend(label_list)
    plt.show()


class LF():
    """LF data and methods."""

    def __init__(self, samp, colname, bins, norm=1,
                 Vmax='Vmax_dec', error='Poisson', sel_dict='None'):
        """Initialise new LF instance from specified table and column."""

        self.sel_dict = sel_dict
        self.error = error

        self.bins = bins
        nbin = len(bins) - 1
        self.Mbin = bins[:-1] + 0.5*np.diff(bins)
        self.comp = np.ones(nbin, dtype=bool)
        self.comp_min = bins[0]
        self.comp_max = bins[-1]

        if colname == 'logmstar':
            absval = samp.tsel()[colname]
        else:
            absval = samp.abs_mags(colname)
        wt = samp.tsel()['cweight']/samp.tsel()[Vmax]

        if error == 'mock':
            # Mean and sd of several mocks
            nmock = 9
            self.njack = nmock
            ngal = np.zeros((nmock, nbin), dtype=np.int)
            self.phi_jack = np.zeros((nmock, nbin))
            for ivol in range(nmock):
                sel = samp.tsel()['Volume'] == ivol + 1
                ngal[ivol, :], edges = np.histogram(absval[sel], bins)
                self.phi_jack[ivol, :], edges = np.histogram(
                        absval[sel], bins, weights=wt[sel])
            self.phi_jack *= norm/np.diff(bins)
            self.ngal = np.mean(ngal, axis=0)
            self.phi = np.mean(self.phi_jack, axis=0)
            self.phi_err = np.std(self.phi_jack, axis=0)
        else:
            self.ngal, edges = np.histogram(absval, bins)
            self.phi, edges = np.histogram(absval, bins, weights=wt)
            self.phi *= norm/np.diff(bins)
            if error == 'Poisson':
                self.phi_err = self.phi/np.sqrt(self.ngal)

            if error == 'jackknife':
                # Jackknife errors - this assumes all regions have the same area!
                njack = gs.njack
                self.njack = njack
                self.phi_jack = np.zeros((njack, len(self.phi)))
                for jack in range(njack):
                    idx = samp.tsel()['jack'] != jack
                    self.phi_jack[jack, :], edges = np.histogram(
                        absval[idx], bins, weights=wt[idx])
                    self.phi_jack[jack, :] *= norm*njack/(njack-1)/np.diff(bins)
                self.phi_err = np.sqrt((njack-1) * np.var(self.phi_jack, axis=0))

    def comp_limits(self, samp, zlo, zhi):
        """Set completeness limits in magnitude (Loveday+2012 sec 3.3)."""
        self.comp_min = samp.Mvol(samp.mlimits[0], zhi)
        self.comp_max = samp.Mvol(samp.mlimits[1], zlo)
        self.comp *= (self.comp_min <= self.Mbin) * (self.Mbin < self.comp_max)
        print('Mag completeness limits:', self.comp_min, self.comp_max)

    def comp_limit_mass(self, samp, zlo):
        """Mass completeness at given redshift.  Uses results from
        group_lf.gal_mass_z"""
        p = [1.17442222,  29.68880365, -22.58489171]
        a = 1/(1 + zlo)
        self.comp_min = np.polynomial.polynomial.polyval(a, p)
        self.comp *= (self.Mbin >= self.comp_min)
        print('Mass completeness limit:', self.comp_min)

    def write(self, f, label):
        """Output to specified file."""
        print('# ', label, file=f)
        for i in range(len(self.Mbin)):
            if self.comp[i]:
                print(self.Mbin[i], self.ngal[i], self.phi[i], self.phi_err[i],
                      file=f)

    def fn_fit_old(self, fn=None, p0=(-1, -20, -2.5),
               Mmin=None, Mmax=None, verbose=0):
        """Fit specfied function."""

        if fn:
            self.fn = fn
        else:
            self.fn = self.Schechter_mag
        if (Mmin and Mmax):
            self.Mmin_fit = Mmin
            self.Mmax_fit = Mmax
        else:
            self.Mmin_fit = self.comp_min
            self.Mmax_fit = self.comp_max
        idx = (self.comp * (self.phi_err > 0) *
               (self.Mmin_fit <= self.Mbin) * (self.Mbin < self.Mmax_fit))
        self.ndof = len(self.Mbin[idx]) - 3
        res = scipy.optimize.fmin(self.lf_resid, p0, xtol=0.001, ftol=0.001,
                                  full_output=1, disp=0)
        if verbose:
            print(res)
        self.fit_par = res[0]
        self.chi2 = res[1]
        warnflag = res[4]
        if warnflag:
            print('warnflag =', warnflag)
        fit_jack = []
        for jack in range(self.njack):
            res = scipy.optimize.fmin(
                    self.lf_resid, res[0], args=(jack,),
                    xtol=0.001, ftol=0.001, full_output=1, disp=0)
            fit_jack.append(res[0])
        self.fit_err = np.std(fit_jack, axis=0)
        if self.error == 'jackknife':
            self.fit_err *= np.sqrt(self.njack-1)

    def fn_fit_ap(self, fn, M0=-21, Mmin=None, Mmax=None, verbose=0):
        """Fit function fn to LF data.
        M0 is guess at characteristic magnitude (if negaitive) or
        log stellar mass (if positive)."""

        self.M0 = M0
#        self.fn = fn
        if (Mmin and Mmax):
            self.Mmin_fit = Mmin
            self.Mmax_fit = Mmax
        else:
            self.Mmin_fit = self.comp_min
            self.Mmax_fit = self.comp_max
        idx = (self.comp * (self.phi_err > 0) *
               (self.Mmin_fit <= self.Mbin) * (self.Mbin < self.Mmax_fit))

        if M0 < 0:
            xfac = 2.5  # luminosity
        else:
            xfac = 1.0  # stellar mass
        x = 10**((M0-self.Mbin)/xfac)
        npar = 3
        if fn.alpha.fixed:
            npar = 2
        self.ndof = len(self.Mbin[idx]) - npar
        fit = fitting.LevMarLSQFitter()
#        fit = fitting.SLSQPLSQFitter()

        fit_fn = fit(fn, x[idx], self.phi[idx], weights=1.0/self.phi_err[idx])
#        pdb.set_trace()
        self.fit_fn = fit_fn
        self.fit_par = (-fit_fn.alpha.value,
                        M0 - xfac**math.log10(fit_fn.x_0.value),
                        fit_fn.amplitude.value)
        if verbose:
#            print(self.fit_par)
            print(fit_fn)
        self.chi2 = np.sum((self.phi[idx] - fit_fn(x[idx])/self.phi_err[idx])**2)

        fit_jack = []
        for jack in range(self.njack):
            fit_j = fit(fit_fn, x[idx], self.phi_jack[jack, idx],
                          weights=1.0/self.phi_err[idx])
            fit_jack.append((-fit_j.alpha.value,
                             M0 - xfac*math.log10(fit_j.x_0.value),
                             fit_j.amplitude.value))
        self.fit_err = np.std(fit_jack, axis=0)
        if self.error == 'jackknife':
            self.fit_err *= np.sqrt(self.njack-1)

        return fit_fn

    def fn_fit_sherpa(self, fn, M0=-21, Mmin=None, Mmax=None, verbose=0):
        """Fit function fn to LF data using Sherpa.
        M0 is guess at characteristic magnitude (if negative) or
        log stellar mass (if positive)."""

        self.M0 = M0
#        self.fn = fn
        if (Mmin and Mmax):
            self.Mmin_fit = Mmin
            self.Mmax_fit = Mmax
        else:
            self.Mmin_fit = max(self.bins[0], self.comp_min)
            self.Mmax_fit = min(self.bins[-1], self.comp_max)
        idx = (self.comp * (self.phi_err > 0) *
               (self.Mmin_fit <= self.Mbin) * (self.Mbin < self.Mmax_fit))

        if M0 < 0:  # luminosity with 1 mag bins
            x = 10**(0.4*(M0-self.Mbin))
            xu = 10**(0.4*(1+M0-self.Mbin))
#            dx = np.fabs(np.diff(10**(0.4*(M0-self.bins))))
        else:  # stellar mass with 1 dex bins
            x = 10**(self.Mbin - M0)
            xu = 10**(self.Mbin - M0 + 1)
#            dx = np.fabs(np.diff(10**(self.bins - M0)))
        from sherpa.data import Data1DInt
        from sherpa.astro.models import Schechter
        from sherpa.fit import Fit
        from sherpa.optmethods import LevMar, NelderMead
        from sherpa.stats import Chi2
        from sherpa.estmethods import Confidence

        d = Data1DInt('All', x[idx], xu[idx], self.phi[idx],
                      self.phi_err[idx])
        sfit = Fit(d, fn, stat=Chi2(), method=NelderMead())
        res = sfit.fit()
#        pdb.set_trace()
        self.fit_fn = fn
        sfit.estmethod = Confidence()
        self.fit_errors = sfit.est_errors()
#        self.fit_par = (res.alpha.val,
#                        M0 - xfac**math.log10(res.ref.value),
#                        res.norm.value)
        self.fit_par = res.parvals
        if verbose:
            print(res)
        self.chi2 = res.statval
        self.ndof = res.dof

        fit_jack = []
        for jack in range(self.njack):
            d = Data1DInt('All', x[idx], xu[idx],
                          self.phi_jack[jack, idx], self.phi_err[idx])
            sfit = Fit(d, fn, stat=Chi2(), method=NelderMead())
            resj = sfit.fit()
            fit_jack.append(resj.parvals)
        self.fit_err = np.std(fit_jack, axis=0)
        if self.error == 'jackknife':
            self.fit_err *= np.sqrt(self.njack-1)

        return self.fit_fn

    def fn_fit(self, fn, Mmin=None, Mmax=None, verbose=0):
        """Fit function fn to LF data using Sherpa."""

        self.Mmin_fit = max(self.bins[0], self.comp_min)
        self.Mmax_fit = min(self.bins[-1], self.comp_max)
        if Mmin:
            self.Mmin_fit = max(Mmin, self.Mmin_fit)
        if Mmax:
            self.Mmax_fit = min(Mmax, self.Mmax_fit)
        idx = (self.comp * (self.phi_err > 0) *
               (self.Mmin_fit <= self.Mbin) * (self.Mbin < self.Mmax_fit))

        d = Data1D('All', self.Mbin[idx], self.phi[idx],
                   staterror=self.phi_err[idx])
        sfit = Fit(d, fn, stat=Chi2(), method=NelderMead())
        self.res = sfit.fit()
        sfit.estmethod = Confidence()
        sfit.estmethod.max_rstat = 100
        try:
            self.errors = sfit.est_errors()
        except EstErr:
            print('Warning: reduced chi2 exceeds ', sfit.estmethod.max_rstat)
#            pdb.set_trace()

        self.fit = sfit
        self.fn = fn
#        self.fit_par = (res.alpha.val,
#                        M0 - xfac**math.log10(res.ref.value),
#                        res.norm.value)
#        self.fit_par = res.parvals
        if verbose:
            print(self.res)

#        rproj = RegionProjection()
#        rproj.prepare(nloop=(11, 11))
#        rproj.calc(self.fit, self.fn.Mstar, self.fn.lgps)
##                rproj.contour(overplot=1, clearwindow=0)
#        plt.clf()
#        rproj.contour()
#        plt.show()

#        self.chi2 = res.statval
#        self.ndof = res.dof

#        fit_jack = []
#        for jack in range(self.njack):
#            d = Data1DInt('All', x[idx], xu[idx],
#                          self.phi_jack[jack, idx], self.phi_err[idx])
#            sfit = Fit(d, fn, stat=Chi2(), method=NelderMead())
#            resj = sfit.fit()
#            fit_jack.append(resj.parvals)
#        self.fit_err = np.std(fit_jack, axis=0)
#        if self.error == 'jackknife':
#            self.fit_err *= np.sqrt(self.njack-1)

        return self.fn

    def like_cont_old(self, pp=(0, 1), mp=2, ax=None, label=None,
                  lc_step=32, lc_limits=4,
                  dchisq=[4, ], c=None, ls='-', verbose=0):
        """Plot likelihood contours for given parameter pair pp
        (default alpha-Mstar), marginalising over mp (default log phi*).
        lc_limits may be specified as four lower and upper limits,
        two ranges, or a single sigma multiplier."""

        self.chi2map = np.zeros([lc_step, lc_step])

        try:
            if len(lc_limits) == 4:
                xmin, xmax, ymin, ymax = lc_limits
            if len(lc_limits) == 2:
                xrange, yrange = lc_limits
                xmin = self.fit_par[pp[0]] - xrange
                xmax = self.fit_par[pp[0]] + xrange
                ymin = self.fit_par[pp[1]] - yrange
                ymax = self.fit_par[pp[1]] + yrange
        except TypeError:
            xmin = self.fit_par[pp[0]] - lc_limits*self.fit_err[pp[0]]
            xmax = self.fit_par[pp[0]] + lc_limits*self.fit_err[pp[0]]
            ymin = self.fit_par[pp[1]] - lc_limits*self.fit_err[pp[1]]
            ymax = self.fit_par[pp[1]] + lc_limits*self.fit_err[pp[1]]
        dx = (xmax - xmin)/lc_step
        dy = (ymax - ymin)/lc_step
        self.lc_limits = [xmin, xmax, ymin, ymax]
        if verbose:
            print(self.lc_limits)
#        pdb.set_trace()

        # chi2 minimum
        chi2min = self.lf_resid(self.fit_par)
        self.v = chi2min + dchisq
        for ix in range(lc_step):
            x = xmin + (ix+0.5)*dx
            for iy in range(lc_step):
                y = ymin + (iy+0.5)*dy
                if mp == 0:
                    # Marginalise over alpha
                    res = scipy.optimize.fmin(
                            lambda alpha: self.lf_resid((alpha, x, y)),
                            1, xtol=0.001, ftol=0.001, full_output=1, disp=0)
                    self.chi2map[iy, ix] = res[1]
                    if res[4] != 0:
                        pdb.set_trace()
                if mp == 2:
                    # Marginalise over log phi*
                    res = scipy.optimize.fmin(
                            lambda lpstar: self.lf_resid((x, y, lpstar)),
                            1, xtol=0.001, ftol=0.001, full_output=1, disp=0)
                    self.chi2map[iy, ix] = res[1]
                    if res[4] != 0:
                        pdb.set_trace()
                if mp is None:
                    # Assume fixed alpha
                    self.chi2map[iy, ix] = self.lf_resid((x, y))

        if ax:
            if not c:
                c = next(ax._get_lines.prop_cycler)['color']
            return ax.contour(self.chi2map, self.v, aspect='auto',
                              origin='lower', extent=self.lc_limits,
                              linestyles=ls, colors=c, label=label)
#            pdb.set_trace()

    def like_cont(self, px, py, ax=None, label=None,
                  lc_step=32, lc_limits=4,
                  dchisq=[4, ], c=None, ls='-', verbose=0):
        """Plot likelihood contours for given parameter pair,
        marginalising over any unfrozen parameters in the model.
        lc_limits may be specified as four lower and upper limits,
        two ranges, or a single sigma multiplier."""

#        pdb.set_trace()
        chi2min = self.res.statval
        v = chi2min + np.array(dchisq)
        try:
            if len(lc_limits) == 4:
                xmin, xmax, ymin, ymax = lc_limits
            if len(lc_limits) == 2:
                xrange, yrange = lc_limits
                xmin = self.fit_par[px] - xrange
                xmax = self.fit_par[px] + xrange
                ymin = self.fit_par[py] - yrange
                ymax = self.fit_par[py] + yrange
        except TypeError:
            dvals = zip(self.errors.parnames, self.errors.parvals,
                        self.errors.parmins, self.errors.parmaxes)
            pvals = {d[0]: {'val': d[1], 'loerr': d[2], 'hierr': d[3]}
                     for d in dvals}
            xmin = pvals[px]['val'] + lc_limits*pvals[px]['loerr']
            xmax = pvals[px]['val'] + lc_limits*pvals[px]['hierr']
            ymin = pvals[py]['val'] + lc_limits*pvals[py]['loerr']
            ymax = pvals[py]['val'] + lc_limits*pvals[py]['hierr']
#        dx = (xmax - xmin)/lc_step
#        dy = (ymax - ymin)/lc_step
        self.lc_limits = [xmin, xmax, ymin, ymax]
        if verbose:
            print(self.lc_limits)

        from sherpa.plot import RegionProjection
        rproj = RegionProjection()
        rproj.prepare(min=[xmin, ymin], max=[xmax, ymax],
                      nloop=[lc_step, lc_step])
        rproj.calc(self.fit, self.fn.Mstar, self.fn.lgps)
        rproj.contour()
        x0, x1, chi2 = rproj.x0, rproj.x1, rproj.y
        chi2.resize(rproj.nloop)

        # chi2 minimum
#        chi2min = self.lf_resid(self.fit_par)
#        self.v = chi2min + dchisq
#        for ix in range(lc_step):
#            x = xmin + (ix+0.5)*dx
#            for iy in range(lc_step):
#                y = ymin + (iy+0.5)*dy
#                if mp == 0:
#                    # Marginalise over alpha
#                    res = scipy.optimize.fmin(
#                            lambda alpha: self.lf_resid((alpha, x, y)),
#                            1, xtol=0.001, ftol=0.001, full_output=1, disp=0)
#                    self.chi2map[iy, ix] = res[1]
#                    if res[4] != 0:
#                        pdb.set_trace()
#                if mp == 2:
#                    # Marginalise over log phi*
#                    res = scipy.optimize.fmin(
#                            lambda lpstar: self.lf_resid((x, y, lpstar)),
#                            1, xtol=0.001, ftol=0.001, full_output=1, disp=0)
#                    self.chi2map[iy, ix] = res[1]
#                    if res[4] != 0:
#                        pdb.set_trace()
#                if mp is None:
#                    # Assume fixed alpha
#                    self.chi2map[iy, ix] = self.lf_resid((x, y))

        if ax:
            if not c:
                c = next(ax._get_lines.prop_cycler)['color']
#            pdb.set_trace()
#            ax.imshow(y, origin='lower', cmap='viridis_r', aspect='auto',
#                      extent=(x0.min(), x0.max(), x1.min(), x1.max()))
            pdb.set_trace()
            return ax.contour(chi2, v, aspect='auto',
                              origin='lower', extent=self.lc_limits,
                              linestyles=ls, colors=c, label=label)

    def lf_resid(self, x, jack=-1):
        """Return chi^2 residual for functional fit to binned phi estimate."""

        M = self.Mbin
        fit = self.fn(M, x)
#        if sigma > 0:
#            scale = sigma/np.mean(np.diff(M))
#            ng = int(math.ceil(3*scale))
#            gauss = scipy.stats.norm.pdf(np.arange(-ng, ng+1), scale=scale)
#            fit = np.convolve(fit, gauss, 'same')

        idx = (self.phi_err > 0) * (self.Mmin_fit <= M) * (M < self.Mmax_fit)
        if jack >= 0:
            fc = np.sum(((self.phi_jack[jack, idx]-fit[idx]) /
                         self.phi_err[idx])**2)
        else:
            fc = np.sum(((self.phi[idx]-fit[idx]) / self.phi_err[idx])**2)
        return fc

    def gaussian(self, M, pars):
        mu, sigma, norm = pars[0], pars[1], 10**pars[2]
        return norm * np.exp(-(M - mu)**2) / (2 * sigma**2)

    def Schechter_mag(self, M, pars):
        alpha, Mstar, phistar = pars[0], pars[1], 10**pars[2]
        L = 10**(0.4*(Mstar-M))
        schec = 0.4*ln10*phistar*L**(alpha+1)*np.exp(-L)
        return schec

    def Schechter_mag_fixed_alpha(self, M, pars):
        Mstar, phistar = pars[0], 10**pars[1]
        L = 10**(0.4*(Mstar-M))
        schec = 0.4*ln10*phistar*L**(self.alpha+1)*np.exp(-L)
        return schec

    def Schechter_mass(self, logM, pars):
        alpha, logMstar, phistar = pars[0], pars[1], 10**pars[2]
        M = 10**(logM-logMstar)
        return ln10 * np.exp(-M) * phistar*M**(alpha+1)

    def Schechter_dbl_mass(self, logM, pars):
        logMstar, alpha1, alpha2, ps1, ps2 = pars
        M = 10**(logM-logMstar)
        return ln10 * np.exp(-M) * (ps1*M**(alpha1+1) + ps2*M**(alpha2+1))

    def plot(self, ax=None, nmin=1, label=None, xlim=None, ylim=None,
             fmt='o', ls='-', clr=None, mfc=None, show_fit=True,
             schecp=None, finish=False):
        """Plot LF and optionally the Schechter fn fit."""

        if ax is None:
            plt.clf()
            ax = plt.subplot(111)
#        c = next(ax._get_lines.prop_cycler)['color']
#        c = 'k'
        comp = self.comp
        comp *= (self.ngal >= nmin)
        h = ax.errorbar(self.Mbin[comp], self.phi[comp], self.phi_err[comp],
                        fmt=fmt, color=clr, mfc=mfc, label=label)
#        print(self.Mbin[comp], self.phi[comp])
#        if show_fit and hasattr(self, 'fit_par'):
#            x = np.linspace(self.Mmin_fit, self.Mmax_fit, 100)
#            y = self.fn(x, self.fit_par)
#            show = y > 1e-10
#            ax.plot(x[show], y[show], ls=ls, color=clr)
        if show_fit and hasattr(self, 'fn'):
            Mbin = np.linspace(self.Mmin_fit, self.Mmax_fit, 100)
#            Mbin = bins[:-1] + 0.5*np.diff(bins)
#            x = 10**(0.4*(self.M0-Mbin))
#            xu = 10**(0.4*(1+self.M0-Mbin))
#            dx = np.fabs(np.diff(10**(0.4*(self.M0-bins))))
            y = self.fn(Mbin)
            show = y > 1e-10
            ax.plot(Mbin[show], y[show], ls=ls, color=clr)
#            print(x, y)
#            pdb.set_trace()
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if schecp:
            self.fn_plot(ax, schecp, ls)
            x = np.linspace(self.Mmin_fit, self.Mmax_fit, 100)
        if finish:
            ax.semilogy(basey=10, nonposy='clip')
            ax.set_xlabel(r'$M_r$')
            ax.set_ylabel(r'$\phi$')
            plt.show()
        return h

    def fn_plot(self, ax, par, ls='-', c=None):
        """Plot functional fit."""

        if c is None:
            c = next(ax._get_lines.prop_cycler)['color']
        x = np.linspace(self.Mmin_fit, self.Mmax_fit, 100)
        y = self.fn(x, par)
        show = y > 1e-10
        ax.plot(x[show], y[show], ls, color=c)


class LF2():
    """Bivariate LF data and methods."""

    def __init__(self, t, cols, bins, arange, norm=1, Vmax='Vmax_dec'):
        """Initialise new LF instance from specified table and column.
        Note that the 2d LF array holds the first specified column along
        the first dimension, and the second along the second dimension.
        When plotting, the first dimension corresponds to the vertical axis,
        the second to the horizontal."""

        self.cols, self.bins, self.arange = cols, bins, arange
        wt = t['cweight']/t[Vmax]
        self.ngal, xedges, yedges = np.histogram2d(
                t[cols[0]], t[cols[1]], bins, arange)
        self.phi, xedges, yedges = np.histogram2d(
                t[cols[0]], t[cols[1]], bins, arange, weights=wt)
        self.Mbin1 = xedges[:-1] + 0.5*np.diff(xedges)
        self.Mbin2 = yedges[:-1] + 0.5*np.diff(yedges)
        binsize = (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])
        self.phi *= norm/binsize
        vol, xedges, yedges = np.histogram2d(
            t[cols[0]], t[cols[1]], bins, arange, weights=t['cweight']*t[Vmax])
        cwt, xedges, yedges = np.histogram2d(
            t[cols[0]], t[cols[1]], bins, arange, weights=t['cweight'])
        self.vol = vol/cwt

        # Jackknife errors
        njack = gs.njack
        self.njack = njack
        self.phi_jack = np.zeros((njack, bins[0], bins[1]))
        for jack in range(njack):
            idx = t['jack'] != jack
            self.phi_jack[jack, :, :], xedges, yedges = np.histogram2d(
                t[cols[0]][idx], t[cols[1]][idx], bins, arange, weights=wt[idx])
            self.phi_jack[jack, :, :] *= float(njack)/(njack-1)/binsize
        self.phi_err = norm*np.sqrt((njack-1) * np.var(self.phi_jack, axis=0))

        # Transpose arrays since 1st, 2nd dims correspond to y, x axes on plots
        self.ngal = self.ngal.T
        self.phi = self.phi.T
        self.phi_err = self.phi_err.T
        self.phi_jack = self.phi_jack.T
        self.vol = self.vol.T

    def write(self, f, label):
        """Output to specified file."""
        print('# ', label, file=f)
        for i in range(len(self.Mbin)):
            print(self.Mbin[i], self.phi[i], self.phi_err[i], file=f)

    def plot(self, ax=None, label=None, ngmin=5, ncont=16, vcont=1800,
             vmin=-6, vmax=-1.5, chol_fit=0, finish=1):
        """Plot bivariate LF."""

        if ax is None:
            plt.clf()
            ax = plt.subplot(111)
        extent = self.arange[0] + self.arange[1]
        log_phi = np.log10(self.phi)
        log_phi = np.ma.array(log_phi, mask=np.isnan(log_phi))
        plt.imshow(log_phi, aspect='auto', origin='lower',
                   extent=extent, interpolation='nearest',
                   vmin=vmin, vmax=vmax)
        cb = plt.colorbar()
        cb.set_label(r'$\log_{10} \phi$')

        """Contour for volume vcont."""
        nlow = self.vol < vcont
        print(len(self.vol[nlow]), 'bins have volume below', vcont)
        plt.contour(self.vol, (vcont,), colors=('r',),
                    aspect='auto', origin='lower', extent=extent)

        """Contour for ncont galaxies per bin."""
        plt.contour(self.ngal, (ncont,), colors=('r',), linestyles='dashed',
                    aspect='auto', origin='lower', extent=extent)

        """Least-squares Choloniewski fn fit to phi(M, mu)."""
        if chol_fit:
            chol_par_name = ('alpha', '   M*', ' phi*', ' beta', '  mu*',
                             'log sigma')
            M = np.tile(self.Mbin1, (len(self.Mbin2), 1))
            mu = np.tile(self.Mbin2, (len(self.Mbin1), 1)).T

            def chol_resid(chol_par, phi, phi_err):
                """Return residual between BBD and Choloniewski fit."""
                diff = phi - chol_eval(chol_par)
#                pdb.set_trace()
                return (diff/phi_err).flatten()

            def chol_eval(chol_par):
                """Choloniewski function."""

                alpha, Mstar, phistar, beta, mustar, log_sigma = chol_par
                sigma = 10**log_sigma
                fac = 0.4*math.log(10)/math.sqrt(2*math.pi)/sigma*phistar
                lum = 10**(0.4*(Mstar - M))
                gauss = np.exp(-0.5*((mu - mustar - beta*(M - Mstar))/sigma)**2)
                chol = fac*lum**(alpha + 1)*np.exp(-lum)*gauss
                return chol

            prob = 0.32
            phi = self.phi
            phi_err = self.phi_err
            exclude = self.ngal < ngmin
            phi_err[exclude] = 1e6
            use = self.ngal >= ngmin
            nbin = len(phi[use])
            nu = nbin - 6
            dchisq = scipy.special.chdtri(nu, prob)
            print(nu, dchisq)

            p0 = [-1.2, -20.5, 0.01, 0.3, 20.0, -0.3]
            res = scipy.optimize.leastsq(chol_resid, p0, (phi, phi_err),
                                         xtol=0.001, ftol=0.001, full_output=1)
            popt, cov, info, mesg, ier = res
            print(mesg)
            chi2 = (info['fvec']**2).sum()
            cov *= (chi2/nu)

            for i in range(6):
                print('{} = {:7.3f} +- {:7.3f}'.format(chol_par_name[i],
                      popt[i], math.sqrt(cov[i, i])))
            print('chi2, nu: ', chi2, nu)
            chol_arr = np.log10(chol_eval(popt))
            v = np.linspace(vmin, vmax, int(2*(vmax - vmin)) + 1)
            print('contours ', v)
            plt.contour(chol_arr, v, aspect='auto', origin='lower',
                        extent=extent)

        if finish:
            ax.set_xlabel(self.cols[0])
            ax.set_ylabel(self.cols[1])
            plt.show()


#def lf_resid(x, fn, M, phi, phi_err, Mmin=-99, Mmax=99, sigma=0):
#    """Return chi^2 residual for functional fit to binned phi estimate."""
#
#    fit = fn(M, x)
#    if sigma > 0:
#        scale = sigma/np.mean(np.diff(M))
#        ng = int(math.ceil(3*scale))
#        gauss = scipy.stats.norm.pdf(np.arange(-ng, ng+1), scale=scale)
#        fit = np.convolve(fit, gauss, 'same')
#
#    idx = (phi_err > 0) * (Mmin <= M) * (M < Mmax)
#    fc = np.sum(((phi[idx]-fit[idx]) / phi_err[idx])**2)
#    return fc


def wake_kcorr_test(zrange=(0.15, 0.35, 20), girange=(1.5, 2.5, 20),
                    what='k_r'):
    """Wake LRG k-corrections."""

    z = np.linspace(*zrange)
    gi = np.linspace(*girange)
    nz = zrange[-1]
    ngi = girange[-1]
    wc = wakeKcorr()
    k = np.zeros((nz, ngi))
    w = np.zeros((nz, ngi))
    for iz in range(nz):
        for igi in range(ngi):
            k[nz-iz-1, igi] = wc.interp(z[iz], gi[igi], what)[0]
            w[nz-iz-1, igi] = wc.interp(z[iz], gi[igi], what)[1]
    plt.clf()
    plt.imshow(k, extent=(girange[0], girange[1], zrange[0], zrange[1]),
               aspect='auto')
    plt.xlabel('(g-i)')
    plt.ylabel('z')
    plt.colorbar()
    plt.show()

    plt.clf()
    plt.imshow(w, extent=(girange[0], girange[1], zrange[0], zrange[1]),
               aspect='auto')
    plt.xlabel('(g-i)')
    plt.ylabel('z')
    plt.colorbar()
    plt.show()


def lowz_kcorr(infile='lowz.fits', zrange=(0.15, 0.35), nz=20, what='ke_r',
               H0=70, omega_l=0.718):
    """Wake LRG k-corrections to LOWZ sample."""

    cosmo = gs.CosmoLookup(H0, omega_l, zrange)
    lowz = Table.read(infile)
    z = lowz['z']
    sel = (zrange[0] <= z) * (z < zrange[1])
    lowz = lowz[sel]
    z = lowz['z']
    gi = lowz['modelMag_g'] - lowz['modelMag_i']
    ngal = len(z)
    Mr = np.zeros(ngal)
    wc = wakeKcorr()
    for i in range(ngal):
        Mr[i] = (lowz['cmodelMagCor_r'][i] - cosmo.dist_mod(z[i]) -
                 wc.interp(z[i], gi[i], what)[0])
    zbins = np.linspace(*zrange, nz+1)
    zcen = zbins[:-1] + 0.5*(zbins[1] - zbins[0])
    M_av = np.zeros(nz)
    M_std = np.zeros(nz)
    for iz in range(nz):
        sel = (zbins[iz] <= z) * (z < zbins[iz+1])
        M_av[iz] = np.mean(Mr[sel])
        M_std[iz] = np.std(Mr[sel])
    plt.clf()
    plt.scatter(z, Mr, s=0.01)
    plt.errorbar(zcen, M_av, M_std)
    plt.ylabel('M_r')
    plt.xlabel('z')
    plt.show()


class wakeKcorr():
    """Wake LRG k-corrections."""

    def __init__(self):
        self.A1 = np.loadtxt(lf_data + 'Wake2006/A1.txt')
        self.A2 = np.loadtxt(lf_data + 'Wake2006/A2.txt')
        self.z1 = self.A1[:, 0]
        self.z2 = self.A2[:, 0]
        self.gi1 = self.A1[:, 1]
        self.gi2 = self.A2[:, 1]

    def interp(self, z, gi, what):
        """Interpolate specified k/k+e corr between models at given
        redshift z and g-i colour.
        what is one of k_u, k_g, k_r, k_i, ke_u, ke_g, ke_r, ke_i."""

        idict = {'k_u': 2, 'k_g': 3, 'k_r': 4, 'k_i': 5,
                 'ke_u': 6, 'ke_g': 7, 'ke_r': 8, 'ke_i': 9}
        i = idict[what]

        # First interpolate in redshift to find (g-i)_mod for each model
        assert(z >= self.z1[0] and z <= self.z1[-1])
        gi_mod1 = np.interp(z, self.z1, self.gi1)
        gi_mod2 = np.interp(z, self.z2, self.gi2)

        # Now interpolate desired quantity bteween observed and model colours
        wt1 = (gi_mod2 - gi) / (gi_mod2 - gi_mod1)
        ans = (wt1*np.interp(z, self.z1, self.A1[:, i]) +
               (1-wt1)*np.interp(z, self.z2, self.A2[:, i]))
        return ans, wt1


def lowz_sim(outfile='lowz_sim.fits',
             alpha=-1.23, Mstar=-22, phistar=1e-3, Q=0, P=0,
             Mrange=(-24, -21), mrange=(16, 19.6), zrange=(0.16, 0.36), nz=50,
             H0=100, omega_l=0.7, z0=0.0, area_dg2=8000, nblock=10, pord=4):
    """Generate simulated LOWZ catalogue."""

    def gam_dv(z):
        """Gamma function times volume element to integrate."""
        kc = np.interp(z, ztab, krtab)
        M1 = mrange[1] - cosmo.dist_mod(z) - kc + Q*(z-z0)
        M1 = max(min(Mrange[1], M1), Mrange[0])
        M2 = mrange[0] - cosmo.dist_mod(z) - kc + Q*(z-z0)
        M2 = max(min(Mrange[1], M2), Mrange[0])
        L1 = 10**(0.4*(Mstar - M1))
        L2 = 10**(0.4*(Mstar - M2))
        dens = phistar * 10**(0.4*P*(z-z0)) * mpmath.gammainc(alpha+1, L1, L2)
        ans = area * cosmo.dV(z) * dens
        return ans

    def schec(M):
        """Schechter function."""
        L = 10**(0.4*(Mstar - M))
        ans = 0.4 * ln10 * phistar * L**(alpha+1) * np.exp(-L)
        return ans

    def schec_ev(M, z):
        """Evolving Schechter function."""
        L = 10**(0.4*(Mstar - Q*(z-z0) - M))
        ans = 0.4 * ln10 * phistar * L**(alpha+1) * np.exp(-L)
        return ans

    def vol_ev(z):
        """Volume element multiplied by density evolution."""
        pz = cosmo.dV(z) * 10**(0.4*P*(z-z0))
        return pz

    def zM_pdf(z, M):
        """PDF for joint redshift-luminosity distribution.

        Don't use this.  Generate z and M distributions separately."""
        pz = cosmo.dV(z) * 10**(0.4*P*(z-z0))
        pM = schec_ev(M, z)
        return pz*pM

    """Read Maraston+09 SEDs."""
    sedfile = lf_data+'Maraston2009/M09_models/M09_composite_bestfitLRG.sed'
    data = np.loadtxt(sedfile)
    ages, idxs = np.unique(data[:, 0], return_index=True)
    m09_dir = {}
    for i in range(len(idxs)-1):
        ilo = idxs[i]
        ihi = idxs[i+1]
        spec = astSED.SED(data[ilo:ihi, 1], data[ilo:ihi, 2])
        m09_dir[ages[i]] = spec
    spec = m09_dir[12.]

    # Read Doi+2010 SDSS passbands
    pbfile = lf_data + 'Doi2010/ugriz_atmos.txt'
    doi_g = astSED.Passband(pbfile, normalise=0, transmissionColumn=2)
    doi_r = astSED.Passband(pbfile, normalise=0, transmissionColumn=3)
    doi_i = astSED.Passband(pbfile, normalise=0, transmissionColumn=4)

    area = area_dg2*(math.pi/180.0)*(math.pi/180.0)
    cosmo = util.CosmoLookup(H0, omega_l, zrange)

    # K-correction and colour lookup tables
    ztab = np.linspace(zrange[0], zrange[1], nz)
    kgtab = np.zeros(nz)
    krtab = np.zeros(nz)
    kitab = np.zeros(nz)
    grtab = np.zeros(nz)
    ritab = np.zeros(nz)
    for i in range(len(ztab)):
        specz = spec.copy()
        specz.redshift(ztab[i])
        g_0 = spec.calcMag(doi_g, addDistanceModulus=False, magType='AB')
        r_0 = spec.calcMag(doi_r, addDistanceModulus=False, magType='AB')
        i_0 = spec.calcMag(doi_i, addDistanceModulus=False, magType='AB')
        g_z = specz.calcMag(doi_g, addDistanceModulus=False, magType='AB')
        r_z = specz.calcMag(doi_r, addDistanceModulus=False, magType='AB')
        i_z = specz.calcMag(doi_i, addDistanceModulus=False, magType='AB')
        kgtab[i] = g_z - g_0
        krtab[i] = r_z - r_0
        kitab[i] = i_z - i_0
        grtab[i] = g_z - r_z
        ritab[i] = r_z - i_z

    pcoeffg = np.polynomial.polynomial.polyfit(ztab, kgtab, pord)
    pcoeffr = np.polynomial.polynomial.polyfit(ztab, krtab, pord)
    pcoeffi = np.polynomial.polynomial.polyfit(ztab, kitab, pord)

    plt.clf()
    plt.plot(ztab, kgtab, label='g')
    plt.plot(ztab, np.polynomial.polynomial.polyval(ztab, pcoeffg), label='gp')
    plt.plot(ztab, krtab, label='r')
    plt.plot(ztab, np.polynomial.polynomial.polyval(ztab, pcoeffr), label='rp')
    plt.plot(ztab, kitab, label='i')
    plt.plot(ztab, np.polynomial.polynomial.polyval(ztab, pcoeffi), label='ip')
    plt.xlabel(r'Redshift')
    plt.ylabel(r'$k(z)$')
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(ztab, grtab, label='g-r')
    plt.plot(ztab, ritab, label='r-i')
    plt.xlabel(r'Redshift')
    plt.ylabel(r'Colour')
    plt.legend()
    plt.show()

    # Integrate evolving LF for number of simulated galaxies
    nsim, err = scipy.integrate.quad(gam_dv, zrange[0], zrange[1],
                                     epsabs=1e-3, epsrel=1e-3)
    nsim = int(nsim)
    print('Generating', nsim, 'galaxies')

#    pdb.set_trace()
    nrem = nsim
    nout = 0
    g_out, r_out, i_out, z_out = array('d'), array('d'), array('d'), array('d')
    while nrem > 0:
        z = util.ran_fun(vol_ev, zrange[0], zrange[1], nsim*nblock)
        Mabs = util.ran_fun(schec, Mrange[0], Mrange[1], nsim*nblock) - Q*(z-z0)

        r_obs = Mabs + cosmo.dist_mod(z) + np.interp(z, ztab, krtab)
        g_obs = r_obs + np.interp(z, ztab, grtab)
        i_obs = r_obs - np.interp(z, ztab, ritab)

        # apparent magnitude limits
        sel = (r_obs >= mrange[0]) * (r_obs < mrange[1])
        z, r_obs, g_obs, i_obs = z[sel], r_obs[sel], g_obs[sel], i_obs[sel]
        nsel = len(z)
        if nsel > nrem:
            nsel = nrem
            z, r_obs, g_obs, i_obs = z[:nrem], r_obs[:nrem], g_obs[:nrem], i_obs[:nrem]

        # remaining selection limits
        c_par = 0.7*(g_obs - r_obs) + 1.2*(r_obs - i_obs - 0.18)
        c_perp = np.abs((r_obs - i_obs) - (g_obs - r_obs)/4.0 - 0.18)
        sel = (r_obs < 13.5 + c_par/0.3) * (c_perp < 0.2)
        z, r_obs, g_obs, i_obs = z[sel], r_obs[sel], g_obs[sel], i_obs[sel]
        nobs = len(z)
        g_out.extend(g_obs)
        r_out.extend(r_obs)
        i_out.extend(i_obs)
        z_out.extend(z)
#        t['MODELMAG_G'][nout:nout+nobs] = g_obs
#        t['MODELMAG_R'][nout:nout+nobs] = r_obs
#        t['CMODELMAGCOR_R'][nout:nout+nobs] = r_obs
#        t['MODELMAG_I'][nout:nout+nobs] = i_obs
#        t['Z'][nout:nout+nobs] = z
        nout += nobs
        nrem -= nsel
        print(nrem)

    print(nout, 'out of', nsim, 'galaxies output')
    # Write out as FITS file
    zz = np.zeros(nout)
    ra = 360*np.random.rand(nout)
    dec = (180/math.pi)*np.arccos(2*np.random.rand(nout) - 1) - 90
#    pdb.set_trace()
    t = Table([ra, dec, g_out, r_out, i_out, r_out, z_out, zz,
               np.tile(pcoeffg, (nout, 1)),
               np.tile(pcoeffr, (nout, 1)), np.tile(pcoeffi, (nout, 1))],
              names=('RA', 'DEC', 'MODELMAG_G', 'MODELMAG_R', 'MODELMAG_I',
                     'CMODELMAGCOR_R',
                     'Z', 'CHI2', 'PCOEFF_G', 'PCOEFF_R', 'PCOEFF_I'),
              meta={'omega_l': omega_l, 'z0': z0, 'area': area_dg2,
                    'alpha': alpha, 'Mstar': Mstar, 'phistar': phistar,
                    'Q': Q, 'P': P})
    t.write(outfile, format='fits', overwrite=True)


def area(infile='lowzDR12.fits', nside=64):
    """Estimate survey area using healpix."""
    npixel = hp.nside2npix(nside)
    pixarea = hp.nside2pixarea(nside, degrees=True)
    pixsize = hp.pixelfunc.nside2resol(nside, arcmin=True)
    print(npixel, 'total pixels of size', pixsize, 'arcmin and area',
          pixarea, 'sq deg')
    t = Table.read(infile)
    phi = math.pi/180*t['ra']
    theta = math.pi/180*(90-t['dec'])
    pix = hp.pixelfunc.ang2pix(nside, theta, phi)
    m = np.bincount(pix, minlength=npixel)
    nuse = len(m[m > 0])
    area = nuse*pixarea
    hp.mollview(m, title=infile)
    plt.show()

    nmax = np.max(m)
    binsize = int(max(math.ceil(nmax/50), 1))
    nbins = int(nmax/binsize)
    hmax = nbins*binsize
    print(nmax, binsize, nbins, hmax)
    hist, edges = np.histogram(m[m > 0], bins=np.linspace(0, hmax, nbins))
    mode = scipy.stats.mode(m[m > 0])[0]
#    print(m)
#    pdb.set_trace()
    weight = np.clip(m/mode, 0, 1)
#    print(weight)
    wtsum = np.sum(weight)
    wtarea = wtsum*pixarea
    print(nuse, 'of', npixel, 'occupied pixels, area = ', area)
    print('mode =', mode, 'weighted area = ', wtarea)

    plt.clf()
    plt.hist(m[m > 0], bins=np.linspace(0, hmax, nbins))
    plt.xlabel('counts')
    plt.ylabel('Frequency')
    plt.show()
    plt.clf()
    plt.hist(m[m > 0], bins=np.logspace(0, np.log10(hmax), nbins))
    plt.xlabel('log counts')
    plt.ylabel('Frequency')
    plt.show()
#    hp.zoomtool.mollzoom(m)
