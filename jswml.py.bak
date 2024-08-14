# Joint stepwise maximum likelihood luminosity function and radial density
# estimation using the method of Cole (2011) but allowing for
# individual k-corections
#
# Revision history
#
# 1.0 21-may-13  Based on mlum.py and Shaun's rancat_jswml.f90
# 1.1 08-aug-14  Much simplified by separating out evolution fitting and
#                Vdc_max calculation from LF fitting

import contextlib
import os
import matplotlib
from matplotlib.ticker import MaxNLocator
if not(os.environ.has_key('DISPLAY')):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import AxesGrid
import itertools
import lum
import math
import mpmath
import multiprocessing
import numpy as np
import pmap
import pickle
import pdb
import astropy.io.fits as fits
#import pyqt_fit.kde
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import scipy.stats
import time
import util


# Allow customisation of printed array format,
# see http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)

# Avoid excessive space around markers in legend
matplotlib.rcParams['legend.handlelength'] = 0

# Treatment of numpy errors
np.seterr(all='warn')

# Global parameters
par = {'progName': 'jswml.py', 'version': 1.1, 'ev_model': 'z',
       'clean_photom': True, 'use_wt': True, 'kc_use_poly': True}
cosmo = None
sel_dict = {}
chol_par_name = ('alpha', '   M*', ' phi*', ' beta', '  mu*', 'sigma')
methods = ('lfchi', 'denchi', 'post', 'min_slope', 'zero_slope')
mass_limits = (8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12)
mass_zlimits = (0.15, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12)
mag_limits = (-23, -22, -21, -20, -19, -18, -17, -16, -15)
wmax = 5.0  # max incompleteness weighting

# Constants
lg2pi = math.log10(2 * math.pi)
ln10 = math.log(10)
J3 = 30000.0

# Jacknife regions are 4 deg segments starting at given RA
njack = 9
ra_jack = (129, 133, 137, 174, 178, 182, 211.5, 215.5, 219.5)

# Determined from GAMA-I.  Early/late cut at n = 1.9
Q_all = 1.59
P_all = 0.14
Q_early = 1.31
P_early = 0.14
Q_late = 1.98
P_late = 0.13

# Determined from GAMA-II.
Q_clr = {'c': 0.78, 'b': 0.23, 'r': 0.83}
P_clr = {'c': 1.72, 'b': 3.55, 'r': 1.10}
Qdef, Pdef = 1.0, 1.0

# Factor by which to multiply apparent radius in arcsec to get
# absolute radius in kpc when distance measured in Mpc
radfac = math.pi/180.0/3.6

# Solar magnitudes from Blanton et al 2003 for ^{0.1}ugriz bands
Msun_ugriz = [6.80, 5.45, 4.76, 4.58, 4.51]

# FNugrizYJHK (z0=0) Solar magnitudes from Driver et al 2012
Msun_z0 = {'F': 16.02, 'N': 10.18, 'u': 6.38, 'g': 5.15, 'r': 4.71,
           'i': 4.56, 'z': 4.54, 'Y': 4.52, 'J': 4.57, 'H': 4.71, 'K': 5.19}

# Imaging completeness from Blanton et al 2005, ApJ, 631, 208, Table 1
# Modified to remove decline at bright end and to prevent negative
# completeness values at faint end
sb_tab = (18, 19, 19.46, 19.79, 20.11, 20.44, 20.76, 21.09, 21.41,
          21.74, 22.06, 22.39, 22.71, 23.04, 23.36, 23.69, 24.01,
          24.34, 26.00)
comp_tab = (1.0, 1.0, 0.99, 0.97, 0.98, 0.98, 0.98, 0.97, 0.96, 0.96,
            0.97, 0.94, 0.86, 0.84, 0.76, 0.63, 0.44, 0.33, 0.01)

# Polynomial fits to mass completeness limits from misc.mass_comp()
mass_comp_pfit = {'c': (50.96, -57.42, 23.57, 7.32),
                  'b': (44.40, -51.90, 22.22, 7.21),
                  'r': (25.88, -32.11, 15.62, 8.13)}

# Standard symbol and colour order for plots
sym_list = ('ko', 'bs', 'g^', 'r<', 'mv', 'y>', 'cp')
clr_list = 'bgrck'

# Plot labels
mag_petro_label = r'$^{0.1}M_{r_{\rm Petro}} -\ 5 \lg h$'
mag_sersic_label = r'$^{0.1}M_{r_{\rm Sersic}} -\ 5 \lg h$'
den_mag_label = r'$\phi(M)\ (h^3 {\rm Mpc}^{-3} {\rm mag}^{-1})$'
den_mass_label = r'$\phi(M)\ (h^3 {\rm Mpc}^{-3} {\rm dex}^{-1})$'

# Directory to save plots
plot_dir = os.environ['HOME'] + '/Documents/tex/papers/gama/jswml/'

#------------------------------------------------------------------------------
# Driver routines
#------------------------------------------------------------------------------


def apollo_jobs():
    """Submit several jobs on apollo."""

    dir = '/research/astro/gama/loveday/gama/jswml/auto'
#    for ireal in range(26):
    python_commands = ['import jswml',
                       '''jswml.ev_fit_samples("kcorrz00.fits",
                       "ev_z00_{}_petro_fit_{}.dat")''']
    util.apollo_job(dir, python_commands)


def ev_fit_samples(infile='kcorrz01.fits', outroot='ev_{}_petro_fit_{}.dat',
                   colour='c', param='r_petro', method='lfchi', ev_model='z'):
    "Determine ev parameters and density-corrected Vmax for specified sample."
    par['ev_model'] = ev_model
    par['clean_photom'] = True
    clr_limits = ('a', 'z')
    if (colour == 'b'): clr_limits = ('b', 'c')
    if (colour == 'r'): clr_limits = ('r', 's')
    sel_dict['colour'] = clr_limits
    outfile = outroot.format(method, colour)
    ev_fit(infile, outfile, Pbins=(0.0, 2.5, 25), Qbins=(0.0, 1.5, 30),
           param=param, method=method)
    del sel_dict['colour']


def ev_fit_test(infile='kcorrz01.fits', outroot='ev_test_Mlt{}.dat',
                colour='c', param='r_petro', method='lfchi', ev_model='z',
                Mmax=-12):
    """Determine ev parameters and density-corrected Vmax for specified sample
    using small number of P, Q bins and no optimization."""
    par['ev_model'] = ev_model
    par['clean_photom'] = True
    clr_limits = ('a', 'z')
    if (colour == 'b'): clr_limits = ('b', 'c')
    if (colour == 'r'): clr_limits = ('r', 's')
    sel_dict['colour'] = clr_limits
#    outfile = outroot.format(method, colour)
    outfile = outroot.format(Mmax)
    ev_fit(infile, outfile, Pbins=(0.0, 2.5, 25), Qbins=(0.0, 2.0, 20), opt=0,
           param=param, method=method, Mmax=Mmax)


def ev_fit_mock(infile='kcorrz01.fits', outroot='ev_{:02d}.dat', ireal=0,
                param='SDSS_R_OBS_APP', method='lfchi', ev_model='z'):
    """Evolution fitting for mocks."""
    par['ev_model'] = ev_model
    par['clean_photom'] = False
    par['ireal'] = ireal
    outfile = outroot.format(ireal)
    print(ireal)
    ev_fit(infile, outfile, Pbins=(-1.0, 2.0, 15), Qbins=(-1.0, 1.0, 10),
           param=param, method=method)


def ev_sim_par(inroot='sim_z01_{}.fits', outroot='lf_kde_z01_{}_{}.dat'):
    """Evolution fitting for several simulations run in parallel."""
    i = int(os.environ['SGE_TASK_ID']) - 1
    infile = inroot.format(i)
    for method in ('lfchi', 'post'):
        outfile = outroot.format(method, i)
        print infile, method
        lf_PQ(inFile=infile, outFile=outfile,
              param_list=(('r_petro',
                           (10, 19.8), (10, 19.8, 29), (-23, -15, 32), 0),),
              zmin=0.002, zmax=0.65, nz=65, lf_zbins=((0, 20), (20, 64)),
              Pbins=(0, 3.5, 35), Qbins=(0, 1.5, 30),
              P_prior=(2, 1), Q_prior=(1, 1), method=method, err_type='j3',
              use_mp=True, lf_est='kde')


def Vmax(infile='kcorrz01.fits', evroot='ev_{}_{}_fit_{}.dat',
         outroot='Vmax_{}_{}.fits'):
    """Vmax calculation."""
    par['clean_photom'] = False
    magtype = 'petro'
    for method in ('lfchi', 'post'):
        for colour in 'cbr':
            evfile = evroot.format(method, magtype, colour)
            outfile = outroot.format(method, colour)
            # Output Vmax values for ALL objects
            # sel_dict['colour'] = ('a', 'z')
            Vmax_out(infile, evfile, outfile)


def Vmax_z00(infile='kcorrz00.fits', evfile='ev_z00_lfchi_petro_fit_c.dat',
             outfile='Vmax_z00.fits'):
    """Vmax calculation with k-correction to z=0.0."""
    par['clean_photom'] = True
    Vmax_out(infile, evfile, outfile)


def Vmax_noclean(infile='kcorrz01.fits', evroot='ev_lfchi_{}_fit_c.dat',
                 outroot='Vmax_{}_noclean.fits'):
    """Vmax calculation."""
    par['clean_photom'] = False
    for magtype in ('petro', 'sersic'):
        evfile = evroot.format(magtype)
        outfile = outroot.format(magtype)
        Vmax_out(infile, evfile, outfile)


def Vmax_z_slices(infile='kcorrz01.fits', evroot='ev_lfchi_petro_fit_{}.dat',
                  outroot='Vmax_{}_z_{}_{}.fits'):
    """Vmax in redshift slices."""
    for colour in 'cbr':
        evfile = evroot.format(colour)
        for zrange in ((0.002, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
                       (0.4, 0.5), (0.5, 0.6)):
#        for zrange in ((0.002, 0.06), (0.002, 0.1), (0.1, 0.2), (0.2, 0.3),
#                       (0.3, 0.65)):
            nz = int(math.ceil(100 * (zrange[1] - zrange[0])))
            outfile = outroot.format(colour, zrange[0], zrange[1])
            Vmax_out(infile, evfile, outfile, zmin=zrange[0], zmax=zrange[1],
                     nz=nz)


def Vmax_sim(inroot='sim_z01_{}.fits', evroot='lf_kde_z01_{}_{}.dat',
             outroot='Vmax_{}_{}.fits'):
    """Vmax for simulations."""
    for method in ('post', 'lfchi'):
        for i in xrange(10):
            infile = inroot.format(i)
            evfile = evroot.format(method, i)
            outfile = outroot.format(method, i)
            Vmax_out(infile, evfile, outfile)


def Vmax_mocks(infile='kcorrz01.fits', evroot='ev_{:02d}.dat',
               outroot='Vmax_{:02d}.fits', param='SDSS_R_OBS_APP'):
    """Vmax for mocks."""
    par['clean_photom'] = False
    for ireal in (1, 2, 5, 6, 14, 20):
        par['ireal'] = ireal
        evfile = evroot.format(ireal)
        outfile = outroot.format(ireal)
        Vmax_out(infile, evfile, outfile, param=param)


def Vmax_pinch_test(infile='Vmax_lfchi_c.fits', nz=65, idebug=1,
                    param='r_petro',
                    plot_file='Vmax_pinch_test.png', plot_size=(5, 8)):
    """Vmax pinch test."""

    global par

    hdulist = fits.open(infile)
    header = hdulist[1].header
    par['H0'] = header['H0']
    par['omega_l'] = header['OMEGA_L']
    par['z0'] = header['Z0']
    par['area'] = header['AREA'] * (math.pi/180.0)**2
    par['ev_model'] = header['ev_model']
    Q = header['Q']
    P = header['P']
    mlims = [0, 19.8]
    mlims[0] = header['mlim_0']
    mlims[1] = header['mlim_1']
    zmin = header['zmin']
    zmax = header['zmax']
    ev_model = 'z'
    hdulist.close()

    print 'P, Q, nz, zmin, zmax =', P, Q, nz, zmin, zmax

    par.update({'infile': infile, 'param': param, 'zmin': zmin, 'zmax': zmax,
                'mlims': mlims, 'dmlim': 2, 'Mmin': -99, 'Mmax': 99, 'Mbin': 1,
                'ev_model': ev_model, 'idebug': idebug})

    # Volume out to galaxy redshifts
    # First calculate volumes without evolution
    samp = Sample(infile, par, sel_dict, 0, nqmin=2)
    gala = samp.calc_limits(0)
    zbin, zhist, V, V_int = z_binning(gala, nz, (zmin, zmax))
    zstep = zbin[1] - zbin[0]
    S_obs, S_vis = vis_calc(gala, nz, zmin, zstep, V, V_int)
    converged, Npred, delta, den_var, Pz, Vmax_dc, niter = delta_solve(
        0, 0, gala, nz, (zmin, zmax), zbin, zhist, V, V_int, S_vis)
    V_raw = np.dot(V, S_obs)
    V_dc = np.dot(delta * V, S_obs)

    # Now volume with evolution
    samp = Sample(infile, par, sel_dict, Q, nqmin=2)
    gala = samp.calc_limits(Q)
    zbin, zhist, V, V_int = z_binning(gala, nz, (zmin, zmax))
    zstep = zbin[1] - zbin[0]
    S_obs, S_vis = vis_calc(gala, nz, zmin, zstep, V, V_int)
    converged, Npred, delta, den_var, Pz, Vmax_dec, niter = delta_solve(
        P, Q, gala, nz, (zmin, zmax), zbin, zhist, V, V_int, S_vis)
    V_dec = np.dot(delta * Pz * V, S_obs)

    plt.clf()
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=True, num=1)
    plt.subplots_adjust(hspace=0.0)
    ax = axes[0]
    ax.scatter(gala['appval_sel'], gala['zhi']/gala['z'], 0.1)
    ax.semilogy(basey=10, nonposy='clip')
    ax.set_ylabel('zmax/z')

    ax = axes[1]
    ax.scatter(gala['appval_sel'], samp.Vmax_raw/V_raw, 0.1)
    ax.semilogy(basey=10, nonposy='clip')
    ax.text(0.1, 0.1, 'Raw', transform=ax.transAxes)

    ax = axes[2]
    ax.scatter(gala['appval_sel'], samp.Vmax_dc/V_dc, 0.1)
    ax.semilogy(basey=10, nonposy='clip')
    ax.text(0.1, 0.1, 'DC', transform=ax.transAxes)
    ax.set_ylabel('Vmax/V')
    odd = samp.Vmax_dc/V_dc < 1
    print (samp.Vmax_dc/V_dc)[odd]
#    pdb.set_trace()

    ax = axes[3]
    ax.scatter(gala['appval_sel'], samp.Vmax_dec/V_dec, 0.1)
    ax.semilogy(basey=10, nonposy='clip')
    ax.text(0.1, 0.1, 'DEC', transform=ax.transAxes)
    ax.set_xlim(19, 19.9)
    ax.set_ylim(0.5, 5)
    ax.set_xlabel('r_petro')
    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
        print 'plot saved to ', plot_dir + plot_file


def lf1(infile='Vmax_lfchi_c.fits', outroot='lf_{}_wt{}.dat',
        param='z_sersic', Mmin=-25, Mmax=-12, nbin=26, clean_photom=1,
        use_wt=1):
    """LF in specified band."""
    par['clean_photom'] = clean_photom
    par['use_wt'] = use_wt
    outfile = outroot.format(param, use_wt)
    lf_1d(infile, outfile, param, Mmin, Mmax, nbin)


def lfr(inroot='Vmax_{}_{}.fits', outroot='lf_{}_{}_{}.dat',
        method='lfchi', param='r_petro', Mmin=-25, Mmax=-12, nbin=52,
        clean_photom=True):
    """Petrosian r-band LF."""
    par['clean_photom'] = clean_photom
    for colour in 'cbr':
        clr_limits = ('a', 'z')
        if (colour == 'b'): clr_limits = ('b', 'c')
        if (colour == 'r'): clr_limits = ('r', 's')
        sel_dict['colour'] = clr_limits
        infile = inroot.format(method, colour)
        outfile = outroot.format(param, method, colour)
        lf_1d(infile, outfile, param, Mmin, Mmax, nbin)
        del sel_dict['colour']


def lfr_z00(infile='Vmax_z00.fits', outfile='lf_r_z00.dat',
            param='r_petro', Mmin=-25, Mmax=-12, nbin=52,
            clean_photom=True):
    """Petrosian r-band LF k-corrected to z=0.0."""
    par['clean_photom'] = clean_photom
    lf_1d(infile, outfile, param, Mmin, Mmax, nbin)


def lfr_mocks(inroot='Vmax_{:02d}.fits', outroot='lf_{:02d}.dat',
              method='lfchi', param='SDSS_R_OBS_APP', Mmin=-25, Mmax=-12,
              nbin=52):
    """Petrosian r-band LF for mocks."""
    par['clean_photom'] = False
    for ireal in (1, 2, 5, 6, 14, 20):
        par['ireal'] = ireal
        infile = inroot.format(ireal)
        outfile = outroot.format(ireal)
        lf_1d(infile, outfile, param, Mmin, Mmax, nbin)


def lfr_ev(inroot='Vmax_{}_z_{}_{}.fits', outroot='lf_{}_{}_z_{}_{}.dat',
           param='r_petro', Mmin=-25, Mmax=-12, nbin=52):
    """Evolution of r-band LF."""
    par['clean_photom'] = True
    for colour in 'cbr':
        clr_limits = ('a', 'z')
        if (colour == 'b'): clr_limits = ('b', 'c')
        if (colour == 'r'): clr_limits = ('r', 's')
        sel_dict['colour'] = clr_limits
        for zrange in ((0.002, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.65)):
            infile = inroot.format(colour, zrange[0], zrange[1])
            outfile = outroot.format(param, colour, zrange[0], zrange[1])
            lf_1d(infile, outfile, param, Mmin, Mmax, nbin)
        del sel_dict['colour']


def lfr_zslices(inroot='Vmax_{}_z_{}_{}.fits', outroot='lfr_zslice_{}_{}.dat',
                param='r_petro', Mmin=-25, Mmax=-12, nbin=52,
                Vmax_type='Vmax_raw'):
    """r-band LF in redshift slices, without any evolution corrections."""
    par['clean_photom'] = True
    colour = 'c'
    for zrange in ((0.002, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
                   (0.4, 0.5), (0.5, 0.6)):
        infile = inroot.format(colour, zrange[0], zrange[1])
        outfile = outroot.format(zrange[0], zrange[1])
        lf_1d(infile, outfile, param, Mmin, Mmax, nbin, Vmax_type=Vmax_type,
              Q=0)


def smf(inroot='Vmax_lfchi_{}.fits', outroot='smf_{}.dat',
        param='logmstar_fluxscale', Mmin=6, Mmax=13, nbin=28,
        schec_guess=(-1.0, 10.6, -2)):
    "Stellar mass function."
    global par

    par['clean_photom'] = True
    for colour in 'cbr':
        par['colour'] = colour
        clr_limits = ('a', 'z')
        if (colour == 'b'): clr_limits = ('b', 'c')
        if (colour == 'r'): clr_limits = ('r', 's')
        sel_dict['colour'] = clr_limits
        infile = inroot.format(colour)
        outfile = outroot.format(colour)
        lf_1d(infile, outfile, param, Mmin, Mmax, nbin,
              schec_guess=schec_guess)


def smf_ev(inroot='Vmax_z_{}_{}.fits', outroot='smf_{}_{}_{}.dat',
           param='logmstar_fluxscale', Mmin=6, Mmax=13, nbin=28,
           schec_guess=(-1.0, 10.6, -2)):
    "Evolution of stellar mass function."
    global par

    # par['use_wt'] = False
    par['clean_photom'] = True
    for colour in 'cbr':
        par['colour'] = colour
        clr_limits = ('a', 'z')
        if (colour == 'b'): clr_limits = ('b', 'c')
        if (colour == 'r'): clr_limits = ('r', 's')
        sel_dict['colour'] = clr_limits
        for zrange in ((0.002, 0.06), (0.002, 0.1), (0.1, 0.2), (0.2, 0.3),
                       (0.3, 0.65)):
            infile = inroot.format(zrange[0], zrange[1])
            outfile = outroot.format(colour, zrange[0], zrange[1])
            lf_1d(infile, outfile, param, Mmin, Mmax, nbin,
                  schec_guess=schec_guess)
            

def lf_sim(inroot='Vmax_{}_{}.fits', outroot='lf_bin_{}_{}.fits', 
           param='r_petro', Mmin=-24, Mmax=-12, nbin=48):
    """Petrosian r-band LF for simulations."""
    for method in ('post', 'lfchi'):
        for i in xrange(10):
            infile = inroot.format(method, i)
            outfile = outroot.format(method, i)
            lf_1d(infile, outfile, param, Mmin, Mmax, nbin)


def delta_lf_plots(infile='kcorrz01.fits',
                   param_list=(('r_petro', (14, 19.8), (14, 19.8, 27),
                                (-23, -18, 20), 0),),
                   zmin=0.002, zmax=0.65, nz=65, lf_zbins=((0, 20), (20, 64)),
                   idebug=1, Pbins=(0.0, 4.0, 20), Qbins=(0.0, 2.0, 20),
                   P_prior=(1,1), Q_prior=(1,1), method='lfchi', err_type='jack'):
    "Delta and LF plots to demonstrate lfchi method."

    global par
    par = {'zmin': zmin, 'zmax': zmax, 'idebug': idebug}
    samp = read_gama(infile, param_list)
    qty = samp.qty_list[0]
    lf_bins = np.linspace(qty.absMin, qty.absMax, qty.nabs+1)

    costfn = Cost(samp, nz, (zmin, zmax), lf_bins, lf_zbins, method, 
                  P_prior, Q_prior, Qbins[0], Qbins[1], err_type)
    c = costfn((0.0, 0.0))
    fig = plt.gcf()
    fig.set_size_inches(3, 6)
    plt.savefig(plot_dir + 'delta_lf_0_0.pdf', bbox_inches='tight')
    c = costfn((1.7, 0.7))
    fig = plt.gcf()
    fig.set_size_inches(3, 6)
    plt.savefig(plot_dir + 'delta_lf_1.7_0.7.pdf', bbox_inches='tight')


def ran_gen_sample(infile='kcorrz01.fits', Q=Qdef, P=Pdef,
                   outfile='ranz.dat', param='lum', limits=(-25, -10),
                   colour='c', zmin=0.002, zmax=0.65, nz=65, nfac=30):
    """Generate random distribution for single specified sample."""
    clr_limits = ('a', 'z')
    if (colour == 'b'):
        clr_limits = ('b', 'c')
    if (colour == 'r'):
        clr_limits = ('r', 's')
    sel_dict['colour'] = clr_limits
    Mmin = -25
    Mmax = -10
    if param == 'lum':
        Mmin = limits[0]
        Mmax = limits[1]
    if param == 'mass':
        M_h_corr = 2*math.log10(1.0/0.7)
        sel_dict['logmstar'] = (limits[0] + M_h_corr, limits[1] + M_h_corr)

    par.update({'zmin': zmin, 'zmax': zmax, 'nz': nz,
                'Mmin': Mmin, 'Mmax': Mmax, 'Mbin': 1,
                'param': 'r_petro', 'dmlim': 2, 'mlims': (0, 19.8),
                'idebug': 1})
    ran_gen(infile, outfile, nfac=nfac, Q=Q, P=P, vol=0)


# -----------------------------------------------------------------------------
# Main procedures
# -----------------------------------------------------------------------------


def ev_fit(infile, outfile, mlims=(0, 19.8), param='r_petro',
           Mmin=-24, Mmax=-12, Mbin=48, dmlim=2,
           zmin=0.002, zmax=0.65, nz=65,
           lf_zbins=((0, 20), (20, 65)),
           Pbins=(-0.5, 4.0, 45), Qbins=(0.0, 1.5, 30),
           P_prior=(2, 1), Q_prior=(1, 1),
           idebug=1, method='lfchi', err_type='jack', use_mp=False, opt=True,
           lf_est='kde'):
    """Fit evolution parameters and radial overdensities.
    Searches over both Q and P values,
    rather than trying to estimate P from Cole eqn (25).

    Elements of P_prior and Q_prior are mean and variance."""

    global par
    par.update({'infile': infile, 'mlims': mlims,
                'param': param, 'dmlim': dmlim,
                'zmin': zmin, 'zmax': zmax,
                'Mmin': Mmin, 'Mmax': Mmax, 'Mbin': Mbin,
                'idebug': idebug, 'method': method, 'lf_est': lf_est})

    print '\n************************\njswml.py version ', par['version']
    print method
    print sel_dict
    assert method in methods
    lf_bins = np.linspace(Mmin, Mmax, Mbin+1)

    samp = Sample(infile, par, sel_dict)
    costfn = Cost(samp, nz, (zmin, zmax), lf_bins, lf_zbins, method,
                  P_prior, Q_prior, Qbins[0], Qbins[1], err_type)
    out = {'par': par}

    if par['idebug'] > 0:
        print 'Q, P chi^2 grid using', method

    # Calculate chi^2 on (P,Q) grid to get likelihood contours and to find
    # starting point for minimization
    Qmin = Qbins[0]
    Qmax = Qbins[1]
    nQ = Qbins[2]
    Qstep = float(Qmax - Qmin)/nQ
    Pmin = Pbins[0]
    Pmax = Pbins[1]
    nP = Pbins[2]
    Pstep = float(Pmax - Pmin)/nP
    Qa = np.linspace(Qmin, Qmax, nQ, endpoint=False) + 0.5*Qstep
    Pa = np.linspace(Pmin, Pmax, nP, endpoint=False) + 0.5*Pstep
    plt.clf()

    if use_mp:
        chi2grid = np.array(pmap.parallel_map(lambda Q: [costfn((P, Q)) 
                                                         for P in Pa], Qa))
    else:
        chi2grid = np.array(map(lambda Q: [costfn((P, Q)) for P in Pa], Qa))

    
    extent = (Pmin, Pmax, Qmin, Qmax)
    cmap = matplotlib.cm.jet
    ax = plt.subplot(313)
    im = ax.imshow(chi2grid, cmap=cmap, aspect='auto', origin='lower', 
                   extent=extent, interpolation='nearest')
    cb = plt.colorbar(im, ax=ax)
    (j, i) = np.unravel_index(np.argmin(chi2grid), chi2grid.shape)
    P_maxl = Pmin + (i+0.5)*Pstep
    Q_maxl = Qmin + (j+0.5)*Qstep
    ax.plot(P_maxl, Q_maxl, '+')
    ax.set_xlabel('P')
    ax.set_ylabel('Q')
    plt.draw()

    if opt:
        if par['idebug'] > 0:
            print 'Simplex optimization ...'
        # Use simplex method to optimize ev parameters (P,Q)
        res = scipy.optimize.fmin(costfn, (P_maxl, Q_maxl), xtol=0.1, ftol=0.1, 
                                  full_output=True)
        Popt = res[0][0]
        Qopt = res[0][1]
    else:
        c = costfn((P_maxl, Q_maxl))
        Popt = P_maxl
        Qopt = Q_maxl

    out['Pbins'] = Pbins
    out['Qbins'] = Qbins
    out['chi2grid'] = chi2grid
    out['Pa'] = Pa
    out['Qa'] = Qa
    out['P'] = Popt
    out['P_err'] = 0
    out['Q'] = Qopt
    out['Q_err'] = 0
    out['zbin'] = costfn.zbin
    out['delta'] = costfn.delta
    out['delta_err'] = costfn.delta_err
    out['den_var'] = costfn.den_var
    out['lf_bins'] = lf_bins
    out['phi'] = costfn.phi
    out['phi_err'] = costfn.phi_err
    out['Mbin'] = costfn.Mbin
    out['Mhist'] = costfn.Mhist
    out['whist'] = costfn.whist
    out['ev_fit_chisq'] = costfn.chisq
    out['ev_fit_nu'] = costfn.nu
    fout = open(outfile, 'w')
    pickle.dump(out, fout)
    fout.close()


def Vmax_out(infile, evfile, outfile, param='r_petro',
             zmin=0.002, zmax=0.65, nz=65, mlims=(0, 19.8), idebug=1):
    """Output file of density-corrected Vmax."""

    global par

    dat = pickle.load(open(evfile, 'r'))
    P = dat['P']
    Q = dat['Q']
    try:
        ev_model = dat['par']['ev_model']
    except:
        ev_model = 'z'

    print 'P, Q, nz, zmin, zmax =', P, Q, nz, zmin, zmax

    par.update({'infile': infile, 'param': param, 'zmin': zmin, 'zmax': zmax,
                'mlims': mlims, 'dmlim': 2, 'Mmin': -99, 'Mmax': 99, 'Mbin': 1,
                'ev_model': ev_model, 'idebug': idebug})

    # First calculate Vmax values without evolution
    samp = Sample(infile, par, sel_dict, 0, nqmin=2)
    gala = samp.calc_limits(0)
    zbin, zhist, V, V_int = z_binning(gala, nz, (zmin, zmax))
    zstep = zbin[1] - zbin[0]
    S_obs, S_vis = vis_calc(gala, nz, zmin, zstep, V, V_int)
    Vmax_raw = np.dot(V, S_vis)
    converged, Npred, delta, den_var, Pz, Vmax_dc, niter = delta_solve(
        0, 0, gala, nz, (zmin, zmax), zbin, zhist, V, V_int, S_vis)
#    Vmax_dc = np.dot(delta * V, S)

    # Now include evolution
    samp = Sample(infile, par, sel_dict, Q, nqmin=2)
    gala = samp.calc_limits(Q)
    S_obs, S_vis = vis_calc(gala, nz, zmin, zstep, V, V_int)
    converged, Npred, delta, den_var, Pz, Vmax_dec, niter = delta_solve(
        P, Q, gala, nz, (zmin, zmax), zbin, zhist, V, V_int, S_vis)
#    Vmax_ec = np.dot(Pz * V, S)

    # Output a new file of selected objects
    header = samp.header
    header['H0'] = (par['H0'], 'Hubble parameter (km/s/Mpc)')
    header['mlim_0'] = (mlims[0], 'Petrosian r-band mag bright limit')
    header['mlim_1'] = (mlims[1], 'Petrosian r-band mag faint limit')
    header['ev_model'] = (ev_model, 'ev z-dependence, z1z = z/(1+z)')
    header['Q'] = (Q, 'Luminosity evolution')
    header['P'] = (P, 'Density evolution')
    header['zmin'] = zmin
    header['zmax'] = zmax
    for ik in xrange(5):
        header['kc_{}'.format(ik)] = (samp.kmean[ik],
                                      'Mean kcorr coeff {}'.format(ik))

    hdu = fits.BinTableHDU(data=samp.tbdata, header=header)
    hdu.writeto(outfile, clobber=True)

    # Add Vmax columns
    hdulist = fits.open(outfile)
    cols = hdulist[1].data.columns
    c1 = fits.Column(name='Vmax_raw', format='E', unit='(Mpc/h)^3',
                     array=Vmax_raw)
    c2 = fits.Column(name='Vmax_dc', format='E', unit='(Mpc/h)^3',
                     array=Vmax_dc)
#    c3 = fits.Column(name='Vmax_ec', format='E', unit='(Mpc/h)^3',
#                     array=Vmax_ec)
    c4 = fits.Column(name='Vmax_dec', format='E', unit='(Mpc/h)^3',
                     array=Vmax_dec)
    # pdb.set_trace()
    hdu = fits.BinTableHDU.from_columns(cols + c1 + c2 + c4, header=header)
    hdu.writeto(outfile, clobber=True)


def lf_1d(infile, outfile, param, Mmin=-24, Mmax=-12, nbin=48, dmlim=2,
          Vmax_type='Vmax_dec', Q='read', schec_guess=(-1.0, -20.0, -2),
          saund_guess=(-1.0, -20.0, 0.4, -2), idebug=1, lf_est='bin'):
    """Univariate LF using density-corrected Vmax."""

    global par
    par.update({'infile': infile, 'param': param, 'dmlim': dmlim,
                'idebug': idebug, 'lf_est': lf_est})
    out = {'infile': infile, 'param': param, 'Mmin': Mmin, 'Mmax': Mmax,
           'nbin': nbin, 'Vmax_type': Vmax_type, 'lf_est': lf_est}

    print '\n************************\njswml.py version ', par['version']
    print sel_dict

    gala = read_vmax(infile, sel_dict, param, Vmax_type, Q=Q)
    lf_bins = np.linspace(Mmin, Mmax, nbin+1)

    # Find completeness limits in magnitude (Loveday+2012 sec 3.3)
    Mbright = par['mlims'][0] - dmodk(par['zmax'], par['kc_mean'], par['Q'])
    Mfaint = par['mlims'][1] - dmodk(par['zmin'], par['kc_mean'], par['Q'])
    print 'Mag completeness limits:', Mbright, Mfaint

    if param in ('logmstar', 'logmstar_fluxscale'):
        # 95 percentile completeness in stellar mass
        pfit = mass_comp_pfit[par['colour']]
        Mmin = np.polyval(pfit, par['zmin'])
        comp = (lf_bins[:-1] > Mmin)
        print 'mass completeness limit:', Mmin
        out['Mmin'] = Mmin
    else:
        comp = (lf_bins[1:] < Mfaint) * (lf_bins[:-1] > Mbright)
        out['Mbright'] = Mbright
        out['Mfaint'] = Mfaint

    lf = lf1d(gala, gala['Vmax'], lf_bins)
    schec = util.schec_fit(lf['Mbin'][comp], lf['phi'][comp],
                           lf['phi_err'][comp],
                           schec_guess, sigma=lf['kde_bandwidth'], loud=1)
    saund = util.saund_fit(lf['Mbin'][comp], lf['phi'][comp],
                           lf['phi_err'][comp], saund_guess)
    out['par'] = par
    out['comp'] = comp
    out['alpha'] = schec['alpha']
    out['alpha_err'] = schec['alpha_err']
    out['Mstar'] = schec['Mstar']
    out['Mstar_err'] = schec['Mstar_err']
    out['lpstar'] = schec['lpstar']
    out['lpstar_err'] = schec['lpstar_err']
    out['lf_chi2'] = schec['chi2']
    out['lf_nu2'] = schec['nu']
    out['kde_bandwidth'] = lf['kde_bandwidth']

    plt.clf()
    plt.semilogy(basey=10, nonposy='clip')
    plt.errorbar(lf['Mbin'][comp], lf['phi'][comp], lf['phi_err'][comp],
                 fmt='o')
    util.schec_plot(schec['alpha'], schec['Mstar'], 10**schec['lpstar'],
                    lf['Mbin'][0], lf['Mbin'][-1], lineStyle='--')
    util.saund_plot(saund['alpha'], saund['Mstar'], saund['sigma'],
                    10**saund['lpstar'],
                    lf['Mbin'][0], lf['Mbin'][-1], lineStyle=':')
    plt.xlabel(plot_label(param)[1])
    plt.ylabel(r'$\Phi({})$'.format(plot_label(param)[1]))
    plt.ylim(2e-7, 1)
    plt.draw()

    out['Mbin'] = lf['Mbin']
    out['Mhist'] = lf['Mhist']
    out['phi'] = lf['phi']
    out['phi_err'] = lf['phi_err']
    fout = open(outfile, 'w')
    pickle.dump(out, fout)
    fout.close()


def lfnd(inFile, outFile, param_list, zmin=0.002, zmax=0.65, nz=65,
         idebug=1, lf_est='bin'):
    """Multivariate distribution function given evolution parameters."""

    global par, plot
    par = {'progName': 'jswml.py', 'version': '1.0', 'inFile': inFile,
           'zmin': zmin, 'zmax': zmax, 'idebug': idebug, 'lf_est': lf_est}

    # Default log radius and sb limits
    # par['rad_min'] = -0.6
    # par['rad_max'] = 1.6
    # par['mu_min'] = 15
    # par['mu_max'] = 26
    par['rad_min'] = -99
    par['rad_max'] = 99
    par['mu_min'] = -99
    par['mu_max'] = 99
            
    print '\n************************\njswml.py version ', par['version']
    print param_list, sel_dict

    samp = read_gama(inFile, param_list)
    qty = samp.qty_list[0]
    lf_bins = np.linspace(qty.absMin, qty.absMax, qty.nabs+1)
    Q = qty.Q
    print 'Q =', Q
    out = {'par': par, 'qty_list': samp.qty_list}

    gala = samp.calc_limits()
    zbin, zhist, V, V_int = z_binning(gala, nz, (zmin, zmax))
    zstep = zbin[1] - zbin[0]
    S_obs, S_vis = vis_calc(gala, nz, zmin, zstep, V, V_int)
    converged, Npred, delta, den_var, Pz, Vdc_max, niter = delta_solve(
        P, Q, gala, nz, (zmin, zmax), zbin, zhist, V, V_int, S_vis)
 
    # Jacknife errors on delta
    delta_jack = np.zeros((njack, nz))
    for jack in range(njack):
        idx = (gala['ra'] < ra_jack[jack]) + (gala['ra'] >= ra_jack[jack] + 4.0)
        zhist, bin_edges = np.histogram(
            gala['z'][idx], nz, (zmin, zmax), weights=gala['weight'][idx])
        xx, xx, delta_jack[jack, :], xx, xx, xx, xx = delta_solve(
            P, Q, gala[idx], nz, (zmin, zmax), zbin, zhist, 
            V, V_int, S[:, idx])
    delta_err = np.sqrt((njack-1) * np.var(delta_jack, axis=0))
    delta_poiss_err = delta/np.sqrt(zhist)

    plt.clf()
    ax = plt.subplot(211)
    ax.step(zbin, delta, where='mid')
    ax.errorbar(zbin, delta, delta_err, fmt='none')
    ax.errorbar(zbin, delta, delta_poiss_err, fmt='none')
    ax.errorbar(zbin, delta, np.sqrt(den_var), fmt='none')
    ax.plot([zmin, zmax], [1.0, 1.0], ':')
    ax.set_ylim(0.3, 1.7)
    ax.set_xlabel('Redshift z')          
    ax.set_ylabel(r'$\Delta(z)$')

    if samp.nq > 1:
        lf_bins = []
        lf_range = []
        absStep = 1.0
        for qty in samp.qty_list:
            absStep *= qty.absStep
            lf_bins.append(qty.nabs)
            lf_range.append((qty.absMin, qty.absMax))
        Mhist, whist, phi, phi_err, edges = lfnd(gala, Vdc_max, lf_bins, 
                                                 lf_range, absStep)
        Mbin = edges[0][:-1] + 0.5 * samp.qty_list[0].absStep
        phi_proj = (np.sum(phi, axis=tuple(range(1, samp.nq))) *
                    absStep/samp.qty_list[0].absStep)
        Mhist_proj = np.sum(Mhist, axis=tuple(range(1, samp.nq)))
    else:
        lf = LF1d(gala, Vdc_max, lf_bins)
        (Mbin, Mhist, whist, phi, phi_err) = (
            lf.Mbin, lf.Mhist, lf.whist, lf.phi, lf.phi_err)
        phi_proj = phi
        edges = lf_bins
        # pdb.set_trace()
        schec = util.schec_fit(lf.Mbin, lf.phi, lf.phi_err, 
                           (-1.0, -20.0, -2), sigma=lf.kde_bandwidth)
        out['alpha'] = schec['alpha']
        out['alpha_err'] = schec['alpha_err']
        out['Mstar'] = schec['Mstar']
        out['Mstar_err'] = schec['Mstar_err']
        out['lpstar'] = schec['lpstar']
        out['lpstar_err'] = schec['lpstar_err']
        out['lf_chi2'] = schec['chi2']
        out['lf_nu2'] = schec['nu']
        out['kde_bandwidth'] = lf.kde_bandwidth
 
    ax = plt.subplot(2, 1, 2)
    ax.semilogy(basey=10, nonposy='clip')        
    ax.plot(Mbin, phi_proj)
    fit = Mhist > 0
    ax.set_xlabel(samp.qty_list[0].name)          
    ax.set_ylabel(r'$\Phi({})$'.format(samp.qty_list[0].name))
    ax.set_ylim(2e-7, 1)
    plt.draw()

    out['P'] = P
    out['P_err'] = 0
    out['Q'] = samp.qty_list[0].Q
    out['Q_err'] = 0
    out['zbin'] = zbin
    out['delta'] = delta
    out['delta_err'] = delta_err
    out['den_var'] = den_var
    out['phi'] = phi
    out['phi_err'] = phi_err
    out['edges'] = edges
    out['Mbin'] = Mbin
    out['Mhist'] = Mhist
    out['whist'] = whist
    out['ev_fit_chisq'] = 0
    out['ev_fit_nu'] = 1
    fout = open(outFile, 'w')
    pickle.dump(out, fout)
    fout.close()


def ran_gen(gala, outfile, nfac, Q=Qdef, P=Pdef, vol=0):
    """Generate random distribution nfac times larger than input catalogue."""

    def vol_ev(z):
        """Volume element multiplied by density evolution."""
        pz = cosmo.dV(z) * den_evol(z, P)
        return pz

    global par
    nz = par['nz']
    zmin = par['zmin']
    zmax = par['zmax']

    print par
    print sel_dict

#    f = open(evfile, 'r')
#    dat = pickle.load(f)
#    f.close()
#    P = dat['P']
#    Q = dat['Q']
#    print 'P, Q =', P, Q

#    samp = Sample(infile, par, sel_dict)
#    gala = samp.calc_limits(Q)
    zbin, zhist, V, V_int = z_binning(gala, nz, (zmin, zmax))
    zstep = zbin[1] - zbin[0]
    S_obs, S_vis = vis_calc(gala, nz, zmin, zstep, V, V_int)
    converged, Npred, delta, den_var, Pz, Vdc_max, niter = delta_solve(
        P, Q, gala, nz, (zmin, zmax), zbin, zhist, V, V_int, S_vis)
    V_max = np.dot(Pz * V, S_vis)
    ndupe = np.round(nfac * V_max / Vdc_max).astype(np.int32)

    ngal = len(gala['z'])
    nran = np.sum(ndupe)
    galhist, zbins = np.histogram(gala['z'], nz, (zmin, zmax))
    ranhist = np.zeros(nz)
    zcen = zbins[:-1] + 0.5 * (zbins[1]-zbins[0])
    zstep = (zmax - zmin)/nz

    info = par.copy()
    info.update({'P': P, 'Q': Q, 'nfac': nfac, 
            'ngal': ngal, 'nran': nran,'sel_dict': sel_dict, 
            'galhist': list(galhist), 'zbins': list(zbins), 'zcen': list(zcen)})
    fout = open(outfile, 'w')
    print >> fout, info
    for i in xrange(ngal):
        if vol:
            z = util.ran_fun(vol_ev, zmin, zmax, ndupe[i])
        else:
            # Avoid tail of randoms beyond highest z galaxy
            # zhi = min(zmaxg, gala['zhi'][i])
            z = util.ran_fun(vol_ev, gala['zlo'][i], gala['zhi'][i], ndupe[i])
        for j in xrange(len(z)):
            print >> fout, z[j], V_max[i], gala['weight'][i]
            jz = int((z[j] - zmin)/zstep)
            ranhist[jz] += 1
    fout.close()
    print nran, ' redshifts output'

    plt.clf()
    plt.step(zcen, galhist, where='mid')
    plt.plot(zcen, ranhist*float(ngal)/nran)
    plt.xlabel('Redshift')
    plt.ylabel('Frequency')
    plt.draw()


def vol_limit(Mmin, Mmax, zmin, zmax, infile='kcorrz01.fits', Q=0):
    """Form volume-limited sample between specified absolute magnitiude
    and redshift limits."""

    global par
    par['clean_photom'] = 0
    par['nz'] = 1
    par['param'] = 'r_petro'
    par['Mbin'] = 1
    par['Mmin'] = Mmin
    par['Mmax'] = Mmax
    par['zmin'] = zmin
    par['zmax'] = zmax

    samp = Sample(infile, par, sel_dict, Q=Q)
    z = samp.gal_arr['z']
    M = samp.gal_arr['absval_sel']
    idx = (zmin < z) * (z < zmax)
    print('{:d} galaxies in redshift range'.format(len(z[idx])))
    idx = (zmin < z) * (z < zmax) * (Mmin < M) * (M < Mmax)
    print('{:d} galaxies in vol limit'.format(len(z[idx])))


def z_binning(gala, nz, (zmin, zmax)):
    """Redshift binning and histogram"""
    zhist, bin_edges = np.histogram(gala['z'], nz, (zmin, zmax), 
                                    weights=gala['weight'])
    zstep = bin_edges[1] - bin_edges[0]
    zbin = bin_edges[:-1] + 0.5 * zstep
    V_int = par['area'] / 3.0 * cosmo.dm(bin_edges)**3
    V = np.diff(V_int)
    return zbin, zhist, V, V_int

def vis_calc(gala, nz, zmin, zstep, V, V_int):
    """Arrays S_obs and S_vis contain volume-weighted fraction of 
    redshift bin iz in which galaxy igal lies and is visible."""

    afac = par['area'] / 3.0
    ngal = len(gala)
    S_obs = np.zeros((nz, ngal))
    S_vis = np.zeros((nz, ngal))

    for igal in xrange(ngal):
        ilo = min(nz-1, int((gala['zlo'][igal] - zmin) / zstep))
        ihi = min(nz-1, int((gala['zhi'][igal] - zmin) / zstep))
        iob = min(nz-1, int((gala['z'][igal] - zmin) / zstep))
        S_obs[ilo+1:iob, igal] = 1
        S_vis[ilo+1:ihi, igal] = 1
        Vp = V_int[ilo+1] - afac*cosmo.dm(gala['zlo'][igal])**3
        S_obs[ilo, igal] = Vp/V[ilo]
        S_vis[ilo, igal] = Vp/V[ilo]
        Vp = afac*cosmo.dm(gala['z'][igal])**3 - V_int[iob]
        S_obs[iob, igal] = Vp/V[ihi]
        Vp = afac*cosmo.dm(gala['zhi'][igal])**3 - V_int[ihi]
        S_vis[ihi, igal] = Vp/V[ihi]
    return S_obs, S_vis

def delta_solve(P, Q, gala, nz, (zmin, zmax), zbin, zhist, V, V_int, S, 
                nitermax=50, delta_tol=1e-4):
    """Solve for overdensity delta for given P, Q."""

    converged = False
    niter = 0
    Npred = np.zeros(nz)
    delta_old = np.ones(nz)
    Pz = den_evol(zbin, P)
    # pdb.set_trace()
    if par['idebug'] > 1:
        print 'iteration  max(delta Delta)'
    while (not(converged) and niter < nitermax):
        # Density-corrected Vmax estimates for each galaxy
        Vdc_max = np.dot(delta_old * Pz * V, S)
            
        # Predicted mean galaxy number per redshift bin
        for iz in xrange(nz):
            Npred[iz] = (Pz[iz] * V[iz] * S[iz,:] * gala['weight'] / Vdc_max).sum()

        # Overdensity = weighted sum of galaxies in bin / predicted
        delta = np.ones(nz)
        occ = Npred > 0
        delta[occ] = zhist[occ]/Npred[occ]

        # Check for convergance
        delta_err = np.max(np.absolute(delta - delta_old))
        if par['idebug'] > 1:
            print niter, delta_err
        delta_old = delta
        if delta_err < delta_tol:
            converged = True
        niter += 1
    den_var = (1 + J3*Npred/V) / Npred

    return converged, Npred, delta, den_var, Pz, Vdc_max, niter

def delta_P_solve(Q, gala, zbin, zhist, V, V_int, S, P_prior, nitermax=50, 
                  delta_tol=1e-4, P_tol=1e-3):
    """Solve for LF, density variation and density evolution P for given  
    luminosity evolution parameter Q."""

    converged = False
    niter = 0
    P = 0.0
    mu = 0.0
    nz = len(zbin)
    Npred = np.zeros(nz)
    delta = np.ones(nz)
    if par['idebug'] > 1:
        print 'iteration  Q   P    mu    max delta change'
    while (not(converged) and niter < nitermax):
        P_old = P
        mu_old = mu
        delta_old = delta
        Pz = den_evol(zbin, P)
        # pdb.set_trace()
        # Regular and density-corrected Vmax estimates for each galaxy
        V_max = np.dot(Pz * V, S)
        Vdc_max = np.dot(delta * Pz * V, S)

        # Lagrange multiplier mu
        try:
            mu = scipy.optimize.newton(mufunc, mu_old, fprime=muprime, 
                                       args=(V_max, Vdc_max), tol=1e-3)
        except:
            mu = 0.0

        # Predicted mean galaxy number and variance per redshift bin
        for iz in xrange(nz):
            Npred[iz] = (Pz[iz] * V[iz] * S[iz,:] * gala['weight'] / 
                         (Vdc_max + mu*V_max)).sum()
        den_var = (1 + J3*Npred/V) / Npred

        # Overdensity delta via solution of quadratic eqn (23)
        delta = np.ones(nz)
        arg = (1 - Npred*den_var)**2 + 4*zhist*den_var
        idx = arg >= 0
        delta[idx] = 0.5 * (1 - Npred[idx]*den_var[idx] + arg[idx]**0.5)

        # Solve for density evolution parameter P via eqn (25)
        P = 0.4*ln10*P_prior[1]/nz * np.sum(zbin * (zhist - Npred*(delta+mu)))
        P = min(max(P, -5), 5)

        # Check for convergance
        P_err = abs(P - P_old)
        delta_err = np.max(np.absolute(delta - delta_old))
        if par['idebug'] > 1:
            print niter, Q, P, mu, delta_err
        if P_err < P_tol and delta_err < delta_tol:
            converged = True
        niter += 1

    V_max = np.dot(Pz * V, S)
    Vdc_max = np.dot(delta * Pz * V, S)
    V_max_corr = Vdc_max + mu*V_max
    return converged, P, mu, Npred, delta, den_var, Pz, V_max_corr


def lf1d(gala, V_max_corr, lf_bins):
    """Calculates univariate LF."""
    
    absval = gala['absval_lf']
    Mbin = lf_bins[:-1] + 0.5*np.diff(lf_bins)
    # for i in xrange(len(Mbin)):
    #     idx = (lf_bins[i] <= absval) * (absval < lf_bins[i+1])
    #     Mbin[i] = np.mean(absval[idx])

    Mhist, edges = np.histogram(absval, lf_bins)
    whist, edges = np.histogram(absval, lf_bins, weights=gala['weight'])
    wt = gala['weight']/V_max_corr
    if par['lf_est'] == 'bin':
        phi, edges = np.histogram(absval, lf_bins, weights=wt)
        phi /= np.diff(lf_bins)
        kde_bandwidth = 0
    if par['lf_est'] == 'kde':
        kde = pyqt_fit.kde.KDE1D(absval, lower=lf_bins[0], 
                                 upper=lf_bins[-1], weights=wt)
        phi = kde(Mbin) * wt.sum()
        kde_bandwidth = kde.bandwidth

    # Jackknife errors
    phi_jack = np.zeros((njack, len(phi)))
    for jack in range(njack):
        idx = (gala['ra'] < ra_jack[jack]) + (gala['ra'] >= ra_jack[jack] + 4.0)
        if par['lf_est'] == 'bin':
            phi_jack[jack, :], edges = np.histogram(
                absval[idx], lf_bins, weights=wt[idx])
            phi_jack[jack, :] *= float(njack)/(njack-1)/np.diff(lf_bins)
        if par['lf_est'] == 'kde':
            kde = pyqt_fit.kde.KDE1D(absval[idx], lower=lf_bins[0], 
                                     upper=lf_bins[-1], weights=wt[idx])
            phi_jack[jack, :] = kde(Mbin) * wt[idx].sum() * njack / (njack-1)
    phi_err = np.sqrt((njack-1) * np.var(phi_jack, axis=0))
    lf = {'Mbin': Mbin, 'Mhist': Mhist, 'whist': whist, 
          'phi': phi, 'phi_err': phi_err, 'kde_bandwidth': kde_bandwidth}
    return lf

def lfnd(gala, V_max_corr, lf_bins, lf_range, absStep):
    """Returns n-d LF with errors from jackknife sampling."""

    absval = gala['absval']
    Mhist, edges = np.histogramdd(absval, lf_bins, lf_range)
    whist, edges = np.histogramdd(absval, lf_bins, lf_range, weights=gala['weight'])
    phi, edges = np.histogramdd(absval, lf_bins, 
                                weights=gala['weight'] / V_max_corr)
    phi /= absStep
    # phi_err = phi/np.sqrt(Mhist)
    # phi_err[np.isnan(phi_err)] = 99  # for empty bins
    # phi_err[np.isinf(phi_err)] = 99  # for empty bins

    # Jackknife errors
    phi_jack = np.zeros(([njack] + list(phi.shape)))
    for jack in range(njack):
        idx = (gala['ra'] < ra_jack[jack]) + (gala['ra'] >= ra_jack[jack] + 4.0)
        phi_jack[jack, :], edges = np.histogramdd(
            absval[idx], lf_bins, weights=gala['weight'][idx] / V_max_corr[idx])
        phi_jack[jack, :] *= float(njack)/(njack-1)/absStep
    phi_err = np.sqrt((njack-1) * np.var(phi_jack, axis=0))
    return Mhist, whist, phi, phi_err, edges

class Cost(object):
    """Cost function and associated parameters."""

    def __init__(self, samp, nz, (zmin, zmax), lf_bins, lf_zbins, 
                 method, P_prior, Q_prior, Qmin, Qmax, err_type='jack'):
        self.samp = samp
        self.nz = nz
        self.zmin = zmin
        self.zmax = zmax
        self.zbin_edges, self.zstep = np.linspace(zmin, zmax, nz+1, retstep=True)
        self.zbin = self.zbin_edges[:-1] + 0.5 * self.zstep
        self.dist_mod = cosmo.dist_mod(self.zbin)
        self.V_int = par['area'] / 3.0 * cosmo.dm(self.zbin_edges)**3
        self.V = np.diff(self.V_int)
        self.lf_bins = lf_bins
        self.lf_zbins = lf_zbins
        self.method = method
        self.P_prior = P_prior
        self.Q_prior = Q_prior
        self.Qmin = Qmin
        self.Qmax = Qmax
        self.err_type = err_type
        self.Q = -99.0
        self.delta_old = np.ones(nz)

        # Mag bin limits for LF
        if par['idebug'] > 0:
            print 'Setting LF bin limits Qmin, Qmax = ', Qmin, Qmax
        if self.method == 'post':
            self.binidx = np.ones(len(self.lf_bins) - 1, dtype=bool)
        if self.method == 'lfchi':
            self.binidx = np.ones((len(self.lf_zbins), len(self.lf_bins) - 1), 
                                  dtype=bool)
            zstep = (zmax - zmin)/nz
            if par['idebug'] > 0:
                print 'zlo, zhi, Mmin, Mmax, nbins'
        for Q in (Qmin, Qmax):
            gala = samp.calc_limits(Q)
            if self.method == 'post':
                Mhist, edges = np.histogram(gala['absval_lf'], lf_bins)
                self.binidx *= (Mhist > 0)
                if par['idebug'] > 0 and Q == self.Qmax:
                    print 'Non-zero LF bins: ', Mhist[self.binidx]

            if self.method == 'lfchi':
                for iz in range(len(lf_zbins)):
                    zlo = zmin + lf_zbins[iz][0]*zstep
                    zhi = zmin + lf_zbins[iz][1]*zstep
                    idx = (zlo <= gala['z']) * (gala['z'] < zhi)
                    Mhist, edges = np.histogram(gala['absval_lf'][idx], lf_bins)
                    Mmin = par['mlims'][0] - dmodk(zhi, samp.kmean, Q)
                    Mmax = par['mlims'][1] - dmodk(zlo, samp.kmean, Q)
                    Mlo = edges[:-1]
                    Mhi = edges[1:]
                    self.binidx[iz, :] *= (Mhi < Mmax) * (Mlo > Mmin) * (Mhist > 9)
                    if par['idebug'] > 0 and Q == self.Qmax:
                        print zlo, zhi, Mmax, len(Mhist[self.binidx[iz, :]])

    def __call__(self, (P, Q)):
        """Returns cost for evolution parameters (P, Q).  
        If P is None then solve for P."""

        if Q != self.Q:
            self.Q = Q
            self.gala = self.samp.calc_limits(Q)
            self.S_obs, self.S_vis = vis_calc(
                self.gala, self.nz, self.zmin, self.zstep, self.V, self.V_int)
            self.zhist, bin_edges = np.histogram(
                self.gala['z'], self.nz, (self.zmin, self.zmax), 
                weights=self.gala['weight'])
            Mhi = (
                par['Mmax'] - self.dist_mod - 
                kcorr(self.zbin, self.gala['kcoeff'].transpose()) + 
                ecorr(self.zbin, Q))
            self.hi_bin, self.hi_frac = self.samp.abs_bin(Mhi)
            Mlo = (
                par['Mmin'] - self.dist_mod - 
                kcorr(self.zbin, self.gala['kcoeff'].transpose()) + 
                ecorr(self.zbin, Q))
            self.lo_bin, self.lo_frac = self.samp.abs_bin(Mlo)
            # pdb.set_trace()
        if P is None:
            (converged, self.P, self.mu, Npred, self.delta, 
             self.den_var, Pz, Vdc_max) = delta_P_solve(
                Q, self.gala, self.zbin, self.zhist, self.V, self.V_int, 
                self.S_vis, self.P_prior, self.delta_old)
        else:
            self.P = P
            (converged, Npred, self.delta, self.den_var, 
             Pz, Vdc_max, niter) = delta_solve(
                P, Q, self.gala, self.nz, (self.zmin, self.zmax), self.zbin, 
                self.zhist, self.V, self.V_int, self.S_vis)
        self.delta_old = self.delta

        # Jacknife errors on delta
        if self.err_type == 'jack':
            delta_jack = np.zeros((njack, self.nz))
            for jack in range(njack):
                idx = ((self.gala['ra'] < ra_jack[jack]) + 
                       (self.gala['ra'] >= ra_jack[jack] + 4.0))
                zhist, bin_edges = np.histogram(
                    self.gala['z'][idx], self.nz, (self.zmin, self.zmax), 
                    weights=self.gala['weight'][idx])
                if P is None:
                    xx, xx, delta_jack[jack, :], xx, xx, xx = delta_P_solve(
                        P, Q, self.gala[idx], self.zbin, zhist, 
                        self.V, self.V_int, self.S_vis[:, idx], self.delta_old)
                else:
                    xx, xx, delta_jack[jack, :], xx, xx, xx, xx = delta_solve(
                        P, Q, self.gala[idx], self.nz, (self.zmin, self.zmax), 
                        self.zbin, zhist, self.V, self.V_int, 
                        self.S_vis[:, idx])
                self.delta_err = np.sqrt((njack-1) * np.var(delta_jack, axis=0))
            del_var = self.delta_err**2
        else:
            self.delta_err = np.zeros(self.nz)
            del_var = self.den_var

        ax = plt.subplot(311)
        plt.cla()
        ax.step(self.zbin, self.delta, where='mid')
        ax.errorbar(self.zbin, self.delta, self.delta_err, fmt='none')
        ax.bar(self.zbin - 0.5*self.zstep, 2*np.sqrt(self.den_var), 
               width=self.zstep, bottom=self.delta - np.sqrt(self.den_var), 
               alpha=0.1, ec='none')
        ax.plot([self.zmin, self.zmax], [1.0, 1.0], ':')
        ax.set_ylim(0, 5)
        ax.set_xlabel('Redshift z')          
        ax.set_ylabel(r'$\Delta(z)$')
        ax.text(0.1, 0.9, r'$P = {:4.2f},\ Q = {:4.2f}$'.format(P, Q),
                transform = ax.transAxes)

        lf = lf1d(self.gala, Vdc_max, self.lf_bins)
        (self.Mbin, self.Mhist, self.whist, self.phi, self.phi_err) = (
            lf['Mbin'], lf['Mhist'], lf['whist'], lf['phi'], lf['phi_err'])

        ax = plt.subplot(312)
        plt.cla()
        ax.errorbar(lf['Mbin'], lf['phi'], lf['phi_err'])
        ax.set_xlabel(r'$M_r$')
        ax.set_ylabel(r'$\Phi(M_r)$')
        ax.semilogy(basey=10, nonposy='clip')
        ax.set_ylim(1e-5, 0.05)
        plt.subplots_adjust(hspace=0.25)
        plt.draw()
        
        if self.method == 'post':
            idx = (self.delta > 0) * (del_var > 0) 
            densum = np.sum(self.zhist[idx] *
                            np.log(self.V[idx]*Pz[idx]*self.delta[idx]))
            # phisum = np.sum(self.whist[self.binidx]*np.log(self.phi[self.binidx]))
            phisum = np.sum(self.whist*np.log(self.phi))

            sum1 = np.zeros((len(self.gala), len(Pz)))
            for igal in xrange(len(self.gala)):
                for iz in xrange(len(Pz)):
                    sum1[igal, iz] = (
                        self.lo_frac[igal, iz]*self.phi[self.lo_bin[igal, iz]] + 
                        self.phi[self.lo_bin[igal, iz]+1:self.hi_bin[igal, iz]-1].sum() +
                        self.hi_frac[igal, iz]*self.phi[self.hi_bin[igal, iz]])
            sum2 = np.dot(sum1, self.V*Pz*self.delta)
            xsum = np.sum(self.gala['weight'] * np.log(sum2))

            lnL = ((densum + phisum - xsum)/len(self.gala) -
                   ((self.delta[idx]-1)**2/(2*del_var[idx])).sum() - 
                   (self.P-self.P_prior[0])**2/(2*self.P_prior[1]) - 
                   (self.Q-self.Q_prior[0])**2/(2*self.Q_prior[1]))

            self.chisq = -2*lnL
            self.nu = len(self.gala) - 2
            if par['idebug'] > 0:
                print '{:5.2f} {:5.2f} {:6d} {:e} {:e} {:e} {:e} {:6d} {:b} {:3d}'.format(
                    self.Q, self.P, len(self.gala), densum, phisum, xsum, lnL, 
                    len(self.zhist[idx]), converged, niter)
                # pdb.set_trace()
            if math.isnan(self.chisq):
                self.chisq = 1e9
            return self.chisq

        if self.method == 'denchi':
            self.chisq = ((self.delta-1)**2 / del_var).sum()
            self.nu = len(self.delta) - 2
            if par['idebug'] > 0:
                print '{:5.2f} {:5.2f} {:6d} {:e} {:b} {:3d}'.format(
                    self.Q, self.P, len(self.gala), self.chisq, 
                    converged, niter)
            if not(converged):
                self.chisq *= 10
            return self.chisq

        if self.method == 'lfchi':
            zstep = (self.zmax - self.zmin)/self.nz
            phiz = np.zeros((len(self.lf_zbins), len(self.lf_bins) - 1))
            phiz_err = np.zeros((len(self.lf_zbins), len(self.lf_bins) - 1))
            for iz in range(len(self.lf_zbins)):
                izlo = self.lf_zbins[iz][0]
                izhi = self.lf_zbins[iz][1]
                zlo = self.zmin + izlo*zstep
                zhi = self.zmin + izhi*zstep
                galidx = (zlo <= self.gala['z']) * (self.gala['z'] < zhi)
                galz = self.gala[galidx]
                V_max = np.dot(self.delta[izlo:izhi] * 
                               Pz[izlo:izhi] * self.V[izlo:izhi],
                               self.S_vis[izlo:izhi, galidx])
                lfz = lf1d(galz, V_max, self.lf_bins)
                phiz[iz, :] = lfz['phi']
                phiz_err[iz, :] = lfz['phi_err']
                ax.errorbar(lfz['Mbin'][self.binidx[iz, :]], 
                            phiz[iz, self.binidx[iz, :]], 
                            phiz_err[iz, self.binidx[iz, :]])
                plt.draw()

            idx = del_var > 0
            self.chisq = ((self.delta[idx]-1)**2 / del_var[idx]).sum()
            self.nu = len(self.delta[idx]) - 2
            for iz in range(len(self.lf_zbins) - 1):
                for jz in range(iz+1, len(self.lf_zbins)):
                    idx = self.binidx[iz] * self.binidx[jz]
                    self.nu += len(lfz['Mbin'][idx])
                    self.chisq += np.sum((phiz[iz, idx] - phiz[jz, idx])**2 /
                                  (phiz_err[iz, idx]**2 + phiz_err[jz, idx]**2))

            ax.text(0.1, 0.9, r'$\chi^2 = {:6.1f}$'.format(self.chisq),
                    transform = ax.transAxes)
            if par['idebug'] > 0:
                print '{:5.2f} {:5.2f} {:6d} {:e} {:4d} {:b} {:3d}'.format(
                    self.Q, self.P, len(self.gala), self.chisq, self.nu, 
                    converged, niter)
                # pdb.set_trace()
            # if not(converged):
            #     self.chisq *= 10
            return self.chisq

        self.chisq = 0
        self.nu = 0
        popt, pcov = scipy.optimize.curve_fit(lambda x, m, c: m*x + c,
                                              self.zbin, self.delta, p0=(0, 1),
                                              sigma = np.sqrt(del_var))
        if par['idebug'] > 0:
            print '{:5.2f} {:5.2f} {:6d} {:e} {:b} {:3d}'.format(
                    self.Q, self.P, len(self.gala), popt[0], converged, niter)

        if par['idebug'] > 1:
            ax = plt.subplot(2, 1, 1)
            ax.plot((0.0, 0.5), np.polyval(popt, (0.0, 0.5)), '--')
            plt.draw()

        if self.method == 'min_slope':
            return abs(popt[0])
        else:
            if not(converged):
                if Q <= self.Qmin:
                    return 9
                else:
                    return -9
            return popt[0]

def M_binning(inFile, param_list, zmin=0.002, zmax=0.5, idebug=1, 
              bins=(-25, -22, -21, -20, -19, -18, -17, -16, -15, -12)):
    """Test bin occupancy for different mag bins."""

    global par, plot
    par = {'progName': 'jswml.py', 'version': '1.0', 'inFile': inFile,
           'zmin': zmin, 'zmax': zmax, 'idebug': idebug}

    samp = read_gama(inFile, param_list)
    for Q in (0, 1, 2, 3):
        samp.qty_list[0].Q = Q
        samp.calc_limits()
        Mhist, bin_edges = np.histogram(samp.absval[:,0], bins)
        print 'Q = ', Q, Mhist
    
    
def den_evol(z, P):
    """Density evolution at redshift z."""
    assert par['ev_model'] in ('z', 'z1z')
    if par['ev_model'] == 'z':
        return 10**(0.4*P*z)
    if par['ev_model'] == 'z1z':
        return 10**(0.4*P*z/(1+z))

def read_gama(file, param_list):
    """Read GAMA data from fits file."""

    global par, cosmo
    zmin, zmax = par['zmin'], par['zmax']
    hdulist = fits.open(file)
    header = hdulist[1].header
    par['H0'] = 100.0
    if 'Mock' in file:
        samp_type = 'mock'
        par['omega_l'] = 0.75
        par['z0'] = 0.2
        par['area'] = 180*(math.pi/180.0)*(math.pi/180.0)
    else:
        samp_type = 'gama'
        par['omega_l'] = header['OMEGA_L']
        par['z0'] = header['Z0']
        par['area'] = header['AREA']
        # par['omega_l'] = 0.7
        # par['z0'] = 0.1
        # par['area'] = 180.0
        par['area'] *= (math.pi/180.0)*(math.pi/180.0)
    cosmo = util.CosmoLookup(par['H0'], par['omega_l'], (zmin, zmax))

    print 'H0, omega_l, z0, area/Sr = ', par['H0'], par['omega_l'], par['z0'], par['area']

    # Read simulation parameters if present
    try:
        par['sim_par'] = (header['alpha'], header['Mstar'], header['phistar'],
                          header['Q'], header['P'])
    except:
        par['sim_par'] = None
        
    qty_list = []
    for i in xrange(len(param_list)):
        qty_list.append(Qty(param_list[i]))
        
    samp = Sample(samp_type, qty_list)
    samp.read(hdulist[1].data)
    napp = samp.ngal
    print napp, 'objects satisfy apparent limits'

    # Select objects with good redshifts
    if samp_type == 'gama':
        idx = samp.gal_arr['q'] > 2
        samp = samp.subset(idx)
        nz = len(samp.gal_arr)
        print '{} objects ({:4.1f}%) with reliable redshifts'.format(
            nz, 100.0*nz/napp)
        print 'mean weight = ', np.mean(samp.gal_arr['weight'])
    
    # Apply redshift limits
    idx = (zmin <= samp.gal_arr['z']) * (samp.gal_arr['z'] < zmax)
    samp = samp.subset(idx)
    ngal = len(samp.gal_arr)
    par['ngal'] = ngal
    print ngal, 'objects selected'
    samp.ngal = ngal
        
    hdulist.close()

    return samp

#------------------------------------------------------------------------------
# Support classes and functions
#------------------------------------------------------------------------------


class Sample(object):
    """A sample of galaxies, whose attributes are stored in
    structured array gal_arr."""

    def __init__(self, infile, selpar, sel_dict, Q=0, chi2max=10, nqmin=3):
        """Read selected objects from FITS table."""

        global cosmo, par

        par = selpar
        zmin, zmax = par['zmin'], par['zmax']
        self.Mmin = par['Mmin']
        self.Mmax = par['Mmax']
        self.Mbin = par['Mbin']
        self.Mstep = float(self.Mmax - self.Mmin)/self.Mbin

        hdulist = fits.open(infile)
        header = hdulist[1].header
        tbdata = hdulist[1].data
        cols = hdulist[1].columns
        par['H0'] = 100.0
        par['omega_l'] = header['OMEGA_L']
        par['z0'] = header['Z0']
        par['area'] = header['AREA'] * (math.pi/180.0)**2
        cosmo = util.CosmoLookup(par['H0'], par['omega_l'], (zmin, zmax))
        self.par = par
        self.cosmo = cosmo
        print('H0, omega_l, z0, area/Sr = ',
              par['H0'], par['omega_l'], par['z0'], par['area'])

        try:
            alpha = header['alpha']
            sim = True
            print 'Simulated data'
        except:
            sim = False

        if 'IREAL' in cols.names:
            mock = True
            sel = ((tbdata['ireal'] == par['ireal']) *
                   (tbdata['redshift_obs'] >= zmin) *
                   (tbdata['redshift_obs'] < zmax))
        else:
            mock = False
            sel = ((tbdata['survey_class'] > 3) *
                   (tbdata['nq'] >= nqmin) *
                   (tbdata['z_tonry'] >= zmin) *
                   (tbdata['z_tonry'] < zmax))

        # Apply other selection limits in sel_dict
        for key, limits in sel_dict.iteritems():
            print key, limits
            sel *= ((tbdata[key] >= limits[0]) *
                    (tbdata[key] < limits[1]))
            par[key] = limits

        # Exclude objects with suspect photometry
        # pdb.set_trace()
        if par['clean_photom']:
            ncand = len(tbdata[sel])
            sel *= ((tbdata['bn_objid'] < 0) *
                    (np.fabs(tbdata['r_petro'] - tbdata['r_sersic']) <
                    par['dmlim']))
            nclean = len(tbdata[sel])
            print nclean, 'out of', ncand, 'targets with clean photometry'

        tbdata = tbdata[sel]
        ngal = len(tbdata)
        nk = tbdata['pcoeff_r'].shape[1]
        gal_arr = np.zeros(
            ngal,
            dtype=[('cataid', 'int32'),
                   ('appval_sel', 'float32'), ('absval_sel', 'float32'),
                   ('appval_lf', 'float32'), ('absval_lf', 'float32'),
                   ('ra', 'float32'), ('dec', 'float32'),
                   ('weight', 'float32'),
                   ('kc', 'float32'), ('kcoeff', 'float32', nk),
                   ('z', 'float32'), ('zlo', 'float32'), ('zhi', 'float32')
                   ])
        if mock:
            z = tbdata['redshift_obs']
            gal_arr['cataid'] = 0
            gal_arr['ra'] = tbdata['ra']
            gal_arr['dec'] = tbdata['dec']
            gal_arr['appval_sel'] = tbdata['SDSS_r_obs_app']
            gal_arr['appval_lf'] = tbdata[par['param']]
            gal_arr['z'] = z
            gal_arr['kc'] = tbdata['kcorr_r']
            gal_arr['kcoeff'] = tbdata['pcoeff_r']
        else:
            z = tbdata['z_tonry']
            gal_arr['cataid'] = tbdata['cataid']
            gal_arr['ra'] = tbdata['ra']
            gal_arr['dec'] = tbdata['dec']
            gal_arr['appval_sel'] = tbdata['r_petro']
            gal_arr['appval_lf'] = tbdata[par['param']]
            gal_arr['z'] = z
            gal_arr['kc'] = tbdata['kcorr_r']
            gal_arr['kcoeff'] = tbdata['pcoeff_r']

        if par['kc_use_poly']:
            gal_arr['kc'] = np.polynomial.polynomial.polyval(
                z - par['z0'], gal_arr['kcoeff'].transpose(), tensor=False)
        if sim:
            # Reverse coeffs given in old (high -> low) order
            gal_arr['kcoeff'] = gal_arr['kcoeff'][:, ::-1]

        # Fit polynomial to median K(z) for good fits
        good = np.isfinite(gal_arr['kc']) * (tbdata['chi2'] < chi2max)
        zbin = np.linspace(par['zmin'], par['zmax'], 50) - par['z0']
        k_array = np.polynomial.polynomial.polyval(
            zbin, gal_arr['kcoeff'][good].transpose())
        k_median = np.median(k_array, axis=0)
        self.kmean = np.polynomial.polynomial.polyfit(zbin, k_median, nk-1)

        # Set any missing or bad k-corrs to median values
        bad = np.logical_not(good)
        nbad = len(z[bad])
        if nbad > 0:
            gal_arr['kc'][bad] = np.polynomial.polynomial.polyval(
                z[bad] - par['z0'], self.kmean)
            gal_arr['kcoeff'][bad] = self.kmean
            print nbad, 'missing/bad k-corrections replaced with mean'
            f = open('bad_kcorr.txt', 'w')
            for ibad in xrange(nbad):
                print >> f, gal_arr['cataid'][bad][ibad]
            f.close()

        gal_arr['absval_sel'] = (gal_arr['appval_sel'] - cosmo.dist_mod(z) -
                                 gal_arr['kc'] + ecorr(z, Q))
        gal_arr['absval_lf'] = (gal_arr['appval_lf'] - cosmo.dist_mod(z) -
                                gal_arr['kc'] + ecorr(z, Q))

        self.header = header
        self.tbdata = tbdata
        self.gal_arr = gal_arr
        self.ngal = ngal
        print ngal, 'galaxies selected'

        # Completeness weight
        # sb = (tbdata[('r_petro') + 2.5*lg2pi +
        #                   5*np.log10(tbdata[('petror50_r')))
        try:
            sb = tbdata['r_sb']
            imcomp = np.interp(sb, sb_tab, comp_tab)
        except:
            print 'No column r_sb; ignoring SB completeness'
            imcomp = np.ones(ngal)
        try:
            r_fibre = tbdata['fibermag_r']
            zcomp = z_comp(r_fibre)
        except:
            print 'No column fibermag_r; ignoring redshift completeness'
            zcomp = np.ones(ngal)
        self.gal_arr['weight'] = np.clip(1.0/(imcomp*zcomp), 1, wmax)

        # Read Vmax values if present
        try:
            self.Vmax_raw = tbdata['Vmax_raw']
            self.Vmax_dc = tbdata['Vmax_dc']
            self.Vmax_dec = tbdata['Vmax_dec']
        except:
            pass

    def calc_limits(self, Q, vis=True):
        """Calculate absolute values and visibilty limits for each galaxy,
        returning a view of gal_arr for galaxies within absolute limits."""

        zmin, zmax = par['zmin'], par['zmax']
        ngal = self.ngal
        z = self.gal_arr['z']
        sel = (z > 0)

        kc = self.gal_arr['kc']
        kcoeff = self.gal_arr['kcoeff']
        self.gal_arr['absval_sel'] = (self.gal_arr['appval_sel'] -
                                      cosmo.dist_mod(z) - kc + ecorr(z, Q))
        self.gal_arr['absval_lf'] = (self.gal_arr['appval_lf'] -
                                     cosmo.dist_mod(z) - kc + ecorr(z, Q))
        if vis:
            self.gal_arr['zlo'] = map(
                lambda i: zdm(par['mlims'][0] - self.gal_arr['absval_sel'][i],
                              kcoeff[i], (zmin, zmax), Q),
                xrange(ngal))
            self.gal_arr['zhi'] = map(
                lambda i: zdm(par['mlims'][1] - self.gal_arr['absval_sel'][i],
                              kcoeff[i], (zmin, zmax), Q),
                xrange(ngal))

        # Galaxies within absolute limits
        absm = self.gal_arr['absval_lf']
        sel *= (par['Mmin'] <= absm) * (absm < par['Mmax'])

        gala = self.gal_arr[sel]
        if par['idebug'] > 1:
            print len(gala), 'galaxies satisfy absolute limits'
        return gala

    def abs_bin(self, absval):
        """Returns bin number and fraction for given absval, such that:
        absval = absMin + (iabs+frac)*absStep."""
        
        absval = np.clip(absval, self.Mmin, self.Mmax)
	iabs = np.floor((absval - self.Mmin)/self.Mstep).astype(np.int32)
        iabs = np.clip(iabs, 0, self.Mbin - 1)
        frac = (absval - (self.Mmin + iabs*self.Mstep))/self.Mstep
        return iabs, frac

    def subset(self, idx):
        """Return subset of gala with given indices."""
        subset = Sample(self.type, self.qty_list)
        subset.gal_arr = self.gal_arr[idx]
        subset.ngal = len(subset.gal_arr)
        return subset

    def resample(self):
        """Bootstrap resampling"""
        idx = np.random.randint(0, self.ngal, self.ngal)
        return self.subset(idx)
    
    def jacknife(self, jack):
        """Return a subsample with jacknife region jack omitted"""

        idx = (self.gal_arr['ra'] < ra_jack[jack]) + (self.gal_arr['ra'] >= ra_jack[jack] + 4.0)
        subset = self.subset(idx)
        subset.area *= 8.0/9.0
        return subset


class Qty(object):
    """Class for a given quantity, with methods to convert between observed
    ('apparent') and derived ('absolute') values, distance limits and
    absolute value limits."""
    
    def __init__(self, params):
        name, limits, app_binning, abs_binning, Q = params
        self.name = name
        self.limits = limits
        self.appMin = app_binning[0]
        self.appMax = app_binning[1]
        self.napp = app_binning[2]
        self.appStep = float(self.appMax - self.appMin)/self.napp
        self.absMin = abs_binning[0]
        self.absMax = abs_binning[1]
        self.nabs = abs_binning[2]
        self.absStep = float(self.absMax - self.absMin)/self.nabs
        self.Q = Q

        # Kind of quantity: mag, radius, sb, colour, mass
        self.kind = None
        if 'index' in self.name:
            self.kind = 'index'
        if 'mag' in self.name or 'r_petro' in self.name or 'Rpetro' in self.name:
            self.kind = 'mag'
        if 'mass' in self.name:
            self.kind = 'mass'
        if 'logmstar' in self.name:
            self.kind = 'logmass'
        if 'r50' in self.name or 'gal_re' in self.name:
            self.kind = 'radius'
            par['rad_min'] = self.limits[0]
            par['rad_max'] = self.limits[1]
        if 'sb' in self.name:
            self.kind = 'sb'
        if 'mu' in self.name:
            self.kind = 'mu'
        if 'sb' in self.name or 'mu' in self.name:
            par['mu_min'] = self.limits[0]
            par['mu_max'] = self.limits[1]

        # Band if relevant
        if self.kind in ('mag', 'radius', 'sb', 'mu'):
            self.band = self.name[-1]
        if self.name in ('r_petro', 'r_sb'):
            self.band = 'r'
        
    def app_calc(self, absval, z, kcoeff):
        """Calculate apparent values from absolute values and redshift."""
        
        # magnitude
        if self.kind == 'mag':
            appval = (absval + cosmo.dist_mod(z) + 
                      kcorr(z, kcoeff) - ecorr(z, self.Q))

        # log radius: convert abs (kpc) to apparent (arcsec)
        if self.kind == 'radius':
            appval = absval - math.log10(radfac * cosmo.da(z)) - ecorr(z, self.Q)
            
        # log mass or Sersic index
        if self.kind in ('mass', 'index'):
            appval = absval + ecorr(z, self.Q)
            
        # Surface brightness
        if self.kind in ('sb', 'mu'):
            appval = (absval + 10*np.log10(1 + z) + 
                      kcorr(z, kcoeff) - ecorr(z, self.Q))

        return appval

    def abs_bin(self, absval):
        """Returns bin number and fraction for given absval, such that:
        absval = absMin + (iabs+frac)*absStep."""
        
        absval = np.clip(absval, self.absMin, self.absMax)
	iabs = np.floor((absval - self.absMin)/self.absStep).astype(np.int32)
        iabs = np.clip(iabs, 0, self.nabs - 1)
        frac = (absval - (self.absMin + iabs*self.absStep))/self.absStep
        return iabs, frac


    def app_bin(self, appval):
        """Returns bin number and fraction for given appval, such that:
        appval = appMin + (iapp+frac)*appStep."""
        
        appval = np.clip(appval, self.appMin, self.appMax)
	iapp = np.floor((appval - self.appMin)/self.appStep).astype(np.int32)
        iapp = np.clip(iapp, 0, self.napp - 1)
        frac = (appval - (self.appMin + iapp*self.appStep))/self.appStep
        return iapp, frac


def read_vmax(infile, sel_dict, param, Vmax_type, Q='read'):
    """Read data from Vmax file for given parameter."""

    global par, cosmo

    hdulist = fits.open(infile)
    header = hdulist[1].header
    tbdata = hdulist[1].data
    cols = hdulist[1].columns
    par['H0'] = 100.0
    par['omega_l'] = header['OMEGA_L']
    par['z0'] = header['Z0']
    par['area'] = header['AREA'] * (math.pi/180.0)**2
    par['mlims'] = np.zeros(2)
    par['mlims'][0] = header['mlim_0']
    par['mlims'][1] = header['mlim_1']
    par['ev_model'] = header['ev_model']
    if Q == 'read':
        par['Q'] = header['Q']
    else:
        par['Q'] = Q
    par['zmin'] = header['zmin']
    par['zmax'] = header['zmax']
    par['kc_mean'] = np.zeros(5)
    for ik in xrange(5):
        par['kc_mean'][ik] = header['kc_{}'.format(ik)]
    cosmo = util.CosmoLookup(par['H0'], par['omega_l'],
                             (par['zmin'], par['zmax']))
    print('Q, H0, omega_l, z0, area/Sr, zmin, zmax = ',
          par['Q'], par['H0'], par['omega_l'], par['z0'],
          par['area'], par['zmin'], par['zmax'])

    if 'IREAL' in cols.names:
        mock = True
        zfield = 'redshift_obs'
        sel = ((tbdata['ireal'] == par['ireal']) *
               (tbdata[param] >= par['mlims'][0]) *
               (tbdata[param] < par['mlims'][1]) *
               (tbdata['redshift_obs'] >= par['zmin']) *
               (tbdata['redshift_obs'] < par['zmax']))
    else:
        mock = False
        zfield = 'z_tonry'
        sel = ((tbdata['survey_class'] > 3) *
               (tbdata['nq'] >= 3) *
               (tbdata[param] >= par['mlims'][0]) *
               (tbdata[param] < par['mlims'][1]) *
               (tbdata['z_tonry'] >= par['zmin']) *
               (tbdata['z_tonry'] < par['zmax']))

    # Apply other selection limits in sel_dict
    for key, limits in sel_dict.iteritems():
        print key, limits
        sel *= ((tbdata.field(key) >= limits[0]) *
                (tbdata.field(key) < limits[1]))
#        par[key] = limits

    # Exclude objects with suspect photometry
    if par['clean_photom']:
        ncand = len(tbdata[sel])
        sel *= ((tbdata.field('bn_objid') < 0) *
                (np.fabs(tbdata.field('r_petro') -
                 tbdata.field('r_sersic')) < par['dmlim']))
        nclean = len(tbdata[sel])
        print nclean, 'out of', ncand, 'targets with clean photometry'

    tbdata = tbdata[sel]
    ngal = len(tbdata)
    print ngal, 'galaxies selected'

    gal_arr = np.zeros(
        ngal,
        dtype=[('ra', 'float32'), ('weight', 'float32'),
               ('absval_lf', 'float32'),  # for param
               ('Mr', 'float32'), ('Vmax', 'float32')])  # for Vmax_type
    z = tbdata[zfield]
    par['zmean'] = np.mean(z)
    gal_arr['ra'] = tbdata['ra']

    # Completeness weight
    if par['use_wt']:
        try:
            sb = tbdata.field('r_sb')
            imcomp = np.interp(sb, sb_tab, comp_tab)
        except:
            print 'No column r_sb; ignoring SB completeness'
            imcomp = np.ones(ngal)
        try:
            r_fibre = tbdata.field('fibermag_r')
            zcomp = z_comp(r_fibre)
        except:
            print 'No column fibermag_r; ignoring redshift completeness'
            zcomp = np.ones(ngal)
        gal_arr['weight'] = np.clip(1.0/(imcomp*zcomp), 1, wmax)
    else:
        gal_arr['weight'] = 1.0

    gal_arr['Vmax'] = tbdata[Vmax_type]

    gal_arr['Mr'] = (
        tbdata[param] - tbdata['kcorr_r'] -
        cosmo.dist_mod(z) + ecorr(z, par['Q']))

    gal_arr['absval_lf'] = gal_arr['Mr']

    if param == 'r_kron':
        gal_arr['absval_lf'] = (
            tbdata['r_kron'] - tbdata['kcorr_r'] -
            cosmo.dist_mod(z) + ecorr(z, par['Q']))

    if param == 'r_sersic':
        gal_arr['absval_lf'] = (
            tbdata['r_sersic'] - tbdata['kcorr_r'] -
            cosmo.dist_mod(z) + ecorr(z, par['Q']))

    if param == 'logmstar':
        gal_arr['absval_lf'] = (
            tbdata['logmstar'] - 2*math.log10(par['H0']/70.0))

    if param == 'logmstar_fluxscale':
        gal_arr['absval_lf'] = (
            tbdata['logmstar'] - 2*math.log10(par['H0']/70.0) +
            np.log10(tbdata['fluxscale']))

    # Output list of hyperluminous galaxies (M_r < -24)
    # f = open('bright.txt', 'w')
    # bright = gal_arr['absval_lf'] < -24
    # cataid_bright = tbdata[bright].field('cataid')
    # M_bright = gal_arr['absval'][bright]
    # m_bright = tbdata.field(param)[bright]
    # for i in xrange(len(M_bright)):
    #     print >> f, cataid_bright[i], M_bright[i], m_bright[i]
    # f.close()

    return gal_arr


def zdm(dmod, kcoeff, zRange, Q):
    """Calculate redshift z corresponding to distance modulus dmod, solves
    dmod = m - M = DM(z) + K(z) - Q(z-z0),
    ie. including k-correction and luminosity evolution Q.
    z is constrained to lie in range zRange."""

    if dmodk(zRange[0], kcoeff, Q) - dmod > 0:
        return zRange[0]
    if dmodk(zRange[1], kcoeff, Q) - dmod < 0:
        return zRange[1]
    z = scipy.optimize.brentq(lambda z: dmodk(z, kcoeff, Q) - dmod,
                              zRange[0], zRange[1], xtol=1e-5, rtol=1e-5)
    return z


def z_s_S(s, S, kcoeff, zRange):
    """Calculate redshift of galaxy with absolute and apparent
    surface brightnesses S and s.

    Solves S = s - 10*lg(1+z) - k(z),
    z is constrained to lie in range zRange."""

    tol = 1e-5

    zlo = zRange[0]
    zhi = zRange[1]
    if (S > s - 10*math.log10(1+zlo) - kcorr(zlo, kcoeff)): return zlo
    if (S < s - 10*math.log10(1+zhi) - kcorr(zhi, kcoeff)): return zhi
    
    z = 0.5*(zlo + zhi)
    err = zhi - zlo
    while(err > tol):
        if (S > s - 10*math.log10(1+z) - kcorr(z, kcoeff)):
            zhi = z
        else:
            zlo = z
        z = 0.5*(zlo + zhi)
        err = zhi - zlo
    return z


def z_m_r_S(m, r, S, kcoeff, zRange):
    """Calculate redshift at which galaxy of apparent mag m and log radius r
    has absolute SB S.
    Solves S = S_fun(m, r, z)
    z is constrained to lie in range zRange."""

    def S_fun(m, r, z, kcoeff):
        return m + 2.5*lg2pi + 5*r - 10*math.log10(1+z) - kcorr(z, kcoeff)
    tol = 1e-5

    zlo = zRange[0]
    zhi = zRange[1]
    if (S < S_fun(m, r, zlo, kcoeff)): return zlo
    if (S > S_fun(m, r, zhi, kcoeff)): return zhi
    
    z = 0.5*(zlo + zhi)
    err = zhi - zlo
    while(err > tol):
        if (S < S_fun(m, r, z, kcoeff)):
            zhi = z
        else:
            zlo = z
        z = 0.5*(zlo + zhi)
        err = zhi - zlo
    return z




def s_m_r(m, r):
    """Apparent sb as function of app mag and log app radius."""
    return m + 2.5 * lg2pi + 5 * r
    
def z_S_M_r(S, M, r, zRange):
    """Calculate redshift at which galaxy of abs sb S and abd mag M has
    log app radius r.
    Solves S = M + DM(z) + 2.5log(2pi r^2) - 10log(1+z)
    z is constrained to lie in range zRange."""

    def S_fun(M, r, z):
        """Abs sb as function of abs mag, radius and redshift.
        See Driver et al 2005 eqn (1)."""
        return (M + cosmo.dist_mod(z) + 2.5*lg2pi + 5*r -
                10*np.log10(1 + z))
    
    tol = 1e-5
    zlo = zRange[0]
    zhi = zRange[1]
    if (S < S_fun(M, r, zlo)): return zlo
    if (S > S_fun(M, r, zhi)): return zhi
    
    z = 0.5*(zlo + zhi)
    err = zhi - zlo
    while(err > tol):
        if (S < S_fun(M, r, z)):
            zhi = z
        else:
            zlo = z
        z = 0.5*(zlo + zhi)
        err = zhi - zlo
    return z

## def rad_Mzsb(M, z, mu):
##     """Radius as function of abs mag, redshift and sb."""
##     return (10**(0.4*(mu - M - cosmo.dist_mod(z) +
##                       10*np.log10(1 + z)))/(2*math.pi))**0.5
    
def r_m_s(m, s):
    """log app radius as function of app mag and sb."""
    return 0.2*(s - m - 2.5 * lg2pi)
    

def z_r_R(rapp, rabs, zRange):
    """Calculate redshift at which galaxy with log-radius rabs/kpc
    has apparent log-radius rapp/arcsec.  Solves rabs = rapp*Da(z) iteratively,
    where Da(z) is ang diameter distance."""

    # Convert rapp from log-arcsec to radians, rabs from log-kpc to Mpc
    rapp = math.pi/180.0/3600.0*10**rapp
    rabs = 10**rabs/1000.0

    tol = 1e-5
    zlo = zRange[0]
    zhi = zRange[1]
    z = 0.5*(zlo + zhi)
    err = zhi - zlo
    while(err > tol):
        if cosmo.da(z)*rapp > rabs:
            zhi = z
        else:
            zlo = z
        z = 0.5*(zlo + zhi)
        err = zhi - zlo

    return z;

def z_m_s_R(m, s, R, zRange):
    """Calculate redshift at which a galaxy of apparent mag m and
    surface brightness s has given absolute log radius R.
    Solves R = R_fun(m, s, z)
    z is constrained to lie in range zRange."""

    def R_fun(m, s, z):
        return 0.2*(s - m - 2.5 * lg2pi) + math.log10(radfac * cosmo.da(z))
    
    tol = 1e-5
    zlo = zRange[0]
    zhi = zRange[1]
    if (R < R_fun(m, s, zlo)): return zlo
    if (R > R_fun(m, s, zhi)): return zhi
    
    z = 0.5*(zlo + zhi)
    err = zhi - zlo
    while(err > tol):
        if (R < R_fun(m, s, z)):
            zhi = z
        else:
            zlo = z
        z = 0.5*(zlo + zhi)
        err = zhi - zlo
    return z

def z_M_s_R(M, s, R, kcoeff, Q, zRange):
    """Calculate redshift at which a galaxy of absolute mag M and
    apparent surface brightness s has given absolute log radius R.
    Solves R = R_fun(M, s, z)
    z is constrained to lie in range zRange."""

    def R_fun(M, s, z):
        return (0.2*(s - (M + dmodk(z, kcoeff, Q)) - 2.5 * lg2pi) +
                math.log10(radfac * cosmo.da(z)))
    
    tol = 1e-5
    zlo = zRange[0]
    zhi = zRange[1]
    if (R < R_fun(M, s, zlo)): return zlo
    if (R > R_fun(M, s, zhi)): return zhi
    
    z = 0.5*(zlo + zhi)
    err = zhi - zlo
    while(err > tol):
        if (R < R_fun(M, s, z)):
            zhi = z
        else:
            zlo = z
        z = 0.5*(zlo + zhi)
        err = zhi - zlo
    return z

def dmodk(z, kcoeff, Q):
    """Returns the K- and e-corrected distance modulus 
    DM(z) + k(z) - e(z)."""
    dm =  cosmo.dist_mod(z) + kcorr(z, kcoeff) - ecorr(z, Q)
    return dm

def kcorr(z, kcoeff):
    """K-correction from polynomial fit."""
    return np.polynomial.polynomial.polyval(z - par['z0'], kcoeff)

def ecorr(z, Q):
    """e-correction."""
    assert par['ev_model'] in ('z', 'z1z')
    if par['ev_model'] == 'z':
        return Q*(z - par['z0'])
    if par['ev_model'] == 'z1z':
        return Q*z/(1+z)

def z_comp(r_fibre):
    """Sigmoid function fit to redshift succcess given r_fibre, from misc.zcomp."""
    p = (22.42, 2.55, 2.24)
    return (1.0/(1 + np.exp(p[1]*(r_fibre-p[0]))))**p[2]

#------------------------------------------------------------------------------
# Routines for making simulated catalogues
#------------------------------------------------------------------------------


def simcat(infile='kcorr.fits', outfile='jswml_sim.fits',
           alpha=-1.23, Mstar=-20.70, phistar=0.01, Q=0.7, P=1.8, chi2max=10,
           Mrange=(-24, -12), mrange=(10, 19.8), zrange=(0.002, 0.65), nz=65,
           fbad=0.03, do_kcorr=True, area_fac=1.0, nblock=500000, schec_nz=0):
    """Generate test data for jswml - see Cole (2011) Sec 5."""

    def gam_dv(z):
        """Gamma function times volume element to integrate."""
        M1 = mrange[1] - cosmo.dist_mod(z) - kcorr(z, pc_med) + Q*(z-par['z0'])
        M1 = max(min(Mrange[1], M1), Mrange[0])
        M2 = mrange[0] - cosmo.dist_mod(z) - kcorr(z, pc_med) + Q*(z-par['z0'])
        M2 = max(min(Mrange[1], M2), Mrange[0])
        L1 = 10**(0.4*(Mstar - M1))
        L2 = 10**(0.4*(Mstar - M2))
        dens = (phistar * 10**(0.4*P*(z-par['z0'])) *
                mpmath.gammainc(alpha+1, L1, L2))
        ans = area * cosmo.dV(z) * dens
        return ans

    def schec(M):
        """Schechter function."""
        L = 10**(0.4*(Mstar - M))
        ans = 0.4 * ln10 * phistar * L**(alpha+1) * np.exp(-L)
        return ans

    def schec_ev(M, z):
        """Evolving Schechter function."""
        L = 10**(0.4*(Mstar - Q*(z-par['z0']) - M))
        ans = 0.4 * ln10 * phistar * L**(alpha+1) * np.exp(-L)
        return ans

    def vol_ev(z):
        """Volume element multiplied by density evolution."""
        pz = cosmo.dV(z) * 10**(0.4*P*(z-par['z0']))
        return pz

    def zM_pdf(z, M):
        """PDF for joint redshift-luminosity distribution.

        Don't use this.  Generate z and M distributions separately."""
        pz = cosmo.dV(z) * 10**(0.4*P*(z-par['z0']))
        pM = schec_ev(M, z)
        return pz*pM

    # Read k-corrections and survey params from input file
    hdulist = fits.open(infile)
    header = hdulist[1].header
    H0 = 100.0
    omega_l = header['OMEGA_L']
    par['z0'] = header['Z0']
    area_dg2 = area_fac*header['AREA']
    area = area_dg2*(math.pi/180.0)*(math.pi/180.0)
    cosmo = util.CosmoLookup(H0, omega_l, zrange)
#    rmin, rmax = cosmo.dm(zrange[0]), cosmo.dm(zrange[1])

    if do_kcorr:
        tbdata = hdulist[1].data
        sel = ((tbdata.field('survey_class') > 3) *
               (tbdata.field('z_tonry') >= zrange[0]) *
               (tbdata.field('z_tonry') < zrange[1]) *
               (tbdata.field('nQ') > 2) * (tbdata['chi2'] < chi2max))
        for ic in xrange(5):
            sel *= np.isfinite(tbdata.field('pcoeff_r')[:, ic])
        tbdata = tbdata[sel]
        nk = len(tbdata)
        ra_gal = tbdata.field('ra')
        pc = tbdata.field('pcoeff_r').transpose()
        pdim = (pc.shape)[1]  # number of coeffs
        hdulist.close()

        # For median k-correction, find median of K(z) and fit poly to this,
        # rather than taking median of coefficients
        zbin = np.linspace(zrange[0], zrange[1], 50)
        k_array = np.polynomial.polynomial.polyval(zbin, pc)
        kmin = np.min(k_array)
        kmax = np.max(k_array)
        print 'kmin, kmax =', kmin, kmax
#        pdb.set_trace()
        k_median = np.median(k_array, axis=0)
        pc_med = np.polynomial.polynomial.polyfit(zbin, k_median, pdim-1)
        k_fit = np.polynomial.polynomial.polyval(zbin, pc_med)
        plt.clf()
        plt.plot(zbin + par['z0'], k_median)
        plt.plot(zbin + par['z0'], k_fit, '--')
        plt.xlabel('z')
        plt.ylabel('K(z)')
        plt.draw()
    else:
        nk = 1
        ra_gal = np.zeros(1)
        pdim = 5
        pc = np.zeros((nk, pdim))
        pc_med = np.zeros(pdim)

    # zbins = np.linspace(zrange[0], zrange[1], nz+1)
    # V_int = area/3.0 * cosmo.dm(zbins)**3
    # V = np.diff(V_int)
    # zcen = zbins[:-1] + 0.5 * (zbins[1]-zbins[0])
    # hist_gen = np.zeros(nz)

    # Predicted N(z) from integrating evolving Schechter function
    if schec_nz:
        zlims = np.linspace(0.0, 0.65, 66)
        nz = len(zlims) - 1
        fout = open(schec_nz, 'w')
        print >> fout, 'alpha {}, M* {}, phi* {}, Q {}, P {}'.format(
                alpha, Mstar, phistar, Q, P)
        for i in xrange(nz):
            zlo = zlims[i]
            zhi = zlims[i+1]
            schec_num, err = scipy.integrate.quad(
                gam_dv, zlo, zhi, epsabs=1e-3, epsrel=1e-3)
            print >> fout, 0.5*(zlo+zhi), schec_num
        fout.close()
        return

    # Integrate evolving LF for number of simulated galaxies
    nsim, err = scipy.integrate.quad(gam_dv, zrange[0], zrange[1],
                                     epsabs=1e-3, epsrel=1e-3)
    nsim = int(nsim)
    print 'Simulating', nsim, 'galaxies'
    mapp_out = np.zeros(nsim)
    Mabs_out = np.zeros(nsim)
    z_out = np.zeros(nsim)
    ra_out = np.zeros(nsim)
    kc_out = np.zeros(nsim)
    pc_out = np.zeros((nsim, 5))

    nrem = nsim
    nout = 0
    while nrem > 0:
        # z, Mabs = util.ran_fun2(zM_pdf, zrange[0], zrange[1], 
        #                         Mrange[0], Mrange[1], nblock)
        z = util.ran_fun(vol_ev, zrange[0], zrange[1], nblock)
        Mabs = util.ran_fun(schec, Mrange[0], Mrange[1], nblock) - Q*(z-par['z0'])

        # Calculate apparent mag and test for visibility
        # First do a crude cut without k-corrections
        mapp = Mabs + cosmo.dist_mod(z)
        sel = (mapp >= mrange[0] - kmax) * (mapp < mrange[1] - kmin)
        z, Mabs, mapp = z[sel], Mabs[sel], mapp[sel]
        nsel = len(z)

        # K-corrections for remaining objects
        kidx = np.random.randint(0, nk, nsel)
        pc_ran = pc[kidx, :]
        kc = np.array(map(lambda i: kcorr(z[i], pc_ran[i, :]), xrange(nsel)))
        mapp += kc
        ra = ra_gal[kidx]
        sel = (mapp >= mrange[0]) * (mapp < mrange[1])
        z, Mabs, mapp, ra, kc, pc_ran = z[sel], Mabs[sel], mapp[sel], ra[sel], kc[sel], pc_ran[sel, :]
        nsel = len(z)
        if nsel > nrem:
            nsel = nrem
            z, Mabs, mapp, ra, kc, pc_ran = z[:nrem], Mabs[:nrem], mapp[:nrem], ra[:nrem], kc[:nrem], pc_ran[:nrem, :]

        mapp_out[nout:nout+nsel] = mapp
        Mabs_out[nout:nout+nsel] = Mabs
        z_out[nout:nout+nsel] = z
        ra_out[nout:nout+nsel] = ra
        kc_out[nout:nout+nsel] = kc
        pc_out[nout:nout+nsel, :] = pc_ran

        nout += nsel
        nrem -= nsel
        print nrem

    # Randomly resample in redshift bins to induce density fluctuations
    zbins = np.linspace(zrange[0], zrange[1], nz+1)
    V_int = area/3.0 * cosmo.dm(zbins)**3
    V = np.diff(V_int)
    zcen = zbins[:-1] + 0.5 * (zbins[1]-zbins[0])
    zhist, bin_edges = np.histogram(z_out, bins=zbins)
    hist_gen = np.zeros(nz)
    # mapp_samp = []
    # Mabs_samp = []
    # z_samp = []
    # ra_samp = []
    # kc_samp = []
    # pc_samp = []
    samp_list = []
    nsamp = 0
    print 'iz  delta  zhist  nsel'
    for iz in xrange(nz):
        zlo = zbins[iz]
        zhi = zbins[iz+1]
        delta = np.random.normal(0.0, math.sqrt(J3/V[iz]))
        nsel = int(round((1+delta) * zhist[iz]))
        hist_gen[iz] = nsel
        print iz, delta, zhist[iz], nsel
        if nsel > 0:
            sel = (zlo <= z_out) * (z_out < zhi)
            idx = np.where(sel)
            samp = np.random.randint(0, zhist[iz], nsel)
            for i in xrange(nsel):
                samp_list.append(idx[0][samp[i]])
                # mapp_samp.append(mapp_out[idx][samp[i]])
                # Mabs_samp.append(Mabs_out[idx][samp[i]])
                # z_samp.append(z_out[idx][samp[i]])
                # ra_samp.append(ra_out[idx][samp[i]])
                # kc_samp.append(kc_out[idx][samp[i]])
                # pc_samp.append(pc_out[idx][samp[i]])
            nsamp += nsel

    nQ = 4*np.ones(nsamp)
    survey_class = 6*np.ones(nsamp)
    vis_class = np.zeros(nsamp)
    post_class = np.zeros(nsamp)
    A_r = np.zeros(nsamp)
    print nsamp, ' galaxies after resampling'

    # Assign surface brightness and fibre mag from fits to observed relations
    # (Loveday+ 2012, App A1)
    mapp = np.array([mapp_out[i] for i in samp_list])
    Mabs = np.array([Mabs_out[i] for i in samp_list])
    sb = 22.42 + 0.029*Mabs + np.random.normal(0.0, 0.76, nsamp)
    imcomp = np.interp(sb, sb_tab, comp_tab)
    r_fibre = 5.84 + 0.747*mapp + np.random.normal(0.0, 0.31, nsamp)
    zcomp = z_comp(r_fibre)
    bad = (imcomp * zcomp < np.random.random(nsamp))
    nQ[bad] = 0
    nbad = len(nQ[bad])
    print nbad, 'out of', nsamp, 'redshifts marked as bad', float(nbad)/nsamp

    # Write out as FITS file
    cols = [fits.Column(name='r_petro', format='E', array=mapp),
            fits.Column(name='fibermag_r', format='E', array=r_fibre),
            fits.Column(name='r_sb', format='E', array=sb),
            fits.Column(name='z_tonry', format='E', 
                          array=[z_out[i] for i in samp_list]),
            fits.Column(name='nQ', format='I', array=nQ),
            fits.Column(name='survey_class', format='I', array=survey_class),
            fits.Column(name='vis_class', format='I', array=vis_class),
            fits.Column(name='post_class', format='I', array=post_class),
            fits.Column(name='ra', format='E', 
                          array=[ra_out[i] for i in samp_list]),
            fits.Column(name='kcorr_r', format='E', 
                          array=[kc_out[i] for i in samp_list]),
            fits.Column(name='pcoeff_r', format='{}E'.format(pdim), 
                          array=[pc_out[i, :] for i in samp_list]),
            fits.Column(name='A_r', format='E', array=A_r)]
    tbhdu = fits.new_table(cols)
    # Need PyFits 3.1 to add new header parameters in this way
    hdr = tbhdu.header
    hdr['omega_l'] = header['omega_l']
    hdr['z0'] = header['z0']
    hdr['area'] = area_dg2
    hdr['alpha'] = alpha
    hdr['Mstar'] = Mstar
    hdr['phistar'] = phistar
    hdr['Q'] = Q
    hdr['P'] = P
    tbhdu.writeto(outfile, clobber=True)

    plt.clf()
    ax = plt.subplot(2, 1, 1)
    ax.plot(zcen, zhist)
    ax.step(zcen, hist_gen, where='mid')
    ax.set_xlabel('z')
    ax.set_ylabel('N(z)')
    ax = plt.subplot(2, 1, 2)
    ax.hist(Mabs, 
            bins=int(4*(Mrange[1] - Mrange[0])), range=(Mrange[0], Mrange[1]))
    ax.set_xlabel('Abs mag M')          
    ax.set_ylabel(r'$N(M)$')    
    plt.draw()

def multisim(infile='kcorrz01.fits', outfile='sim_z01_{}.fits', do_kcorr=True):
    """Generate several simulated catalogues."""
    for i in xrange(10):
        simcat(infile, outfile=outfile.format(i), do_kcorr=do_kcorr, 
               area_fac=1.0)
        
def sim_av(evroot='lf_kde_z01_{}_{}.dat', lfroot='lf_bin_{}_{}.fits', 
           out_file='sim_table.tex', nlo=0, nhi=10):
    """Average results from simulations."""
    for method in ('post', 'lfchi'):
        nsim = nhi - nlo
        alpha = np.zeros(nsim)
        Mstar = np.zeros(nsim)
        lpstar = np.zeros(nsim)
        P = np.zeros(nsim)
        Q = np.zeros(nsim)
        delta = []
        delta_err = []
        den_var = []
        phi = []
        phi_err = []
        for i in xrange(nsim):
            infile = evroot.format(method, i+nlo)
            f = open(infile, 'r')
            dat = pickle.load(f)
            f.close()
            P[i] = dat['P']
            Q[i] = dat['Q']
            zbin = dat['zbin']
            delta.append(dat['delta'])
            delta_err.append(dat['delta_err'])
            den_var.append(dat['den_var'])
            sim_par = dat['par']['sim_par']

            infile = lfroot.format(method, i+nlo)
            f = open(infile, 'r')
            dat = pickle.load(f)
            f.close()
            alpha[i] = dat['alpha']
            Mstar[i] = dat['Mstar']
            lpstar[i] = dat['lpstar']
            Mbin = dat['Mbin']
            phi.append(dat['phi'])
            phi_err.append(dat['phi_err'])

        print method
        print '  alpha {:6.2f} +/- {:6.2f}'.format(np.mean(alpha), np.std(alpha))
        print '     M* {:6.2f} +/- {:6.2f}'.format(np.mean(Mstar), np.std(Mstar))
        print 'lg phi* {:6.2f} +/- {:6.2f}'.format(np.mean(lpstar), np.std(lpstar))
        print '      P {:6.2f} +/- {:6.2f}'.format(np.mean(P), np.std(P))
        print '      Q {:6.2f} +/- {:6.2f}'.format(np.mean(Q), np.std(Q))
        print 'cov(Q,P) ', np.cov(Q, P)
        plt.clf()
        ax = plt.subplot(211)
        ax.errorbar(zbin, np.mean(delta, axis=0), np.std(delta, axis=0), fmt='none')
        ax.errorbar(zbin, np.mean(delta, axis=0), np.mean(delta_err, axis=0), fmt='none')
        dz = zbin[1] - zbin[0]
        den_sig = np.sqrt(np.mean(den_var, axis=0))
        ax.bar(zbin - 0.5*dz, 2*den_sig, width=dz, 
               bottom=np.mean(delta, axis=0)-den_sig, alpha=0.1, ec='none')
        ax.plot([zbin[0], zbin[-1]], [1.0, 1.0], ':')
        ax.set_xlabel('Redshift z')          
        ax.set_ylabel(r'$\Delta(z)$')
        ax.set_ylim(0.3, 1.7)
        ax = plt.subplot(212)
        ax.axis([dat['Mmin'], dat['Mmax'], 2e-7, 1])
        ax.semilogy(basey=10, nonposy='clip')
        ax.errorbar(Mbin, np.mean(phi, axis=0), np.std(phi, axis=0), fmt='none')
        ax.errorbar(Mbin, np.mean(phi, axis=0), np.mean(phi_err, axis=0), fmt='none')
        label = plot_label(dat['param'])[1]
        ax.set_xlabel(label)
        ax.set_ylabel(r'$\Phi(M) / h^3 {\rm Mpc}^{-3}$')

        plt.draw()

def sim_test(file='sim.fits'):
    """Check k-corrections in simcat"""

    hdulist = fits.open(file)
    header = hdulist[1].header
    par['z0'] = header['Z0']

    tbdata = hdulist[1].data
    z = tbdata.field('z_tonry')
    kc = tbdata.field('kcorr_r')
    pc = tbdata.field('pcoeff_r')[:, ::-1]  # NB sims still have reversed coeffs
    ngal = len(pc)
    kc_poly = kcorr(z, pc)
    plt.clf()
    plt.scatter(kc, kc - kc_poly, 0.1)
    plt.draw()
    hdulist.close()

def ktest(file='kcorrz01.fits', zrange=(0.0, 0.5)):
    """Check k-corrections."""
    hdulist = fits.open(file)
    header = hdulist[1].header
    par['z0'] = header['Z0']

    tbdata = hdulist[1].data
    sel = ((tbdata.field('z_tonry') >= zrange[0]) * 
           (tbdata.field('z_tonry') < zrange[1]) *
           (tbdata.field('nQ') > 2))
    tbdata = tbdata[sel]
    pc = tbdata.field('pcoeff_r').transpose()
    zbin = np.linspace(0.0, 0.5, 50) - par['z0']
    k_array = np.polynomial.polynomial.polyval(zbin, pc)
    k_median = np.median(k_array, axis=0)
    pc_k_med = np.polynomial.polynomial.polyfit(zbin, k_median, 5)
    k_fit = np.polynomial.polynomial.polyval(zbin, pc_k_med)
    pc_median = np.median(pc, axis=1)
    k_fit2 = np.polynomial.polynomial.polyval(zbin, pc_median)
    plt.clf()
    plt.plot(zbin + par['z0'], k_median)
    plt.plot(zbin + par['z0'], k_fit, '--')
    plt.plot(zbin + par['z0'], k_fit2, ':')
    plt.xlabel('z')
    plt.ylabel('K(z)')
    plt.draw()
    pdb.set_trace()

def reformat(infile='sim_noev_nok.fits', outfile='sim_noev_nok.txt'):
    """Reformat simulated catalogue to be read by Shaun's code."""

    hdulist = fits.open(infile)
    header = hdulist[1].header
    par['z0'] = header['Z0']

    tbdata = hdulist[1].data
    mag = tbdata.field('gal_mag_10re_r')
    z = tbdata.field('z_tonry')
    ngal = len(z)
    area = (math.pi/180.0)**2 * header['area']
    hdulist.close()

    fout = open(outfile, 'w')
    print >> fout, '# Selection criteria:'
    print >> fout, '# zmin=  0.002 zmax=  0.5'
    print >> fout, '# faint maglimit= 19.8'
    print >> fout, '# solid angle=  {}  Sr'.format(area)
    print >> fout, '# Catalogue: ncat= {}'.format(ngal)
    print >> fout, '#     mag     z      id'
    for i in xrange(ngal):
        print >> fout, mag[i], z[i], i+1
    fout.close()

def plot_ascii():
    """Plot delta and LF from ascii files output by Shaun's code."""
    delta = np.loadtxt('TestData/delta.txt')
    lf = np.loadtxt('TestData/LF.txt')
    plt.clf()
    ax = plt.subplot(2, 1, 1)
    ax.plot(delta[:,0], delta[:,1])
    ax.plot([0.0, 0.5], [1.0, 1.0], ':')
    ax.set_xlabel('Redshift z')          
    ax.set_ylabel(r'$\Delta(z)$')
    ax = plt.subplot(2, 1, 2)
    ax.semilogy(basey=10, nonposy='clip')
    idx = lf[:,1] > 0
    ax.plot(lf[:,0][idx], lf[:,1][idx])
    ax.set_xlabel('Abs mag M')          
    ax.set_ylabel(r'$\Phi(M)$')
    phi_err = 0.05*lf[:,1][idx] # Assume 5% errors in each bin
    fitpars = lum.schecFit(lf[:,0][idx], lf[:,1][idx], phi_err, 
                           (-1.0, -20.0, -2))
    alpha, alphaErr, Mstar, MstarErr, lpstar, lpstarErr, chi2, nu = fitpars
    lum.plotSchec(alpha, Mstar, 10**lpstar, lf[:,0][0], lf[:,0][-1], 
                  lineStyle='--', axes=ax)
    plt.draw()


def jack_err(ests):
    """Jackknife error from array of estimates."""
    nest = len(ests)
    if nest > 1:
        err = math.sqrt((nest-1)*np.var(ests))
    else:
        err = ests[0]
    return err

#------------------------------------------------------------------------------
# Plotting Routines
#------------------------------------------------------------------------------

def plot(infile='jswml.dat', Mlimits=(-11, -24), plot_file=None):
    """Plot delta(z) and phi(M) from jswml output."""
    f = open(infile, 'r')
    dat = pickle.load(f)
    f.close()

    plt.clf()
    ax = plt.subplot(2, 1, 1)
    ax.step(dat['zbin'], dat['delta'], where='mid')
    ax.errorbar(dat['zbin'], dat['delta'], dat['delta_err'], fmt='none')
    # ax.errorbar(dat['zbin'], dat['delta'], np.sqrt(dat['den_var']), fmt='none')
    dz = dat['zbin'][1] - dat['zbin'][0]
    ax.bar(dat['zbin'] - 0.5*dz, 2*np.sqrt(dat['den_var']), width=dz, 
           bottom = dat['delta'] - np.sqrt(dat['den_var']), alpha=0.05, ec='none')
    ax.plot([0.0, 0.5], [1.0, 1.0], ':')
    ax.set_xlabel('Redshift z')          
    ax.set_ylabel(r'$\Delta(z)$')
    ax.set_ylim(0.4, 1.6)
    ax = plt.subplot(2, 1, 2)
    ax.axis((Mlimits[0], Mlimits[1], 1e-7, 1))
    ax.semilogy(basey=10, nonposy='clip')
    ndim = len(dat['phi'].shape)
    if ndim > 1:
        step = dat['edges'][0][1] - dat['edges'][0][0]
        Mbin = dat['edges'][0][:-1] + 0.5*step
        absStep = 1.0
        for i in xrange(ndim):
            absStep *= (dat['edges'][i][1] - dat['edges'][i][0])
        phi = np.sum(dat['phi'], axis=tuple(range(1, ndim))) * absStep/step
        phi_err = (np.sum(dat['phi_err']**2, axis=tuple(range(1, ndim)))**0.5 * 
                        absStep/step)
    else:
        step = dat['edges'][1] - dat['edges'][0]
        Mbin = dat['edges'][:-1] + 0.5*step
        phi = dat['phi']
        phi_err = dat['phi_err']
    ax.errorbar(Mbin, phi, phi_err, fmt='o')
    fitpars = lum.schecFit(Mbin, phi, phi_err, (-1.0, -20.0, -2))
    alpha, alphaErr, Mstar, MstarErr, lpstar, lpstarErr, chi2, nu = fitpars
    lum.plotSchec(alpha, Mstar, 10**lpstar, Mbin[0], Mbin[-1],
                  lineStyle='--', axes=ax)
    label = plot_label(dat['qty_list'][0].name)[1]
    ax.set_xlabel(label)
    ax.set_ylabel(r'$\Phi(M) / h^3 {\rm Mpc}^{-3}$')
    if plot_file is None:
        ax.text(0.1, 0.6, r'$\chi^2/\nu = {:4.2f}$'.format(chi2/nu),
                transform = ax.transAxes)
        ax.text(0.1, 0.5, r'$\alpha = {:4.2f} \pm {:4.2f}$'.format(
                alpha, np.mean(alphaErr)),
                transform = ax.transAxes)
        ax.text(0.1, 0.4, r'$M^* = {:5.2f} \pm {:5.2f}$'.format(
                Mstar, np.mean(MstarErr)),
                transform = ax.transAxes)
        ax.text(0.1, 0.3, r'$\log \phi^* = {:4.2f} \pm {:4.2f}$'.format(
                lpstar, np.mean(lpstarErr)),
                transform = ax.transAxes)
        ax.text(0.1, 0.2, r'$P = {:5.2f} \pm {:5.2f}$'.format(
                dat['P'], dat['P_err']),
                transform = ax.transAxes)
        ax.text(0.1, 0.1, r'$Q = {:5.2f} \pm {:5.2f}$'.format(
                dat['Q'], dat['Q_err']),
                transform = ax.transAxes)
    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def plot_lfs(filelist=('lf_sersic_clean.dat', 'lf_sersic_noclean.dat'),
             xlab=mag_sersic_label, ylab=den_mag_label,
             plot_size=(5, 3.5), plot_file='lf_sersic.pdf'):
    """Plot LFs in single panel."""
    panel = []
    symbols = itertools.cycle(sym_list)
    for fname in filelist:
        lf = pickle.load(open(fname, 'r'))
        lf.update({'props': symbols.next(), 'err_plot': 1, 'label': None,
                   'schec_plot': 0, 'schec_label': 0})
        panel.append(lf)
        panels = [panel, ]
        plabels = [None, ]
        plot1d(panels, plabels, xlimits=(-25, -12), xlab=xlab, ylab=ylab,
               plot_size=plot_size, plot_file=plot_file)


def plot_lfs_colour(temp_list=('lf_r_petro_{}.dat','lf_r_sersic_{}.dat'), 
                    llabels=('Petro', 'Sersic'), plot_file='lf_r.pdf'):
    """Plot r-band LFs by colour.  
    Comparison Schechter parameters from Loveday+ 2012, Table 5."""

    global par
    par['H0'] = 100.0
    par['z0'] = 0.1

    plabels = {'c': 'All', 'b': 'Blue', 'r': 'Red'}
    schec_comps = {'c': (-1.23, -20.70, 0.94e-2), 
                   'b': (-1.49, -20.45, 0.38e-2), 
                   'r': (-0.57, -20.34, 1.11e-2)}
    panels = []
    for colour in 'cbr':
        files =[]
        for filetemp in temp_list:
            files.append(filetemp.format(colour))
        comp = None
        if colour == 'c':
            comp = Blanton2005
        plabel = plabels[colour]
        schec_comp = schec_comps[colour]
        panels.append((files, llabels, comp, schec_comp, plabel))

    plot1d(panels, xlimits=(-25, -12), ylimits=(2e-8, 1), 
           xlab=mag_petro_label, ylab=den_mag_label, plot_file=plot_file)

def plot_lfs_morph(template='lf_lfchi_{}.dat', plot_file='lf_r.pdf'):
    """Plot r-band LFs by morphology."""
    plot1d(((template.format('c'), (-1.26, -20.73, 0.90e-2), 'All'), 
               (template.format('e'), (-1.45, -20.28, 0.55e-2), 'Early'),
               (template.format('s'), (-0.53, -20.28, 0.98e-2), 'Late')), 
              xlimits=(-24, -11), plot_file='lf_r.pdf')


def plot_lfs_mock(filetemp='lf_{:02d}.dat', plot_file='lf_r.pdf'):
    """Plot r-band LFs from mocks."""
    lf = pickle.load(open('../../jswml/auto/lf_r_sersic_lfchi_c.dat', 'r'))
    lf.update({'props': 'bo', 'err_plot': 1, 'label': None,
               'schec_plot': 0, 'schec_label': 0})
    panel = [lf, ]
    for ireal in (1, 2, 5, 6, 14, 20):
        lf = pickle.load(open(filetemp.format(ireal), 'r'))
        lf.update({'props': 'k-', 'err_plot': 0, 'label': None,
                   'schec_plot': 0, 'schec_label': 0})
        panel.append(lf)
    panels = [panel, ]
    plabels = [None, ]
    plot1d(panels, plabels, xlimits=(-24, -12),
           plot_size=(8, 8), plot_file='lf_r.pdf')


def tab_results(evroot='ev_{}_petro_fit_{}.dat', 
                lfroot='lf_r_petro__{}_{}.dat', 
                outfile='ev_table.tex'):
    """Table of evolution and LF parameters by colour."""

    label = {'c': 'All', 'b': 'Blue', 'r': 'Red'}
    header = {'post': 'Mean Probability', 'lfchi': 'LF-redshift'}
    fout = open(plot_dir + outfile, 'w')
    fout.write(r'''
    \begin{math}
    \begin{array}{lccccccccc}
    \hline
    {\rm Sample} & Q_e & P_e & m & c & \chi^2_{\rm ev}/\nu & \alpha & M^* - 5 \log h & \log \phi^*/\denunit & \chi^2_\Phi/\nu\\
    ''')

    for method in ('post', 'lfchi'):
        fout.write(r'''
        \hline
        \multicolumn{10}{c}{\mbox{''' + header[method] + r'''}} \\
        \hline
        ''')

        for colour in 'cbr':
            infile = evroot.format(method, colour)
            f = open(infile, 'r')
            dat = pickle.load(f)
            f.close()
            Q = dat['Q']
            P = dat['P']
            evchi2 = dat['ev_fit_chisq']/dat['ev_fit_nu']

            # Q, P errors from quadratic fits to binned chi2 around minimum
            chi2grid = dat['chi2grid']
            Qmin = dat['Qbins'][0]
            Qmax = dat['Qbins'][1]
            nQ = dat['Qbins'][2]
            Qstep = float(Qmax - Qmin)/nQ
            iqvals = np.arange(nQ)
            Qvals = Qmin + (iqvals + 0.5)*Qstep
            iqlo = int((Q - Qmin)/Qstep)
            Pmin = dat['Pbins'][0]
            Pmax = dat['Pbins'][1]
            nP = dat['Pbins'][2]
            Pstep = float(Pmax - Pmin)/nP
            ipvals = np.arange(nP)
            Pvals = Pmin + (ipvals + 0.5)*Pstep
            iplo = int((P - Pmin)/Pstep)
            chi2q = chi2grid[:,iplo]
            fit = util.chisq_quad_fit(Qvals, chi2q)
            Qerr = 0.5 * (fit['xhi'] - fit['xlo'])
            chi2p = chi2grid[iqlo,:]
            fit = util.chisq_quad_fit(Pvals, chi2p)
            Perr = 0.5 * (fit['xhi'] - fit['xlo'])
            m, c = ev_lin_fit(infile, nfit=2)

            infile = lfroot.format(method, colour)
            f = open(infile, 'r')
            dat = pickle.load(f)
            f.close()

            if method == 'lfchi':
                fout.write(r'''{} & {:4.2f} \pm {:4.2f} & {:4.2f} \pm {:4.2f} & {:4.2f} & {:4.2f} & {:4.2f} & {:4.2f} \pm {:4.2f} & {:4.2f} \pm {:4.2f} & {:4.2f} \pm {:4.2f} & {:4.2f}\\
                '''.format(
                    label[colour], Q, Qerr, P, Perr, m, c, evchi2, 
                    dat['alpha'], np.mean(dat['alpha_err']), 
                    dat['Mstar'], np.mean(dat['Mstar_err']),
                    dat['lpstar'], np.mean(dat['lpstar_err']),
                    dat['lf_chi2']/dat['lf_nu2']))
            else:
                fout.write(r'''{} & {:4.2f} \pm {:4.2f} & {:4.2f} \pm {:4.2f} & {:4.2f} & {:4.2f} & \ldots & {:4.2f} \pm {:4.2f} & {:4.2f} \pm {:4.2f} & {:4.2f} \pm {:4.2f} & {:4.2f}\\
                '''.format(
                    label[colour], Q, Qerr, P, Perr, m, c, 
                    dat['alpha'], np.mean(dat['alpha_err']), 
                    dat['Mstar'], np.mean(dat['Mstar_err']),
                    dat['lpstar'], np.mean(dat['lpstar_err']),
                    dat['lf_chi2']/dat['lf_nu2']))

    fout.write(r'''
    \hline
    \end{array}
    \end{math}
    ''')

    fout.close()

def ev_table(evroot='ev_{}_petro_fit_{}.dat', outfile='ev_table.tex'):
    """Table of evolution parameters by colour."""

    label = {'c': 'All', 'b': 'Blue', 'r': 'Red'}
    header = {'post': 'Mean Probability', 'lfchi': 'LF-redshift'}
    fout = open(plot_dir + outfile, 'w')
    fout.write(r'''
    \begin{math}
    \begin{array}{lcccccc}
    \hline
    {\rm Sample} & Q_e & P_e & Q_e + P_e & m & c & \chi^2_\nu \\
    ''')
    plt.clf()

    for method in ('post', 'lfchi'):
        fout.write(r'''
        \hline
        \multicolumn{7}{c}{\mbox{''' + header[method] + r'''}} \\
        \hline
        ''')

        for colour in 'cbr':
            infile = evroot.format(method, colour)
            f = open(infile, 'r')
            dat = pickle.load(f)
            f.close()
            Q = dat['Q']
            P = dat['P']
            chi2min = dat['ev_fit_chisq']
            evchi2 = dat['ev_fit_chisq']/dat['ev_fit_nu']

            chi2grid = dat['chi2grid']
            Qmin = dat['Qbins'][0]
            Qmax = dat['Qbins'][1]
            nQ = dat['Qbins'][2]
            Qstep = float(Qmax - Qmin)/nQ
            iqvals = np.arange(nQ)
            Qvals = Qmin + (iqvals + 0.5)*Qstep
            Pmin = dat['Pbins'][0]
            Pmax = dat['Pbins'][1]
            nP = dat['Pbins'][2]
            Pstep = float(Pmax - Pmin)/nP
            ipvals = np.arange(nP)
            Pvals = Pmin + (ipvals + 0.5)*Pstep

            # Bounding box for chi2 < chi2_min + 1 -> Qerr, Perr
            B = np.argwhere(chi2grid < dat['ev_fit_chisq'] + 1)
            (iqlo, iplo), (iqhi, iphi) = B.min(0), B.max(0) + 1
            Qerr = 0.5*(Qvals[iqhi] - Qvals[iqlo])
            Perr = 0.5*(Pvals[iphi] - Pvals[iplo])

            print method, colour
            print 'iqlo, iqhi, iplo, iphi', iqlo, iqhi, iplo, iphi
            print Qvals[iqlo], Qvals[iqhi]
            print Pvals[iplo], Pvals[iphi]
            # pdb.set_trace()

            # Spline fit to chi2 map to find error on P+Q
            spline = scipy.interpolate.RectBivariateSpline(
                Qvals, Pvals, chi2grid, bbox=[Qmin, Qmax, Pmin, Pmax])
            def delta_chi2(delta, spline, P, Q, chi2min):
                """Delta chi2 minus 1"""
                return spline(Q-delta, P+delta) - chi2min - 1

            dlo = scipy.optimize.brentq(delta_chi2, -1, 0, 
                                        args=(spline, P, Q, chi2min),
                                        xtol=0.01, rtol=0.01)
            dhi = scipy.optimize.brentq(delta_chi2, 0, 1, 
                                        args=(spline, P, Q, chi2min),
                                        xtol=0.01, rtol=0.01)
            QPerr = 0.5*(dhi - dlo)

            m, c = ev_lin_fit(infile, nfit=2)

            if method == 'lfchi':
                fout.write(r'''{} & {:4.2f} \pm {:4.2f} & {:4.2f} \pm {:4.2f} & {:4.2f} \pm {:4.2f} & {:4.2f} & {:4.2f} & {:4.2f} \\
                '''.format(label[colour], Q, Qerr, P, Perr, Q+P, QPerr, m, c, evchi2))
                if colour == 'c':
                    Pgrid, Qgrid = np.meshgrid(Pvals, Qvals)
                    plt.plot(Pgrid + Qgrid, chi2grid)
            else:
                fout.write(r'''{} & {:4.2f} \pm {:4.2f} & {:4.2f} \pm {:4.2f} & {:4.2f} \pm {:4.2f} & {:4.2f} & {:4.2f} & \ldots \\
                '''.format(label[colour], Q, Qerr, P, Perr, Q+P, QPerr, m, c))


    fout.write(r'''
    \hline
    \end{array}
    \end{math}
    ''')

    fout.close()
    plt.xlabel(r'$P + Q$')
    plt.ylabel(r'$\chi^2$')
    plt.draw()

def lf_table(lfroot='lf_{}_{}.dat', outfile='lf_table.tex'):
    """Table of LF parameters by colour."""

    label = {'c': 'All', 'b': 'Blue', 'r': 'Red'}
    header = {'r_petro_post': 'Mean Probability Petrosian', 
              'r_petro_lfchi': 'LF-redshift Petrosian',
              'r_sersic': 'LF-redshift Sersic'}
    fout = open(plot_dir + outfile, 'w')
    fout.write(r'''
    \begin{math}
    \begin{array}{lcccc}
    \hline
    {\rm Sample} & \alpha & M^* - 5 \log h & \log \phi^*/\denunit & \chi^2_\nu\\
    ''')

    for est in ('r_petro_post', 'r_petro_lfchi', 'r_sersic'):
        fout.write(r'''
        \hline
        \multicolumn{5}{c}{\mbox{''' + header[est] + r'''}} \\
        \hline
        ''')

        for colour in 'cbr':
            infile = lfroot.format(est, colour)
            dat = pickle.load(open(infile, 'r'))

            fout.write(r'''{} & {:4.2f} \pm {:4.2f} & {:4.2f} \pm {:4.2f} & {:4.2f} \pm {:4.2f} & {:4.2f}\\
                '''.format(
                    label[colour], 
                    dat['alpha'], np.mean(dat['alpha_err']), 
                    dat['Mstar'], np.mean(dat['Mstar_err']),
                    dat['lpstar'], np.mean(dat['lpstar_err']),
                    dat['lf_chi2']/dat['lf_nu2']))

    fout.write(r'''
    \hline
    \end{array}
    \end{math}
    ''')

    fout.close()


def plot_smf(temp_list=('smf_{}.dat',)):
    """Plot stellar mass functions."""

    global par
    par['H0'] = 100.0

    labels = {'c': 'All', 'b': 'Blue', 'r': 'Red'}
    panels = []
    # for colour in 'cbr':
    for colour in 'c':
        files = []
        for filetemp in temp_list:
            files.append(filetemp.format(colour))
        comp = None
        schec_comp = None
        if colour == 'c':
            comp = Baldry2011
        label = labels[colour]
        panels.append((files, None, comp, schec_comp, label))

    plot1d(panels, xlimits=(6,13), xlab=r'$\lg(M_*/M_\odot)$', 
           ylab=r'$\phi(M)$')


def plot_smf_loz(files=('smf_fs_z_0.002_0.06.dat',
                        'smf_fs_nowt_z_0.002_0.06.dat'),
                 plot_file='smf_loz.pdf'):
    """Plot low-z stellar mass function."""

    global par
    par['H0'] = 100.0

    panels = []
    schec_comp = None
    comp = Baldry2011
    llabels = ('Weighted', 'Unweighted')
    panels.append((files, llabels, comp, schec_comp, None))

    plot1d(panels, xlimits=(6, 12), ylimits=(1e-5, 1), 
           xlab=r'$\lg(M_*/M_\odot) + 2 \lg h$', 
           ylab=r'$\phi(M)\ (h^3 {\rm Mpc}^{-3})$', schec_plot=False,
           plot_size=(5,4), plot_file=plot_file)


def plot_lf_z(lftemp='lf_r_{}_{}.dat', ztemp='lf_r_{}_{}_z_{}_{}.dat',
              mag_type='petro', colour='c', Mlims=(-24, -17), plot_file=None,
              plot_size=(5, 8)):
    """Plot LF/SMF in redshift bins."""

    global par
    par['H0'] = 100.0
    zlims = (0.002, 0.1, 0.2, 0.3, 0.65)
    labeltemp = r'${} < z < {}$'
    sym = ('ov^<>s')
    clr_label = {'c': 'All', 'b': 'Blue', 'r': 'Red'}
    plt.clf()
    fig, axes = plt.subplots(2, sharex=True, num=1)
    plt.subplots_adjust(hspace=0.0)
    lf = pickle.load(open(lftemp.format(mag_type, colour), 'r'))
    lf_chi2 = 0
    nu = 0
    
    for iz in xrange(len(zlims)-1):
        zlo = zlims[iz]
        zhi = zlims[iz+1]
        infile = ztemp.format(mag_type, colour, zlo, zhi)
        label = labeltemp.format(zlo, zhi)
        lfz = pickle.load(open(infile, 'r'))
        cmp = lfz['comp']
        cmp *= (lfz['phi'] > 0)
        nu += len(lfz['Mbin'][cmp])
        lf_chi2 += np.sum((lf['phi'][cmp] - lfz['phi'][cmp])**2 /
                              (lf['phi_err'][cmp]**2 + lfz['phi_err'][cmp]**2))
        axes[0].errorbar(lfz['Mbin'][cmp], lfz['phi'][cmp],
                         lfz['phi_err'][cmp], fmt=sym[iz], label=label)
        axes[1].errorbar(lfz['Mbin'][cmp], lfz['phi'][cmp]/lf['phi'][cmp],
                         lfz['phi_err'][cmp]/lf['phi'][cmp], fmt=sym[iz])

    axes[0].set_xlim(Mlims)
    axes[0].set_ylim(1e-6, 0.1)
    axes[0].set_ylabel(den_mag_label)
    axes[0].semilogy(basey=10, nonposy='clip')
    axes[0].legend(loc=4)
    axes[0].text(0.1, 0.8, clr_label[colour], transform=axes[0].transAxes)

    axes[1].plot(Mlims, (1, 1), ':')
    axes[1].set_xlim(Mlims)
    axes[1].set_ylim(0, 1.99)
    axes[1].set_ylabel(r'$\phi(M_z)/\phi(M_{\rm tot})$')
    axes[1].set_xlabel(plot_label(lf['param'])[1])
    plt.draw()
    print 'chi^2, nu =', lf_chi2, nu

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
        print 'plot saved to ', plot_dir + plot_file


def plot_lf_zslices(lftemp='lfr_zslice_{}_{}.dat', Mlims=(-24, -17),
                    plot_size=(4, 8)):
    """Plot LF in redshift slices."""

    global par
    par['H0'] = 100.0
    zlims = (0.002, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    labeltemp = r'${} < z < {}$'
    plt.clf()
    fig, axes = plt.subplots(6, sharex=True, num=1)
    plt.subplots_adjust(hspace=0.0)
    zlist = []
    fitlist = []
    for iz in xrange(len(zlims)-1):
        ax = axes[iz]
        zlo = zlims[iz]
        zhi = zlims[iz+1]
        infile = lftemp.format(zlo, zhi)
        label = labeltemp.format(zlo, zhi)
        lfz = pickle.load(open(infile, 'r'))
        cmp = lfz['comp']
        Mbin = lfz['Mbin'][cmp]
        phi = lfz['phi'][cmp]
        phi_err = lfz['phi_err'][cmp]
        ax.errorbar(Mbin, phi, phi_err, fmt='o', label=label)
        if iz == 0:
            fitpars = lum.schecFit(Mbin, phi, phi_err, (-1.0, -20.0, -2))
        else:
            fitpars = lum.schecFit(Mbin, phi, phi_err, (alpha, -20.0, -2),
                                   afix=True)
        zlist.append(lfz['par']['zmean'])
        fitlist.append(fitpars)
        alpha, alphaErr, Mstar, MstarErr, lpstar, lpstarErr, chi2, nu = fitpars
        lum.plotSchec(alpha, Mstar, 10**lpstar, Mlims[0], Mlims[-1],
                      lineStyle='--', axes=ax)

        ax.set_xlim(Mlims)
        ax.set_ylim(1e-6, 0.1)
        ax.set_ylabel(den_mag_label)
        ax.semilogy(basey=10, nonposy='clip')
        ax.text(0.1, 0.8, label, transform=ax.transAxes)
    ax.set_xlabel(r'$M_r$')
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    plot_file = 'lf_zslices.pdf'
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    print 'plot saved to ', plot_dir + plot_file

    plt.clf()
    fig, axes = plt.subplots(2, sharex=True, num=1)
    plt.subplots_adjust(hspace=0.0)
    ax = axes[0]
    ax.errorbar(zlist, [fitlist[iz][2] for iz in xrange(6)],
                np.array([fitlist[iz][3] for iz in xrange(6)]).T)
    ax.set_ylabel(r'$M^*$')
    ax = axes[1]
    ax.errorbar(zlist, [fitlist[iz][4] for iz in xrange(6)],
                np.array([fitlist[iz][5] for iz in xrange(6)]).T)
    ax.set_ylabel(r'$\log \phi^*$')
    ax.set_xlabel('Redshift')
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    plot_file = 'lf_ztrends.pdf'
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    print 'plot saved to ', plot_dir + plot_file


def plot_smf_z(reftemp='smf_{}.dat', filetemp='smf_{}_{}_{}.dat',
               sep_panels=False, Mlims=(8.1, 12), den_range=(2e-7, 0.1),
               plot_file='smf_z.pdf', plot_size=(8, 5)):
    """Plot SMF in colour and redshift bins."""

    global par
    par['H0'] = 100.0
    zlims = (0.002, 0.1, 0.2, 0.3, 0.65)
    labeltemp = r'${} < z < {}$'
    sym = ('ov^<>s')
    plt.clf()
    fig, axes = plt.subplots(2, 3, sharex=True, num=1)
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    for ic in xrange(3):
        colour = 'cbr'[ic]
        clabel = ('All', 'Blue', 'Red')[ic]
        print clabel
        reffile = reftemp.format(colour)
        lf = pickle.load(open(reffile, 'r'))

        if colour == 'c':
            comp = Baldry2011
            cd = comp()
            axes[0, ic].plot(cd[0], cd[1], 'wD', label=cd[3])
            axes[0, ic].errorbar(cd[0], cd[1], cd[2], fmt='none', ecolor='k')

#        axes[0, ic].errorbar(lf['Mbin'], lf['phi'], lf['phi_err'],
#                             fmt='s', color='k')
        for iz in xrange(len(zlims)-1):
            zlo = zlims[iz]
            zhi = zlims[iz+1]
            infile = filetemp.format(colour, zlo, zhi)
            label = labeltemp.format(zlo, zhi)

            lfz = pickle.load(open(infile, 'r'))
            cmp = lfz['comp'] * (lfz['phi'] > 0)
            Mbin = lfz['Mbin'][cmp]
            phin = lf['phi'][cmp]
            phiz = lfz['phi'][cmp]
            phiz_err = lfz['phi_err'][cmp]
            axes[0, ic].errorbar(Mbin, phiz, phiz_err, fmt=sym[iz], label=label)
            axes[1, ic].errorbar(Mbin, phiz/phin, phiz_err/phin, fmt=sym[iz])
#            with printoptions(precision=1, suppress=True):
#                print zlo, zhi, lfz['phi']/lf['phi']
        ax = axes[0, ic]
        ax.set_xlim(Mlims)
        ax.set_ylim(den_range)
        ax.semilogy(basey=10, nonposy='clip')
        ax.text(0.9, 0.9, clabel, ha='right', transform=ax.transAxes)
        if colour == 'c':
            ax.legend(loc=3, numpoints=1)

        axes[1, ic].plot(Mlims, (1, 1), ':')
        axes[1, ic].set_xlim(Mlims)
#        axes[1, ic].set_ylim(0.1, 10)
#        axes[1, ic].semilogy(basey=10, nonposy='clip')
        axes[1, ic].set_ylim(0.0, 1.99)
        axes[1, ic].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0, 0].set_ylabel(den_mass_label)
    axes[1, 0].set_ylabel(r'$\phi(M_z)/\phi(M_{\rm tot})$')
    axes[1, 1].set_xlabel(plot_label(lf['param'])[1])

    for icol in (1, 2):
        for irow in (0, 1):
            axes[irow, icol].yaxis.set_major_formatter(
                matplotlib.ticker.NullFormatter())

    plt.draw()

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
        print 'plot saved to ', plot_dir + plot_file


def plot1d_old(panels, xlimits=(-24, -12), ylimits=(2e-7, 1), 
           xlab=r'$M - 5 \lg h$', ylab=r'$\phi(M)$', schec_plot=True, 
           schec_label=False, plot_size=(5,8), plot_file=None):
    """Plot 1d LFs: panels contains list of things to be plotted 
    within each panel: a list of input files, list of line labels,
    comparison data, comparison Schechter function and panel label."""


    npanel = len(panels)
    plt.clf()
    if npanel > 1:
        bot = 0.07
    else:
        bot = 0.1
    fig, axes = plt.subplots(npanel, sharex=True, num=1)
    fig.subplots_adjust(bottom=bot, left=0.125, hspace=0, wspace=0)
    fig.text(0.5, 0.01, xlab, ha='center', va='center')
    fig.text(0.01, 0.5, ylab, ha='center', va='center', rotation='vertical')

    ip = 0
    for panel in panels:
        try:
            ax = axes[ip]
        except:
            ax = axes
            
        files, llabels, comp, schec_comp, plabel = panel

        # Plot any comparison data first so it doesn't overplot main results
        if comp:
            cd = comp()
            ax.plot(cd[0], cd[1], 'wD', label=cd[3])
            ax.errorbar(cd[0], cd[1], cd[2], fmt='none', ecolor='k')
        if schec_comp:
            util.schec_plot(schec_comp[0], schec_comp[1], schec_comp[2], 
                            xlimits[0], xlimits[1], lineStyle='r:', axes=ax)

        i = 0
        symbols = itertools.cycle(sym_list)
        colours = itertools.cycle(clr_list)
        for infile in files:
            f = open(infile, 'r')
            dat = pickle.load(f)
            f.close()
            sym = symbols.next()
            clr = colours.next()
            cmp = dat['comp']
            if llabels:
                llabel = llabels[i]
            else:
                llabel = None

            ax.errorbar(dat['Mbin'][cmp], dat['phi'][cmp], dat['phi_err'][cmp], 
                        fmt=sym, mfc=clr, ecolor=clr, label=llabel)
            if schec_plot:
                util.schec_plot(dat['alpha'], dat['Mstar'], 10**dat['lpstar'],
                                xlimits[0], xlimits[1], lineStyle=clr+'-',
                                axes=ax)
            i += 1

        if plabel:
            ax.text(0.1, 0.9, plabel, ha='right', transform=ax.transAxes)
        if ip == 0:
            ax.legend(loc=0)
        ax.axis(xlimits + ylimits)
        ax.semilogy(basey=10, nonposy='clip')
        if schec_label:
            ax.text(0.2, 0.4, r'$\alpha = {:4.2f} \pm {:4.2f}$'.format(
                alpha, np.mean(alphaErr)),
                    transform = ax.transAxes)
            ax.text(0.2, 0.3, r'$M^* = {:5.2f} \pm {:5.2f}$'.format(
                Mstar, np.mean(MstarErr)),
                    transform = ax.transAxes)
            ax.text(0.2, 0.2, r'$\log \phi^* = {:4.2f} \pm {:4.2f}$'.format(
                lpstar, np.mean(lpstarErr)),
                    transform = ax.transAxes)
            ax.text(0.2, 0.1, r'$\chi^2/\nu = {:4.2f}$'.format(chi2/nu),
            transform = ax.transAxes)
        ip += 1
    plt.draw()

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def plot1d(panels, plabels, xlimits=(-24, -12), ylimits=(2e-7, 1),
           xlab=r'$M - 5 \lg h$', ylab=r'$\phi(M)$', schec_plot=True,
           schec_label=False, plot_size=(5, 8), plot_file=None):
    """Plot 1d LFs: panels contains list of things to be plotted
    within each panel: [[lf1, lf2, ...], [lf1, lf2, ...], ...]
    Each lf is a dictionary including 'Mbin', 'phi', 'phi_err', 'props'
    props gives the line/symbol properties, e.g. 'r+' or 'k-'
    err_plot = True to show error bars
    schec_plot = True to show Schechter fit
    schec_label = True to show Schechter parameters."""

    npanel = len(panels)
    plt.clf()
    if npanel > 1:
        bot = 0.07
    else:
        bot = 0.1
    fig, axes = plt.subplots(npanel, sharex=True, num=1)
    fig.subplots_adjust(bottom=bot, left=0.125, hspace=0, wspace=0)
    fig.text(0.5, 0.01, xlab, ha='center', va='center')
    fig.text(0.01, 0.5, ylab, ha='center', va='center', rotation='vertical')

    ip = 0
    for panel, plabel in zip(panels, plabels):
        try:
            ax = axes[ip]
        except:
            ax = axes

        for lf in panel:
            cmp = lf['comp']
            ax.plot(lf['Mbin'][cmp], lf['phi'][cmp], lf['props'],
                    label=lf['label'])
            if lf['err_plot']:
                ax.errorbar(lf['Mbin'][cmp], lf['phi'][cmp],
                            lf['phi_err'][cmp], fmt='none',
                            ecolor=lf['props'][0])
            if lf['schec_plot']:
                util.schec_plot(lf['alpha'], lf['Mstar'], 10**lf['lpstar'],
                                xlimits[0], xlimits[1], lineStyle=lf['props'],
                                axes=ax)
            if lf['schec_label']:
                ax.text(0.2, 0.4, r'$\alpha = {:4.2f} \pm {:4.2f}$'.format(
                    lf['alpha'], np.mean(lf['alphaErr'])),
                        transform=ax.transAxes)
                ax.text(0.2, 0.3, r'$M^* = {:5.2f} \pm {:5.2f}$'.format(
                    lf['Mstar'], np.mean(lf['MstarErr'])),
                        transform=ax.transAxes)
                ax.text(0.2, 0.2, r'$\log \phi^* = {:4.2f} \pm {:4.2f}$'.format(
                    lf['lpstar'], np.mean(lf['lpstarErr'])),
                        transform=ax.transAxes)
                ax.text(0.2, 0.1, r'$\chi^2/\nu = {:4.2f}$'.format(
                    lf['chi2/nu']), transform=ax.transAxes)

        if plabel:
            ax.text(0.1, 0.9, plabel, ha='right', transform=ax.transAxes)
        if ip == 0:
            ax.legend(loc=0)
        ax.axis(xlimits + ylimits)
        ax.semilogy(basey=10, nonposy='clip')
        ip += 1
    plt.draw()

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')

def plotnd(param_list, schec_guess=None, plot_lab=None, plot_file=None):
    """Plot nd LFs in single panel."""

    plt.clf()
    ax = plt.subplot(111)
    for param in param_list:
        infile = param[0]
        iq = param[1]
        f = open(infile, 'r')
        dat = pickle.load(f)
        f.close()

        ax.axis([dat['qty_list'][iq].absMin, dat['qty_list'][iq].absMax, 
                 2e-7, 1])
        ax.semilogy(basey=10, nonposy='clip')

        qtyi = dat['qty_list'][iq]
        step = dat['edges'][iq][1] - dat['edges'][iq][0]
        Mbin = dat['edges'][iq][:-1] + 0.5*step
        ndim = len(dat['phi'].shape)
        if ndim > 1:
            absStep = 1.0
            for i in xrange(ndim):
                absStep *= (dat['edges'][i][1] - dat['edges'][i][0])
            bincorr = absStep/qtyi.absStep
            sumdims = range(ndim)
            sumdims.remove(iq)
            sumdims = tuple(sumdims)
            phi = np.sum(dat['phi'], axis=sumdims) * bincorr
            phi_err = np.sum(dat['phi_err']**2, axis=sumdims)**0.5 * bincorr
        else:
            phi = dat['phi']
            phi_err = dat['phi_err']
        ax.errorbar(Mbin, phi, phi_err, fmt='o')
        if schec_guess:
            fitpars = lum.schecFit(Mbin, phi, phi_err, schec_guess)
            alpha, alphaErr, Mstar, MstarErr, lpstar, lpstarErr, chi2, nu = fitpars
            lum.plotSchec(alpha, Mstar, 10**lpstar, Mbin[0], Mbin[-1],
                          lineStyle='--', axes=ax)
            ax.text(0.1, 0.6, r'$\chi^2/\nu = {:4.2f}$'.format(chi2/nu),
                    transform = ax.transAxes)
            ax.text(0.1, 0.5, r'$\alpha = {:4.2f} \pm {:4.2f}$'.format(
                    alpha, np.mean(alphaErr)),
                    transform = ax.transAxes)
            ax.text(0.1, 0.4, r'$M^* = {:5.2f} \pm {:5.2f}$'.format(
                    Mstar, np.mean(MstarErr)),
                    transform = ax.transAxes)
            ax.text(0.1, 0.3, r'$\log \phi^* = {:4.2f} \pm {:4.2f}$'.format(
                    lpstar, np.mean(lpstarErr)),
                    transform = ax.transAxes)
        label = plot_label(dat['qty_list'][iq].name)[1]
        ax.set_xlabel(label)
        ax.set_ylabel(r'$\Phi(M) / h^3 {\rm Mpc}^{-3}$')
        if plot_lab:
            ax.text(0.8, 0.9, plot_lab, transform = ax.transAxes)
        try:
            ax.text(0.1, 0.2, r'$P = {:5.2f} \pm {:5.2f}$'.format(dat['P'], dat['P_err']),
                    transform = ax.transAxes)
            ax.text(0.1, 0.1, r'$Q = {:5.2f} \pm {:5.2f}$'.format(dat['Q'], dat['Q_err']),
                    transform = ax.transAxes)
        except:
            pass

    plt.draw()

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(3.2, 2.5)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def delta_plot_morph(param_list=(('../jswml/lf_z065_lfchi_c.dat', 'All'), 
                           ('lf_lfchi_s.dat', 'S'), 
                           ('lf_lfchi_e.dat', 'E')), 
               plot_file='delta_morph.pdf'):
    delta_plot(param_list, plot_file)

def delta_plot_post_morph(param_list=(('../jswml/lf_z065_post_post_c.dat', 'All'), 
                           ('lf_post_s.dat', 'S'), 
                           ('lf_post_e.dat', 'E')), 
               plot_file='delta_post_morph.pdf'):
    delta_plot(param_list, plot_file)

def delta_plot_colour(template='ev_lfchi_petro_fit_{}.dat', 
                      plot_file='delta.pdf'):
    delta_plot(((template.format('c'), 'All'), 
                (template.format('b'), 'Blue'),
                (template.format('r'), 'Red')), 
               plot_file)


def delta_plot_sim(template='lf_kde_z01_lfchi_{}.dat',
                   plot_file='delta_sim.pdf'):
    param_list = []
    for isim in xrange(10):
        param_list.append((template.format(isim), ' '))
    delta_plot(param_list, plot_file)


def delta_plot(param_list, plot_file=None):
    """Plot radial overdensities for list of files."""

    nplot = len(param_list)
    plt.clf()
    fig, axes = plt.subplots(nplot, sharex=True, num=1)
    ip = 0
    for param in param_list:
        infile = param[0]
        label = param[1]

        f = open(infile, 'r')
        dat = pickle.load(f)
        f.close()

        try:
            ax = axes[ip]
        except:
            ax = axes

        ax.step(dat['zbin'], dat['delta'], where='mid')
        ax.errorbar(dat['zbin'], dat['delta'], dat['delta_err'], fmt='none')
        dz = dat['zbin'][1] - dat['zbin'][0]
#        ax.bar(dat['zbin'] - 0.5*dz, 2*np.sqrt(dat['den_var']), width=dz, 
#               bottom=dat['delta'] - np.sqrt(dat['den_var']), alpha=0.1, 
#               ec='none')
        ax.bar(dat['zbin'] - 0.5*dz, 2*np.sqrt(dat['den_var']), width=dz, 
               bottom=1.0-np.sqrt(dat['den_var']), alpha=0.1, 
               ec='none')
        ax.plot([0.0, dat['zbin'][-1]], [1.0, 1.0], ':')
        if ip == nplot/2:
            ax.set_ylabel(r'$\Delta(z)$')
#        ax.set_ylim(0.3, 1.7)
        ax.semilogy(basey=10, nonposy='clip')
        ax.set_ylim(0.3, 10.0)
        ax.text(0.05, 0.9, label, transform = ax.transAxes)
        # ax.text(0.95, 0.15, r'$Q = {:4.2f}$'.format(dat['Q']), 
        #         ha='right', transform = ax.transAxes)
        # ax.text(0.95, 0.05, r'$P = {:4.2f}$'.format(dat['P']), 
        #         ha='right', transform = ax.transAxes)
        ip += 1
    ax.set_xlabel(r'Redshift $z$')          
    fig.subplots_adjust(hspace=0)

    plt.draw()

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(5, 8)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def gridplot_colour(templates=('ev_post_petro_fit_{}.dat', 'ev_lfchi_petro_fit_{}.dat'), 
                    plabels=('Mean prob', 'LF-redshift'),
                    colours='kbr', styles=('solid', 'dashed', 'dotted'),
                    llabels=('All', 'Blue', 'Red'),
                    clevels=[4], comp_data=True,
                    plot_limits=(0.0, 2.5, 0.2, 1.5), plot_file='chigrid.pdf'):
    panels = []
    for template, plabel in zip(templates, plabels):
        files = []
        for clr in 'cbr':
            files.append(template.format(clr))
        panels.append((files, plabel))
    gridplot(panels, clevels=clevels, colours=colours, styles=styles,
             llabels=llabels, comp_data=comp_data,
             plot_limits=plot_limits, plot_file=plot_file)


def gridplot_morph(param_list=(('../jswml/lf_z065_post_post_c.dat', 'All'),
                               ('lf_post_e.dat', 'E'),
                               ('lf_post_s.dat', 'S')),
                   plot_file='chigrid_morph.pdf'):
    gridplot(param_list, plot_file)


def gridplot_lum(inroot='lf_lum_Qfix_{}_{}.dat',
                 Mlimits = (-23, -22, -21, -20, -19, -18, -17, -16, -15),
                 colour='kkkkkkkk', nplot=8, plot_file=None):
    """(P,Q) contour plots for lum bins."""
    param_list = []
    for i in range(len(Mlimits)-1):
        Mlo = Mlimits[i]
        Mhi = Mlimits[i+1]
        infile = inroot.format(Mlo, Mhi)
        label = r'$M_r = [{}, {})$'.format(Mlo, Mhi)
        param_list.append((infile, colour[i], label))
    gridplot(param_list, nplot, plot_file)

def gridplot_mass(param_list=(('lf_mass_5_9.5.dat', 'k', '5-9.5'),
                              ('lf_mass_9.5_10.dat', 'b', '9.5-10'),
                              ('lf_mass_10_10.5.dat', 'g', '10-10.5'), 
                              ('lf_mass_10.5_11.dat', 'r', '10.5-11'), 
                              ('lf_mass_11_11.5.dat', 'm', '11-11.5'), 
                              ('lf_mass_11.5_15.dat', 'c', '11.5-15')), 
                  nplot=6, plot_file=None):
    gridplot(param_list, nplot, plot_file)

def gridplot(panels, clevels=[1, 4, 9], colours='kbr', 
             styles=('solid', 'dashed', 'dotted'),
             llabels=('All', 'Blue', 'Red'), 
             comp_data=False, greyscale=False,
             plot_limits=None, plot_file=None):
    """Plot chi^2 grids.  
    panels contains list of things to be plotted within each panel: 
    a list of input files and panel label.
    clevels gives the contour levels above minimum chi^2, 
    the defaults corresponding to 1, 2 and 3 sigma."""

    npanel = len(panels)

    plt.clf()
    fig, axes = plt.subplots(npanel, sharex=True, num=1)
    fig.subplots_adjust(left=0.125, hspace=0, wspace=0)
    ip = 0
    for panel in panels:
        try:
            ax = axes[ip]
        except:
            ax = axes

        lines = []
        labels = []
        files, plabel = panel
        for infile, colour, style, label in zip(files, colours, styles, llabels):
            f = open(infile, 'r')
            dat = pickle.load(f)
            f.close()

            Pstep = float(dat['Pbins'][1] - dat['Pbins'][0])/dat['Pbins'][2]
            Qstep = float(dat['Qbins'][1] - dat['Qbins'][0])/dat['Qbins'][2]
            extent = (dat['Pbins'][0], dat['Pbins'][1], 
                      dat['Qbins'][0], dat['Qbins'][1])
            print 'extent', extent
            if greyscale:
                cmap = matplotlib.cm.Greys
                im = ax.imshow(dat['chi2grid'], cmap=cmap, aspect='auto', 
                               origin='lower', extent=extent, 
                               interpolation='nearest')
                cb = plt.colorbar(im, ax=ax)
            chi2min = np.min(dat['chi2grid'])
            cs = ax.contour(dat['chi2grid'], chi2min + clevels,
                            colors=colour, linestyles=style, 
                            aspect='auto', origin='lower', extent=extent)
            lines.append(cs.collections[0])
            labels.append(label)
            ax.plot(dat['P'], dat['Q'], 'o'+colour)
            print dat['P'], dat['Q'], dat['P_err'], dat['Q_err']
        if plot_limits:
            ax.axis(plot_limits)
        if plabel:
            ax.text(0.95, 0.9, plabel, ha='right', transform = ax.transAxes)
        if ip == 0:
            ax.legend(lines, labels, loc=3)
        if comp_data:
            ax.errorbar(1.8, 0.7, xerr=0.5, yerr=0.2, ecolor='k')
        ax.set_ylabel(r'$Q_e\ ({\rm mag})$')
        ip += 1
    ax.set_xlabel(r'$P_e$')
    plt.draw()

    if plot_file:
        fig = plt.gcf()
        if npanel > 1:
            fig.set_size_inches(5, 8)
        else:
            fig.set_size_inches(5, 5)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
        print 'plot saved to ', plot_dir + plot_file


def gridplot_av(filenames=('lf_kde_z01_post_{}.dat', 'lf_kde_z01_lfchi_{}.dat'),
                sims=xrange(10), label=('Mean prob', 'LF-redshift'),
                plot_file='sim_ev_contours.pdf'):
    """Average of multiple chi-squared grids."""
    nsim = len(sims)
    P = np.zeros(nsim)
    Q = np.zeros(nsim)
    plt.clf()
    nplot = len(filenames)
    for iplot in xrange(nplot):
        ax = plt.subplot(nplot+1, 1, iplot+1)
        first = True

        for i in xrange(nsim):
            infile = filenames[iplot].format(sims[i])
            f = open(infile, 'r')
            dat = pickle.load(f)
            f.close()

            chi2grid = dat['chi2grid']
            if first:
                chi2av = chi2grid
                first = False
            else:
                chi2av += chi2grid

            Qmin = dat['Qbins'][0]
            Qmax = dat['Qbins'][1]
            nQ = dat['Qbins'][2]
            Qstep = float(Qmax - Qmin)/nQ
            Pmin = dat['Pbins'][0]
            Pmax = dat['Pbins'][1]
            nP = dat['Pbins'][2]
            Pstep = float(Pmax - Pmin)/nP
            (iq, ip) = np.unravel_index(np.argmin(chi2grid), chi2grid.shape)
            P[i] = Pmin + (ip+0.5)*Pstep
            Q[i] = Qmin + (iq+0.5)*Qstep

            extent = (Pmin, Pmax, Qmin, Qmax)
            chi2min = np.min(chi2grid)
            ax.contour(chi2grid, (chi2min+4, ), # colors=('k'),
                       aspect='auto', origin='lower', extent=extent,
                       linewidths=0.2)

        chi2av /= nsim
        chi2min = np.min(chi2av)
        ax.contour(chi2av, (chi2min+4, ), colors=('k'),
                   aspect='auto', origin='lower', extent=extent)
        ax.errorbar(P.mean(), Q.mean(), xerr=P.std(), yerr=Q.std())
        ax.text(0.7, 0.9, label[iplot], transform=ax.transAxes)
        ax.set_ylabel(r'$Q_e\ ({\rm mag})$')
        print 'min chi^2 (P,Q) = ({} +- {}, {} +- {})'.format(
            P.mean(), P.std(), Q.mean(), Q.std())
    ax.set_xlabel(r'$P_e$')
    plt.subplots_adjust(hspace=0.0)
    plt.draw()

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(5, 10)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
        print 'plot saved to ', plot_dir + plot_file


def gridplot_mocks(template=('ev_{:02d}.dat',),
                   sims=(1, 2, 5, 6, 14, 20)):
    gridplot_av(template, sims, label=(' ',), plot_file=None)


def ev_lin_fit(infile, nfit=2, plot=False):
    """Linear fit to minimum (P, Q) chi2 values."""

    f = open(infile, 'r')
    dat = pickle.load(f)
    f.close()

    chi2grid = dat['chi2grid']
    Qmin = dat['Qbins'][0]
    Qmax = dat['Qbins'][1]
    nQ = dat['Qbins'][2]
    Qstep = float(Qmax - Qmin)/nQ
    iqvals = np.arange(nQ)
    Qvals = Qmin + (iqvals + 0.5)*Qstep
    Pmin = dat['Pbins'][0]
    Pmax = dat['Pbins'][1]
    nP = dat['Pbins'][2]
    Pstep = float(Pmax - Pmin)/nP
    Pfit = []
    Qfit = []
    Qerr_fit = []
    chi2min = []
    dchi2 = 1
    for ip in xrange(nP):
        P = Pmin + (ip+0.5)*Pstep
        chi2vals = chi2grid[:,ip]
        iq = np.argmin(chi2vals)
        # Fit quadratic to chi2(Q) at (2*nfit + 1) bins around minimum 
        if iq > nfit-1 and iq < nQ - nfit:
            bins = range(iq-nfit, iq+nfit+1)
            (a, b, c) = np.polyfit(bins, chi2vals[bins], 2)
            xmin = -b/(2*a)
            chi2min.append(np.interp(xmin, iqvals, chi2vals))
            xlo = (-b - math.sqrt(4*a*dchi2))/(2*a)
            xhi = (-b + math.sqrt(4*a*dchi2))/(2*a)
            Qerr = 0.5 * (np.interp(xhi, iqvals, Qvals) - 
                          np.interp(xlo, iqvals, Qvals))
            Pfit.append(P)
            Qfit.append(np.interp(xmin, iqvals, Qvals))
            Qerr_fit.append(Qerr)
            # pdb.set_trace()
    (m, c) = np.polyfit(Pfit, Qfit, 1, w=1.0/(np.array(Qerr_fit)**2))
    print 'Q = {:5.2f} * P + {:5.2f}'.format(m,c)

    if plot:
        plt.clf()
        ax = plt.subplot(111)
        extent = (Pmin, Pmax, Qmin, Qmax)
        im = ax.imshow(dat['chi2grid'], aspect='auto', vmax=max(chi2min) + 1,
                       origin='lower', extent=extent, 
                       interpolation='nearest')
        cb = plt.colorbar(im, ax=ax)
        ax.errorbar(Pfit, Qfit, yerr=Qerr_fit)
        plt.plot((Pmin, Pmax), (m*Pmin + c, m*Pmax + c))
        plt.ylim(Qmin, Qmax)
        ax.set_ylabel(r'$Q$')
        ax.set_xlabel(r'$P$')
        plt.draw()
    return m, c

def ev_lin_fit_sims(filetemp='lf_kde_z01_{}_{}.dat'):
    """Linear fit to minimum (P, Q) chi2 values averaged over sims."""

    marr = np.zeros(10)
    carr = np.zeros(10)
    for method in ('post', 'lfchi'):
        for i in xrange(10):
            marr[i], carr[i] = ev_lin_fit(filetemp.format(method, i))
        print method
        print 'm = {:5.2f} +/- {:5.2f}, c = {:5.2f} +/- {:5.2f}'.format(
            np.mean(marr), np.std(marr), np.mean(carr), np.std(carr))

def likeplot(infile):
    """Plot 1d likelihood distribution."""
    f = open(infile, 'r')
    (Qa, Pa, like_arr) = pickle.load(f)
    f.close()
    
    plt.clf()
    ax = plt.subplot(2, 1, 1)
    ax.plot(Qa, like_arr)
    ax.set_xlabel('Q')
    ax.set_ylabel('-cost')
    ax = plt.subplot(2, 1, 2)
    ax.plot(Qa, Pa)
    ax.set_xlabel('Q')
    ax.set_ylabel('P')
    plt.draw()

def likeplot_multi(filename='like_post_{}.dat', nsim=10):
    """Analysis of multiple 1d likelihood distributions."""
    P = np.zeros(nsim)
    Q = np.zeros(nsim)
    for i in xrange(nsim):
        infile = filename.format(i)
        f = open(infile, 'r')
        (Qa, Pa, like_arr) = pickle.load(f)
        f.close()

        if (i == 0):
            likeav = like_arr
            Pav = Pa
        else:
            likeav += like_arr
            Pav += Pa
            
        if 'zero_slope' in filename:
            iq = np.argmin(np.abs(like_arr))
        else:
            iq = np.argmax(like_arr)
        P[i] = Pa[iq]
        Q[i] = Qa[iq]

    likeav /= nsim
    Pav /= nsim
    print 'MaxL (P,Q) = ({} +- {}, {} +- {})'.format(P.mean(), P.std(), Q.mean(), Q.std())

    plt.clf()
    ax = plt.subplot(2, 1, 1)
    ax.plot(Qa, likeav)
    ax.set_xlabel('Q')
    ax.set_ylabel('-cost')
    ax = plt.subplot(2, 1, 2)
    ax.plot(Qa, Pav)
    ax.set_xlabel('Q')
    ax.set_ylabel('P')
    plt.draw()
    
def phi2d(infile, iq=0, jq=1, ired=1, jred=1):
    """Return 2d distribution fn, projecting over any other dimensions,
    and reducing number of bins by (ired, jred)."""
    global par, cosmo

    f = open(infile, 'r')
    dat = pickle.load(f)
    f.close()

    for qty in dat['qty_list']:
        print qty.name
    par = dat['par']
    ndim = len(dat['phi'].shape)
    qtyi = dat['qty_list'][iq]
    qtyj = dat['qty_list'][jq]
    qtyi.nabs /= ired
    qtyi.absStep *= ired
    qtyj.nabs /= jred
    qtyj.absStep *= jred
    vic = np.linspace(qtyi.absMin + 0.5*qtyi.absStep,
                      qtyi.absMax - 0.5*qtyi.absStep,
                      qtyi.nabs)
    vjc = np.linspace(qtyj.absMin + 0.5*qtyj.absStep,
                      qtyj.absMax - 0.5*qtyj.absStep,
                      qtyj.nabs)
    if ndim > 2:
        print 'raw phi_err range:', np.min(dat['phi_err']), np.max(dat['phi_err'])
        absStep = 1.0
        for i in xrange(ndim):
            absStep *= (dat['edges'][i][1] - dat['edges'][i][0])
        bincorr = absStep/qtyi.absStep/qtyj.absStep
        sumdims = range(ndim)
        sumdims.remove(iq)
        sumdims.remove(jq)
        sumdims = tuple(sumdims)
        Mhist = np.sum(dat['Mhist'], axis=sumdims)
        phi = np.sum(dat['phi'], axis=sumdims) * bincorr
        phi_err = np.sum(dat['phi_err']**2, axis=sumdims)**0.5 * bincorr
    else:
        Mhist = dat['Mhist']
        phi = dat['phi']
        phi_err = dat['phi_err']

    Mhist = util.rebin_sum(Mhist, ired, jred)
    phi = util.rebin_mean(phi, ired, jred)
    phi_err = util.rebin_quad(phi_err, ired, jred)

    # Transpose arrays since 1st, 2nd dims correspond to y, x axes on plots
    Mhist = Mhist.transpose()
    phi = phi.transpose()
    phi_err = phi_err.transpose()

    # Volume probed by each phi bin
    # Use completeness limits (appMin, appMax) rather than selection limits,
    # as these are more indicative of the effective limits
    # (e.g. no selection on Sersic mag, but range will be similar to Petrosian)
    if qtyi.kind == 'mag' and qtyj.kind in ('sb', 'mu', 'radius'):
        cosmo = util.CosmoLookup(par['H0'], par['omega_l'], 
                                 (par['zmin'], par['zmax']))
        zmin, zmax = par['zmin'], par['zmax']
        vol = np.zeros((qtyj.nabs, qtyi.nabs))
        for j in xrange(qtyj.nabs):
            for i in xrange(qtyi.nabs):
                zloi = zdm(qtyi.appMin-vic[i], qtyi.kmean, (zmin, zmax), qtyi.Q)
                zhii = zdm(qtyi.appMax-vic[i], qtyi.kmean, (zmin, zmax), qtyi.Q)

                # mag-radius
                if qtyj.kind == 'radius':
                    z1 = z_r_R(qtyj.appMax, vjc[j], (zmin, zmax))
                    z2 = z_M_s_R(vic[i], par['mu_max'], vjc[j], qtyj.kmean,
                                 qtyj.Q, (zmin, zmax))
                    zloj = max(z1, z2)
                    z1 = z_r_R(qtyj.appMin, vjc[j], (zmin, zmax))
                    z2 = z_M_s_R(vic[i], par['mu_min'], vjc[j], qtyj.kmean,
                                 qtyj.Q, (zmin, zmax))
                    zhij = min(z1, z2)

                # mag-SB
                if qtyj.kind in ('sb', 'mu'):
                    z1 = z_s_S(qtyj.appMin, vjc[j], qtyj.kmean, (zmin, zmax))
                    z2 = z_S_M_r(vjc[j], vic[i], par['rad_max'], (zmin, zmax))
                    zloj = max(z1, z2)
                    z1 = z_s_S(qtyj.appMax, vjc[j], qtyj.kmean, (zmin, zmax))
                    z2 = z_S_M_r(vjc[j], vic[i], par['rad_min'], (zmin, zmax))
                    zhij = min(z1, z2)

                zlo = max(zloi, zloj)
                zhi = min(zhii, zhij)
                vol[j,i] = par['area']/3.0*(cosmo.dm(zhi)**3 - cosmo.dm(zlo)**3)
    else:
        vol = None
    return qtyi, qtyj, vic, vjc, Mhist, phi, phi_err, vol

def plot2d(infile, iq=0, jq=1, ired=1, jred=1, ngmin=16, vol_contour=(1800,), 
           vmin=-6, vmax=-1.5, plot_file=None):
    """Plot 2d distribution, projecting over any other dimensions."""
    global par, cosmo

    qtyi, qtyj, vic, vjc, Mhist, phi, phi_err, vol = phi2d(infile, iq, jq)
    print 'phi_err range:', np.min(phi_err), np.max(phi_err)
    extent = (qtyi.absMin, qtyi.absMax, qtyj.absMin, qtyj.absMax)
    log_phi = np.log10(phi)
    log_phi = np.ma.array(log_phi, mask=np.isnan(log_phi))
    cmap = matplotlib.cm.jet
    cmap.set_bad('w',1.)

    plt.clf()
    plt.imshow(log_phi, cmap=cmap, aspect='auto', origin='lower', 
               extent=extent, interpolation='nearest',
               vmin=vmin, vmax=vmax)
    cb = plt.colorbar()
    cb.set_label(r'$\log_{10} \phi$')
    plt.contour(Mhist, (ngmin,), colors=('k',), linestyles='dashed',
                aspect='auto', origin='lower', extent=extent)

    if qtyi.kind == 'mag' and qtyj.kind in ('sb', 'mu', 'radius'):
        plt.contour(vol, vol_contour, colors=('k',),
                    aspect='auto', origin='lower', extent=extent)

    # Fit Choloniewski function to bbd
    chol_par = None
    if qtyj.kind in ('sb', 'mu') and qtyi.kind == 'mag':
        M = np.tile(vic, (qtyj.nabs, 1))
        mu = np.tile(vjc, (qtyi.nabs, 1)).transpose()
        chol_par, chi2 = chol_fit(M, mu, phi, phi_err, 
                                  Mhist, ngmin)
        chol_arr = np.log10(choloniewski(M, mu, chol_par))
        v = np.linspace(vmin, vmax, int(2*(vmax - vmin)) + 1)
        print 'contours ', v
        plt.contour(chol_arr, v, aspect='auto', origin='lower', extent=extent)

        # If simulation parameters are given, plot as dashed contours
        try:
            chol_arr = np.log10(choloniewski(M, mu, par['sim_par']))
            plt.contour(chol_arr, v, aspect='auto', origin='lower',
                        extent=extent, linestyles='dashed')
        except:
            pass

    plt.xlabel(plot_label(qtyi.name)[1])
    plt.ylabel(plot_label(qtyj.name)[1])
    print 'ylabel:', plot_label(qtyj.name)[1], qtyj.name
    plt.draw()

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(5, 4)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')

def bbd_petro_plot():
    """Plot Petrosian BBD."""
    plot2d('bbd_c.dat', 0, 1, plot_file='bbd_petro.pdf')

def bbd_sersic_plot():
    """Plot Sersic BBD."""
    plot2d('bbd_sersic.dat', 1, 2, plot_file='bbd_sersic.pdf')

def lum_size_plot():
    """Plot Sersic lum-size relation."""
    plot2d('lum_size_c.dat', 2, 3, vmax=-1, plot_file='lum_size.pdf')
    plot2d('lum_size_b.dat', 2, 3, vmax=-1, plot_file='lum_size_b.pdf')
    plot2d('lum_size_r.dat', 2, 3, vmax=-1, plot_file='lum_size_r.pdf')

def sb_size_plot():
    """Plot Sersic sb-size relation."""
    plot2d('sb_size_c.dat', 2, 3, vmax=-1, plot_file='sb_size.pdf')
    plot2d('sb_size_b.dat', 2, 3, vmax=-1, plot_file='sb_size_b.pdf')
    plot2d('sb_size_r.dat', 2, 3, vmax=-1, plot_file='sb_size_r.pdf')

def bbd_slices():
    """Plot and fit Gaussians to Petrosian and Sersic SB in mag bins."""
    plot_slices(param_list=(('bbd_c.dat', 0, 1, 'b'),
                            ('bbd_sersic.dat', 1, 2, 'g')),
                ired=2, jred=1, p0=[20, 0.6, 0.001], 
                vali_min=-23, vali_max=-15,
                Vmin=1800, ngmin=5, 
                phi_range=[1e-6, 0.9], trends_range=(19, 24),
                slice_label=r'$M_r = {:6.2f}$',
                ilabel=r'$M_r$', jlabel=r'$\mu_e / $ mag arcsec$^{-2}$', 
                xlab=0.1, ylab=0.85, slice_file='bbd_slices.pdf', 
                trends_file='bbd_trends.pdf', plot_comp=Driver_BBD)

def Driver_BBD():
    """Plot Driver et al 2005 lum-SB relation from their Table 3.
    (B - r) = 1.2 offset comes from misc.mgc_phot()."""
    
    file = os.environ['HOME'] + '/Documents/Research/LFdata/Driver2005/table3.dat'
    data = np.loadtxt(file)
    M = data[:,0] - 1.2
    mu = data[:,1] - 1.2
    sigma = data[:,3]
    plt.errorbar(M, mu, sigma, fmt='k^', ecolor='black')
    
def lum_rad_slices():
    """Plot and fit Gaussians to log Re in Sersic mag bins."""
    plot_slices(param_list=(('lum_size_b.dat', 2, 3, 'b'),
                            ('lum_size_r.dat', 2, 3, 'r')),
                ired=2, jred=1, p0=[1, 0.3, 0.001], 
                vali_min=-23, vali_max=-15,
                Vmin=1800, ngmin=5, 
                phi_range=[1e-6, 0.9], trends_range=(-0.5, 1.1),
                slice_label=r'$M_r = {:6.2f}$',
                ilabel=r'$M_r - 5\ \lg\ h$', 
                jlabel=r'$\log (R_e / h^{-1} {\rm kpc})$', 
                xlab=0.4, ylab=0.85, slice_file='lum_rad_slices.pdf', 
                trends_file='lum_rad_trends.pdf', plot_comp=shen_2003)

def shen_2003():
    """Plot Shen et al 2003 lum-radius relation from their Fig 6."""

    def early(M):
        return -0.4*0.65*M - 5.06

    def late(M):
        return -0.4*0.26*M + (0.51-0.26)*np.log10(1 + 10**(-0.4*(M+20.91))) -1.71
    lg07 = math.log10(0.7)
    Mmin, Mmax = plt.xlim()
    M = np.linspace(Mmin, Mmax)
    M_7 = M + 5*lg07
    R = early(M_7) + lg07
    plt.plot(M, R, 'r-')
    R = late(M_7) + lg07
    plt.plot(M, R, 'b-')

def lum_rad_slices_ev(colour='c'):
    """Gaussians trends of log Re with mag for late types in z slices."""
    plot_slices(param_list=(('lum_size_ev_0.002_0.1_{}.dat'.format(colour), 2, 3, 'b'), 
                            ('lum_size_ev_0.1_0.2_{}.dat'.format(colour), 2, 3, 'g'),
                            ('lum_size_ev_0.2_0.4_{}.dat'.format(colour), 2, 3, 'r'), 
                            ('lum_size_ev_0.4_0.65_{}.dat'.format(colour), 2, 3, 'c')), 
    # plot_slices(param_list=(('lum_size_ev_0.002_0.2_{}.dat'.format(colour), 2, 3, 'b'), 
    #                         ('lum_size_ev_0.2_0.65_{}.dat'.format(colour), 2, 3, 'g')), 
                ired=2, jred=1, p0=[0.6, 0.2, 1e-4],
                plimits=((-0.6, 1.5), (0.001, 1), (1e-8, 1)),
                vali_min=-23, vali_max=-18,
                Vmin=1800, ngmin=5, 
                phi_range=[1e-6, 0.09], trends_range=None, # (-0.1, 0.1),
                slice_label=r'$M_r = {:6.2f}$',
                ilabel=r'$M_r - 5\ \lg\ h$', 
                jlabel=r'$\Delta \log (R_e / h^{-1} {\rm kpc})$', 
                xlab=0.6, ylab=0.85, slice_file='lum_rad_slice_ev_{}.pdf'.format(colour), 
                trends_file='lum_rad_trends_ev_{}.pdf'.format(colour), offsets=None)
        # offsets='lum_size_{}_trends.dat'.format(colour))
        

def mass_rad_slices(
    file_list=('jswml_mass_size_s.dat', 'jswml_mass_size_e.dat'),
               iq=3, jq=4, ired=2, jred=1, p0=[-0.5, 0.2, 0.1],
               plimits=((-1, 2), (0.001, 2), (1e-8, 1)),
               vali_min=7, vali_max=11.5,
               Vmin=1800, ngmin=5, 
               phi_range=[1e-5, 0.5], trends_range=(-0.6, 1),
               xlab=0.1, ylab=0.85):
    """Plot and fit Gaussians to Sersic SB in Sersic mag bins."""
    plot_slices(file_list=file_list,
                iq=iq, jq=jq, ired=ired, jred=jred, p0=p0, plimits=plimits,
                Vmin=Vmin, ngmin=ngmin, vali_min=vali_min, vali_max=vali_max, 
                phi_range=phi_range, trends_range=trends_range, 
                xlab=xlab, ylab=ylab)

def mass_rad_slices_ev_s():
    mass_rad_slices(
        file_list=('jswml_mass_size_0.002_0.1_s.dat', 'jswml_mass_size_0.1_0.2_s.dat',
                   'jswml_mass_size_0.2_0.3_s.dat', 'jswml_mass_size_0.3_0.5_s.dat'))

def mass_rad_slices_ev_e():
    mass_rad_slices(
        file_list=('jswml_mass_size_0.002_0.1_e.dat', 'jswml_mass_size_0.1_0.2_e.dat',
                   'jswml_mass_size_0.2_0.3_e.dat', 'jswml_mass_size_0.3_0.5_e.dat'))

def plot_slices(param_list, ired=1, jred=1, p0=(0.5, 0.1, 0.01), 
                plimits=None, Vmin=1800, ngmin=5, vali_min=-23, vali_max=-15, 
                phi_range=[1e-5, 0.1], trends_range=None,
                slice_label=r'$M_r = {:6.2f}$',
                ilabel=r'$M_r$', jlabel=r'$\mu_e / $ mag arcsec$^{-2}$', 
                xlab=0.1, ylab=0.85, slice_file=None, trends_file=None, 
                plot_comp=None, offsets=None):
    """Plot and fit Gaussians to phi(jq) in bins of iq"""

    fit_pars = []
    clr_list = []
    ifile = 0
    for param in param_list:
        infile = param[0]
        iq = param[1]
        jq = param[2]
        clr = param[3]
        clr_list.append(clr)
        qtyi, qtyj, vic, vjc, Mhist, phi, phi_err, vol = phi2d(infile, iq, jq,
                                                               ired, jred)
        rangej=[qtyj.absMin+0.01, qtyj.absMax]
        xvals = np.linspace(qtyj.absMin, qtyj.absMax, 50)
        idx = (vali_min < vic) * (vali_max > vic)
        nplot = len(vic[idx])
        Mbin = []
        mean = []
        sigma = []
        mean_err = []
        sigma_err = []
        if ifile == 0:
            plt.clf()
            fig = plt.figure(1, figsize=(12, 8))
            nrow, ncol = util.two_factors(nplot)
            axes = AxesGrid(fig, 111, nrows_ncols = (nrow, ncol), axes_pad=0.0,
                            share_all=False, aspect=False)
            fig.text(0.5, 0.04, jlabel, ha='center', va='center')
            fig.text(0.06, 0.5, r'$\phi / h^3 {\rm Mpc}^{-3}$', 
                     ha='center', va='center', rotation='vertical')

        iplot = 0
        for i in xrange(qtyi.nabs):
            if vali_min < vic[i] and vali_max > vic[i]:
                ax = axes[iplot]
                ax.errorbar(vjc, phi[:,i], phi_err[:,i], fmt=symbols.next(),
                            color=clr)
                # White out points outside Vmin contour or with too few galaxies
                if vol is not None:
                    idx = (vol[:,i] < Vmin) + (Mhist[:,i] < ngmin)
                    ax.plot(vjc[idx], phi[idx,i], linestyle='None',
                            marker=symbols.next(), color='white')

                # Fit Gaussian to rest
                idx = (Mhist[:,i] >= ngmin)
                if vol is not None:
                    idx *= (vol[:,i] >= Vmin)
                res = gauss_fit(vjc[idx], phi[idx,i], phi_err[idx,i], p0, plimits)
                if res:
                    popt, pcov, chi2, nu = res
                    if (pcov[0,0] > 0 and pcov[0,0] < 10 and
                        pcov[1,1] > 0 and pcov[1,1] < 10):
                        phivals = gaussian(xvals, popt[0], popt[1], popt[2])
                        ax.plot(xvals, phivals, color=clr)
                        Mbin.append(vic[i])
                        mean.append(popt[0])
                        sigma.append(popt[1])
                        mean_err.append(math.sqrt(pcov[0,0]))
                        sigma_err.append(math.sqrt(pcov[1,1]))
                        print '{:6.2f} mu {:6.2f} +- {:6.2f}, sig {:6.2f} +- {:6.2f}, chi2 {:6.2f}, nu {:d}'.format(Mbin[-1], mean[-1], mean_err[-1], sigma[-1], sigma_err[-1], chi2, nu)
               
                ax.axis(rangej + phi_range)
                ax.semilogy(basey=10, nonposy='clip')
                if ifile == 0:
                    ax.text(xlab, ylab, slice_label.format(vic[i]),
                            transform = ax.transAxes)
                iplot += 1
        ifile += 1
        fit_pars.append((Mbin, mean, mean_err, sigma, sigma_err))
    plt.draw()

    if slice_file:
        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        plt.savefig(plot_dir + slice_file, bbox_inches='tight')

    # a = raw_input('t for trends plot, any other key to stop here: ')
    # if a != 't':
    #     return
    
    # Read offsets if specified
    if offsets:
        f = open(offsets, 'r')
        line = f.readline()
        M, mu, mu_err, sig, sig_err = eval(line)
        f.close()
    plt.clf()
    ifile = 0
    for fp in fit_pars:
        infile = param_list[ifile][0]
        outfile = infile.replace('.dat', '_trends.dat')
        fout = open(outfile, 'w')
        print >> fout, fp
        fout.close()
        fp = list(fp)
        if offsets:
            fp[1] -= np.interp(fp[0], M, mu)
        plt.errorbar(fp[0], fp[1], fp[2], fmt=symbols.next(), 
                     color=colours[ifile]) # , markersize=8
        # plt.errorbar(fp[0], fp[1], fp[3], fmt='none')
        plt.fill_between(fp[0], np.array(fp[1]) - 0.5*np.array(fp[3]),
                         np.array(fp[1]) + 0.5*np.array(fp[3]), 
                         facecolor=colours[ifile], alpha=0.05)
        ifile += 1
    plt.xlabel(ilabel, fontsize=14)
    plt.ylabel(jlabel, fontsize=14)
    plt.xlim((vali_max, vali_min))
    if trends_range:
        plt.ylim(trends_range)
    if plot_comp:
        plot_comp()
    plt.draw()

    if trends_file:
        fig = plt.gcf()
        fig.set_size_inches(5, 4)
        plt.savefig(plot_dir + trends_file, bbox_inches='tight')

def gauss_fit(x, y, yerr, p0, plimits):
    """Least-squares Gaussian fit to y(x).
    p0 = initial guess of (mean, sigma, amp)."""

    nbin = len(x)
    nu = nbin - 3
    if nu < 0:
        print 'Not enough data points for Gaussian fit'
        return None
    # Find maximum to get starting position and amplitude
    imax = np.argmax(y)
    p0[0] = x[imax]
    p0[2] = y[imax]
    print p0
    
    try:
        # First do simplex fit, then leastsq fit to get covariance matrix
        res = scipy.optimize.fmin(
            gauss_chi2, p0, args=(plimits, x, y, yerr), ftol=0.001, full_output=1, disp=0)
        print 'fmin: ', res
        res = scipy.optimize.leastsq(gauss_resid, res[0], (x, y, yerr),
                                     xtol=0.001, ftol=0.001, full_output=1)
        popt, cov, info, mesg, ier = res
        if ier > 4:
            print mesg
        chi2 = (info['fvec']**2).sum()
        cov *= (chi2/nu)
    
        return popt, cov, chi2, nu
    except:
        print 'Failed to fit Gaussian'
        return None
    
def gauss_chi2(gauss_par, plimits, x, y, yerr):
    """Chi2 of Gaussian fit."""
    if plimits:
        if (plimits[0][0] > gauss_par[0] or plimits[0][1] < gauss_par[0] or
            plimits[1][0] > gauss_par[1] or plimits[1][1] < gauss_par[1] or
            plimits[2][0] > gauss_par[2] or plimits[2][1] < gauss_par[2]):
                return 1e9
    diff = y - gaussian(x, gauss_par[0], gauss_par[1], gauss_par[2])
    return np.sum((diff/yerr)**2)

def gauss_resid(gauss_par, x, y, yerr):
    """Return residual between y(x) and Gaussian fit."""

    diff = y - gaussian(x, gauss_par[0], gauss_par[1], gauss_par[2])
    return diff/yerr

def gaussian(x, mean, sigma, amp):
    """Gaussian function."""
    
    fac = amp/math.sqrt(2*math.pi)/sigma
    gauss = np.exp(-0.5*((x - mean)/sigma)**2)
    return fac*gauss


def plot_label(name):
    """Return apparent and absolute plot labels for given quantity name."""
    plot_label_dict = {
        'r_petro': (r'$r_{\rm Petro}/{\rm mag}$',
                    r'$^{0.1}M_{r_{\rm Petro}} -\ 5 \lg h$'),
        'petromagcor_r': (r'$r_{\rm Petro}/{\rm mag}$',
                          r'$^{0.1}M_{r_{\rm Petro}} -\ 5 \lg h$'),
        'SDSS_R_OBS_APP': (r'$r/{\rm mag}$',
                           r'$^{0.1}M_r -\ 5 \lg h$'),
        'r_kron': (r'$r_{\rm Kron}/{\rm mag}$',
                   r'$M_{r_{\rm Kron}} - 5 \lg h$'),
        'r_sersic': (r'$r_{\rm Sersic}/{\rm mag}$',
                     r'$^{0.1}M_{r_{\rm Sersic}} -\ 5 \lg h$'),
        'sb_50_r': (r'$\mu_{50,{\rm Petro}}/{\rm mag\ arcsec}^{-2}$',
                    r'$\mu_{50,{\rm Petro}}/{\rm mag\ arcsec}^{-2}$'),
        'r_sb': (r'$\mu_{50,{\rm Petro}}/{\rm mag\ arcsec}^{-2}$',
                 r'$\mu_{50,{\rm Petro}}/{\rm mag\ arcsec}^{-2}$'),
        'gal_mag_10re_r': (r'$r_{\rm Sersic}/{\rm mag}$',
                           r'$M_{r_{\rm Sersic}} -\ 5 \lg h$'),
        'm01_galmag10re_01_r': (r'$r_{\rm Sersic}/{\rm mag}$',
                                r'$M_{r_{\rm Sersic}} -\ 5 \lg h$'),
        'gal_mu_e_avg_r': (r'${\mu_e}_{\rm Sersic}/{\rm mag\ arcsec}^{-2}$',
                           r'${\mu_e}_{\rm Sersic}/{\rm mag\ arcsec}^{-2}$'),
        'm01_galmueavg_01_r': (r'${\mu_e}_{\rm Sersic}/{\rm mag\ arcsec}^{-2}$',
                               r'${\mu_e}_{\rm Sersic}/{\rm mag\ arcsec}^{-2}$'),
        'gal_re_c_r': (r'$\log_{10}(r_e/{\rm arcsec})$',
                       r'$\log_{10}(R_e/ h^{-1} {\rm kpc})$'),
        'petror50_r': (r'$\log_{10}(r_{50}/{\rm arcsec})$',
                       r'$\log_{10}(R_{50}/ h^{-1} {\rm kpc})$'),
        'mass': (r'$\log_{10}(M/M_\odot)$', r'$\log_{10}(M/M_\odot)$'),
        'logmstar': (r'$\log_{10}(M/M_\odot)$', r'$\log_{10}(M/M_\odot)$'),
        'logmstar_fluxscale': (r'$\log_{10}(M/M_\odot)$', r'$\log_{10}(M/M_\odot)$'),
        'fibermag_r': (r'$r_{\rm fibre}$', r'${M_r}_{\rm fibre}$')
        }
    try:
        return plot_label_dict[name]
    except:
        return name

def chol_fit(M, mu, phi, phi_err, ngal, ngmin=5):
    """Least-squares Choloniewski fn fit to phi(M, mu)."""

    prob = 0.32
    sel = ngal >= ngmin
    nbin = len(M[sel])
    nu = nbin - 6
    dchisq = scipy.special.chdtri(nu, prob)
    print nu, dchisq

    p0 = (-1.2, -20.5, 0.01, 0.3, 20.0, 0.5)

    res = scipy.optimize.leastsq(chol_resid, p0, (M[sel], mu[sel],
                                                  phi[sel], phi_err[sel]),
                                 xtol=0.001, ftol=0.001, full_output=1)
    popt, cov, info, mesg, ier = res
    print mesg
    chi2 = (info['fvec']**2).sum()
    cov *= (chi2/nu)

    for i in xrange(6):
        print '{} = {:7.3f} +- {:7.3f}'.format(chol_par_name[i],
                                               popt[i], math.sqrt(cov[i,i]))
    print 'chi2, nu: ', chi2, nu
##     print 'cov ', cov
    return popt, chi2

def chol_resid(chol_par, M, mu, phi, phi_err):
    """Return residual between phi(M,mu) estimate and Choloniewski fit."""
    # Trap for sigma <= 0
    if chol_par[-1] <= 0:
        return 1e8*np.ones(len(M))
    diff = phi - choloniewski(M, mu, chol_par)
    return diff/phi_err

def choloniewski(M, mu, chol_par):
    """Choloniewski function."""
    
    alpha, Mstar, phistar, beta, mustar, sigma = chol_par
    fac = 0.4*math.log(10)/math.sqrt(2*math.pi)/sigma*phistar
    lum = 10**(0.4*(Mstar - M))
    gauss = np.exp(-0.5*((mu - mustar - beta*(M - Mstar))/sigma)**2)
    chol = fac*lum**(alpha + 1)*np.exp(-lum)*gauss
    return chol

def test_zlim(m=18, z=0.1, mlim=19.8, zmin=0.002, zmax=0.5, kcoeff=(0.0,)):
    """Investigate zlim dependence on luminosity evolution parameter Q."""
    global par, cosmo
    par['z0'] = 0
    cosmo = util.CosmoLookup(100, 0.7, (zmin, zmax))
    Ql = []
    zlim = []
    for Q in np.linspace(0.0, 2.0):
        Ql.append(Q)
        M = m - cosmo.dist_mod(z) - kcorr(z, kcoeff) + ecorr(z, Q)
        zlim.append(zdm(mlim - M, kcoeff, (zmin, zmax), Q))
    plt.clf()
    plt.plot(Ql, zlim)
    plt.xlabel('Q')
    plt.ylabel('zlim')
    plt.draw()


def Nz_plot_lum(inroot='ranz_lum_c_{}_{}.dat', plot_file=None,
                plot_size=(5, 10)):
    """N(z) plots for lum-selected samples."""
    Mlimits = (-23, -22, -21, -20, -19, -18, -17, -16, -15)
    param_list = []
    for i in range(len(Mlimits)-1):
        Mlo = Mlimits[i]
        Mhi = Mlimits[i+1]
        infile = inroot.format(Mlo, Mhi)
        param_list.append((r'$M_r = [{}, {})$'.format(Mlo, Mhi),
                           ((infile, 'k'),)))
    Nz_plot(param_list, plot_file='Nz_lum.pdf', plot_size=plot_size)
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
        print 'plot saved to ', plot_dir + plot_file

def Nz_plot_lumV(inroot='ranz_lumV_{}_{}.dat'):
    """N(z) plots for vol-ltd lum-selected samples."""
    Mlimits = (-23, -22, -21, -20, -19, -18, -17, -16, -15)
    param_list = []
    for i in range(len(Mlimits)-1):
        Mlo = Mlimits[i]
        Mhi = Mlimits[i+1]
        infile = inroot.format(Mlo, Mhi)
        param_list.append((r'$M_r = [{}, {})$'.format(Mlo, Mhi), 
                           ((infile, 'k'),)))
    Nz_plot(param_list, plot_file='Nz_lumV.pdf', plot_size=(5, 10))

def Nz_plot_mass(inroot='ranz_mass_c_{}_{}.dat'):
    """N(z) plots for mass-selected samples."""
    Mlimits = mass_limits
    param_list = []
    for i in range(len(Mlimits)-1):
        Mlo = Mlimits[i]
        Mhi = Mlimits[i+1]
        infile = inroot.format(Mlo, Mhi)
        param_list.append((r'$\log(M/M_\odot) = [{}, {})$'.format(Mlo, Mhi), 
                           ((infile, 'k'),)))
    Nz_plot(param_list, plot_file='Nz_mass.pdf', plot_size=(5, 10))


def Nz_plot(param_list=(('All', (('ranz_colour_c.dat', 'k'),)),
                        ('Blue', (('ranz_colour_b.dat', 'k'),)),
                        ('Red', (('ranz_colour_r.dat', 'k'),))),
            plot_file='Nz.pdf', plot_size=(5, 8)):
    """Plot N(z) for galaxies and randoms generated using ran_gen.
    New version reads galaxy histogram from ranz files."""

    nplot = len(param_list)
    plt.clf()
    fig, axes = plt.subplots(nplot, sharex=True, num=1)
    ip = 0
    for param in param_list:
        label = param[0]
        ax = axes[ip]
        ax.text(0.95, 0.8, label, ha='right', transform=ax.transAxes)

        first = True
        for ranc in param[1]:
            f = open(ranc[0], 'r')
            info = eval(f.readline())
            f.close()
            data = np.loadtxt(ranc[0], skiprows=1)
            ranhist, zbins = np.histogram(data[:, 0], info['zbins'])
            ranhist_norm = ranhist * float(info['ngal'])/info['nran']
            if first:
                ax.step(info['zcen'], info['galhist'], where='mid')
                first = False
            ax.plot(info['zcen'], ranhist_norm, ranc[1])
        PQlab = r'$P = {:4.1f}, Q = {:4.1f}$'.format(info['P'], info['Q'])
        # ax.text(0.95, 0.6, PQlab, ha='right', transform = ax.transAxes)
        # Avoid overlapping tick labels
        if ip > 0:
            nbins = len(ax.get_xticklabels())
            ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))
        if ip == nplot/2:
            ax.set_ylabel('Frequency')
        ip += 1

    fig.subplots_adjust(hspace=0)
    ax.set_xlabel('Redshift')
    plt.draw()

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
        print 'plot saved to ', plot_dir + plot_file


def Nz_plot_schec(ranfile='ranz_colour_c.dat', schecfile='schec_nz.txt',
                  plot_file='Nz_schec.pdf', plot_size=(5, 3)):
    """Plot N(z) for galaxies and randoms generated using both ran_gen
    and from evolving Schechter function (via simcat)."""

    plt.clf()
    f = open(ranfile, 'r')
    info = eval(f.readline())
    f.close()
    data = np.loadtxt(ranfile, skiprows=1)
    ranhist, zbins = np.histogram(data[:, 0], info['zbins'])
    ranhist = ranhist * float(info['ngal'])/info['nran']
    plt.step(info['zcen'], info['galhist'], where='mid')
    plt.plot(info['zcen'], ranhist)
    data = np.loadtxt(schecfile, skiprows=1)
    plt.plot(data[:, 0], data[:, 1])
    plt.xlabel('Redshift')
    plt.ylabel('Frequency')

    plt.draw()

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
        print 'plot saved to ', plot_dir + plot_file

def Nz_plot_mock(inroot='ranz_{}_{}_{}.dat'):
    """N(z) plots for mock lum sub-samples."""
    Mlimits = (-23, -22, -21, -20, -19, -18, -17, -16)
    nrow = len(Mlimits) - 1
    ncol = 9
    plt.clf()
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey='row', num=1)
    fig.text(0.5, 0.04, 'Redshift', ha='center', va='center')
    fig.text(0.06, 0.5, 'Frequency', ha='center', va='center', 
             rotation='vertical')
    for i in range(len(Mlimits)-1):
        Mlo = Mlimits[i]
        Mhi = Mlimits[i+1]
        for ivol in range(1,10):
            infile = inroot.format(ivol, Mlo, Mhi)
            f = open(infile, 'r')
            info = eval(f.readline())
            f.close()
            data = np.loadtxt(infile, skiprows=1)
            ranhist, zbins = np.histogram(data[:,0], info['zbins'])
            ranhist *= float(info['ngal'])/info['nran']
            ax = axes[i, ivol-1]
            ax.step(info['zcen'], info['galhist'], where='mid')
            ax.plot(info['zcen'], ranhist)
#            ax.text(0.95, 0.8, infile, ha='right', transform=ax.transAxes)
    fig.subplots_adjust(hspace=0)
    plt.draw()


def weight_hist(infile='kcorrz01.fits', outfile='weights.fits', nbins=50,
                plot_file='weight.pdf'):
    """Plot weight histogram."""

    hdulist = fits.open(infile)
    tbdata = hdulist[1].data

    # Main targets with reliable redshifts
    sel = ((tbdata.field('survey_class') > 3) *
           (tbdata.field('nq') > 2) * (tbdata.field('z') > 0.002) *
           (tbdata.field('r_petro') < 19.8))
    tbdata = tbdata[sel]
    cataid = tbdata.field('cataid')
    sb = tbdata.field('r_sb')
    z = tbdata.field('z')
    imcomp = np.interp(sb, sb_tab, comp_tab)
    r_fibre = tbdata.field('fibermag_r')
    zcomp = z_comp(r_fibre)
    weight = 1.0/(imcomp*zcomp)
    logweight = np.log10(1.0/(imcomp*zcomp))
    hdulist.close()

    print 'mean weight (pre-clipping)= ', np.mean(weight)
    wgt = weight > wmax
    print '{} out of {} galaxies ({}) have weight > {}'.format(
        len(weight[wgt]), len(weight), float(len(weight[wgt]))/len(weight), wmax)
    logwmax = math.log10(wmax)
    plt.clf()
#    plt.subplot(131)
#    plt.scatter(sb, logweight, 0.1)
#    plt.xlabel('r_sb')
#    plt.ylabel(r'log$_{10}$ weight')
#    plt.subplot(132)
#    plt.scatter(r_fibre, logweight, 0.1)
#    plt.xlabel('r_fibre')
    plt.subplot(111)
    plt.scatter(z, logweight, 0.1)
    plt.xlabel('z')
    plt.xlim(0, 0.7)
#    plt.subplots_adjust(wspace=0)
    plt.show()

    plt.clf()
#    plt.hist(weight, nbins, (1, 5), histtype='step')
    plt.hist(logweight, nbins, histtype='step')
    plt.semilogy(basey=10, nonposy='clip')
    plt.plot((logwmax, logwmax), (1, 1e6))
    plt.xlabel(r'log$_{10}$ weight')
    plt.ylabel('Frequency')
    plt.draw()

    hiwt = weight > wmax
    f = open('high_weight.txt', 'w')
    for i in xrange(len(weight[hiwt])):
        print >> f, cataid[hiwt][i]
    f.close()

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(5, 3)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
        print 'plot saved to ', plot_dir + plot_file

    print 'mean weight (post-clipping)= ', np.mean(np.clip(weight, 1, wmax))

    # Output weights to a file
    if outfile:
        tbhdu = fits.BinTableHDU.from_columns(
            [fits.Column(name='CATAID', format='J', array=cataid),
             fits.Column(name='WEIGHT', format='E', array=weight),
             fits.Column(name='IMCOMP', format='E', array=imcomp),
             fits.Column(name='ZCOMP', format='E', array=zcomp),
            ])
        tbhdu.writeto(outfile, clobber=True)


# Routines for reading comparison data

def Blanton2005():
    """Blanton et al 2005 low-z LF."""
    global par
    # Correction for bandpass shift
    delta_M = 2.5*math.log10(1 + par['z0'])
    iband = 2
    fname = (os.environ['HOME'] + 
             '/Documents/Research/LFdata/blanton2005/r.dat')
    data = np.loadtxt(fname)
    M = data[:,0] + delta_M
    phi = data[:,1]
    phi_err = data[:,2]
    return M, phi, phi_err, 'Blanton+ 2005'

def Baldry2011():
    """Baldry et al 2011 SMF."""
    global par
    fname = (os.environ['HOME'] + 
             '/Documents/Research/LFdata/Baldry2011/table1.txt')
    data = np.loadtxt(fname)
    M = data[:,0] - 2*math.log10(par['H0']/70.0)
    phi = data[:,2] * (par['H0']/70.0)**3 / 1e3
    phi_err = data[:,3] * (par['H0']/70.0)**3 / 1e3
    return M, phi, phi_err, 'GAMA-I'


def info(infile):
    """Print parameter info for specified file."""
    print pickle.load(open(infile, 'r'))['par']


def tabulate(files, labels, outfile):
    """Tabulate binned LF results into an ascii file."""
    f = open(outfile, 'w')
    for infile, label in zip(files, labels):
        dat = pickle.load(open(infile, 'r'))
        print >> f, label
        print >> f, 'ngal    M     phi    Err'
        cmp = dat['comp']
        for ibin in range(len(cmp)):
            ng = int(dat['Mhist'][ibin])
            if ng > 0:
                line = '{:4d} {:6.2f} {:4.2e} {:4.2e}'.format(
                    ng, dat['Mbin'][ibin], dat['phi'][ibin],
                    dat['phi_err'][ibin])
                print >> f, line
        print >> f
    f.close()


def tabulate_smfs(files=('smf_fs_z_0.002_0.06.dat',
                         'smf_fs_nowt_z_0.002_0.06.dat'),
                  labels=('Weighted', 'Unweighted'), outfile='Fig11.txt'):
    """Tabulate binned SMF results into an ascii file."""
    tabulate(files, labels, outfile)


def tabulate_lfs(temp_list=('lf_r_petro_{}.dat', 'lf_r_sersic_{}.dat'),
                 outfile='Fig08.txt'):
    """Tabulate binned LF results into an ascii file."""
    files = []
    for colour in 'cbr':
        for filetemp in temp_list:
            files.append(filetemp.format(colour))
    tabulate(files, files, outfile)


def tabulate_lfs_z_petro(ztemp='lf_r_{}_{}_z_{}_{}.dat', mag_type='petro',
                         colour='c', outfile='lf_z_{}.txt'):
    """Tabulate LF/SMF in redshift bins."""

    zlims = (0.002, 0.1, 0.2, 0.3, 0.65)
    labeltemp = r'${} < z < {}$'
    files = []
    labels = []
    for iz in xrange(len(zlims)-1):
        zlo = zlims[iz]
        zhi = zlims[iz+1]
        files.append(ztemp.format(mag_type, colour, zlo, zhi))
        labels.append(labeltemp.format(zlo, zhi))
    tabulate(files, files, outfile.format(mag_type))


def tabulate_lfs_z_sersic(ztemp='lf_r_{}_z_{}_{}.dat', mag_type='sersic',
                          colour='c', outfile='lf_z_{}.txt'):
    """Tabulate LF/SMF in redshift bins."""

    zlims = (0.002, 0.1, 0.2, 0.3, 0.65)
    labeltemp = r'${} < z < {}$'
    files = []
    labels = []
    for iz in xrange(len(zlims)-1):
        zlo = zlims[iz]
        zhi = zlims[iz+1]
        files.append(ztemp.format(mag_type, zlo, zhi))
        labels.append(labeltemp.format(zlo, zhi))
    tabulate(files, files, outfile.format(mag_type))
