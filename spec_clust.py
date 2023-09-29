#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-correlation of GAMA specLine-selected samples with all galaxies
Created on Wed Aug 16 16:03:08 2017

@author: loveday
"""

from astropy.table import Table, join
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import subprocess

import clust_util as cu
import gal_sample as gs
import util

gama_data = os.environ['GAMA_DATA']
gfs_file = gama_data + 'SpecLineSFR/GaussFitSimplev05.fits'
gfc_file = gama_data + 'SpecLineSFR/GaussFitComplexv05.fits'
xi_int = '$BIN/xi '
xi_bat = 'qsub /research/astro/gama/loveday/Research/python/apollo_job.sh $BIN/xi '


def gordon_agn_class(zlimits=[0, 0.3]):
    """Gordon+2017 AGN classifications."""
    
    filename = gama_data + 'SpecLineSFR/GAMA_SpecClass_Gordon2019_v2.fits'
    t = Table.read(filename)
    sel = (((t['SURVEY'] == 'SDSS') + (t['SURVEY'] == 'GAMA')) *
           (t['BAD_FLAG'] < 1) * (t['Z'] < 0.3) * (t['NQ'] > 2))
    t = t[sel]
    print(np.unique(t['EmLineType'], return_counts=True))

    bins = np.linspace(0.0, 0.3, 31)
    zvals = bins[:-1] + 0.5*np.diff(bins)
    hist_all, edges = np.histogram(t['Z'], bins=bins)
    plt.clf()
    for typ in ['BLAGN', 'Seyfert']:
        histt, edges = np.histogram(t['Z'][t['EmLineType'] == typ], bins=bins)
        plt.plot(zvals, histt/hist_all, label=typ)
    plt.xlabel('Redshift')
    plt.ylabel('AGN fraction')
    plt.legend()
    plt.show()

    
def agn_class(zlimits=[0, 0.3]):
    """Classify AGN a la Gordon+2017 sec 3.1."""

    galxs = gs.GalSample(zlimits=zlimits)
    galxs.read_gama()
    galxs.agn_class()


def cone_plots(Mlim=None, zlimits=[0, 0.267], gal_size=0.1, agn_size=10,
               maxgal=20000, plot_size=(9, 6), plot_file=None):
    """Cone plots of all galxs and AGN."""

    galxs = gs.GalSample(zlimits=zlimits)
    galxs.read_gama()
    galxs.specline_props()
    if Mlim:
        galxs.vol_limit(Mlim)
    t = galxs.t
    # idxs = np.arange(len(t))
    # ngal = min(maxgal, len(t))
    # idr = np.random.choice(idxs, size=ngal, replace=False)
    # gal_dict = {'ra': t['RA'][idr], 'z': t['z'][idr], 's': gal_size, 'c': 'k',
    #             'marker': ',', 'alpha': 0.5, 'vmax': None}
    gal_dict = {'ra': t['RA'], 'z': t['z'], 's': gal_size, 'c': 'k',
                'marker': '.', 'alpha': 0.5, 'vmax': None}

    agn = t['bpt_type'] == 'a'
    agn_dict = {'ra': t['RA'][agn], 'z': t['z'][agn], 's': agn_size, 'c': 'r',
                'marker': 'o', 'alpha': 1.0, 'vmax': None}
    util.cone_plot([gal_dict, agn_dict], z_limits=zlimits,
                   plot_size=plot_size, plot_file=plot_file)


def bpt_gen(zlimits=(0.002, 0.35), nfac=10):
    """Cross-correlation of BPT samples."""

    # Galaxy sample
    galxs = gs.GalSample(zlimits=zlimits)
    galxs.read_gama()
    galxs.vis_calc((galxs.sel_mag_lo, galxs.sel_mag_hi))
    galxs.add_vmax()
    cu.xi_sel(galxs, 'gal.dat', 'gal_ran.dat', '',
              nfac, set_vmax=False,
              mask=gama_data+'/mask/zcomp.ply', run=0, J3wt=False)

    # BPT samples
    galxs.specline_props()
    for bpt in 'acsu':
        sel_dict = {'bpt_type': (bpt, chr(ord(bpt)+1))}
        cu.xi_sel(galxs, 'gal{}.dat'.format(bpt), 'gal{}_ran.dat'.format(bpt),
                  '', nfac, sel_dict=sel_dict, set_vmax=False,
                  mask=gama_data+'/mask/zcomp.ply', run=0, J3wt=False)


def bpt_run(xi_cmd=xi_bat, nbin=4):
    """Cross-correlation pair counts of BPT samples."""

    cmd = xi_cmd + 'gal.dat gg_gal_gal.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'gal.dat gal_ran.dat gr_gal_gal.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'gal_ran.dat rr_gal_gal.dat'
    subprocess.call(cmd, shell=True)

    for bpt in 'acsu':
        cmd = xi_cmd + 'gal{0}.dat gg_gal{0}_gal{0}.dat'.format(bpt)
        subprocess.call(cmd, shell=True)
        cmd = xi_cmd + 'gal{0}.dat gal{0}_ran.dat gr_gal{0}_gal{0}.dat'.format(bpt)
        subprocess.call(cmd, shell=True)
        cmd = xi_cmd + 'gal{0}_ran.dat rr_gal{0}_gal{0}.dat'.format(bpt)
        subprocess.call(cmd, shell=True)
        cmd = xi_cmd + 'gal_ran.dat gal{0}_ran.dat rr_gal_gal{0}.dat'.format(bpt)
        subprocess.call(cmd, shell=True)
    
        cmd = xi_cmd + 'gal.dat gal{0}.dat gg_gal_gal{0}.dat'.format(bpt)
        subprocess.call(cmd, shell=True)
        cmd = xi_cmd + 'gal.dat gal{0}_ran.dat gr_gal_gal{0}.dat'.format(bpt)
        subprocess.call(cmd, shell=True)
        cmd = xi_cmd + 'gal{0}.dat gal_ran.dat gr_gal{0}_gal.dat'.format(bpt)
        subprocess.call(cmd, shell=True)


def plot_by_est(key='w_p', binning=1, pi_lim=100, rp_lim=100):
    """Plot the correlations for each estimator."""
    for bpt in 'acsu':
        gg = cu.PairCounts(f'gg_gal_gal.dat')
        Gg = cu.PairCounts(f'gg_gal_gal{bpt}.dat')
        GG = cu.PairCounts(f'gg_gal{bpt}_gal{bpt}.dat')

        gr = cu.PairCounts(f'gr_gal_gal.dat')
        gR = cu.PairCounts(f'gr_gal_gal{bpt}.dat')
        Gr = cu.PairCounts(f'gr_gal{bpt}_gal.dat')
        GR = cu.PairCounts(f'gr_gal{bpt}_gal{bpt}.dat')

        rr = cu.PairCounts(f'rr_gal_gal.dat')
        Rr = cu.PairCounts(f'rr_gal_gal{bpt}.dat')
        RR = cu.PairCounts(f'rr_gal{bpt}_gal{bpt}.dat')

        counts = {'gg': gg, 'Gg': Gg, 'GG': GG,
                  'gr': gr, 'gR': gR, 'Gr': Gr, 'GR': GR,
                  'rr': rr, 'Rr': Rr, 'RR': RR}
        xi = cu.Xi()
        xi0 = xi.est(counts, cu.dpx, key=key, binning=binning,
                     pi_lim=pi_lim, rp_lim=rp_lim)
        xi1 = xi.est(counts, cu.lsx, key=key, binning=binning,
                     pi_lim=pi_lim, rp_lim=rp_lim)
        xi2 = xi.est(counts, cu.lsx2r, key=key, binning=binning,
                     pi_lim=pi_lim, rp_lim=rp_lim)

        plt.clf()
        ax = plt.subplot(111)
        xi0.plot(ax, label=f'{bpt} dpx')
        xi1.plot(ax, label=f'{bpt} lsx')
        xi2.plot(ax, label=f'{bpt} lsx2r')
        ax.loglog(base=10, nonpositive='clip')
        plt.legend()
        plt.xlabel(r'$r_\perp$')
        plt.ylabel(r'$w_p(r_\perp)$')
        plt.show()


def plot_by_type(key='w_p', binning=1, pi_lim=100, rp_lim=100):
    """Plot the correlations."""

    plt.clf()
    ax = plt.subplot(111)

    gg = cu.PairCounts(f'gg_gal_gal.dat')
    gr = cu.PairCounts(f'gr_gal_gal.dat')
    rr = cu.PairCounts(f'rr_gal_gal.dat')
    counts = {'gg': gg, 'gr': gr, 'rr': rr}
    xi = cu.Xi()
    xi0 = xi.est(counts, cu.ls, key=key, binning=binning,
                 pi_lim=pi_lim, rp_lim=rp_lim)
    xi0.plot(ax, label='all')

    for bpt in 'acsu':
        Gg = cu.PairCounts(f'gg_gal_gal{bpt}.dat')
        GG = cu.PairCounts(f'gg_gal{bpt}_gal{bpt}.dat')

        gR = cu.PairCounts(f'gr_gal_gal{bpt}.dat')
        Gr = cu.PairCounts(f'gr_gal{bpt}_gal.dat')
        GR = cu.PairCounts(f'gr_gal{bpt}_gal{bpt}.dat')

        Rr = cu.PairCounts(f'rr_gal_gal{bpt}.dat')
        RR = cu.PairCounts(f'rr_gal{bpt}_gal{bpt}.dat')

        counts = {'gg': gg, 'Gg': Gg, 'GG': GG,
                  'gr': gr, 'gR': gR, 'Gr': Gr, 'GR': GR,
                  'rr': rr, 'Rr': Rr, 'RR': RR}
        xi = cu.Xi()
        xi1 = xi.est(counts, cu.lsx, key=key, binning=binning,
                     pi_lim=pi_lim, rp_lim=rp_lim)

        xi1.plot(ax, label=bpt)
    ax.loglog(base=10, nonpositive='clip')
    plt.legend()
    plt.xlabel(r'$r_\perp$')
    plt.ylabel(r'$w_p(r_\perp)$')
    plt.show()
