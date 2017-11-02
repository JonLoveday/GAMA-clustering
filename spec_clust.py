#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-correlation of GAMA specLine-selected samples with all galaxies
Created on Wed Aug 16 16:03:08 2017

@author: loveday
"""

import matplotlib.pyplot as plt
import os
import pdb
import subprocess

import clust_util as cu
import gal_sample as gs

gama_data = os.environ['GAMA_DATA']
mass_bins = (12, 13, 13.5, 14, 16)
xi_int = '$BIN/xi '
xi_bat = 'qsub /research/astro/gama/loveday/Documents/Research/python/apollo_job.sh $BIN/xi '


def bpt_gen(zlimits=(0.002, 0.35), nfac=10):
    """Cross-correlation of BPT samples."""

    # Galaxy sample
    galxs = gs.GalSample(zlimits=zlimits)
    galxs.read_gama()
    galxs.vis_calc()
    galxs.add_vmax()
    cu.xi_sel(galxs, 'gal.dat', 'gal_ran.dat', '',
              nfac, set_vmax=False,
              mask=gama_data+'/mask/zcomp.ply', run=0)

    # BPT samples
    galxs.specline_props()
    for bpt in 'acsu':
        sel_dict = {'bpt_type': (bpt, chr(ord(bpt)+1))}
        cu.xi_sel(galxs, 'gal{}.dat'.format(bpt), 'gal{}_ran.dat'.format(bpt),
                  '', nfac, sel_dict=sel_dict, set_vmax=False,
                  mask=gama_data+'/mask/zcomp.ply', run=0)


def bpt_run(xi_cmd=xi_bat, nbin=4):
    """Cross-correlation pair counts of BPT samples."""

    cmd = xi_cmd + 'gal.dat gg_gal_gal.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'gal.dat gal_ran.dat gr_gal_gal.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'gal_ran.dat rr_gal_gal.dat'
    subprocess.call(cmd, shell=True)

    for bpt in 'acsu':
        cmd = xi_cmd + 'grp{0}.dat gg_grp{0}_grp{0}.dat'.format(bpt)
        subprocess.call(cmd, shell=True)
        cmd = xi_cmd + 'grp{0}.dat grp{0}_ran.dat gr_grp{0}_grp{0}.dat'.format(bpt)
        subprocess.call(cmd, shell=True)
        cmd = xi_cmd + 'grp{0}_ran.dat rr_grp{0}_grp{0}.dat'.format(bpt)
        subprocess.call(cmd, shell=True)
        cmd = xi_cmd + 'gal_ran.dat grp{0}_ran.dat rr_gal_grp{0}.dat'.format(bpt)
        subprocess.call(cmd, shell=True)
    
        cmd = xi_cmd + 'gal.dat grp{0}.dat gg_gal_grp{0}.dat'.format(bpt)
        subprocess.call(cmd, shell=True)
        cmd = xi_cmd + 'gal.dat grp{0}_ran.dat gr_gal_grp{0}.dat'.format(bpt)
        subprocess.call(cmd, shell=True)
        cmd = xi_cmd + 'grp{0}.dat gal_ran.dat gr_grp{0}_gal.dat'.format(bpt)
        subprocess.call(cmd, shell=True)


def plots(key='w_p', binning=1, pi_lim=100, rp_lim=100):
    """Plot the correlations."""
    gg = cu.PairCounts('gg_gal_gal.dat')
    Gg = cu.PairCounts('gg_gal_grp.dat')
    GG = cu.PairCounts('gg_grp_grp.dat')

    gr = cu.PairCounts('gr_gal_gal.dat')
    gR = cu.PairCounts('gr_gal_grp.dat')
    Gr = cu.PairCounts('gr_grp_gal.dat')
    GR = cu.PairCounts('gr_grp_grp.dat')

    rr = cu.PairCounts('rr_gal_gal.dat')
    Rr = cu.PairCounts('rr_gal_grp.dat')
    RR = cu.PairCounts('rr_grp_grp.dat')

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
    xi0.plot(ax, label='dpx')
    xi1.plot(ax, label='lsx')
    xi2.plot(ax, label='lsx2r')
    ax.loglog(basex=10, basey=10, nonposy='clip')
    plt.legend()
    plt.xlabel(r'$r_\perp$')
    plt.ylabel(r'$w_p(r_\perp)$')
    plt.draw()
