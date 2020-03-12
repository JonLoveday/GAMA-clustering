#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-correlation of GAMA groups with galaxies
Created on Wed Aug 16 16:03:08 2017

@author: loveday
"""

import glob
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import pickle
import scipy.optimize
import subprocess

import astropy.coordinates
from astropy.table import Table, join
import Corrfunc
import illustris as il

import clust_util as cu
import gal_sample as gs
import util

gama_data = os.environ['GAMA_DATA']
g3cgal = gama_data + 'g3cv9/G3CGalv08.fits'
g3cfof = gama_data + 'g3cv9/G3CFoFGroupv09.fits'
g3cmockfof = gama_data + 'g3cv6/G3CMockFoFGroupv06.fits'
g3cmockhalo = gama_data + 'g3cv6/G3CMockHaloGroupv06.fits'
g3cmockgal = gama_data + 'g3cv6/G3CMockGalv06.fits'
mock_Q = 1.75
mock_P = 0.0
mock_mass_bins = (12.0, 13.1, 13.3, 13.5, 13.7, 14.0, 15.2)
#mock_halomass = (12.91, 13.20, 13.40, 13.59, 13.82, 14.23)
mass_bins = (12.0, 13.0, 13.3, 13.6, 14.6)
halomass = (12.82, 13.16, 13.44, 13.85)  # V approx bin centres
gal_mag_bins = (-23, -22, -21, -20, -19, -18, -17)
gal_mass_bins = (8.5, 9.5, 10, 10.5, 11, 11.5)

mock_halomass = halomass
#bad_nodes = "-l h='!node001&!node005&!node037&!node072&!node002&!node041'"
bad_nodes = ''  # should all be working now!
xi_int = '$BIN/xi '
xi_bat = 'qsub ' + bad_nodes + ' /research/astro/gama/loveday/Documents/Research/python/apollo_job.sh $BIN/xi '
plot_dir = '/Users/loveday/Documents/tex/papers/gama/ridwan/'
metadata_conflicts = 'silent'  # Alternatives are 'warn', 'error'


def mass_z(grpfile=g3cfof, nmin=5, edge_min=0.9, vmax=100, zrange=(0, 0.267),
           Mrange=(11.8, 14.8), plot_file='mass_z.pdf', plot_size=(5, 4.5)):
    """Halo mass-redshift plot."""

    # Read and select groups meeting selection criteria
    groups = gs.GalSample(zlimits=zrange)
    groups.read_groups()
    t = groups.t
    print('mass range of selected groups: {:5.2f} - {:5.2f}'.format(
            np.min(t['log_mass']), np.max(t['log_mass'])))
    print('mass quartiles: ', np.percentile(t['log_mass'], (25, 50, 75)))

    plt.clf()
    plt.scatter(t['z'], t['log_mass'], s=2, c=t['Nfof'], vmax=vmax,
                norm=mpl.colors.LogNorm())
    plt.xlim(zrange)
    for m in mass_bins:
        plt.plot(zrange, (m, m), 'k-', linewidth=0.5)
    plt.ylim(Mrange)
#    for z in (0.1, 0.2, 0.3):
#        plt.plot((z, z), Mrange, 'k-', linewidth=0.5)
    plt.xlabel('Redshift')
    plt.ylabel(r'$\log_{10}({\cal M}_h/{\cal M}_\odot h^{-1})$')
    cb = plt.colorbar()
    cb.set_label(r'$N_{\rm FoF}$')
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


def mass_comp_mock(infile=g3cmockhalo, nmin=5, edge_min=0.9, vmax=None,
                   limits=(11.8, 15, 11.8, 15), mbins=np.linspace(12, 15, 7),
                   plot_file='mass_comp_mock.pdf', plot_size=(6, 5)):
    """Compare luminosity-based and true halo mass estimates for mock groups."""

    # Read and select groups meeting selection criteria
    thalo = Table.read(g3cmockhalo)
    thalo = thalo['HaloMass', 'IterCenRA', 'IterCenDEC']
    tfof = Table.read(g3cmockfof)
    t = join(thalo, tfof, keys=('IterCenRA', 'IterCenDEC'),
             metadata_conflicts=metadata_conflicts)

    t['log_mass_lum'] = 13.98 + 1.16*(np.log10(t['LumBfunc']) - 11.5)
    sel = (np.array(t['GroupEdge'] > edge_min) *
           np.logical_not(t['log_mass_lum'].mask) *
           np.array(t['Nfof'] >= nmin))
#           np.array(t['Volume'] == 1))
    t = t[sel]
    lgMh = np.log10(t['HaloMass'])

    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    plt.plot(limits[:2], limits[2:], 'k:')
    sc = plt.scatter(lgMh, t['log_mass_lum'], s=2, c=t['Nfof'], vmax=vmax,
                     norm=mpl.colors.LogNorm())

    nbin = len(mbins)-1
    mhav, mlav, mlstd = np.zeros(nbin), np.zeros(nbin), np.zeros(nbin)
    for ibin in range(nbin):
        mlo, mhi = mbins[ibin], mbins[ibin+1]
        sel = (lgMh >= mlo) * (lgMh < mhi)
        mhav[ibin] = np.mean(lgMh[sel])
        mlav[ibin] = np.mean(t['log_mass_lum'][sel])
        mlstd[ibin] = np.std(t['log_mass_lum'][sel])
    plt.errorbar(mhav, mlav, mlstd, fmt='ro')
    plt.ylabel(r'$\lg {\cal M}_{\rm lum}$')
    plt.xlabel(r'$\lg {\cal M}_{\rm halo}$')
#    ax.axis((11.8, 15, 11.8, 15))
    plt.axis(limits)
#    ax.set_aspect('equal')
#    fig.subplots_adjust(top=0.93)
#    cbar_ax = fig.add_axes([0.13, 0.97, 0.75, 0.02])
#    cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cb = plt.colorbar(sc)
    cb.set_label(r'$N_{\rm FoF}$')
#    cbar_ax.set_title('Redshift')
#    cbar_ax.set_title(r'$N_{\rm FoF}$')

#    cb = plt.colorbar()
    plt.draw()
#    fig = plt.gcf()
#    fig.set_size_inches(plot_size)
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()

    print('nmin    lgMh - lgMl')
    for nlo in range(2, 10):
        sel = t['Nfof'] >= nlo
        print(nlo, np.mean(lgMh[sel] - t['log_mass_lum'][sel]),
              np.std(lgMh[sel] - t['log_mass_lum'][sel]))


def groups_gen(zlimits=(0.002, 0.1), nfac=10):
    """Cross-correlation sample generation for all groups."""

    # Galaxy sample
    galxs = gs.GalSample(zlimits=zlimits)
    galxs.read_gama()
    galxs.vis_calc()
    galxs.vmax_calc()
    cu.xi_sel(galxs, 'gal.dat', 'gal_ran.dat', '',
              nfac, set_vmax=False,
              mask=gama_data+'/mask/zcomp.ply', run=0)

    # Galaxy groups
    groups = gs.GalSample(zlimits=zlimits)
    groups.cosmo = galxs.cosmo
    groups.area = galxs.area
    groups.read_groups()

    cu.xi_sel(groups, 'grp.dat', 'grp_ran.dat', '',
              nfac, set_vmax=False,
              mask=gama_data+'/mask/zcomp.ply', run=0)


def groups_mass(zlimits=(0.002, 0.5), mbins=mass_bins, nfac=10):
    """Cross-correlation sample generation for groups in mass bins."""

    # Galaxy sample
    galxs = gs.GalSample(zlimits=zlimits)
    galxs.read_gama()
    galxs.vis_calc()
    galxs.add_vmax()
    cu.xi_sel(galxs, 'gal.dat', 'gal_ran.dat', '',
              nfac, set_vmax=False,
              mask=gama_data+'/mask/zcomp.ply', run=0)

    # Galaxy groups
    groups = gs.GalSample(zlimits=zlimits)
    groups.cosmo = galxs.cosmo
    groups.area = galxs.area
    groups.read_groups()
    for i in range(len(mbins)-1):
        sel_dict = {'log_mass': (mbins[i], mbins[i+1])}
        cu.xi_sel(groups, 'grp{}.dat'.format(i), 'grp{}_ran.dat'.format(i), '',
                  nfac, sel_dict=sel_dict, set_vmax=False,
                  mask=gama_data+'/mask/zcomp.ply', run=0)


def mock_groups_mass(Mlim=-20.2, zlo=0.002, zhi=0.5, mbins=mass_bins, nfac=10):
    """Cross-correlation sample generation for mock groups in mass bins."""

    # Mock galaxies
    galxs = gs.GalSample(Q=mock_Q, P=mock_P, zlimits=(zlo, zhi))
    galxs.read_gama_mocks()
    print('Mlim =', Mlim)
    plt.clf()
    ax = plt.subplot(111)
    galxs.vol_limit(Mlim, ax=ax)
    plt.show()

    zhi = min(zhi, galxs.zlim)
    dmax = galxs.cosmo.dm(zhi)
    vol = 144*(math.pi/180)**2/3*dmax**3
    den = len(galxs.t)/vol
    print('zlimit, dmax, vol, ngal, density =',
          zhi, dmax, vol, len(galxs.t), den)
    galxs.info.update({'Mlim': Mlim, 'zlimits': (zlo, zhi)})
#    galxs.vis_calc((gs.sel_gama_mag_lo, gs.sel_gama_mag_hi))
#    galxs.vmax_calc(denfile=None)
    for ivol in range(1, 10):
        sel_dict = {'Volume': (ivol, ivol+1)}
        if ivol == 1:
            ranfile = 'mock_gal_ran.dat'
        else:
            ranfile = None
        cu.xi_sel(galxs, 'mock{}_gal.dat'.format(ivol), ranfile, '',
                  nfac, sel_dict=sel_dict, set_vmax=False,
                  mask=gama_data+'../gama1/mask/gama_rect.ply', run=0)

    # Mock groups
    for mass_est, grps in zip(('lum', 'true'), (g3cmockfof, g3cmockhalo)):
        groups = gs.GalSample(zlimits=(zlo, zhi))
        groups.cosmo = galxs.cosmo
        groups.area = galxs.area
        groups.read_groups(grps, mass_est=mass_est)
        groups.info.update({'Mlim': Mlim, 'zlimits': (zlo, zhi)})
#        groups.vis_calc((gs.sel_gama_mag_lo, gs.sel_gama_mag_hi))
#        groups.vmax_calc(denfile=None)
        if mass_est == 'true':
            grpfile = 'hmock{}_grp{}.dat'
        else:
            grpfile = 'fmock{}_grp{}.dat'
        ranfile = None
        for ivol in range(1, 10):
            for i in range(len(mbins)-1):
                sel_dict = {'Volume': (ivol, ivol+1),
                            'log_mass': (mbins[i], mbins[i+1])}
                cu.xi_sel(groups, grpfile.format(ivol, i), ranfile, '',
                          nfac, sel_dict=sel_dict, set_vmax=False,
                          mask=gama_data+'../gama1/mask/gama_rect.ply', run=0,
                          J3wt=False)


def groups_mass_vol(Mlim=-20, zlo=0.002, zhi=0.5, mbins=mass_bins, nfac=10,
                    omega_l=0.75):
    """Cross-correlation sample generation for groups in mass bins and
    volume-limited galaxy sample."""

    # Galaxy sample
    galxs = gs.GalSample(zlimits=(zlo, zhi), omega_l=omega_l)
    galxs.read_gama()
    plt.clf()
    ax = plt.subplot(111)
    galxs.vol_limit(Mlim, ax=ax)
    plt.show()

    zhi = min(zhi, galxs.zlim)
    dmax = galxs.cosmo.dm(zhi)
    vol = 180*(math.pi/180)**2/3*dmax**3
    den = len(galxs.t)/vol
    print('zlimit, dmax, vol, ngal, density =',
          zhi, dmax, vol, len(galxs.t), den)
    galxs.info.update({'Mlim': Mlim, 'zlimits': (zlo, zhi)})
    cu.xi_sel(galxs, 'gal.dat', 'gal_ran.dat', '',
              nfac, set_vmax=False,
              mask=gama_data+'/mask/zcomp.ply', run=0)

    # Galaxy groups
    groups = gs.GalSample(zlimits=(zlo, zhi))
    groups.cosmo = galxs.cosmo
    groups.area = galxs.area
    groups.read_groups()
    groups.info.update({'Mlim': Mlim, 'zlimits': (zlo, zhi)})
    for i in range(len(mbins)-1):
        sel_dict = {'log_mass': (mbins[i], mbins[i+1])}
        cu.xi_sel(groups, 'grp{}.dat'.format(i), None, '',
                  nfac, sel_dict=sel_dict, set_vmax=False,
                  mask=gama_data+'/mask/zcomp.ply', run=0)


def group_mass_tab(out_file='group_mass.tex'):
    """Tabulate group mass statistics."""

    def group_stats(infile):
        f = open(infile, 'r')
        info = eval(f.readline())
        args = f.readline().split()
        ngrp = int(args[0])
        return ngrp, info

    nmbin = len(mass_bins) - 1
    f = open(plot_dir + out_file, 'w')
    print(r'''
\begin{table}
\caption{Group bin names and log-mass limits, with number of groups,
mean log-mass, and mean redshift for GAMA-II data and mocks.
Note that each mock realisation has about 11 per cent larger volume than
the GAMA-II fields, due to a slightly higher redshift limit
($z < 0.300$ versus $z < 0.267$).
\label{tab:group_mass_def}}
\begin{tabular}{ccccccccc}
\hline
& & \multicolumn{3}{c}{GAMA} & & \multicolumn{3}{c}{Mocks} \\
\cline{3-5} \cline{7-9} \\[-2ex]
& $\lg {\cal M}_{h, {\rm limits}}$ & $N$ & $\overline{\lg {\cal M}}$ &
$\overline z$ & & $N$ & $\overline{\lg {\cal M}}$ &  $\overline z$ \\
\hline''', file=f)
    for i in range(nmbin):
        ngrp_gama, info_gama = group_stats(f'grp{i}.dat')
        nlist = []
        mlist = []
        zlist = []
        for ivol in range(1, 10):
            ngrp_mock, info_mock = group_stats(f'fmock{ivol}_grp{i}.dat')
            nlist.append(ngrp_mock)
            mlist.append(info_mock['mean_log_mass'])
            zlist.append(info_mock['mean_z'])
        ngrp_mock = np.mean(nlist)
#        print(r'${\cal M}' + str(i+1) + r'$ & ' +
#              '[{:4.1f}, {:4.1f}] & '.format(mass_bins[i], mass_bins[i+1]) +
#              str(ngrp_gama) + ' & ' + str(info_gama['mean_log_mass']) +
#              ' & ' + str(info_gama['mean_z']) + ' & & ' +
#              str(np.mean(nlist)) + ' & ' + str(np.mean(mlist)) +
#              ' & ' + str(np.mean(zlist)) + r'\\', file=f)
        print(rf"${{\cal M}}{i+1}$ & "
              rf"[{mass_bins[i]:4.1f}, {mass_bins[i+1]:4.1f}] & "
              rf"{ngrp_gama} & {info_gama['mean_log_mass']:5.2f} & "
              rf"{info_gama['mean_z']:4.2f} & & {int(np.mean(nlist))} & "
              rf"{np.mean(mlist):5.2f}  & {np.mean(zlist):4.2f} \\", file=f)
    print(r'''
\hline
\end{tabular}
\end{table}
''', file=f)
    f.close()


def group_mass_tab2(out_file='group_mass.tex'):
    """Tabulate group mass statistics, including both halo and FoF mocks."""

    def group_stats(infile):
        f = open(infile, 'r')
        info = eval(f.readline())
        args = f.readline().split()
        ngrp = int(args[0])
        return ngrp, info

    # These numbers from running illustris_halo()
    tng_n = [31826, 1788, 890, 738]
    tng_lgm = [12.32, 13.13, 13.44, 13.88]
    nmbin = len(mass_bins) - 1
    f = open(plot_dir + out_file, 'w')
    print(r'''
\begin{table*}
\caption{Group bin names and log-mass limits, number of groups,
mean log-mass, and mean redshift for GAMA-II data, intrinsic mock haloes,
FoF mock groups, and TNG300 simulation ($z=0.15$ snapshot).
Note that each mock realisation has about 11 per cent larger volume than
the GAMA-II fields, due to a slightly higher redshift limit ---
see Table~\ref{tab:gal_def}.
\label{tab:group_mass_def}}
\begin{tabular}{ccccccccccccccccc}
\hline
& & \multicolumn{3}{c}{GAMA} & & \multicolumn{3}{c}{Halo Mocks} & &
\multicolumn{3}{c}{FoF Mocks} & & \multicolumn{2}{c}{TNG300} \\
\cline{3-5} \cline{7-9} \cline{11-13} \cline{15-16} \\[-2ex]
& $\lg {\cal M}_{h, {\rm limits}}$ & $N$ & $\overline{\lg {\cal M}}$ &
$\overline z$ & &
$N$ & $\overline{\lg {\cal M}}$ &  $\overline z$ & &
$N$ & $\overline{\lg {\cal M}}$ &  $\overline z$ & &
$N$ & $\overline{\lg {\cal M}}$ \\
\hline''', file=f)
    for i in range(nmbin):
        ngrp_gama, info_gama = group_stats(f'grp{i}.dat')
        nmean, mmean, zmean = {}, {}, {}
        for mock_type in 'hf':
            nlist, mlist, zlist = [], [], []
            for ivol in range(1, 10):
                ngrp_mock, info_mock = group_stats(
                        f'{mock_type}mock{ivol}_grp{i}.dat')
                nlist.append(ngrp_mock)
                mlist.append(info_mock['mean_log_mass'])
                zlist.append(info_mock['mean_z'])
            nmean[mock_type] = int(np.mean(nlist))
            mmean[mock_type] = np.mean(mlist)
            zmean[mock_type] = np.mean(zlist)
        print(rf"${{\cal M}}{i+1}$ & "
              rf"[{mass_bins[i]:4.1f}, {mass_bins[i+1]:4.1f}] & "
              rf"{ngrp_gama} & {info_gama['mean_log_mass']:5.2f} & "
              rf"{info_gama['mean_z']:4.2f} & & "
              rf"{nmean['h']} & {mmean['h']:5.2f} & {zmean['h']:4.2f} & & "
              rf"{nmean['f']} & {mmean['f']:5.2f} & {zmean['f']:4.2f} & & "
              rf"{tng_n[i]} & {tng_lgm[i]} \\",
              file=f)
    print(r'''
\hline
\end{tabular}
\end{table*}
''', file=f)
    f.close()


def groups_run(xi_cmd=xi_bat):
    """Cross-correlation pair counts for all groups."""

    cmd = xi_cmd + 'gal.dat gg_gal_gal.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'gal.dat gal_ran.dat gr_gal_gal.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'gal_ran.dat rr_gal_gal.dat'
    subprocess.call(cmd, shell=True)

    cmd = xi_cmd + 'grp.dat gg_grp_grp.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'grp.dat grp_ran.dat gr_grp_grp.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'grp_ran.dat rr_grp_grp.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'gal_ran.dat grp_ran.dat rr_gal_grp.dat'
    subprocess.call(cmd, shell=True)

    cmd = xi_cmd + 'gal.dat grp.dat gg_gal_grp.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'gal.dat grp_ran.dat gr_gal_grp.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'grp.dat gal_ran.dat gr_grp_gal.dat'
    subprocess.call(cmd, shell=True)


def groups_bin_arun(xi_cmd=xi_bat, nbin=4):
    """Auto-correlation pair counts of binned groups."""

    for i in range(nbin):
        cmd = xi_cmd + 'grp{0}.dat gg_grp{0}.dat'.format(i)
        subprocess.call(cmd, shell=True)
        cmd = xi_cmd + 'grp{0}.dat grp{0}_ran.dat gr_grp{0}.dat'.format(i)
        subprocess.call(cmd, shell=True)
        cmd = xi_cmd + 'grp{0}_ran.dat rr_grp{0}.dat'.format(i)
        subprocess.call(cmd, shell=True)


def groups_bin_xrun(xi_cmd=xi_bat, nbin=len(mass_bins)-1):
    """Cross-correlation pair counts of binned groups."""

    cmd = xi_cmd + 'gal.dat gg_gal_gal.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'gal.dat gal_ran.dat gr_gal_gal.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'gal_ran.dat rr_gal_gal.dat'
    subprocess.call(cmd, shell=True)

    for i in range(nbin):
#        cmd = xi_cmd + 'grp{0}.dat gg_grp{0}_grp{0}.dat'.format(i)
#        subprocess.call(cmd, shell=True)
#        cmd = xi_cmd + 'grp{0}.dat grp{0}_ran.dat gr_grp{0}_grp{0}.dat'.format(i)
#        subprocess.call(cmd, shell=True)
#        cmd = xi_cmd + 'grp{0}_ran.dat rr_grp{0}_grp{0}.dat'.format(i)
#        subprocess.call(cmd, shell=True)
#        cmd = xi_cmd + 'gal_ran.dat grp{0}_ran.dat rr_gal_grp{0}.dat'.format(i)
#        subprocess.call(cmd, shell=True)
    
        cmd = xi_cmd + 'gal.dat grp{0}.dat gg_gal_grp{0}.dat'.format(i)
        subprocess.call(cmd, shell=True)
#        cmd = xi_cmd + 'gal.dat grp{0}_ran.dat gr_gal_grp{0}.dat'.format(i)
#        subprocess.call(cmd, shell=True)
        cmd = xi_cmd + 'grp{0}.dat gal_ran.dat gr_grp{0}_gal.dat'.format(i)
        subprocess.call(cmd, shell=True)


def mock_gal_xrun(xi_cmd=xi_bat):
    """gg, gr, and rr mock galaxy counts for use with group-galaxy cross-correlation."""

    cmd = xi_cmd + 'mock_gal_ran.dat mock_rr_gal_gal.dat'
    subprocess.call(cmd, shell=True)
    for ivol in range(1, 10):
        cmd = xi_cmd + 'mock{0}_gal.dat mock{0}_gg_gal_gal.dat'.format(ivol)
        subprocess.call(cmd, shell=True)
        cmd = xi_cmd + 'mock{0}_gal.dat mock_gal_ran.dat mock{0}_gr_gal_gal.dat'.format(ivol)
        subprocess.call(cmd, shell=True)


def mock_groups_bin_xrun(xi_cmd=xi_bat, nbin=len(mass_bins)-1):
    """Cross-correlation pair counts of binned mock groups."""

    for prefix in 'fh':
        for i in range(nbin):
    #        cmd = xi_cmd + '{0}mock_grp{1}_ran.dat {0}mock_rr_grp{1}_grp{1}.dat'.format(prefix, i)
    #        subprocess.call(cmd, shell=True)
    #        cmd = xi_cmd + '{0}mock_gal_ran.dat {0}mock_grp{1}_ran.dat {0}mock_rr_gal_grp{1}.dat'.format(prefix, i)
    #        subprocess.call(cmd, shell=True)
            for ivol in range(1, 10):
    #            cmd = xi_cmd + '{0}mock{1}_grp{2}.dat {0}mock{1}_gg_grp{2}_grp{2}.dat'.format(prefix, ivol, i)
    #            subprocess.call(cmd, shell=True)
    #            cmd = xi_cmd + '{0}mock{1}_grp{2}.dat {0}mock_grp{2}_ran.dat {0}mock{1}_gr_grp{2}_grp{2}.dat'.format(prefix, ivol, i)
    #            subprocess.call(cmd, shell=True)
            
                cmd = xi_cmd + 'mock{1}_gal.dat {0}mock{1}_grp{2}.dat {0}mock{1}_gg_gal_grp{2}.dat'.format(prefix, ivol, i)
                subprocess.call(cmd, shell=True)
    #            cmd = xi_cmd + '{0}mock{1}_gal.dat {0}mock_grp{2}_ran.dat {0}mock{1}_gr_gal_grp{2}.dat'.format(prefix, ivol, i)
    #            subprocess.call(cmd, shell=True)
                cmd = xi_cmd + '{0}mock{1}_grp{2}.dat mock_gal_ran.dat {0}mock{1}_gr_grp{2}_gal.dat'.format(prefix, ivol, i)
                subprocess.call(cmd, shell=True)


def mock_groups_bin_av():
    """Average cross-correlation pair counts of binned mock groups."""

    cu.counts_av(glob.glob('mock?_gg_gal_gal.dat'), 'mock_gg_gal_gal.dat')
    cu.counts_av(glob.glob('mock?_gr_gal_gal.dat'), 'mock_gr_gal_gal.dat')

    for prefix in 'fh':
        for i in range(len(mass_bins)-1):
    #        cu.counts_av(glob.glob('mock?_gg_grp{0}_grp{0}.dat'.format(i)),
    #                     'mock_gg_grp{0}_grp{0}.dat'.format(i))
    #        cu.counts_av(glob.glob('mock?_gr_grp{0}_grp{0}.dat'.format(i)),
    #                     'mock_gr_grp{0}_grp{0}.dat'.format(i))
            cu.counts_av(glob.glob(prefix+'mock?_gg_gal_grp{0}.dat'.format(i)),
                         prefix+'mock_gg_gal_grp{0}.dat'.format(i))
    #        cu.counts_av(glob.glob('mock?_gr_gal_grp{0}.dat'.format(i)),
    #                     'mock_gr_gal_grp{0}.dat'.format(i))
            cu.counts_av(glob.glob(prefix+'mock?_gr_grp{0}_gal.dat'.format(i)),
                         prefix+'mock_gr_grp{0}_gal.dat'.format(i))


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


def gal_aplots(key='w_p', binning=1, pi_lim=100, rp_lim=100):
    """Plot galaxy auto-correlation function."""

    gg = cu.PairCounts('gg_gal_gal.dat')
    gr = cu.PairCounts('gr_gal_gal.dat')
    rr = cu.PairCounts('rr_gal_gal.dat')

    counts = {'gg': gg, 'gr': gr, 'rr': rr}
    xi = cu.Xi()
    xi_est = xi.est(counts, cu.ls, key=key, binning=binning,
                    pi_lim=pi_lim, rp_lim=rp_lim)

    plt.clf()
    ax = plt.subplot(111)
    xi_est.plot(ax, label='gg')
    ax.loglog(basex=10, basey=10, nonposy='clip')
    plt.legend()
    plt.xlabel(r'$r_\perp$')
    plt.ylabel(r'$w_p(r_\perp)$')
    plt.show()


def bin_aplots(key='w_p', binning=1, pi_lim=100, rp_lim=100, nbin=4):
    """Plot the binned auto-correlations."""

    xi_list = []
    for i in range(nbin):

        gg = cu.PairCounts('gg_grp{0}.dat'.format(i))
        gr = cu.PairCounts('gr_grp{0}.dat'.format(i))
        rr = cu.PairCounts('rr_grp{0}.dat'.format(i))

        counts = {'gg': gg, 'gr': gr, 'rr': rr}
        xi = cu.Xi()
        xi_est = xi.est(counts, cu.ls, key=key, binning=binning,
                        pi_lim=pi_lim, rp_lim=rp_lim)
        xi_list.append(xi_est)

    plt.clf()
    ax = plt.subplot(111)
    for xi, label in zip(xi_list, [0, 1, 2, 3]):
        xi.plot(ax, label='M{}'.format(label))
    ax.loglog(basex=10, basey=10, nonposy='clip')
    plt.legend()
    plt.xlabel(r'$r_\perp$')
    plt.ylabel(r'$w_p(r_\perp)$')
    plt.draw()


def bin_xplots(est=cu.lsx, key='w_p', binning=1, pi_lim=100, rp_lim=100,
               nbin=4, bmin=5, bmax=20, cs=5, plot_file=None):
    """Plot the binned cross-correlations."""

    # Galaxy auto-correlation for reference
    gg = cu.PairCounts('gg_gal_gal.dat')
    gr = cu.PairCounts('gr_gal_gal.dat')
    rr = cu.PairCounts('rr_gal_gal.dat')

    counts = {'gg': gg, 'gr': gr, 'rr': rr}
    xi_gal = cu.Xi()
    xi_gal_est = xi_gal.est(counts, cu.ls, key=key, binning=binning,
                            pi_lim=pi_lim, rp_lim=rp_lim)
    xi_list = []
    for i in range(nbin):

        gg = cu.PairCounts('gg_gal_gal.dat')
        Gg = cu.PairCounts('gg_gal_grp{0}.dat'.format(i))
        GG = cu.PairCounts('gg_grp{0}_grp{0}.dat'.format(i))

        gr = cu.PairCounts('gr_gal_gal.dat')
        gR = cu.PairCounts('gr_gal_grp{0}.dat'.format(i))
        Gr = cu.PairCounts('gr_grp{0}_gal.dat'.format(i))
        GR = cu.PairCounts('gr_grp{0}_grp{0}.dat'.format(i))

        rr = cu.PairCounts('rr_gal_gal.dat')
        Rr = cu.PairCounts('rr_gal_grp{0}.dat'.format(i))
        RR = cu.PairCounts('rr_grp{0}_grp{0}.dat'.format(i))

        counts = {'gg': gg, 'Gg': Gg, 'GG': GG,
                  'gr': gr, 'gR': gR, 'Gr': Gr, 'GR': GR,
                  'rr': rr, 'Rr': Rr, 'RR': RR}
        xi = cu.Xi()
        xi_est = xi.est(counts, est, key=key, binning=binning,
                        pi_lim=pi_lim, rp_lim=rp_lim)
        xi_list.append(xi_est)

    if key == 'xi2':
        plt.clf()
        fig, axes = plt.subplots(4, 1, sharex=True, sharey=True, num=1)
        fig.subplots_adjust(hspace=0.01)
        fig.set_size_inches(4, 16)
        for im in range(4):
            xi = xi_list[im]
            ax = axes.flat[im]
            sc = xi.plot(ax, prange=(-2, 1), aspect='equal', axlabels=False,
                         cbar=True)
            if im == 3:
                ax.set_xlabel(r'$r_\perp\ [h^{-1} {{\rm Mpc}}]$')
            ax.set_ylabel(r'$r_\parallel\ [h^{-1} {{\rm Mpc}}]$')
            ax.text(0.9, 0.9, f'M{im+1}', ha='right', transform=ax.transAxes,
                    color='w')
#        fig.subplots_adjust(top=0.93)
#        cbar_ax = fig.add_axes([0.13, 0.96, 0.75, 0.01])
#        cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
#        cbar_ax.set_title(r'$\log \xi$')
        plt.draw()
        if plot_file:
            plt.savefig(plot_dir + plot_file, bbox_inches='tight')
        plt.show()

    else:

        plt.clf()
        fig, axes = plt.subplots(2, 1, sharex=True, sharey=False,
                                 gridspec_kw={'hspace': 0.05}, num=1)
        fig.set_size_inches(6, 8)
        ax = axes.flat[0]
        for xi, label in zip(xi_list, [1, 2, 3, 4]):
            xi.plot(ax, label=f'M{label}')
        xi_gal_est.plot(ax, label='gal-auto')
        ax.loglog(basex=10, basey=10, nonposy='clip')
        ax.set_xlim(0.1, 50)
        ax.set_ylim(1, 1e4)
        ax.legend()
        ax.set_ylabel(r'$w_p(r_\perp)$')

        # Relative bias
        xiref = xi_gal_est
        b, berr = np.zeros(nbin), np.zeros(nbin)
        bbins = (bmin <= xiref.sep) * (xiref.sep < bmax)
        ax = axes.flat[1]
        ax.semilogx(basex=10, nonposx='clip')
        ax.set_xlim(0.1, 50)
        ax.set_ylim(0, 10)
        for i in range(nbin):
            xi = xi_list[i]
            bratio = xi.est[:, 0] / xiref.est[:, 0]
            bvar = bratio**2 * ((xi.cov.sig[:]/xi.est[:, 0])**2 +
                                (xiref.cov.sig[:]/xiref.est[:, 0])**2)
            b[i] = np.average(bratio[bbins], weights=1.0/bvar[bbins])
            berr[i] = np.sum(1.0/bvar[bbins])**-0.5
            ax.errorbar(xiref.sep, bratio, bvar**0.5, capsize=cs)
        ax.set_xlabel(r'$r_\perp\ [h^{-1} {{\rm Mpc}}]$')
        ax.set_ylabel(r'$b/b_{\rm ref}$')
        plt.draw()
        if plot_file:
            plt.savefig(plot_dir + plot_file, bbox_inches='tight')
        plt.show()

        plt.clf()
        ax = plt.subplot(111)
        plt.errorbar(halomass, b, berr, capsize=cs)
        plt.xlabel(r'$lg M_h$')
        plt.ylabel(r'$b/b_{\rm ref}$')
        plt.draw()
        plt.savefig(plot_dir + 'bias.pdf', bbox_inches='tight')
        plt.show()


def bin_xplotsm(est=cu.lsx, key='w_p', binning=1, pi_lim=100, rp_lim=100,
                nbin=4, bmin=5, bmax=20, cs=5, tng_file='tng_wp.dat',
                plot_file=None):
    """Plot the binned cross-correlations, including mocks."""

    wp_dm = cu.RefXi()
    tng_dir = pickle.load(open(tng_file, 'rb'))
    plt.clf()
    if key == 'xi2':
        fig, axes = plt.subplots(4, 3, sharex=True, sharey=True, num=1)
        fig.subplots_adjust(hspace=0.01)
        fig.set_size_inches(12, 16)
    else:
        fig, axes = plt.subplots(2, 3, sharex='all', sharey='row', num=1,
                                 gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
        fig.set_size_inches(12, 8)

    labels = ('GAMA', 'FoF mock', 'halo mock')
    gpref = ('', 'mock_', 'mock_')
    hpref = ('', 'fmock_', 'hmock_')
    for icol in range(3):

        # Galaxy auto-correlation for reference
        gg = cu.PairCounts(f'{gpref[icol]}gg_gal_gal.dat')
        gr = cu.PairCounts(f'{gpref[icol]}gr_gal_gal.dat')
        rr = cu.PairCounts(f'{gpref[icol]}rr_gal_gal.dat')

        counts = {'gg': gg, 'gr': gr, 'rr': rr}
        xi_gal = cu.Xi()
        xi_gal_est = xi_gal.est(counts, cu.ls, key=key, binning=binning,
                                pi_lim=pi_lim, rp_lim=rp_lim)
        xi_list = []
        for i in range(nbin):

            Gg = cu.PairCounts(f'{hpref[icol]}gg_gal_grp{i}.dat')
            Gr = cu.PairCounts(f'{hpref[icol]}gr_grp{i}_gal.dat')
            counts = {'gg': gg, 'Gg': Gg, 'gr': gr, 'Gr': Gr, 'rr': rr}
            xi = cu.Xi()
            xi_est = xi.est(counts, est, key=key, binning=binning,
                            pi_lim=pi_lim, rp_lim=rp_lim)
            xi_list.append(xi_est)

        if key == 'xi2':
            for im in range(4):
                xi = xi_list[im]
                ax = axes[im, icol]
                sc = xi.plot(ax, prange=(-2, 1), aspect='equal', axlabels=False,
                             cbar=True)
                if im == 3:
                    ax.set_xlabel(r'$r_\perp\ [h^{-1} {{\rm Mpc}}]$')
                ax.set_ylabel(r'$r_\parallel\ [h^{-1} {{\rm Mpc}}]$')
                ax.text(0.9, 0.9, rf'${{\cal M}}${im+1}-gal', ha='right',
                        transform=ax.transAxes, color='w')
#        fig.subplots_adjust(top=0.93)
#        cbar_ax = fig.add_axes([0.13, 0.96, 0.75, 0.01])
#        cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
#        cbar_ax.set_title(r'$\log \xi$')

        else:
            ax = axes[0, icol]
            for im in range(nbin):
                xi = xi_list[im]
                color = next(ax._get_lines.prop_cycler)['color']
                xi.plot(ax, label=rf'${{\cal M}}${im+1}-gal', color=color)
#                if icol == 0:
                wp_tng = tng_dir[f'grp{im}']
                ax.plot(wp_tng.sep, wp_tng.est[:, 0], color=color)
#                    ,
#                            label=f'TNG grp{label}')
            color = next(ax._get_lines.prop_cycler)['color']
            xi_gal_est.plot(ax, color=color, label='gal-auto')
#            if icol == 0:
            wp_tng_gal = tng_dir['gal']
            ax.plot(wp_tng_gal.sep, wp_tng_gal.est[:, 0], color=color)
#            pdb.set_trace()
#                ,
#                        label='TNG gal')
#            ax.plot(wp_dm.xi.sep, wp_dm.xi.est[:, 0], label='DM')
            ax.loglog(basex=10, basey=10, nonposy='clip')
            ax.set_xlim(0.01, 50)
            ax.set_ylim(2, 1e5)
            ax.legend()
            ax.text(0.1, 0.1, labels[icol], ha='left', transform=ax.transAxes)
            if icol == 0:
                ax.set_ylabel(r'$w_p(r_\perp)$')

            # Relative bias
            xiref = xi_gal_est
            b, berr = np.zeros(nbin), np.zeros(nbin)
            bbins = (bmin <= xiref.sep) * (xiref.sep < bmax)
            ax = axes[1, icol]
            ax.semilogx(basex=10, nonposx='clip')
            ax.set_xlim(0.01, 50)
            ax.set_ylim(0, 12)
            for im in range(nbin):
                xi = xi_list[im]
                bratio = xi.est[:, 0] / xiref.est[:, 0]
                bvar = bratio**2 * ((xi.cov.sig[:]/xi.est[:, 0])**2 +
                                    (xiref.cov.sig[:]/xiref.est[:, 0])**2)
#                bratio = xi.est[:, 0] / wp_dm.interp(xi.sep)
#                berr = bratio * (xi.cov.sig[:]/xi.est[:, 0])
                b[im] = np.average(bratio[bbins], weights=1.0/bvar[bbins])
                berr[im] = np.sum(1.0/bvar[bbins])**-0.5
                color = next(ax._get_lines.prop_cycler)['color']
                ax.errorbar(xiref.sep, bratio, bvar**0.5, fmt='o',
                            color=color, capsize=cs)
                ax.plot(wp_tng_gal.sep,
                        tng_dir[f'grp{im}'].est[:, 0]/wp_tng_gal.est[:, 0],
                        color=color)
            ax.set_xlabel(r'$r_\perp\ [h^{-1} {{\rm Mpc}}]$')
            if icol == 0:
                ax.set_ylabel(r'$w_p^{\rm grp}/w_p^{\rm gal}$')
#                ax.set_ylabel(r'$w_p^{\rm grp}/w_p^{\rm DM}$')
    plt.draw()
    if plot_file:
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()

#    plt.clf()
#    ax = plt.subplot(111)
#    plt.errorbar(halomass, b, berr, capsize=cs)
#    plt.xlabel(r'$lg M_h$')
#    plt.ylabel(r'$b/b_{\rm ref}$')
#    plt.draw()
#    plt.savefig(plot_dir + 'bias.pdf', bbox_inches='tight')
#    plt.show()


def mock_bin_xplots(key='w_p', binning=1, pi_lim=100, rp_lim=100, nbin=4,
                    bmin=5, bmax=20, xlim=(0.01, 90), ylim=(0.1, 200000),
                    plot_size=(6, 5), plot_file=None):
    """Plot mock binned cross-correlations."""

    xi_list, xih_list = [], []
    for i in range(nbin):

#        gg = cu.PairCounts('mock_gg_gal_gal.dat')
        Gg = cu.PairCounts('fmock_gg_gal_grp{0}.dat'.format(i))
#        GG = cu.PairCounts('mock_gg_grp{0}_grp{0}.dat'.format(i))

        gr = cu.PairCounts('mock_gr_gal_gal.dat')
#        gR = cu.PairCounts('mock_gr_gal_grp{0}.dat'.format(i))
        Gr = cu.PairCounts('fmock_gr_grp{0}_gal.dat'.format(i))
#        GR = cu.PairCounts('mock_gr_grp{0}_grp{0}.dat'.format(i))

        rr = cu.PairCounts('mock_rr_gal_gal.dat')
#        Rr = cu.PairCounts('mock_rr_gal_grp{0}.dat'.format(i))
#        RR = cu.PairCounts('mock_rr_grp{0}_grp{0}.dat'.format(i))

#        counts = {'gg': gg, 'Gg': Gg, 'GG': GG,
#                  'gr': gr, 'gR': gR, 'Gr': Gr, 'GR': GR,
#                  'rr': rr, 'Rr': Rr, 'RR': RR}
        counts = {'Gg': Gg, 'gr': gr, 'Gr': Gr, 'rr': rr}
        xi = cu.Xi()
        xi_est = xi.est(counts, cu.lsx, key=key, binning=binning,
                        pi_lim=pi_lim, rp_lim=rp_lim)
        xi_list.append(xi_est)

        # True halo ;pair coounts
        Gg = cu.PairCounts('hmock_gg_gal_grp{0}.dat'.format(i))
        Gr = cu.PairCounts('hmock_gr_grp{0}_gal.dat'.format(i))
        counts = {'Gg': Gg, 'gr': gr, 'Gr': Gr, 'rr': rr}
        xi = cu.Xi()
        xi_est = xi.est(counts, cu.lsx, key=key, binning=binning,
                        pi_lim=pi_lim, rp_lim=rp_lim)
        xih_list.append(xi_est)

    sa_left = 0.15
    sa_bot = 0.08
    plt.clf()
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, num=1)
    fig.set_size_inches(plot_size)
    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, r'$r_\perp [h^{-1} {\rm Mpc}]$', ha='center', va='center')
    fig.text(0.06, 0.5, r'$w_p(r_\perp)$', ha='center', va='center',
             rotation='vertical')
    for i in range(nbin):
        ax = axes.flat[i]
        xi_list[i].plot(ax, label='FoF')
        xih_list[i].plot(ax, label='halo')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.text(0.9, 0.9, f'M{i+1}', ha='right', transform=ax.transAxes)
    ax.loglog(basex=10, basey=10, nonposy='clip')
    plt.legend(loc='lower left')
    plt.draw()
    if plot_file:
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()

    # Relative bias
    b, berr = np.zeros(nbin), np.zeros(nbin)
    xiref = xi_list[0]
    bbins = (bmin <= xiref.sep) * (xiref.sep < bmax)
    for i in range(nbin):
        xi = xi_list[i]
        bratio = xi.est[:, 0] / xiref.est[:, 0]
        bvar = bratio**2 * ((xi.cov.sig[:]/xi.est[:, 0])**2 +
                            (xiref.cov.sig[:]/xiref.est[:, 0])**2)
        b[i] = np.average(bratio[bbins], weights=1.0/bvar[bbins])
        berr[i] = np.sum(1.0/bvar[bbins])**-0.5

    plt.clf()
    ax = plt.subplot(111)
    plt.errorbar(mock_halomass, b, berr)
    plt.xlabel(r'$lg M_h$')
    plt.ylabel(r'$b/b_{\rm ref}$')
    plt.show()


def mock_bin_ind_plots(key='w_p', binning=1, pi_lim=100, rp_lim=100, nbin=4,
                       bmin=5, bmax=20, xlim=(0.01, 100), ylim=(0.1, 50000),
                       prefix='fmock', vols=range(1, 10), plot_size=(8, 5)):
    """Plot individual mock binned cross-correlations."""

    plt.clf()
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, num=1)
    fig.set_size_inches(plot_size)
    sa_left = 0.12
    sa_bot = 0.08
    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, r'$r_\perp [h^{-1} {\rm Mpc}$', ha='center', va='center')
    fig.text(0.06, 0.5, r'$w_p(r_\perp)$', ha='center', va='center',
             rotation='vertical')
    for im in range(nbin):
        ax = axes.flat[im]
        for ivol in vols:
#        gg = cu.PairCounts('mock_gg_gal_gal.dat')
            Gg = cu.PairCounts(f'{prefix}{ivol}_gg_gal_grp{im}.dat')
#        GG = cu.PairCounts('mock_gg_grp{0}_grp{0}.dat'.format(i))

            gr = cu.PairCounts(f'mock{ivol}_gr_gal_gal.dat')
#        gR = cu.PairCounts('mock_gr_gal_grp{0}.dat'.format(i))
            Gr = cu.PairCounts(f'{prefix}{ivol}_gr_grp{im}_gal.dat')
#        GR = cu.PairCounts('mock_gr_grp{0}_grp{0}.dat'.format(i))

            rr = cu.PairCounts('mock_rr_gal_gal.dat')
#        Rr = cu.PairCounts('mock_rr_gal_grp{0}.dat'.format(i))
#        RR = cu.PairCounts('mock_rr_grp{0}_grp{0}.dat'.format(i))

#        counts = {'gg': gg, 'Gg': Gg, 'GG': GG,
#                  'gr': gr, 'gR': gR, 'Gr': Gr, 'GR': GR,
#                  'rr': rr, 'Rr': Rr, 'RR': RR}
            counts = {'Gg': Gg, 'gr': gr, 'Gr': Gr, 'rr': rr}
            xi = cu.Xi()
            xi_est = xi.est(counts, cu.lsx, key=key, binning=binning,
                            pi_lim=pi_lim, rp_lim=rp_lim)
            ax.plot(xi_est.sep, xi_est.est[:, 0])
#            xi_est.plot(ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.text(0.9, 0.9, f'{prefix} M{im+1}', ha='right',
                transform=ax.transAxes)
    ax.loglog(basex=10, basey=10, nonposy='clip')
    plt.legend()
    plt.show()


def zhist(file_list=('gal.dat', 'gal_ran.dat', 'grp0.dat', 'grp1.dat',
                     'grp2.dat', 'grp3.dat'),
          nbin=20, plot_size=(4, 12), plot_file='zhist.pdf'):

    def read_file(infile, nskip=3):
        # Read input file into array and return array of distances
        data = np.loadtxt(infile, skiprows=nskip)
        dist = np.sqrt(np.sum(data[:, 0:3]**2, axis=1))
        return dist

    plt.clf()
    fig, axes = plt.subplots(5, 1, sharex=True, sharey=False, num=1)
    fig.subplots_adjust(hspace=0.05)
    fig.set_size_inches(plot_size)
    ax = axes[0]
    dist = read_file(file_list[0])
    bins = np.linspace(0.0, np.max(dist), nbin)
    ax.hist(dist, bins=bins, histtype='step', label='Gal', lw=1)
    dist = read_file(file_list[1])
    ax.hist(dist, bins=bins, histtype='step', weights=0.1*np.ones(len(dist)),
            label='Ran', lw=1)
    ax.legend(loc='upper left')
    ax.set_ylabel('Frequency')
    for i in range(4):
        dist = read_file(file_list[i+2])
        ax = axes[i+1]
        ax.hist(dist, bins=bins, histtype='step', label=f'Grp{i}', lw=1)
        ax.legend(loc='upper left')
        ax.set_ylabel('Frequency')
    plt.xlabel('Distance [Mpc/h]')
    plt.show()


def zhist_gal_ran(nbin=20, density=False, weights=(1, 0.1),
                  files=('gal.dat', 'gal_ran.dat'), labels=('Gal', 'Ran')):
    cu.zhist_list(files, nbin=nbin, labels=labels, density=density,
                  weights=weights)


def cone_plot_cart(infile, H0=100, omega_l=0.7, zlimits=(0, 0.267)):
    """Cone plot from cartesian coordinates."""

    def read_cart(infile):
        """Read the Cartesian coordinates from infile."""
        data = np.loadtxt(infile, skiprows=3)
        return data[:, :3]

    data = read_cart(infile)
    r, lat, lon = astropy.coordinates.cartesian_to_spherical(
            data[:, 0], data[:, 1], data[:, 2])
    cosmo = util.CosmoLookup(H0, omega_l, zlimits)
    z = cosmo.z_at_dm(r)
    util.cone_plot(lon.degree, lat.degree, z, z_limits=zlimits)


def cone_plot_fits(Mlim=-20, zlimits=(0, 0.267), gal_size=0.1, grp_size=10,
                   maxgal=20000,
                   clbl=r'$\log_{10}({\cal M}_h/{\cal M}_\odot h^{-1})$',
                   plot_size=(9, 6), plot_file=plot_dir+'cone.pdf'):
    """Cone plot from FITS file, colour-coded by group mass."""

    galxs = gs.GalSample(zlimits=zlimits)
    galxs.read_gama()
    galxs.vol_limit(Mlim)
    t = galxs.t
    idxs = np.arange(len(t))
    ngal = min(maxgal, len(t))
    idr = np.random.choice(idxs, size=ngal, replace=False)
    gal_dict = {'ra': t['RA'][idr], 'z': t['z'][idr], 's': gal_size, 'c': 'k',
                'marker': ',', 'alpha': 0.5, 'vmax': None}
#
#    ra = t['RA']
#    dec = t['DEC']
#    z = t['z']
#    clr = 12*np.ones(len(ra))
#    sz = 0.1*np.ones(len(ra))
#    util.cone_plot(t['RA'], t['DEC'], t['z'], z_limits=zlimits, size=0.1,
#                   clr='k', clbl=None, alpha=0.5, plot_size=plot_size,
#                   plot_file=plot_file)

    groups = gs.GalSample(zlimits=zlimits)
    groups.read_groups()
    t = groups.t
    grp_dict = {'ra': t['RA'], 'z': t['z'], 's': grp_size, 'c': t['log_mass'],
                'marker': 'o', 'alpha': 1.0, 'vmax': 14.6}
#    ra = np.append(ra, t['RA'])
#    dec = np.append(dec, t['DEC'])
#    z = np.append(z, t['z'])
#    clr = np.append(clr, t['log_mass'])
#    sz = np.append(sz, size*np.ones(len(ra)))
#    util.cone_plot(t['RA'], t['DEC'], t['z'], z_limits=zlimits, size=size,
#                   clr=t['log_mass'], clbl=clbl, alpha=1, plot_size=plot_size,
#                   plot_file=plot_file)
    util.cone_plot([gal_dict, grp_dict], z_limits=zlimits, clbl=clbl,
                   plot_size=plot_size, plot_file=plot_file)
#    util.cone_plot([grp_dict,], z_limits=zlimits, clbl=clbl,
#                   plot_size=plot_size, plot_file=plot_file)


# Galaxy clustering results for Stephen

def gal_lum_vol(bins=gal_mag_bins, nfac=10):
    """Galaxy clustering for volume-limited luminosity bins."""

    selcol = 'r_abs'
    for im in range(len(bins)-1):
        zlimits = [0.002, 0.5]
        galxs = gs.GalSample(zlimits=zlimits)
        galxs.read_gama()
        Mlo, Mhi = bins[im], bins[im+1]
        plt.clf()
        ax = plt.subplot(111)
        galxs.vol_limit(Mhi, ax=ax)
        plt.show()

        print('zlimit =', galxs.zlim)
        zlimits[1] = min(zlimits[1], galxs.zlim)
        galxs.info.update({'Mr': (Mlo, Mhi), 'zlimits': zlimits})
        sel_dict = {selcol: (Mlo, Mhi)}
        cu.xi_sel(galxs, f'gal_{Mlo}_{Mhi}.dat', f'gal_ran_{Mlo}_{Mhi}.dat',
                  '', nfac, sel_dict=sel_dict, set_vmax=False,
                  mask=gama_data+'/mask/zcomp.ply', run=0)


def gal_mass_vol(bins=gal_mass_bins, nfac=10):
    """Galaxy clustering for volume-limited masss bins."""

    def dm(z):
        return Mlo - gs.mass_limit(z)

    selcol = 'logmstar'
    for im in range(len(bins)-1):
        zlimits = [0.002, 0.5]
        galxs = gs.GalSample(zlimits=zlimits)
        galxs.read_gama()
        galxs.stellar_mass()
        galxs.vol_limited = True
        Mlo, Mhi = bins[im], bins[im+1]
        zmax = zlimits[1]
        if (dm(zmax) > 0):
            zhi = zmax
        else:
            zhi = scipy.optimize.brentq(dm, zlimits[0], zmax, xtol=1e-5,
                                        rtol=1e-5)

        print('zlimit =', zhi)
        galxs.zlimits[1] = min(zlimits[1], zhi)
        galxs.info.update({'logmstar': (Mlo, Mhi), 'zlimits': zlimits})
        sel_dict = {selcol: (Mlo, Mhi), 'z': galxs.zlimits}
        cu.xi_sel(galxs, f'gal_{Mlo}_{Mhi}.dat', f'gal_ran_{Mlo}_{Mhi}.dat',
                  '', nfac, sel_dict=sel_dict, set_vmax=False,
                  mask=gama_data+'/mask/zcomp.ply', run=0)


def gal_bin_run(bins=gal_mag_bins, xi_cmd=xi_bat):
    """Pair counts for bins."""

    for im in range(len(bins)-1):
        Mlo, Mhi = bins[im], bins[im+1]
        cmd = xi_cmd + f'gal_{Mlo}_{Mhi}.dat gg_{Mlo}_{Mhi}.dat'
        subprocess.call(cmd, shell=True)
        cmd = xi_cmd + f'gal_{Mlo}_{Mhi}.dat gal_ran_{Mlo}_{Mhi}.dat gr_{Mlo}_{Mhi}.dat'
        subprocess.call(cmd, shell=True)
        cmd = xi_cmd + f'gal_ran_{Mlo}_{Mhi}.dat rr_{Mlo}_{Mhi}.dat'
        subprocess.call(cmd, shell=True)


def gal_bin_plot(bins=gal_mag_bins, key='w_p', binning=1, pi_max=40.0,
                 outfile=None):
    """Plot clustering in luminosity/mass bins."""

    xlab = cu.xlabel[key]
    ylab = cu.ylabel[key]
    plt.clf()
    ax = plt.subplot(111)
    ax.loglog(basex=10, basey=10, nonposy='clip')
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    if outfile:
        fout = open(outfile, 'w')
    else:
        fout = None

    for im in range(len(bins)-1):
        Mlo, Mhi = bins[im], bins[im+1]
        infile = f'xi_{Mlo}_{Mhi}.dat'
        xi = cu.xi_req(infile, key, binning=binning, pi_lim=pi_max)
        xi.plot(ax, fout=fout, label=infile)
    if fout:
        fout.close()
    plt.legend()
    plt.show()


# Illustris halo clustering
def illustris_halo(bins=mass_bins, outfile='tng_wp.dat'):
    """Illustris halo and galaxy correlation functions."""

    lgrbins = np.linspace(-2, 1.8, 20)
    rbins = 10**lgrbins
    rcen = 10**(lgrbins[:-1] + np.diff(lgrbins))
    shellvol = 4/3*math.pi*(rbins[1:]**3 - rbins[:-1]**3)
    boxsize = 205.0
    boxvol = boxsize**3
    nthreads = 1

    basePath = '/Users/loveday/Data/TNG300-1/output/'

    # halo autocorrelation function
    fields = ['GroupMass', 'Group_M_Mean200', 'GroupPos']
    halos = il.groupcat.loadHalos(basePath, 87, fields=fields)
    lgM = 10 + np.log10(halos['Group_M_Mean200'])

    plt.clf()
    for i in range(len(bins)-1):
        Mlo, Mhi = bins[i], bins[i+1]
        sel = (Mlo <= lgM) * (lgM < Mhi)
        coords = 0.001 * halos['GroupPos'][sel, :]
        results = Corrfunc.theory.xi(boxsize, nthreads, rbins, *coords.T,
                                     output_ravg=1)
        plt.plot(results['ravg'], results['xi'], label=fr'[${Mlo}, {Mhi}]$')
    plt.loglog(basx=10, basey=10)
    plt.xlabel('r [Mpc/h]')
    plt.ylabel(r'$\xi(r)$')
    plt.title('Haloes')
    plt.legend()
    plt.show()

    # galaxy auto-correlation function
    subfields = ['SubhaloFlag', 'SubhaloMassType', 'SubhaloPos',
                 'SubhaloStellarPhotometrics']
    subhalos = il.groupcat.loadSubhalos(basePath, 87, fields=subfields)
    M_r = subhalos['SubhaloStellarPhotometrics'][:, 5]
    Mlim = np.sort(M_r)[47383]  # Required number to match density of gama vls
    print('Mlim =', Mlim)
    sel = M_r < Mlim
    galcoords = 0.001 * subhalos['SubhaloPos'][sel, :]
    ngal = len(M_r[sel])
    print(ngal, 'out of', len(M_r), 'galaxies selected')

    results = Corrfunc.theory.xi(boxsize, nthreads, rbins, *galcoords.T,
                                 output_ravg=1)
    nbin = len(rcen)
    xir = cu.Xi1d(nbin, 0, lgrbins[0], lgrbins[-1], 'xir', 'none',
                  {'source': 'TNG300-1'})
    xir.sep = results['ravg']
    xir.est = np.zeros((nbin, 1))
    xir.est[:, 0] = results['xi']
    xir.galpairs = 100*np.zeros(nbin)
    wp = xir.project()
    outdir = {'gal': wp}

    plt.clf()
    plt.plot(rcen, wp.est[:, 0])
    plt.loglog(basx=10, basey=10)
    plt.xlabel('r [Mpc/h]')
    plt.ylabel(r'$w_p(r_\bot)$')
    plt.title('Galaxies')
    plt.show()

    # halo-galaxy cross-correlation function
    plt.clf()
    for i in range(len(bins)-1):
        Mlo, Mhi = bins[i], bins[i+1]
        sel = (Mlo <= lgM) * (lgM < Mhi)
        ngrp = len(lgM[sel])
        print(ngrp, 'haloes in bin', i, 'Mmean =', np.mean(lgM[sel]))

        grpcoords = 0.001 * halos['GroupPos'][sel, :]
        results = Corrfunc.theory.DD(
                0, nthreads, rbins, *grpcoords.T,
                X2=galcoords[:, 0], Y2=galcoords[:, 1], Z2=galcoords[:, 2],
                boxsize=boxsize, output_ravg=1)
        rrpairs = ngal * ngrp * shellvol / boxvol
        xi = results['npairs']/rrpairs - 1
        xir = cu.Xi1d(nbin, 0, lgrbins[0], lgrbins[-1], 'xir', 'none',
                      {'source': 'TNG300-1'})
        xir.sep = results['ravg']
        xir.est = np.zeros((nbin, 1))
        xir.est[:, 0] = xi
        xir.galpairs = 100*np.zeros(nbin)
        wp = xir.project()
        outdir.update({f'grp{i}': wp})
        plt.plot(rcen, wp.est[:, 0], label=fr'$[{Mlo}, {Mhi}]$')
    plt.loglog(basx=10, basey=10)
    plt.xlabel('r [Mpc/h]')
    plt.ylabel(r'$w_p(r_\bot)$')
    plt.title('Cross-corr')
    plt.legend()
    plt.show()
    pickle.dump(outdir, open(outfile, 'wb'))
