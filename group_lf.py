# Plots for group LF paper

# from array import array
import copy
import math
import matplotlib as mpl
import matplotlib.lines as mlines
# import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker
# import mpmath
import numpy as np
import os
import pdb
import pickle
import scipy.optimize
from scipy import stats

# from astLib import astSED
from astropy.io import ascii
#from astropy.modeling import models, fitting
from astropy import table
from astropy.table import Table, join, hstack
# import healpy as hp
import illustris as il
# from sherpa.estmethods import Confidence
from sherpa.models.basic import Const1D, Gauss1D
from sherpa.plot import IntervalProjection, RegionProjection

import gal_sample as gs
import lf
from schec import (LogNormal, SchecMag, SchecMass, SchecMagSq, SchecMassSq,
                   SaundersMag, SaundersMass, SchecMagGen, SchecMassGen)
import util

# Global parameters
gama_data = os.environ['GAMA_DATA']
HOME = os.environ['HOME']
g3cgal = gama_data + 'g3cv9/G3CGalv08.fits'
g3cfof = gama_data + 'g3cv9/G3CFoFGroupv09.fits'
g3cmockfof = gama_data + 'g3cv6/G3CMockFoFGroupv06.fits'
g3cmockhalo = gama_data + 'g3cv6/G3CMockHaloGroupv06.fits'
g3cmockgal = gama_data + 'g3cv6/G3CMockGalv06.fits'
lf_data = os.environ['LF_DATA']
plot_dir = '/Users/loveday/Documents/tex/papers/gama/groupLF/'
kctemp = gama_data + 'kcorr_dmu/v5/kcorr_auto_z{}_vecv05.fits'

mag_label = r'$^{0.0}M_r - 5 \log_{10} h$'
ms_label = r'$\log_{10}\ ({\cal M}_*/{\cal M}_\odot h^{-2})$'

lf_label = r'$\phi(M)\ [h^3\ {\rm Mpc}^{-3}\ {\rm mag}^{-1}]$'
smf_label = r'$\phi({\cal M}_*)\ [h^3\ {\rm Mpc}^{-3}\ {\rm dex}^{-1}]$'
clf_label = r'$\phi_C(M)\ [{\rm group}^{-1}\ {\rm mag}^{-1}]$'
csmf_label = r'$\phi_C({\cal M}_*)\ [{\rm group}^{-1}\ {\rm dex}^{-1}]$'

# Halo mass bin limits and means and central red fraction
mbins_def6 = (12.0, 13.1, 13.3, 13.5, 13.7, 14.0, 15.2)
# mbins_def = (12.0, 13.2, 13.5, 13.8, 15.2)
mbins_def = (12.0, 13.3, 13.7, 14.1, 15.2)
mbins_sim6 = (12.7, 13.1, 13.3, 13.5, 13.7, 14.0, 14.5)
# mbins_sim = (12.8, 13.2, 13.5, 13.8, 14.5)
mbins_sim = (12.8, 13.3, 13.7, 14.1, 14.8)
mbins_z = (13.3, 13.7, 15.2)
# lgm_av6 = (12.91, 13.20, 13.40, 13.59, 13.82, 14.23)
# lgm_av = (12.99, 13.35, 13.64, 14.08)
lgm_av = (13.03, 13.50, 13.88, 14.37)  # Should eventually remove this
# lgm_av_mock6 = (12.84, 13.21, 13.40, 13.60, 13.84, 14.25)
# lgm_av_mock = (12.94, 13.36, 13.65, 14.10)
# lgm_av_mock = (12.97, 13.51, 13.89, 14.34)
# crf = (0.66, 0.69, 0.76, 0.74, 0.74, 0.75)

# Redshift bin limits
zbins = (0.002, 0.1, 0.2, 0.3)

# Constants
ln10 = math.log(10)
fwhm2sigma = (8*math.log(2))**-0.5
ngal_grouped = 25079
ngal_grouped_mock = 209926

metadata_conflicts = 'silent'  # Alternatives are 'warn', 'error'

# Ticks point inwards on all axes
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['mathtext.fontset'] = 'dejavusans'

np.seterr(all='warn')

def mass_z(grpfile=g3cfof, nmin=5, edge_min=0.9, vmax=None, zrange=(0, 0.5),
           Mrange=(11.8, 15.4), plot_file='mass_z.pdf', plot_size=(5, 4.5)):
    """Halo mass-redshift plot."""

    # Read and select groups meeting selection criteria
    t = Table.read(grpfile)
    if 'sim' in grpfile:
        sel = np.array(t['Nfof'] >= nmin)
    else:
        t['log_mass'] = 13.98 + 1.16*(np.log10(t['LumB']) - 11.5)
        sel = (np.array(t['GroupEdge'] > edge_min) *
               np.logical_not(t['log_mass'].mask) *
               np.array(t['Nfof'] >= nmin))

    t = t[sel]
    print('mass range of selected groups: {:5.2f} - {:5.2f}'.format(
            np.min(t['log_mass']), np.max(t['log_mass'])))

    plt.clf()
    plt.scatter(t['IterCenZ'], t['log_mass'], s=2, c=t['Nfof'], vmax=vmax,
                norm=mpl.colors.LogNorm())
    plt.xlim(zrange)
    for m in mbins_def:
        plt.plot(zrange, (m, m), 'k-', linewidth=0.5)
    plt.ylim(Mrange)
    for z in (0.1, 0.2, 0.3):
        plt.plot((z, z), Mrange, 'k-', linewidth=0.5)
    plt.xlabel('Redshift')
    plt.ylabel(r'$\log_{10}({\cal M}_h/{\cal M}_\odot h^{-1})$')
    cb = plt.colorbar()
    cb.set_label(r'$N_{\rm FoF}$')
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    if plot_file:
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()

    # Redshift histograms for each halo mass bin
    plt.clf()
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, num=1)
    fig.set_size_inches(5, 5)
    fig.subplots_adjust(left=0.1, bottom=0.1, top=1.0,
                        hspace=0.0, wspace=0.0)
    fig.text(0.58, 0.0, 'Redshift', ha='center', va='center')
    fig.text(0.0, 0.5, 'Frequency', ha='center', va='center', rotation='vertical')
    nbin = len(mbins_def) - 1
    for i in range(nbin):
        ax = axes.flat[i]
        ax.text(0.9, 0.9, f'M{i}', transform=ax.transAxes, ha='right')
        sel = (mbins_def[i] <= t['log_mass']) * (t['log_mass'] < mbins_def[i+1])
        ax.hist(t['IterCenZ'][sel], bins=np.linspace(0.0, 0.5, 26))
    plt.show()


def mass_comp(infile=g3cfof, nmin=5, edge_min=0.9, plot_file='mass_comp.pdf',
              plot_size=(5, 4)):
    """Compare dynamical and luminosity-based halo mass estimates."""

    # Read and select groups meeting selection criteria
    t = Table.read(infile)
    t['log_mass_lum'] = 13.98 + 1.16*(np.log10(t['LumB']) - 11.5)
    t['log_mass_dyn'] = np.log10(t['MassAfunc'])
    sel = (np.array(t['GroupEdge'] > edge_min) *
           np.logical_not(t['log_mass_lum'].mask) *
           np.array(t['Nfof'] >= nmin))
    t = t[sel]

    plt.clf()
    plt.scatter(t['log_mass_dyn'], t['log_mass_lum'], s=2,
                c=t['IterCenZ'])

    plt.axis((12, 15, 12, 15))
#    plt.xlabel(r'$\log_{10} {\cal M}_{\rm lum}$')
#    plt.ylabel(r'$\log_{10} {\cal M}_{\rm dyn}$')
    plt.xlabel(r'$\lg {\cal M}_{\rm dyn}$')
    plt.ylabel(r'$\lg {\cal M}_{\rm lum}$')
    cb = plt.colorbar()
    cb.set_label('Redshift')
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


def mass_comp_mock(infile=g3cmockhalo, nmin=5, edge_min=0.9, vmax=None,
                   dyn_mass='MassA', lum_mass='LumB',
                   limits=(11.8, 15, 11.8, 15), mbins=np.linspace(12, 15, 7),
                   plot_file='mass_comp_mock.pdf', plot_size=(5, 9.6)):
    """Compare dynamical and luminosity-based halo mass estimates for mock groups."""

    # Read and select groups meeting selection criteria
    thalo = Table.read(g3cmockhalo)
    thalo = thalo['HaloMass', 'IterCenRA', 'IterCenDEC']
    tfof = Table.read(g3cmockfof)
    t = join(thalo, tfof, keys=('IterCenRA', 'IterCenDEC'),
             metadata_conflicts=metadata_conflicts)

    t['log_mass_lum'] = 13.98 + 1.16*(np.log10(t[lum_mass]) - 11.5)
    t['log_mass_dyn'] = np.log10(t[dyn_mass])
    sel = (np.array(t['GroupEdge'] > edge_min) *
           np.logical_not(t['log_mass_lum'].mask) *
           np.array(t['Nfof'] >= nmin))
#           np.array(t['Volume'] == 1))
    t = t[sel]
    lgMh = np.log10(t['HaloMass'])
    nbin = len(mbins)-1
    mhav, mlav, mlstd = np.zeros(nbin), np.zeros(nbin), np.zeros(nbin)

    plt.clf()
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, num=1)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.set_size_inches(plot_size)
    ax = axes[0]
    ax.plot(limits[:2], limits[2:], 'k:')
#    ax.scatter(lgMh, t['log_mass_dyn'], s=2, c=t['IterCenZ'])
    ax.scatter(lgMh, t['log_mass_dyn'], s=2, c=t['Nfof'], vmax=vmax,
               norm=mpl.colors.LogNorm())

    for ibin in range(nbin):
        mlo, mhi = mbins[ibin], mbins[ibin+1]
        sel = (lgMh >= mlo) * (lgMh < mhi)
        mhav[ibin] = np.mean(lgMh[sel])
        mlav[ibin] = np.mean(t['log_mass_dyn'][sel])
        mlstd[ibin] = np.std(t['log_mass_dyn'][sel])
    ax.errorbar(mhav, mlav, mlstd, fmt='ro')
#    hist, xedge, yedge = np.histogram2d(t['log_mass_dyn'], lgMh, bins=38,
#                                        range=[limits[2:4], limits[0:2]])
#    ax.contour(hist, levels=levels, extent=limits, colors='r')
    ax.set_ylabel(r'$\lg {\cal M}_{\rm dyn}$')

    ax = axes[1]
    ax.plot(limits[:2], limits[2:], 'k:')
#    sc = ax.scatter(lgMh, t['log_mass_lum'], s=2, c=t['IterCenZ'])
    sc = ax.scatter(lgMh, t['log_mass_lum'], s=2, c=t['Nfof'], vmax=vmax,
                    norm=mpl.colors.LogNorm())

    for ibin in range(nbin):
        mlo, mhi = mbins[ibin], mbins[ibin+1]
        sel = (lgMh >= mlo) * (lgMh < mhi)
        mhav[ibin] = np.mean(lgMh[sel])
        mlav[ibin] = np.mean(t['log_mass_lum'][sel])
        mlstd[ibin] = np.std(t['log_mass_lum'][sel])
    ax.errorbar(mhav, mlav, mlstd, fmt='ro')
#    clr = ['red', 'orange']
#    for icont in range(2):
#        if icont == 0:
#            sel = t['Nfof'] < 5
#        else:
#            sel = t['Nfof'] >= 5
#    hist, xedge, yedge = np.histogram2d(t['log_mass_lum'], lgMh, bins=38,
#                                        range=[limits[2:4], limits[0:2]])
#    print('hist range ', np.min(hist), np.max(hist))
#    ax.contour(hist, levels=levels, extent=limits, colors='red')
    ax.set_ylabel(r'$\lg {\cal M}_{\rm lum}$')
    ax.set_xlabel(r'$\lg {\cal M}_{\rm halo}$')
#    ax.axis((11.8, 15, 11.8, 15))
    ax.axis(limits)
#    ax.set_aspect('equal')
    fig.subplots_adjust(top=0.93)
    cbar_ax = fig.add_axes([0.13, 0.97, 0.75, 0.02])
    cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
#    cb.set_label('Redshift')
#    cbar_ax.set_title('Redshift')
    cbar_ax.set_title(r'$N_{\rm FoF}$')

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


def group_mass_tab(nmin=5, edge_min=0.9, mbins=mbins_def,
                   out_file='group_mass.tex'):
    """Tabulate group mass statistics."""

    def group_stats(infile):
        t = Table.read(infile)
        try:
            t['Nfof'] = t['Nhalo']
        except KeyError:
            pass
        t['log_mass'] = 13.98 + 1.16*(np.log10(t['LumB']) - 11.5)
        sel = (np.array(t['GroupEdge'] > edge_min) *
               np.logical_not(t['log_mass'].mask) *
               np.array(t['Nfof'] >= nmin))
        t = t[sel]
        nmbin = len(mbins) - 1
        ngrp = []
        ngal = []
        zmean = []
        Mmean = []
        ngrptot, ngaltot = 0, 0
        for i in range(nmbin):
            sel = (mbins[i] <= t['log_mass']) * (t['log_mass'] < mbins[i+1])
            n = len(t[sel])
            ng = np.sum(t['Nfof'][sel])
            if 'Mock' in infile:
                n /= 9
                ng /= 9
            else:
                ngrptot += n
                ngaltot += ng
            ngrp.append('{:d}'.format(int(n)))
            ngal.append('{:d}'.format(int(ng)))
            zmean.append('{:3.2f}'.format(np.mean(t['IterCenZ'][sel])))
            Mmean.append('{:3.2f}'.format(np.mean(t['log_mass'][sel])))
        if 'Mock' not in infile:
            print(ngaltot, 'galaxies in', ngrptot, 'groups')
        return ngrp, ngal, zmean, Mmean

    ngrp_gama, ngal_gama, zmean_gama, Mmean_gama = group_stats(g3cfof)
    ngrp_fmock, ngal_fmock, zmean_fmock, Mmean_fmock = group_stats(g3cmockfof)
    ngrp_hmock, ngal_hmock, zmean_hmock, Mmean_hmock = group_stats(g3cmockhalo)
    nmbin = len(mbins) - 1
    f = open(plot_dir + out_file, 'w')
    print(r'''
\begin{table*}
\caption{Group bin names and log-mass limits, number of groups and galaxies,
mean log-mass, and mean redshift for GAMA-II groups, intrinsic mock haloes,
and FoF mock groups.
Note that each mock realisation has about 20 per cent smaller volume than
the GAMA-II equatorial fields.
\label{tab:group_mass_def}}
\begin{tabular}{ccccccccccccccccc}
\hline
& & \multicolumn{4}{c}{GAMA} & & \multicolumn{4}{c}{Halo Mocks} & &
\multicolumn{4}{c}{FoF Mocks} \\
\cline{3-6} \cline{8-11} \cline{13-16} \\[-2ex]
& $\lg {\cal M}_{h, {\rm limits}}$ & 
$N_{\rm grp}$ & $N_{\rm gal}$ & $\overline{\lg {\cal M}_h}$ & $\overline z$ & &
$N_{\rm grp}$ & $N_{\rm gal}$ & $\overline{\lg {\cal M}_h}$ & $\overline z$ & &
$N_{\rm grp}$ & $N_{\rm gal}$ & $\overline{\lg {\cal M}_h}$ & $\overline z$ \\
\hline''', file=f)
    for i in range(nmbin):
        print(r'${\cal M}' + str(i+1) + r'$ & ' +
              '[{:4.1f}, {:4.1f}] & '.format(mbins[i], mbins[i+1]) +
              ngrp_gama[i] + ' & ' + ngal_gama[i] + ' & ' + Mmean_gama[i] +
              ' & ' + zmean_gama[i] + ' & & ' +
              ngrp_hmock[i] + ' & ' + ngal_hmock[i] + ' & ' + Mmean_hmock[i] +
              ' & ' + zmean_hmock[i] + ' & & ' +
              ngrp_fmock[i] + ' & ' + ngal_fmock[i] + ' & ' + Mmean_fmock[i] +
              ' & ' + zmean_fmock[i] + r'\\', file=f)
    print(r'''
\hline
\end{tabular}
\end{table*}
''', file=f)
    f.close()


def sim_group_mass_tab(mbins=mbins_sim, lgal_file='smf_Lgal.pkl',
                       tng_file='TNG300-1_84_csmf.pkl',
                       eagle_file='/Users/loveday/Data/gama/jswml/auto/eagle_csmf.pkl',
                       out_file='sim_group_mass.tex'):
    """Tabulate group mass statistics for simulations."""

    # Numbers and mean log-masses of simulated groups
    # lgal_nhalo = [39316, 12618, 5665, 2984]
    # lgal_ngal = [39316, 12618, 5665, 2984]
    # lgal_z = [0.14, 0.19, 0.25, 0.29]
    # lgal_lgm = [12.97, 13.33, 13.63, 14.02]
#    eagle_n = [4110, 1054, 658, 425, 350, 183]
    eagle_z = [0.18, 0.18, 0.18, 0.18, 0.18, 0.18]
#    eagle_lgm = [12.90, 13.20, 13.40, 13.60, 13.84, 14.22]
#    tng_n = [4110, 1054, 658, 425, 350, 183]
    # tng_z = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    tng_z = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
#    tng_lgm = [12.90, 13.20, 13.40, 13.60, 13.84, 14.22]

    lgal_dict = pickle.load(open(lgal_file, 'rb'))
    eagle_dict = pickle.load(open(eagle_file, 'rb'))
    tng_dict = pickle.load(open(tng_file, 'rb'))

    nmbin = len(mbins) - 1
    with open(plot_dir + out_file, 'w') as f:
        print(r'''
\begin{table*}
\caption{Halo samples for \lgal, EAGLE and \tng\ simulations.
The log-mass limits (second column) are chosen to give
mean log masses close to those of GAMA galaxies in corresponding halo mass bins
(see Table~\ref{tab:group_mass_def}).
For each simulation, we give the number of haloes and galaxies, mean log-mass,
and snapshot redshift.
The number of galaxies quoted for \lgal\ comprises only those from Millennium,
not Millennium II, i.e. those with $\lg \mass_* > 9.5$.
EAGLE and \tng\ samples give the number of galaxies with $\lg \mass_* > 8.5$.
\label{tab:sim_group_mass_def}}
\begin{tabular}{cccccccccccccccccccc}
\hline
& & \multicolumn{4}{c}{\lgal} & &
\multicolumn{4}{c}{EAGLE} & & \multicolumn{4}{c}{\tng} \\
\cline{3-6} \cline{8-11} \cline{13-16} \\[-2ex]
& $\lg {\cal M}_{h, {\rm limits}}$ & 
$N_{\rm halo}$ & $N_{\rm gal}$ & $\overline{\lg {\cal M}_h}$ & $z$ & &
$N_{\rm halo}$ & $N_{\rm gal}$ & $\overline{\lg {\cal M}_h}$ & $z$ & &
$N_{\rm halo}$ & $N_{\rm gal}$ & $\overline{\lg {\cal M}_h}$ & $z$ \\
\hline''', file=f)
        for i in range(nmbin):
            lgal_lf = lgal_dict[f'M{i}_cen_all']
            eagle_lf = eagle_dict[f'M{i}_sat_all']
            tng_lf = tng_dict[f'M{i}_cen_all']
            print(r'${\cal M}' + str(i+1) + r'$ & ' +
                  '[{:4.1f}, {:4.1f}] & '.format(mbins[i], mbins[i+1]) +
                  rf"{lgal_lf.info['Ngrp']} & {lgal_lf.info['Ngal']} & " +
                  rf"{lgal_lf.info['lgm_av']:5.2f} & {lgal_lf.info['z_av']:5.2f} & & " +
                  rf"{eagle_lf.info['Nhalo']} & {eagle_lf.info['Ngal'] + eagle_lf.info['Nhalo']} & " +
                  rf"{eagle_lf.info['lgm_av']:5.2f} & {eagle_z[i]} & & " +
                  rf"{tng_lf.info['Ngrp']} & {tng_lf.info['Ngal']} & " +
                  rf"{tng_lf.info['lgm_av']:5.2f} & {tng_lf.info['z_av']:5.2f} \\", file=f)
        print(r'''
\hline
\end{tabular}
\end{table*}
''', file=f)


def gal_mass_z(galfile=g3cgal, grpfile=g3cfof, mass_est='lum', fslim=(0.8, 10),
               Mmin=8, Mmax=12, zlimits=(0.0, 0.5), nz=10, pfit_ord=2, nfit=4,
               nboot=100, z0=0.5,
               Mlim_ord=2, plot_file='gal_mass_z.pdf', plot_size=(6, 4),
               hexbin=False, colour_by='gminusi_stars', vmin=0.2, vmax=1.2,
               cb_label=r'$(g - i)^*$', sel_dict=None):
    """Grouped galaxy mass-redshift scatterplot or hexbin."""

    def neg(M):
        """Returns -1*kernel(M) for scipy to minimize."""
        return -1*kernel(M)

    samp = gs.GalSample()
    samp.read_grouped(galfile=galfile, grpfile=grpfile,
                      kref=0.0, mass_est=mass_est, masscomp=False)
    samp.select(sel_dict)
    t = samp.tsel()
    # samp.read_gama(nq_min=2)
    # samp.stellar_mass(fslim=fslim)
    # samp.group_props()
    # t = samp.t

#    zlims = np.percentile(t['z'], np.linspace(0, 100, nz+1))
    zlims = np.linspace(*zlimits, nz+1)
    zmean = np.zeros(nz)
    Mpeak = np.zeros(nz)
    Mstd = np.zeros(nz)
    Mbins = np.linspace(Mmin, Mmax, 41)
    M = Mbins[:-1] + 0.5*np.diff(Mbins)
    Mpboot = np.zeros(nboot)
    plt.clf()
#    fig, axes = plt.subplots(10, 1, sharex=True, sharey=True, num=1)
    fig, axes = plt.subplots(nz, 1, sharex=True, num=1)
    fig.set_size_inches(5, 16)
    fig.subplots_adjust(left=0.15, bottom=0.05, hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, ms_label, ha='center', va='center')
    fig.text(0.06, 0.5, 'Frequency', ha='center', va='center', rotation='vertical')
    for i in range(nz):
        sel = (t['z'] >= zlims[i]) * (t['z'] < zlims[i+1])
        zmean[i] = np.mean(t[sel]['z'])
        ax = axes.flat[i]
        lgM = t[sel]['logmstar']
        ax.scatter(lgM, 0.1*np.random.random(lgM.shape), s=0.01)
        kernel = stats.gaussian_kde(lgM)
        den = kernel(M)
        ax.plot(M, den)
        ipeak = np.argmax(den)
        res = scipy.optimize.fmin(neg, M[ipeak], disp=0)
        Mpeak[i] = res[0]
        # Bootstrap to find uncertainty in peak position
        for iboot in range(nboot):
            kernel = stats.gaussian_kde(np.random.choice(lgM, size=len(lgM)))
#            den = kernel(M)
#            ipeak = np.argmax(den)
            res = scipy.optimize.fmin(neg, M[ipeak], disp=0)
            Mpboot[iboot] = res[0]
        Mstd[i] = np.std(Mpboot)
#        freq, _, _ = ax.hist(t[sel]['logmstar'], bins=Mbins)
#        freq, Mbins, patches = ax.hist(t[sel]['logmstar'], bins='auto')
        # Fit polynomial to nfit bins either side of peak
#        icen = np.argmax(freq)
#        ilo = max(0, icen-nfit)
#        ihi = min(len(M)-1, icen+nfit+1)
#        p = np.polynomial.polynomial.polyfit(M[ilo:ihi], freq[ilo:ihi], pfit_ord)
#        fit = np.polynomial.polynomial.polyval(M[ilo:ihi], p)
#        Mpeak[i] = -p[1]/(2*p[2])
#        ax.plot(M[ilo:ihi], fit)
        yrange = (0, ax.axis()[3])
        ax.plot((Mpeak[i], Mpeak[i]), yrange)
        ax.bar(Mpeak[i], yrange[1], 2*Mstd[i], alpha=0.3)
#        ax.plot((Mpeak[i]-Mstd[i], Mpeak[i]-Mstd[i]), yrange, 'o--')
#        ax.plot((Mpeak[i]+Mstd[i], Mpeak[i]+Mstd[i]), yrange, 'o--')
        ax.text(0.9, 0.8, 'z = {:4.3f}'.format(zmean[i]), ha='right',
                transform=ax.transAxes)
        ax.set_ylim(yrange)
    plt.xlim(Mmin, Mmax)
    plt.show()
#    pdb.set_trace()

    plt.clf()
    if hexbin:
        cs = plt.hexbin(t['z'], t['logmstar'], cmap='Blues', bins='log')
        plt.colorbar(cs, label='log N')
    else:
        try:
            plt.scatter(t['z'], t['logmstar'], s=0.1, c=t[colour_by],
                        vmin=vmin, vmax=vmax, cmap='coolwarm')
            cb = plt.colorbar()
            cb.set_label(cb_label)
        except KeyError:
            plt.scatter(t['z'], t['logmstar'], s=0.1)
    plt.errorbar(zmean, Mpeak, Mstd, fmt='go', color='orange')
#    for Mlim_ord in (2):
#        p = np.polynomial.polynomial.polyfit(1/(1+zmean), Mpeak, Mlim_ord, w=Mstd**-2)
#        fit = np.polynomial.polynomial.polyval(1/(1+zlims), p)
#        plt.plot(zlims, fit, '-', label='{}wtd'.format(Mlim_ord))
    p = np.polynomial.polynomial.polyfit(1/(1+zmean), Mpeak, Mlim_ord)
    fit = np.polynomial.polynomial.polyval(1/(1+zlims), p)
    plt.plot(zlims, fit, 'g-', label='{}unwtd'.format(Mlim_ord))
#    plt.legend()
    plt.xlabel(r'$z$')
    plt.ylabel(ms_label)
    plt.axis((0, 0.5, 7, 12))
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()
    print('Polynomial coeffs for M*[1/(1+z)]', p)

    try:
        plt.clf()
        plt.hist(t['gminusi_stars'], bins=20)
        plt.show()
    except KeyError:
        pass


def gal_mag_z(nmin=5, edge_min=0.9, fslim=(0.8, 10), Mmin=8, Mmax=12, nz=10,
              pfit_ord=2, nfit=4, nboot=100, z0=0.5, Mlim_ord=3):
    """Grouped galaxy mag-redshift scatterplot."""

    samp = gs.GalSample()
    samp.read_gama()
    samp.group_props()
    t = samp.t
    sel = (np.array(t['GroupEdge'] > edge_min) * np.array(t['Nfof'] >= nmin))
    t = t[sel]

    plt.clf()
    plt.scatter(t['z'], samp.abs_mags('r_petro')[sel], s=0.1)
    plt.xlabel(r'$z$')
    plt.ylabel(mag_label)
    plt.axis((0, 0.5, -15, -23))
    plt.show()


def gal_mass_lum(nmin=5, edge_min=0.9, fslim=(0.8, 10), pc=95,
                 plot_file='gal_mass_lum.pdf', plot_size=(5, 4)):
    """Galaxy stellar mass-luminosity scatterplot."""

    samp = gs.GalSample()
    samp.read_grouped()
    mags = np.linspace(-23, -15, 32)
    masses = samp.smf_comp(mags, pc=pc)
    fitpar = np.polynomial.polynomial.polyfit(mags, masses, 2)
    fit = np.polynomial.polynomial.polyval((mags[0], mags[-1]), fitpar)
    print('95-percentile mass = {} + {} M_r'.format(*fitpar))

    # Best fitting linear relation y = a + bx +- sigma,
    # where x = logmstar and y = M_r
    def lnlike(par, x, y):
        """-ive log likelihood to minimise."""
        a, b, sigma = par
        lnnorm = math.log((2*math.pi)**0.5 * sigma) * len(x)
        ans = lnnorm + np.sum((y - (a + b*x))**2/(2*sigma**2))
        return ans

    p0 = [0, -2, 0.5]
    par = scipy.optimize.fmin(
            lnlike, p0, args=(samp.t['logmstar'], samp.abs_mags('r_petro')),
            disp=0)
    lgmstar = np.array((7, 12))
    mr = -1.957*lgmstar
    print(f'Mr = {par[0]} + {par[1]} logm* +/ {par[2]}')

    plt.clf()
    plt.scatter(samp.abs_mags('r_petro'), samp.t['logmstar'], s=0.1,
                c=samp.t['gminusi_stars'], vmin=0.2, vmax=1.2, cmap='coolwarm')
    cb = plt.colorbar()
    cb.set_label(r'$(g - i)^*$')
    plt.plot(mr, lgmstar, 'k-', label='Best fit')
    plt.plot(mr-0.41, lgmstar, 'k:')
    plt.plot(mr+0.41, lgmstar, 'k:')
    plt.plot(mags, masses, 'g-', label='Binned 95-th percentile')
    plt.plot((mags[0], mags[-1]), fit, 'r-', label='Linear fit to above')
    plt.xlabel(r'$^{0.1}M_r - 5 \log\ h$')
    plt.ylabel(r'$\log_{10}({\cal M}_*/{\cal M}_\odot h^{-2})$')
    plt.axis((-15, -23, 7, 12))
    plt.legend()
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


def sersic_hist(plot_file='sersic_hist.pdf', lnrange=(-1, 1.5), ncut=1.9,
                colour='gi_colour', Mlims=(-22.5, -21.5), plot_size=(5, 4)):
    """Sersic-index histogram for blue and red galaxies."""

    samp = gs.GalSample()
    samp.read_gama()
    samp.stellar_mass()
    samp.add_sersic_index()

    plt.clf()
    logn = np.log10(samp.t['GALINDEX_r'])
    logncut = math.log10(ncut)
    blue = samp.t[colour] == 'b'
    red = samp.t[colour] == 'r'
    plt.hist((logn[blue], logn[red]), bins=20, range=lnrange, color=('b', 'r'))
#    plt.hist((logn, logn[blue], logn[red]), bins=20, range=range,
#             color=('k', 'b', 'r'))
    ymin, ymax = plt.ylim()
    plt.plot((logncut, logncut), (ymin, ymax), 'k')
    plt.xlabel(r'$\log_{10}\ n_r$')
    plt.ylabel(r'Frequency')
    ax = plt.subplot(111)
    plt.text(0.1, 0.9, 'Disky', transform=ax.transAxes)
    plt.text(0.9, 0.9, 'Spheroidal', ha='right', transform=ax.transAxes)
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.scatter(samp.t['gminusi_stars'], logn, s=0.1, c='b')
    plt.xlabel(r'$(g - i)^*$')
    plt.ylabel(r'$\log_{10}\ n_r$')
    plt.show()

    plt.clf()
    plt.scatter(samp.t['z'], logn, s=0.1, c='b')
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\log_{10}\ n_r$')
    plt.show()

    z = samp.t['z']
    absval = samp.abs_mags('r_petro')
    zbins = np.linspace(0.05, 0.6, 12)
    zc = zbins[:-1] + 0.5*(zbins[1] - zbins[0])
    nz = len(zc)
    rfrac = np.zeros(nz)
    rfrac_err = np.zeros(nz)
    sfrac = np.zeros(nz)
    sfrac_err = np.zeros(nz)
    sph = logn > logncut
    for iz in range(nz):
        sel = ((zbins[iz] <= z) * (z < zbins[iz+1]) *
               (Mlims[0] <= absval) * (absval < Mlims[1]))
        sfrac[iz] = len(samp.t[sph*sel])/len(samp.t[sel])
        sfrac_err[iz] = sfrac[iz]/math.sqrt(len(samp.t[sph*sel]))
        rfrac[iz] = len(samp.t[red*sel])/len(samp.t[sel])
        rfrac_err[iz] = rfrac[iz]/math.sqrt(len(samp.t[red*sel]))
    plt.clf()
    plt.errorbar(zc, sfrac, sfrac_err, label='Sph fraction')
    plt.errorbar(zc, rfrac, rfrac_err, fmt='--', c='r', label='Red fraction')
    plt.xlabel(r'Redshift')
    plt.ylabel(r'Spheroidal or red fraction')
    plt.legend()
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    plt.savefig(plot_dir + 'sph_red_fraction.pdf', bbox_inches='tight')
    plt.show()


def taylor_colour(infile=gama_data+'StellarMassesLambdarv20.fits',
                  fslim=(0.8, 10), H0=100, limits=(8.5, 12.0, 0.0, 1.3),
                  zbins=(0.002, 0.1, 0.2, 0.3, 0.65),
                  plot_file='colour_mass.pdf', plot_size=(5, 5)):
    """Blue/red colour classification using Taylor+2015 dereddened colours."""

    def colourCut(lgm):
        """Return (g-i)^* colour cut corresponding to log mass"""
        return 0.07*lgm - 0.03

    m = Table.read(infile)
    sel = (m['fluxscale'] >= fslim[0]) * (m['fluxscale'] < fslim[1])
    print(len(m[sel]), 'out of', len(m),
          'galaxies with fluxscale in range', fslim)
    m = m[sel]
    logmstar = m['logmstar'] - 2*math.log10(H0/70.0) + np.log10(m['fluxscale'])
    z = m['Z_TONRY']
    gistar = m['gminusi_stars']
    logmstar = m['logmstar']
    lgm = np.array(limits[:2])
    cutline = colourCut(lgm)

    plt.clf()
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, num=1)
    fig.set_size_inches((8, 8))
    fig.subplots_adjust(left=0.14, bottom=0.07, hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, r'$\log_{10}\ ({\cal M}_*/{\cal M}_\odot h^{-2})$',
             ha='center', va='center')
    fig.text(0.06, 0.5, r'$(g - i)^*$', ha='center', va='center',
             rotation='vertical')
    for iz in range(4):
        zlo, zhi = zbins[iz], zbins[iz+1]
        ax = axes.flat[iz]
        idx = (zlo <= z) * (z < zhi)
        hist, xedges, yedges = np.histogram2d(gistar[idx], logmstar[idx],
                                              50, [limits[2:4], limits[0:2]])
#        ax.scatter(logmstar[idx], gistar[idx], s=0.01, c='k', alpha=0.2)
#        pdb.set_trace()
        ax.contour(hist, 10, extent=limits)
        ax.plot(lgm, cutline, 'r-')
        ax.axis(limits)
        title = r'${:2.1f} < z < {:2.1f}$'.format(zlo, zhi)
        ax.text(0.5, 0.9, title, transform=ax.transAxes)
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


def smf_comp(infile='smf.dat', lf_lims=(6.5, 12, 1e-5, 1), nmin=2,
             xlabel=ms_label, ylabel=smf_label,
             plot_file='smf_comp.pdf', plot_size=(5, 4)):
    """Compare z < 0.1 field galaxy SMF with Baldry+2012 & Wright+2017 dbl
    Schechter fits."""

    h = 0.7
    lgh = math.log10(h)
#    h = 1
    wright = (10.78 + 2*lgh, -0.62, -1.50, 2.93e-3/h**3, 0.63e-3/h**3)
    baldry = (10.66 + 2*lgh, -0.35, -1.47, 3.96e-3/h**3, 0.79e-3/h**3)
    lf_dict = pickle.load(open(infile, 'rb'))
    plt.clf()
    ax = plt.subplot(111)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.semilogy(basey=10, nonposy='clip')
#    phi = lf_dict['all']
#    phi.plot(ax=ax, nmin=nmin, show_fit=False, clr='b', label='All GAMA')
    phi = lf_dict['z0']
    phi.plot(ax=ax, nmin=nmin, show_fit=False, clr='b', label='This paper')
    bdat = np.loadtxt(lf_data + '/Baldry2012/table1.txt')
    logM = bdat[:, 0] + 2*lgh
    phi = bdat[:, 2]/1000/h**3
    phi_err = bdat[:, 3]/1000/h**3
    plt.errorbar(logM, phi, phi_err, fmt='gs', label='Baldry+2012')
    logM = np.linspace(lf_lims[0], lf_lims[1], 100)
#    phi = lf.Schechter_dbl_mass(logM, baldry)
#    print(phi)
#    plt.plot(logM, phi, label='Baldry 2012')
    phi = lf.LF(None, None, logM)
    phi.ref_fn(phi.Schechter_dbl_mass, wright)
    phi.plot(ax=ax, clr='r', fmt='-', label='Wright+2017')
    plt.axis(lf_lims)
    plt.legend()
#    plt.draw()
#    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


def central_f_red(mbins=mbins_def, clrname='gi_colour', zlimits=(0.002, 0.65)):
    """Calculate fraction of red central galaxies in each halo mass bin as
    needed for Yang et al central CLF/CSMF comparison.."""

    samp = gs.GalSample(zlimits=zlimits)
    samp.read_gama()
    samp.stellar_mass()
    samp.group_props()
    samp.add_sersic_index()

    nmbin = len(mbins) - 1
    for i in range(nmbin):
        sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                    'RankIterCen': [1, 2]}
        samp.select(sel_dict)
        ncen = len(samp.t[samp.use])
        sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                    'RankIterCen': [1, 2],
                    clrname: ('r', 's')}
        samp.select(sel_dict)
        ncenred = len(samp.t[samp.use])
        print('mass', mbins[i], mbins[i+1], 'central red frac', ncenred/ncen)


def count_grouped(galfile=g3cgal, grpfile=g3cfof,
                  nmin=5, zlimits=(0.002, 0.65), Q=1, P=1):
    """Count number of galaxies in reliable groups."""

    samp = gs.GalSample(Q=Q, P=P, zlimits=zlimits)
    samp.read_grouped(galfile=galfile, grpfile=grpfile, nmin=nmin)
    print(len(samp.t), 'grouped galaxies')


def count_grouped_mock(galfile=g3cmockgal, grpfile=g3cmockfof,
                       nmin=5, zlimits=(0.002, 0.65), Q=1, P=1):
    """Count number of galaxies in reliable groups."""

    samp = gs.GalSample(Q=Q, P=P, zlimits=zlimits)
    samp.read_grouped(galfile=galfile, grpfile=grpfile, nmin=nmin)
    print(len(samp.t), 'mock grouped galaxies')


def lf_field(kref=0.1, colname='r_petro', clrname='gi_colour',
             bins=np.linspace(-25, -12, 26), zlimits=(0.002, 0.65),
             error='jackknife', outfile='lfr.pkl'):
    """r-band LF using density-corrected Vmax."""

    samp = gs.GalSample()
    samp.read_gama(kref=kref)
    samp.stellar_mass()
    samp.add_sersic_index()
    samp.vis_calc((samp.sel_mag_lo, samp.sel_mag_hi))
    samp.vmax_calc()
    lf_dict = {}
    phi = lf.LF(samp, colname, bins, error=error)
#    lf.fn_fit(fn=lf.Schechter_mag, Mmin=Mmin_fit, Mmax=Mmax_fit, p0=p0)
    lf_dict['all'] = phi
#    lf.plot(finish=True)

    for colour in 'br':
        clr_limits = ('a', 'z')
        if (colour == 'b'):
            clr_limits = ('b', 'c')
        if (colour == 'r'):
            clr_limits = ('r', 's')
        sel_dict = {clrname: clr_limits}
        samp.select(sel_dict)
        phi = lf.LF(samp, colname, bins, error=error, sel_dict=sel_dict)
        lf_dict[colour] = phi

    for lbl, sersic_lims in zip(['nlo', 'nhi'], [[0, 1.9], [1.9, 30]]):
        sel_dict = {'GALINDEX_r': sersic_lims}
        samp.select(sel_dict)
        phi = lf.LF(samp, colname, bins, error=error, sel_dict=sel_dict)
#        lf.fn_fit(fn=lf.Schechter_mag, Mmin=Mmin_fit, Mmax=Mmax_fit, p0=p0)
        lf_dict[lbl] = phi

    pickle.dump(lf_dict, open(outfile, 'wb'))


def smf_field(kref=0.0, masscomp='gama', colname='logmstar', clrname='gi_colour',
              bins=np.linspace(7, 12.5, 23), zlimits=(0.002, 0.65),
              error='jackknife', outfile='smf_field_comp.pkl'):
    """SMF using density-corrected Vmax."""

    samp = gs.GalSample()
    samp.read_gama(kref=kref)
    samp.stellar_mass()
    if masscomp:
        samp.masscomp = masscomp
        samp.mass_limit_sel()
        samp.comp_limit_mass()
        samp.vis_calc((samp.sel_mass_hi, samp.sel_mass_lo,
                       samp.sel_mag_lo, samp.sel_mag_hi))
    else:
        # samp.comp_limit_mag()
        samp.vis_calc((samp.sel_mag_lo, samp.sel_mag_hi))
    samp.add_sersic_index()
    samp.vmax_calc()
    lf_dict = {}
    sel_dict = {'masscomp': masscomp}
    phi = lf.LF(samp, colname, bins, error=error)
#    lf.fn_fit(fn=lf.Schechter_mag, Mmin=Mmin_fit, Mmax=Mmax_fit, p0=p0)
    lf_dict['all'] = phi
    # pdb.set_trace()
#    lf.plot(finish=True)

    for colour in 'br':
        clr_limits = ('a', 'z')
        if (colour == 'b'):
            clr_limits = ('b', 'c')
        if (colour == 'r'):
            clr_limits = ('r', 's')
        sel_dict = {clrname: clr_limits}
        samp.select(sel_dict)
        sel_dict['masscomp'] = masscomp
        phi = lf.LF(samp, colname, bins, error=error, sel_dict=sel_dict)
        lf_dict[colour] = phi

    for lbl, sersic_lims in zip(['nlo', 'nhi'], [[0, 1.9], [1.9, 30]]):
        sel_dict = {'GALINDEX_r': sersic_lims}
        samp.select(sel_dict)
        sel_dict['masscomp'] = masscomp
        phi = lf.LF(samp, colname, bins, error=error, sel_dict=sel_dict)
#        lf.fn_fit(fn=lf.Schechter_mag, Mmin=Mmin_fit, Mmax=Mmax_fit, p0=p0)
        lf_dict[lbl] = phi

    pickle.dump(lf_dict, open(outfile, 'wb'))


def lfr_mock(colname='r_petro', clrname='gi_colour',
             bins=np.linspace(-25, -12, 26), zlimits=(0.002, 0.65),
             error='mock', outfile='lfr_mock.pkl'):
    """Mock r-band LF using density-corrected Vmax."""

    samp = gs.GalSample()
    samp.read_gama_mocks()
    samp.vis_calc((samp.sel_mag_lo, samp.sel_mag_hi))
    samp.vmax_calc()
    lf_dict = {}
    lf = lf.LF(samp, colname, bins, error=error)
#    lf.fn_fit(fn=lf.Schechter_mag, Mmin=Mmin_fit, Mmax=Mmax_fit, p0=p0)
    lf_dict['all'] = lf
#    lf.plot(finish=True)

    pickle.dump(lf_dict, open(outfile, 'wb'))


def field_plot(infile, lf_lims=(-15, -23.5, 1e-7, 0.1), nmin=5, fn=SchecMag(),
               p0=(-1, -21, -2), Mmin_fit=-24, Mmax_fit=-17,
               plot_file='/Users/loveday/Documents/tex/papers/gama/groupLF/lf_field.pdf',
               plot_size=(6, 3)):
    """Plot field LFs."""

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


def gama_mock_plot(gfile='lfr_k00.pkl', mfile='lfr_mock.pkl',
                   tfile='TNG300-1_84_lf.pkl',
                   lf_lims=(-15, -23.5, 1e-6, 0.09), nmin=5, schecdbl=True,
                   p0=(-21, -1.25, -2,), Mmin_fit=-24, Mmax_fit=-15,
                   sigma=[1, 2], lc_limits=5, lc_step=32,
                   plot_file='/Users/loveday/Documents/tex/papers/gama/groupLF/lf_field_comp.pdf',
                   plot_size=(5, 5)):
    """Plot GAMA & mock field LFs."""

#    fn.alpha1 = p0[0]
#    fn.alpha2 = p0[1]
#    fn.Mstar = p0[2]
#    fn.lgps1 = p0[3]
#    fn.lgps2 = p0[4]

    # Fit sum of Schechter functions with common M*
    f1 = SchecMag('schec1')
    f1.mstar = p0[0]
    f1.alpha = p0[1]
    f1.lgps = p0[2]
    if schecdbl:
        f2 = SchecMag('schec2')
        sep = Const1D('sep')
        f2.alpha = f1.alpha + sep.c0
        sep.c0 = -1.5
        sep.c0.min = -5
        sep.c0.max = 0
        f2.mstar = f1.mstar
        f2.lgps = p0[2]
        fn = f1 + f2 + 0*sep
    else:
        fn = f1
    gdict = pickle.load(open(gfile, 'rb'))
    mdict = pickle.load(open(mfile, 'rb'))
    tdict = pickle.load(open(tfile, 'rb'))
    plt.clf()
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, num=1)
    fig.set_size_inches(plot_size)
#    fig.subplots_adjust(left=0, bottom=0.0, hspace=0.0, wspace=0.0)
    plt.semilogy(basey=10, nonposy='clip')
#    ax = axes[0]
    ax.axis(lf_lims)
    ax.set_xlabel(mag_label)
    ax.set_ylabel(lf_label)

    axin = ax.inset_axes([0.25, 0.12, 0.4, 0.4])
#    axin.xaxis.set_label_position("top")
#    axin.yaxis.set_label_position("right")
#    axin.set_xlabel(r'$\alpha$')
#    axin.set_ylabel(r'$M^*$')
    axin.set_ylabel(r'$\chi^2$')
    axin.set_xlabel(r'$M^*$')
    phig = gdict['all']
    phim = mdict['all']
    phit = tdict
    for phi, label, clr in zip(
            (phig, phim, phit), ('GAMA', 'Mock', 'TNG'), 'brg'):
        phi.fn_fit(fn, Mmin=Mmin_fit, Mmax=Mmax_fit)
        print(label, 'chi2=', phi.res.statval, 'dof=', phi.res.dof)
        print(phi.res.parvals)
        phi.plot(ax=ax, nmin=nmin, clr=clr, label=label)
        iproj = IntervalProjection()
        iproj.prepare(min=-21.5, max=-20, nloop=41)
#        iproj.calc(phi.fit, phi.fn.schec1.Mstar)
        iproj.calc(phi.fit, f1.Mstar)
        axin.plot(iproj.x, iproj.y, color=clr)
        axin.set_xlim((-20, -21.5))
        axin.set_ylim((0, 200))

#        rproj = RegionProjection()
#        rproj.prepare(nloop=(lc_step, lc_step), fac=lc_limits,
#                      sigma=sigma)
#        rproj.calc(phi.fit, phi.fn.alpha1, phi.fn.Mstar)
#        axin.plot(rproj.parval0, rproj.parval1, '+', color=clr)
#        xmin, xmax = rproj.min[0], rproj.max[0]
#        ymin, ymax = rproj.min[1], rproj.max[1]
#        nx, ny = rproj.nloop
#        extent = (xmin, xmax, ymin, ymax)
#        y = rproj.y.reshape((ny, nx))
#        v = rproj.levels
#        axin.contour(y, v, origin='lower', extent=extent, colors=(clr,))

#    # EAGLE LF
#    infile = HOME + '/Documents/Research/LFdata/EAGLE/lfr.dat'
#    (mag, phi, phi_err) = np.loadtxt(infile)
#    ax.errorbar(mag, phi, phi_err, label='EAGLE')
#    ax.legend()
#
#    # TNG LF
#    phi = pickle.load(open('tng_lf.pkl', 'rb'))
#    phi.plot(ax=ax, nmin=nmin, label='TNG')

    ax.legend()

    plt.draw()
    plt.savefig(plot_file, bbox_inches='tight')
    plt.show()


def field_smf_plot(gfiles=('smf_field_comp.pkl',),
                   lf_lims=(8, 12.1, 1e-5, 0.2), nmin=5,
                   plot_file='/Users/loveday/Documents/tex/papers/gama/groupLF/smf_field_comp.pdf',
                   plot_size=(5, 5)):
    """Plot GAMA & simulation field SMFs."""

    plt.clf()
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, num=1)
    fig.set_size_inches(plot_size)
#    fig.subplots_adjust(left=0, bottom=0.0, hspace=0.0, wspace=0.0)
    plt.semilogy(basey=10, nonposy='clip')
#    ax = axes[0]
    ax.axis(lf_lims)
    ax.set_xlabel(ms_label)
    ax.set_ylabel(smf_label)

#    axin = ax.inset_axes([0.25, 0.12, 0.4, 0.4])
#    axin.xaxis.set_label_position("top")
#    axin.yaxis.set_label_position("right")
#    axin.set_xlabel(r'$\alpha$')
#    axin.set_ylabel(r'$M^*$')
    for gfile in gfiles:
        gdict = pickle.load(open(gfile, 'rb'))
        phi = gdict['all']
#    clr = 'b'
#    phi.fn_fit(fn, Mmin=Mmin_fit, Mmax=Mmax_fit)
        phi.plot(ax=ax, nmin=nmin, label='This work')
        # phi.plot(ax=ax, nmin=nmin, label=gfile, alpha=alpha)

    # Baldry+2012
    h = 0.7
    lgh = math.log10(h)
    bdat = np.loadtxt(lf_data + '/Baldry2012/table1.txt')
    logM = bdat[:, 0] + 2*lgh
    phi = bdat[:, 2]/1000/h**3
    phi_err = bdat[:, 3]/1000/h**3
    plt.errorbar(logM, phi, phi_err, fmt='s', label='Baldry+2012', alpha=0.8)

    # Wight+2017
    wright = (10.78 + 2*lgh, -0.62, -1.50, 2.93e-3/h**3, 0.63e-3/h**3)
    phi = lf.LF(None, None, logM)
    phi.ref_fn(phi.Schechter_dbl_mass, wright)
    phi.plot(ax=ax, fmt='-', label='Wright+2017')
    plt.axis(lf_lims)

    # L-Galaxies SMF
    infile = lf_data + 'L-Galaxies/MR_MRII_smf_z0.20_h-2.txt'
    data = np.loadtxt(infile)
    mass, phi, phi_err = data[:, 0], data[:, 1], data[:, 2]
    ax.errorbar(mass, phi, phi_err, fmt='^', label='L-Galaxies', alpha=0.8)
    ax.legend()

    # EAGLE SMF
    infile = lf_data + '/EAGLE/smf.dat'
    (mass, phi, phi_err) = np.loadtxt(infile)
    ax.errorbar(mass, phi, phi_err, fmt='v', label='EAGLE', alpha=0.8)
    ax.legend()
#    print(mass, phi, phi_err)

    # TNG SMF
    phi = pickle.load(open('TNG300-1_84_smf.pkl', 'rb'))
    phi.plot(ax=ax, nmin=nmin, fmt='D', label='TNG')

    ax.legend()

    plt.draw()
    plt.savefig(plot_file, bbox_inches='tight')
    plt.show()


def clf_broad(mbins=mbins_def, clrname='gi_colour', Vmax='Vmax_grp',
              bins=(-24, -21, -19, -17), colname='r_petro', error='jackknife',
              outfile='clf_broad.pkl'):
    """Conditional LF in broad mag bins."""
    clf(mbins=mbins, clrname=clrname, Vmax=Vmax,
        bins=bins, colname=colname, error=error, outfile=outfile)


def clf_vol(mbins=mbins_def, clrname='gi_colour', Vmax='Vmax_grp',
            bins=np.linspace(-24, -16, 18), zlimits=(0.002, 0.1), colname='r_petro',
            error='jackknife', outfile='clf_vol.pkl'):
    """Conditional LF for volume-limited sample."""
    clf(mbins=mbins, clrname=clrname, Vmax=Vmax,
        bins=bins, zlimits=zlimits, colname=colname, error=error, outfile=outfile)


def clf_vol_vmax(mbins=mbins_def, clrname='gi_colour', Vmax='Vmax_dec',
                 bins=np.linspace(-24, -16, 18), zlimits=(0.002, 0.1), colname='r_petro',
                 error='jackknife', outfile='clf_vol_vmax.pkl'):
    """Vmax LF for volume-limited sample."""
    clf(mbins=mbins, clrname=clrname, Vmax=Vmax,
        bins=bins, zlimits=zlimits, colname=colname, error=error, outfile=outfile)


def lf_vmax(Vmax='Vmax_dec', outfile='lf_vmax.pkl'):
    """Vmax LF."""
    lf_samp(Vmax=Vmax, outfile=outfile)


def lf_vmaxz02(Vmax='Vmax_dec', zlimits=(0.002, 0.2),
               outfile='lf_vmax_z02.pkl'):
    """Vmax LF."""
    lf_samp(Vmax=Vmax, zlimits=zlimits, outfile=outfile)


def lf_vmax_nmin2(nmin=2, Vmax='Vmax_dec', outfile='lf_vmax_nmin2.pkl'):
    """Vmax LF."""
    lf_samp(nmin=nmin, Vmax=Vmax, outfile=outfile)


def lf_mockfof(galfile=g3cmockgal, grpfile=g3cmockfof, Vmax='Vmax_raw',
               error='mock', outfile='lf_mockfof.pkl'):
    """FOF Mock LFs."""
    lf_samp(galfile=galfile, grpfile=grpfile, Vmax=Vmax, error=error,
            outfile=outfile)


def lf_mockhalo(galfile=g3cmockgal, grpfile=g3cmockhalo, Vmax='Vmax_raw',
                error='mock', outfile='lf_mockhalo.pkl'):
    """Halo Mock LFs."""
    lf_samp(galfile=galfile, grpfile=grpfile, mass_est='true', Vmax=Vmax,
            error=error, outfile=outfile)


def smf_comp_vmax(colname='logmstar', bins=np.linspace(7, 12, 21),
                  masscomp='gama', Vmax='Vmax_dec',
                  outfile='smf_comp_vmax.pkl'):
    """Vmax SMF."""
    lf_samp(colname=colname, bins=bins, masscomp=masscomp, Vmax=Vmax,
            outfile=outfile)


def smf_incomp_vmax(colname='logmstar', bins=np.linspace(7, 12, 21),
                    masscomp=False, Vmax='Vmax_dec',
                    outfile='smf_incomp_vmax.pkl'):
    """Vmax SMF."""
    lf_samp(colname=colname, bins=bins, masscomp=masscomp, Vmax=Vmax,
            outfile=outfile)


def lf_zslice(Vmax='Vmax_dec'):
    """Vmax LF in redshift slices."""
    for iz in range(len(zbins)-1):
        zlo, zhi = zbins[iz], zbins[iz+1]
        outfile = f'lf_vmax_z{zlo}_{zhi}.pkl'
        lf_samp(Vmax=Vmax, mbins=mbins_def[1:], zlimits=(zlo, zhi),
                Q=0, P=0, outfile=outfile)


def lf_zslice_m24(Vmax='Vmax_dec'):
    """Vmax LF in redshift slices for mass-bins 2-4."""
    for iz in range(len(zbins)-1):
        zlo, zhi = zbins[iz], zbins[iz+1]
        outfile = f'lf_m24_vmax_z{zlo}_{zhi}.pkl'
        lf_samp(Vmax=Vmax, mbins=(mbins_def[1], mbins_def[4]),
                zlimits=(zlo, zhi),
                Q=0, P=0, outfile=outfile)


def lf_zslice_m34(Vmax='Vmax_dec'):
    """Vmax LF in redshift slices for mass-bins 3-4."""
    for iz in range(len(zbins)-1):
        zlo, zhi = zbins[iz], zbins[iz+1]
        outfile = f'lf_m34_vmax_z{zlo}_{zhi}.pkl'
        lf_samp(Vmax=Vmax, mbins=(mbins_def[2], mbins_def[4]),
                zlimits=(zlo, zhi),
                Q=0, P=0, outfile=outfile)


def smf_zslice(colname='logmstar', bins=np.linspace(7, 12, 21), masscomp='gama',
               Vmax='Vmax_dec'):
    """Vmax SMF in redshift slices."""
    for iz in range(len(zbins)-1):
        zlo, zhi = zbins[iz], zbins[iz+1]
        outfile = f'smf_comp_vmax_z{zlo}_{zhi}.pkl'
        lf_samp(colname=colname, bins=bins, masscomp=masscomp, Vmax=Vmax,
                mbins=mbins_def[1:], zlimits=(zlo, zhi), Q=0, P=0,
                outfile=outfile)


def smf_incomp_zslice(colname='logmstar', bins=np.linspace(7, 12, 21),
                      masscomp=False, Vmax='Vmax_dec'):
    """Vmax SMF in redshift slices (not stellar mass complete)."""
    for iz in range(len(zbins)-1):
        zlo, zhi = zbins[iz], zbins[iz+1]
        outfile = f'smf_incomp_vmax_z{zlo}_{zhi}.pkl'
        lf_samp(colname=colname, bins=bins, masscomp=masscomp, Vmax=Vmax,
                mbins=mbins_def[1:], zlimits=(zlo, zhi), Q=0, P=0,
                outfile=outfile)


def lf_mock_zslice(galfile=g3cmockgal, grpfile=g3cmockfof, Vmax='Vmax_raw',
                   error='mock'):
    """Mock LFs in redshift slices."""
    for iz in range(len(zbins)-1):
        zlo, zhi = zbins[iz], zbins[iz+1]
        outfile = f'lf_mock_z{zlo}_{zhi}.pkl'
        lf_samp(galfile=galfile, grpfile=grpfile, Vmax=Vmax,
                mbins=mbins_def[1:], error=error,
                zlimits=(zlo, zhi), Q=0, P=0, outfile=outfile)


def lf_samp(galfile=g3cgal, grpfile=g3cfof, kref=0.0,
            colname='r_petro', masscomp=False, renorm=False,
            mass_est='lum', nmin=5, mbins=mbins_def, clrname='gi_colour',
            bins=np.linspace(-24, -16, 17), zlimits=(0.002, 0.65), vol_z=None,
            Vmax='Vmax_dec', Q=1, P=1, error='jackknife', outfile='clf.pkl'):
    """Group LF by central/satellite, galaxy colour and Sersic index.
    Normalised to number of groups if Vmax='Vmax_grp', otherwise using
    density-corrected Vmax and normalised to total grouped galaxy sample
    if renorm=True."""

    samp = gs.GalSample(Q=Q, P=P, zlimits=zlimits)
    samp.read_grouped(galfile=galfile, grpfile=grpfile, kref=kref,
                      mass_est=mass_est, nmin=nmin, masscomp=masscomp)

    # Now done within read_+grouped()
    # if masscomp:
    #     mass_limit(samp)
    #     samp.comp_limit_mass()
    if colname == 'r_petro':
        samp.comp_limit_mag()
    if vol_z:
        samp.vol_limit_z(vol_z)
        samp.group_limit(nmin)
        grps = table.unique(samp.t, keys='GroupID')
        plt.clf()
        plt.scatter(grps['IterCenZ'], grps['log_mass'], s=2,
                    c=grps['Nfof'], norm=mpl.colors.LogNorm())
        plt.xlabel('Redshift')
        plt.ylabel(r'$\log_{10}({\cal M}_h/{\cal M}_\odot h^{-1})$')
        cb = plt.colorbar()
        cb.set_label(r'$N_{\rm fof}$')
        plt.show()
    samp.vmax_calc()

    lf_dict = {}
    nmbin = len(mbins) - 1
    for i in range(nmbin):
        sel_dict = {'log_mass': (mbins[i], mbins[i+1])}
        samp.select(sel_dict)
        norm = 1
        if Vmax == 'Vmax_grp':
            samp.vmax_group(mbins[i], mbins[i+1])
        else:
            if renorm:
                norm = len(samp.t)/len(samp.tsel())

        ngrp = len(np.unique(samp.tsel()['GroupID']))
        ngal = len(samp.tsel())
        if 'Mock' in galfile:
            ngrp /= 9
            ngal /= 9
        lgMh_av = np.mean(samp.tsel()['log_mass'])
        z_av = np.mean(samp.tsel()['z'])
        info = {'Ngrp': ngrp, 'Ngal': ngal, 'lgm_av': lgMh_av, 'z_av': z_av,
                'zlimits': zlimits}

        Mkey = f'M{i}'
        for lbl, rank_lims, p0 in zip(['all', 'cen', 'sat'],
                                      [[1, 500], [1, 2], [2, 500]],
                                      ((-1, -20, 1), (5, -19.5, -2.5), (-1, -20, 1))):
            sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                        'RankIterCen': rank_lims}
            samp.select(sel_dict)
            phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax, error=error,
                        sel_dict=sel_dict, info=info)
            lf_dict[Mkey + '_' + lbl + '_all'] = phi

            if error != 'mock':
                for colour in ('blue', 'red'):
                    clr_limits = ('a', 'z')
                    if (colour == 'blue'):
                        clr_limits = ('b', 'c')
                    if (colour == 'red'):
                        clr_limits = ('r', 's')
                    sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                                'RankIterCen': rank_lims,
                                clrname: clr_limits}
                    samp.select(sel_dict)
                    phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax,
                                error=error, sel_dict=sel_dict, info=info)
                    lf_dict[Mkey + '_' + lbl + '_' + colour] = phi

                for nlbl, sersic_lims in zip(['nlo', 'nhi'], [[0, 1.9], [1.9, 30]]):
                    sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                                'RankIterCen': rank_lims,
                                'GALINDEX_r': sersic_lims}
                    samp.select(sel_dict)
                    phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax, error=error,
                                sel_dict=sel_dict, info=info)
                    lf_dict[Mkey + '_' + lbl + '_' + nlbl] = phi

    pickle.dump(lf_dict, open(outfile, 'wb'))


def ks_samp(gal1file=g3cgal, grp1file=g3cfof,
            gal2file=g3cmockgal, grp2file=g3cmockfof, kcfile=kctemp.format('00'),
            mass_est1='lum', mass_est2='lum', colname='r_petro', masscomp=False,
            nmin=5, mbins=mbins_def, clrname='gi_colour',
            zlimits=(0.002, 0.65),
            Q=1, P=1, error='jackknife', outfile='ks_samp.txt'):
    """KS test of the consistency of two samples, e.g. mocks vs data.
    Test is performed on distribution of colname (e.g. abs mag or
    log stellar mass), without any Vmax weighting."""

    samp1 = gs.GalSample(Q=Q, P=P, zlimits=zlimits)
    samp1.read_grouped(galfile=gal1file, grpfile=grp1file, kcfile=kcfile,
                       mass_est=mass_est1, nmin=nmin, masscomp=masscomp)
    samp2 = gs.GalSample(Q=Q, P=P, zlimits=zlimits)
    samp2.read_grouped(galfile=gal2file, grpfile=grp2file, kcfile=kcfile,
                       mass_est=mass_est2, nmin=nmin, masscomp=masscomp)
    # if masscomp:
    #     mass_limit(samp1)
    #     mass_limit(samp2)

    fout = open(outfile, 'w')
    print(gal1file, grp1file, gal2file, grp2file, colname, file=fout)
    nmbin = len(mbins) - 1
    for i in range(nmbin):
        sel_dict = {'log_mass': (mbins[i], mbins[i+1])}
        for lbl, rank_lims in zip(['all', 'cen', 'sat'],
                                  [[1, 500], [1, 2], [2, 500]]):
            sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                        'RankIterCen': rank_lims}
            samp1.select(sel_dict)
            samp2.select(sel_dict)
            if colname == 'logmstar':
                abs1 = samp1.tsel()[colname]
                abs2 = samp2.tsel()[colname]
            else:
                abs1 = samp1.abs_mags(colname)
                abs2 = samp2.abs_mags(colname)
            D, p = scipy.stats.ks_2samp(abs1, abs2)
            print(i, lbl, D, p)
            print(i, lbl, D, p, file=fout)
    fout.close()


def ks_samp_halo_fof():
    ks_samp(g3cmockgal, g3cmockhalo, g3cmockgal, g3cmockfof, mass_est1='true',
            outfile='ks_halo_fof.txt')


def ks_samp_gama_mock():
    ks_samp(g3cgal, g3cfof, g3cmockgal, g3cmockfof, outfile='ks_gama_mock.txt')


def csmf_alp_comp(nmin=2, fslim=(0.8, 10), Vmax='Vmax_grp',
                  mbins=(12, 13, 14, 16), mass_est='lum',
                  smfile=gama_data + 'StellarMassesv19.fits',
                  p0=(-1.5, 10.5, -2.5), bins=np.linspace(7, 12, 20),
                  Mmin_fit=8.5, Mmax_fit=12, clrname='gi_colour',
                  colname='logmstar', error='jackknife',
                  outfile='csmf_alp_comp.pkl'):
    """Low-z conditional SMF in same mass bins as Alpaslan+2015."""

    samp = gs.GalSample(Q=0, P=0, zlimits=(0.013, 0.1))
    samp.read_gama()
    samp.group_props(mass_est=mass_est, nmin=nmin)
    samp.add_sersic_index()
    samp.stellar_mass(smf=smfile, fslim=fslim)

    samp.vis_calc((lf.sel_gama_mag_lo, lf.sel_gama_mag_hi))
    samp.vmax_calc()
    lf_dict = {}
    nmbin = len(mbins) - 1
    for i in range(nmbin):
        sel_dict = {'log_mass': (mbins[i], mbins[i+1])}
        samp.select(sel_dict)
        if Vmax == 'Vmax_grp':
            samp.vmax_group(mbins[i], mbins[i+1])
            norm = 1
#        else:
            #        norm = len(samp.t)/len(samp.tsel())
        phi = lf.LF(samp, colname, bins, Vmax=Vmax,
                    norm=1, error=error, sel_dict=sel_dict)
        phi.fn_fit(fn=lf.Schechter_mass, p0=p0, Mmin=Mmin_fit, Mmax=Mmax_fit)
        p0 = phi.fit_par
        Mkey = 'M{}'.format(i)
        lf_dict[Mkey] = phi

    pickle.dump(lf_dict, open(outfile, 'wb'))


def csmf_alp_comp_plot(infile='csmf_alp_comp.pkl', nmin=2,
                       xlabel=ms_label, ylabel=smf_label,
                       plot_file='csmf_alp_comp.pdf', plot_size=(5, 4)):
    """Low-z conditional SMF in same mass bins as Alpaslan+2015."""

    alp_par = ((-0.87, 10.86, -2.76),
               (-0.87, 10.93, -2.95),
               (-1.06, 11.16, -3.57))
    lf_lims = (9.5, 12.0, 1e-5, 0.01)
    clrs = 'bgr'
    lf_dict = pickle.load(open(infile, 'rb'))
    plt.clf()
    ax = plt.subplot(111)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.semilogy(basey=10, nonposy='clip')
    plt.axis(lf_lims)
    for i in range(3):
        key = 'M{}'.format(i)
        phi = lf_dict[key]
        Mlo, Mhi = phi.sel_dict['log_mass']
        label = r'${\cal M}' + r'= \ [{}, {}]$'.format(Mlo, Mhi)
        phi.plot(ax=ax, nmin=nmin, show_fit=False, clr=clrs[i], label=label)
        phi.fn_plot(ax, alp_par[i], c=clrs[i])
    plt.legend(loc=3)
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


def smf_z(colname='logmstar', bins=np.linspace(7, 12, 21),
          masscomp='gama', outfile='smf_comp_z.pkl'):
    """field SMF in redshift bins (without evolution corrections)."""
    lfr_z(colname=colname, bins=bins, masscomp=masscomp, outfile=outfile)


def smf_incomp_z(colname='logmstar', bins=np.linspace(7, 12, 21),
                 masscomp=False, outfile='smf_incomp_z.pkl'):
    """field SMF in redshift bins (without evolution corrections)."""
    lfr_z(colname=colname, bins=bins, masscomp=masscomp, outfile=outfile)


def lfr_z(colname='r_petro', bins=np.linspace(-24, -16, 17), kref=0.0,
          zlims=(0.002, 0.1, 0.2, 0.3), masscomp=False, renorm=False,
          clrname='gi_colour', error='jackknife', outfile='lfr_z.pkl'):
    """r-band field LF in redshift bins (without evolution corrections,
    normalised to total number of grouped galaxies)."""

    samp = gs.GalSample(Q=0, P=0)
    samp.read_gama(kref=kref)
    ngal = len(samp.t)
    if renorm:
        norm = ngal_grouped/ngal
    else:
        norm = 1
    print('norm =', norm)

    samp.stellar_mass()
    samp.add_sersic_index()
    if masscomp:
        samp.vis_calc((samp.sel_mass_hi, samp.sel_mass_lo))
        samp.mass_limit_sel()
    else:
        samp.vis_calc((samp.sel_mag_lo, samp.sel_mag_hi))

    lf_dict = {}
    for iz in range(3):
        zlo, zhi = zlims[iz], zlims[iz+1]
        samp.zlimits = (zlo, zhi)
        if colname == 'logmstar':
            samp.comp_limit_mass()
        else:
            samp.comp_limit_mag()
        samp.vmax_calc()
        sel_dict = {'z': (zlo, zhi)}
        samp.select(sel_dict)
        phi = lf.LF(samp, colname, bins,
                    norm=norm, error=error, sel_dict=sel_dict)
        Mkey = f'z{iz}_all'
        lf_dict[Mkey] = phi

        for colour in ('blue', 'red'):
            clr_limits = ('a', 'z')
            if (colour == 'blue'):
                clr_limits = ('b', 'c')
            if (colour == 'red'):
                clr_limits = ('r', 's')
            sel_dict = {'z': (zlo, zhi), clrname: clr_limits}
            samp.select(sel_dict)
            phi = lf.LF(samp, colname, bins,
                        norm=norm, error=error, sel_dict=sel_dict)
            lf_dict[f'z{iz}_{colour}'] = phi

        for lbl, sersic_lims in zip(['nlo', 'nhi'], [[0, 1.9], [1.9, 30]]):
            sel_dict = {'z': (zlo, zhi), 'GALINDEX_r': sersic_lims}
            samp.select(sel_dict)
            phi = lf.LF(samp, colname, bins,
                        norm=norm, error=error, sel_dict=sel_dict)
            lf_dict[f'z{iz}_{lbl}'] = phi

    pickle.dump(lf_dict, open(outfile, 'wb'))


def lfr_z_mock(colname='r_petro', bins=np.linspace(-24, -16, 18),
               zlims=(0.002, 0.1, 0.2, 0.3),
               error='mock', outfile='lfr_z_mock.pkl'):
    """mock r-band field LF in redshift bins (without evolution corrections),
    normalised to total number of mock grouped galaxies."""

    samp = gs.GalSample(Q=0, P=0)
    samp.read_gama_mocks()
    ngal = len(samp.t)
    norm = ngal_grouped_mock/ngal
    print('norm =', norm)
    samp.vis_calc((samp.sel_mag_lo, samp.sel_mag_hi))

    lf_dict = {}
    for iz in range(3):
        zlo, zhi = zlims[iz], zlims[iz+1]
        samp.zlimits = (zlo, zhi)
        samp.vmax_calc(denfile=None)
        sel_dict = {'z': (zlo, zhi)}
        samp.select(sel_dict)
        phi = lf.LF(samp, colname, bins,
                    norm=norm, error=error, sel_dict=sel_dict)
        Mkey = 'z{}mock'.format(iz)
        lf_dict[Mkey] = phi

    pickle.dump(lf_dict, open(outfile, 'wb'))


def clf_plotm(infiles=('lf_vmax.pkl', 'lf_vmax_nmin2.pkl'),
              labels=(rf'$N_{{\rm fof}} \geq 5$', rf'$N_{{\rm fof}} \geq 2$'),
              loc=0, npmin=5, lf_lims=(-15.5, -23.9, 5e-6, 0.2),
              plot_file='clf_nmin.pdf', plot_size=(6, 7.5)):
    """Plot LFs from multiple input files."""

    nfiles = len(infiles)
    lf_list = []
    for i in range(nfiles):
        lf_dict = pickle.load(open(infiles[i], 'rb'))
        lf_list.append(lf_dict)
#    clff = SchecMag()
    plt.clf()
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, num=1)
    fig.set_size_inches(plot_size)
    fig.subplots_adjust(left=0.15, bottom=0.05, hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, mag_label, ha='center', va='center')
    if 'vmax' in infiles[0]:
        ylabel = r'$\phi(M) / \phi_{\rm sim}(M)$'
    else:
        ylabel = r'$\phi_C(M) / \phi_{C, {\rm sim}}(M)$'
    fig.text(0.06, 0.5, ylabel, ha='center', va='center', rotation='vertical')
    for ibin in range(6):
        ax = axes.flat[ibin]
        for i in range(nfiles):
            phi = lf_list[i]['M{}'.format(ibin)]
            phi.plot(ax=ax, nmin=npmin, label=labels[i])
        if lf_lims:
            ax.axis(lf_lims)
        ax.semilogy(basey=10)
        label = r'${\cal M}' + r'{}\ [{}, {}]$'.format(
                ibin+1, mbins_def[ibin], mbins_def[ibin+1])
        ax.text(0.9, 0.9, label, ha='right', transform=ax.transAxes)
        if ibin == 0:
#            ax.legend(loc=loc)
            ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncol=len(infiles), handlelength=2, borderaxespad=0.)
    plt.draw()
    if plot_file:
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


# def csmf_plots(infile='smf_incomp_vmax.pkl', mock_file='tng_csmf.pkl',
#                schec_fn=SchecMass(), gauss_fn=Gauss1D(), schecdbl=False,
#                schec_p0=[[-1, -2, 1], [10.5, 9, 12], [-2, -8, 0]],
#                gauss_p0=[[0.5, 0.1, 1], [11, 10.4, 12], [1e-3, 1e-5, 1e-1]],
#                Mmin_fit=8.0, Mmax_fit=12.0,
#                nmin=2, lf_lims=(8.2, 12.2, 2e-7, 0.5),
#                schec_lims=(-1.6, 0.1, 9.9, 11.2),
#                tab_file='csmf_schec.tex', mock_tab_file='tng_csmf_schec.tex',
#                cen_sat_plot='csmf_cen_sat.pdf',
#                colour_plot='csmf_colour.pdf',
#                sersic_plot='csmf_sersic.pdf',
#                schec_plot='csmf_schec.pdf',
#                xlabel=ms_label, ylabel=smf_label, lc_step=32, zand_comp=False):
#     clf_plots(infile, mock_file, schec_fn=schec_fn, gauss_fn=gauss_fn,
#               schec_p0=schec_p0,
#               gauss_p0=gauss_p0, Mmin_fit=Mmin_fit,
#               Mmax_fit=Mmax_fit, nmin=nmin, lf_lims=lf_lims,
#               schec_lims=schec_lims,
#               tab_file=tab_file, mock_tab_file=mock_tab_file,
#               cen_sat_plot=cen_sat_plot, colour_plot=colour_plot,
#               sersic_plot=sersic_plot,
#               schec_plot=schec_plot, xlabel=xlabel, ylabel=ylabel,
#               lc_step=lc_step, zand_comp=zand_comp)


# def clf_plots(infile='lf_vmax.pkl', mock_file='lf_mockfof.pkl',
#               schec_fn=SchecMag(), gauss_fn=Gauss1D(), schecdbl=False,
#               schec_p0=[[-1.2, -2, 1], [-20.2, -22, -19], [-2, -8, 0]],
#               gauss_p0=[[1, 0.1, 2], [-21, -23, -19], [1e-4, 1e-6, 1e-1]],
#               Mmin_fit=-24, Mmax_fit=-17,
#               nmin=2, lf_lims=(-15.5, -23.9, 2e-7, 0.2),
#               schec_lims=(-1.9, 0.8, -19.4, -21.9),
#               tab_file='clf_schec.tex', mock_tab_file='mock_clf_schec.tex',
#               cen_sat_plot='clf_cen_sat.pdf',
#               colour_plot='clf_colour.pdf',
#               sersic_plot='clf_sersic.pdf',
#               schec_plot='clf_schec.pdf', xlabel=mag_label, ylabel=lf_label,
#               sigma=[1, ], lc_limits=5, lc_step=32, zand_comp=True,
#               mmin=12.4, mmax=14.3):
#     """Plot and tabulate galaxy CLFs by cen/sat, colour and Sersic index."""

#     # For Yang et al. comparison
#     mbins = np.linspace(min(lf_lims[:2]), max(lf_lims[:2]), 50)
#     if 'clf' in infile:
#         Mr_sun = 4.76
#         lbins = 0.4*(Mr_sun - mbins)
#     if zand_comp:
#         zand_schec = zandivarez()
#     lf_dict = pickle.load(open(infile, 'rb'))
#     if mock_file:
#         mock_dict = pickle.load(open(mock_file, 'rb'))
#         lf_dict = {**lf_dict, **mock_dict}
#         labels = ['mock', '', 'b', 'r', 'nlo', 'nhi']
#         descriptions = ['Mock', 'All', 'Blue', 'Red', 'low-n', 'high-n']
#     else:
#         labels = ['', 'b', 'r', 'nlo', 'nhi']
#         descriptions = ['All', 'Blue', 'Red', 'low-n', 'high-n']
#     if 'smf' in infile:
#         mstar = r'\lg\ {\cal M}^*'
#         mstard = r'$\lg\ {\cal M}^*$'
#     else:
#         mstar = r'M^*'
#         mstard = r'$M^*$'

#     f1 = schec_fn
# #    f1 = SchecMag('schec1')
#     f1.alpha = schec_p0[0][0]
#     f1.alpha.min = schec_p0[0][1]
#     f1.alpha.max = schec_p0[0][2]
#     f1.mstar = schec_p0[1][0]
#     f1.Mstar.min = schec_p0[1][1]
#     f1.Mstar.max = schec_p0[1][2]
#     f1.lgps = schec_p0[2][0]
#     f1.lgps.min = schec_p0[2][1]
#     f1.lgps.max = schec_p0[2][2]
#     if schecdbl:
#         f2 = SchecMag('schec2')
#         sep = Const1D('sep')
#         f2.alpha = f1.alpha + sep.c0
#         sep.c0 = -1.5
#         sep.c0.min = -5
#         sep.c0.max = 0
#         f2.mstar = f1.mstar
#         f2.lgps = schec_p0[2][0]
#         schec_fn = f1 + f2 + 0*sep
#     else:
#         schec_fn = f1

# #    schec_fn.alpha = schec_p0[0][0]
# #    schec_fn.alpha.min = schec_p0[0][1]
# #    schec_fn.alpha.max = schec_p0[0][2]
# #    schec_fn.Mstar = schec_p0[1][0]
# #    schec_fn.Mstar.min = schec_p0[1][1]
# #    schec_fn.Mstar.max = schec_p0[1][2]
# #    schec_fn.lgps = schec_p0[2][0]
# #    schec_fn.lgps.min = schec_p0[2][1]
# #    schec_fn.lgps.max = schec_p0[2][2]

#     gauss_fn.fwhm = gauss_p0[0][0]
#     gauss_fn.fwhm.min = gauss_p0[0][1]
#     gauss_fn.fwhm.max = gauss_p0[0][2]
#     gauss_fn.pos = gauss_p0[1][0]
#     gauss_fn.pos.min = gauss_p0[1][1]
#     gauss_fn.pos.max = gauss_p0[1][2]
#     gauss_fn.ampl = gauss_p0[2][0]
#     gauss_fn.ampl.min = gauss_p0[2][1]
#     gauss_fn.ampl.max = gauss_p0[2][2]

#     plot_size = (6, 7.5)
#     sa_left = 0.15
#     sa_bot = 0.05
#     nbin = 6
#     nrow, ncol = util.two_factors(nbin)

# #   Tabulate Schechter parameter fits
#     labels = ['', 'cen', 'sat', 'b', 'r', 'nlo', 'nhi']
#     descriptions = ['All', 'Central', 'Satellite', 'Blue', 'Red',
#                     'low-n', 'high-n']
#     with open(plot_dir + tab_file, 'w') as f:
#         print(r"""
# \begin{math}
# \begin{array}{crccc}
# \hline
#  & N_{\rm gal} & \alpha &""", mstar, r"""& \chi^2/\nu \\[0pt]
# """, file=f)
#         for lbl, desc in zip(labels, descriptions):
#             print(r"""
# \hline
# \multicolumn{5}{c}{\mbox{""", desc, r"""}} \\
# """, file=f)
#             for i in range(6):
#                 key = 'M{}'.format(i) + lbl
#                 phi = lf_dict[key]
#                 if lbl == 'cen':
#                     fn = phi.fn_fit(gauss_fn, Mmin=Mmin_fit, Mmax=Mmax_fit,
#                                     verbose=0)
#                 else:
#                     fn = phi.fn_fit(schec_fn, Mmin=Mmin_fit, Mmax=Mmax_fit,
#                                     verbose=0)
#                 lf_dict[key] = copy.deepcopy(phi)
#                 try:
#                     fit_errs = 0.5*(np.array(phi.errors.parmaxes) -
#                                              np.array(phi.errors.parmins))
#                     print(r'\mass{} & {} & {:5.2f}\pm{:5.2f} & {:5.2f}\pm{:5.2f} & {:5.2f}/{:2d} \\[0pt]'.format(
#                             i+1, int(phi.ngal.sum()), phi.res.parvals[0], fit_errs[0],
#                             phi.res.parvals[1], fit_errs[1], phi.res.statval, phi.res.dof), file=f)
#                 except:
#     #            except TypeError or AttributeError:
#                     print('bad error estimate')                    
#                     pdb.set_trace()
#         print(r"""
# \hline
# \end{array}
# \end{math}""", file=f)

#     #   Tabulate Schechter parameter fits for mocks
#     if mock_file:
#         glabels = ['', 'cen', 'sat']
#         mlabels = ['mock', 'mockcen', 'mocksat']
#         mdescriptions = ['Mock', 'Mock central', 'Mock satellite']
#         with open(plot_dir + mock_tab_file, 'w') as f:
#             print(r"""
# \begin{math}
# \begin{array}{crccc}
# \hline
#  & N_{\rm gal} & \alpha & M^* & \chi^2/\nu \\[0pt]
# """, file=f)
#             for glbl, lbl, desc in zip(glabels, mlabels, mdescriptions):
#                 print(r"""
# \hline
# \multicolumn{5}{c}{\mbox{""", desc, r"""}} \\
# """, file=f)
#                 for i in range(6):
#                     key = 'M{}'.format(i) + lbl
#                     phi = lf_dict[key]
# #                    phig = lf_dict['M{}'.format(i) + glbl]
# #                    c, nu, p = phi.chi2(phig)
#                     if lbl == 'mockcen':
#                         fn = phi.fn_fit(gauss_fn, Mmin=Mmin_fit, Mmax=Mmax_fit,
#                                         verbose=0)
#                     else:
#                         fn = phi.fn_fit(schec_fn, Mmin=Mmin_fit, Mmax=Mmax_fit,
#                                         verbose=0)
#                     lf_dict[key] = copy.deepcopy(phi)
#                     print(r'\mass{} & {} & {:5.2f}\pm{:5.2f} & {:5.2f}\pm{:5.2f} & {:5.2f}/{:2d}  \\[0pt]'.format(
#                             i+1, int(phi.ngal.sum()), phi.res.parvals[0], fit_errs[0],
#                             phi.res.parvals[1], fit_errs[1], phi.res.statval, phi.res.dof), file=f)
#     #                print(r'\mass{} & {} & {:5.2f}\pm{:5.2f} & {:5.2f}\pm{:5.2f} & {:5.2f}/{:2d} \\[0pt]'.format(
#     #                        i+1, int(phi.ngal.sum()), phi.fit_par[0], phi.fit_err[0],
#     #                        phi.fit_par[1], phi.fit_err[1], phi.chi2, phi.ndof), file=f)
#             print(r"""
# \hline
# \end{array}
# \end{math}""", file=f)

# #   CLF by central/satellite, including mocks and Yang+ comparison
# #   (plot first, so they don't overwrite GAMA results)
#     plt.clf()
#     fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
#     fig.set_size_inches(plot_size)
#     fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
#     fig.text(0.5, 0.0, xlabel, ha='center', va='center')
#     fig.text(0.06, 0.5, ylabel, ha='center', va='center', rotation='vertical')
#     plt.semilogy(basey=10, nonposy='clip')
#     for i in range(nbin):
#         key = 'M{}'.format(i)
#         label = r'${\cal M}' + r'{}\ [{}, {}]$'.format(
#                 i+1, mbins_def[i], mbins_def[i+1])
#         ax = axes.flat[i]
#         ax.axis(lf_lims)
#         ax.text(0.9, 0.9, label, ha='right', transform=ax.transAxes)
# #        if 'clf' in infile:
# #            yang_file = 'clf_all.txt'
# #            yang_cen, yang_sat = yang(yang_file, halomass[i], lbins)
# #        else:
# #            yang_file = 'csmf_all.txt'
# #            yang_cen, yang_sat = yang(yang_file, halomass[i], mbins)
# #        ax.plot(mbins, yang_cen, 'r--')
# #        ax.plot(mbins, yang_sat, 'b--')
#         if mock_file:
#             phi = lf_dict[key + 'mock']
#             phi.plot(ax=ax, nmin=nmin, ls=':', clr='k', mfc='w')
#         phi = lf_dict[key]
#         phi.plot(ax=ax, nmin=nmin, clr='k', label='all')
# #        pdb.set_trace()
#         for cs, colour, label in zip(['cen', 'sat'], 'rb',
#                                      ['central', 'satellite']):
#             if mock_file:
#                 phi = lf_dict[key + 'mock' + cs]
#                 phi.plot(ax=ax, nmin=nmin, ls=':', clr=colour, mfc='w')
#             phi = lf_dict[key + cs]
#             phi.plot(ax=ax, nmin=nmin, clr=colour, label=label)
#         if i == 4:
#             ax.legend(loc=3)
#     plt.draw()
#     plt.savefig(plot_dir + cen_sat_plot, bbox_inches='tight')
#     plt.show()

# #   LF by mass/colour
#     plt.clf()
#     fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
#     fig.set_size_inches(plot_size)
#     fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
#     fig.text(0.5, 0.0, xlabel, ha='center', va='center')
#     fig.text(0.06, 0.5, ylabel, ha='center', va='center', rotation='vertical')
#     plt.semilogy(basey=10, nonposy='clip')
#     for i in range(nbin):
#         key = 'M{}'.format(i)
#         phi = lf_dict[key]
#         label = r'${\cal M}' + r'{}\ [{}, {}]$'.format(
#                 i+1, mbins_def[i], mbins_def[i+1])
#         ax = axes.flat[i]
#         ax.axis(lf_lims)
#         ax.text(0.9, 0.9, label, ha='right', transform=ax.transAxes)
#         phi.plot(ax=ax, nmin=nmin, clr='k', label='all')
#         for colour, label in zip('br', ['blue', 'red']):
# #            if 'clf' in infile:
# #                yang_file = 'clf_{}.txt'.format(label)
# #                yang_cen, yang_sat = yang(yang_file, halomass[i], lbins)
# #            else:
# #                yang_file = 'csmf_{}.txt'.format(label)
# #                yang_cen, yang_sat = yang(yang_file, halomass[i], mbins)
# #            # Scale central CLF/CSMF my blue or red fraction
# #            if colour == 'b':
# #                yang_cen *= (1 - crf[i])
# #            else:
# #                yang_cen *= crf[i]
# #            ax.plot(mbins, yang_cen, colour+'--')
# #            ax.plot(mbins, yang_sat, colour+'--')
#             phi = lf_dict[key + colour]
#             phi.plot(ax=ax, nmin=nmin, clr=colour, label=label)
# #        if mock_file:
# #            phi = lf_dict[key + 'mock']
# #            phi.plot(ax=ax, nmin=nmin, clr='g', label='Mock')
#         if i == 4:
#             ax.legend(loc=3)
#     plt.draw()
#     plt.savefig(plot_dir + colour_plot, bbox_inches='tight')
#     plt.show()

# #   LF by mass/sersic index
#     plt.clf()
#     fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
#     fig.set_size_inches(plot_size)
#     fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
#     fig.text(0.55, 0.0, xlabel, ha='center', va='center')
#     fig.text(0.06, 0.5, ylabel, ha='center', va='center', rotation='vertical')
#     plt.semilogy(basey=10, nonposy='clip')
#     for i in range(nbin):
#         key = 'M{}'.format(i)
#         phi = lf_dict[key]
#         label = r'${\cal M}' + r'{}\ [{}, {}]$'.format(
#                 i+1, mbins_def[i], mbins_def[i+1])
#         ax = axes.flat[i]
#         ax.axis(lf_lims)
#         ax.text(0.9, 0.9, label, ha='right', transform=ax.transAxes)
#         phi.plot(ax=ax, nmin=nmin, clr='k', label='all')
#         for sersic, colour, label in zip(['nlo', 'nhi'], 'br',
#                                          ['low-n', 'high-n']):
#             phi = lf_dict[key + sersic]
#             phi.plot(ax=ax, nmin=nmin, clr=colour, label=label)
#         if i == 4:
#             ax.legend(loc=3)
#     plt.draw()
#     plt.savefig(plot_dir + sersic_plot, bbox_inches='tight')
#     plt.show()

# #   Schechter parameters - mass panels
# #    plt.clf()
# #    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
# #    fig.set_size_inches(plot_size)
# #    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
# #    fig.text(0.55, 0.0, r'$\alpha$', ha='center', va='center')
# #    fig.text(0.06, 0.5, r'$M^*$', ha='center', va='center', rotation='vertical')
# #    for i in range(nbin):
# #        key = 'M{}'.format(i)
# #        phi = lf_dict[key]
# #        mlo = phi.sel_dict['log_mass'][0]
# #        mhi = phi.sel_dict['log_mass'][1]
# #        label = r'$M_h = [{}, {}]$'.format(mlo, mhi)
# #        ax = axes.flat[i]
# #        ax.text(0.1, 0.9, label, transform=ax.transAxes)
# #        phi.like_cont(ax=ax, label=label, c='k')
# #        for colour in 'br':
# #            phi = lf_dict[key + colour]
# #            phi.like_cont(ax=ax, label=label, c=colour)
# #        for sersic, colour in zip(['nlo', 'nhi'], 'br'):
# #            phi = lf_dict[key + sersic]
# #            phi.like_cont(ax=ax, label=label, c=colour, ls='--')
# #    plt.axis((-1.7, -0.3, -21.5, -18.5))
# ##    plt.legend(label_list)
# #    plt.show()


# #   Schechter parameters - type panels
#     plt.clf()
#     fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, num=1)
#     fig.set_size_inches(plot_size)
#     fig.subplots_adjust(left=sa_left, bottom=sa_bot, top=0.85,
#                         hspace=0.0, wspace=0.0)
#     fig.text(0.52, 0.0, r'$\alpha$', ha='center', va='center')
#     fig.text(0.06, 0.45, mstard, ha='center', va='center', rotation='vertical')
#     lines = []

#     # Set the colormap and norm to correspond to the data for which
#     # the colorbar will be used.
#     cmap = mpl.cm.viridis
#     norm = mpl.colors.Normalize(vmin=mmin, vmax=mmax)
#     scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
# #    colors = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
# #              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan')
#     iy = 0
#     labels = ['', 'mock', 'b', 'r', 'nlo', 'nhi']
#     descriptions = ['All', 'Mock', 'Blue', 'Red', 'low-n', 'high-n']
#     zand_samp = ['all', None, 'blue', 'red', 'late', 'early']
# #    if mock_file:
# #        fig.text(0.06, 0.5, r'$M^*$', ha='center', va='center',
# #                 rotation='vertical')
# #    else:
# #        fig.text(0.06, 0.5, r'$\lg\ {\cal M}^*$', ha='center', va='center',
# #                 rotation='vertical')
# #        labels = ['', 'b', 'r', 'nlo', 'nhi']
# #        descriptions = ['All', 'Blue', 'Red', 'low-n', 'high-n']
#     for lbl, desc in zip(labels, descriptions):
#         ax = axes.flat[iy]
#         ax.axis(schec_lims)
#         if mock_file or lbl != 'mock':
#             ax.text(0.9, 0.9, desc, transform=ax.transAxes, ha='right')
#             for i in range(nbin):
#                 key = 'M{}'.format(i) + lbl
#                 phi = lf_dict[key]
#                 label = r'${\cal M}' + r'{}$'.format(i+1)
#     #            clr = scalarMap.to_rgba(
#     #                    np.clip((lgm_av[i] - mmin) / (mmax - mmin), 0, 1))
#                 clr = scalarMap.to_rgba(lgm_av[i])
#                 rproj = RegionProjection()
#                 rproj.prepare(nloop=(lc_step, lc_step), fac=lc_limits,
#                               sigma=sigma)
#                 rproj.calc(phi.fit, phi.fn.alpha, phi.fn.Mstar)
#     #                rproj_dict[key] = rproj
#                 ax.plot(rproj.parval0, rproj.parval1, '+', color=clr)
#                 xmin, xmax = rproj.min[0], rproj.max[0]
#                 ymin, ymax = rproj.min[1], rproj.max[1]
#                 nx, ny = rproj.nloop
#     #                hx = 0.5 * (xmax - xmin) / (nx - 1)
#     #                hy = 0.5 * (ymax - ymin) / (ny - 1)
#                 extent = (xmin, xmax, ymin, ymax)
#                 y = rproj.y.reshape((ny, nx))
#                 v = rproj.levels
#     #            pdb.set_trace()
#                 sc = ax.contour(y, v, origin='lower', extent=extent,
#                                 colors=(clr,))
#     #            phi.like_cont(ax=ax, label=label, lc_limits=schec_lims)
#     #            phi.like_cont(ax=ax, label=label, lc_limits=lc_limits,
#     #                          lc_step=lc_step, dchisq=dchisq)
#                 if iy == 0:
#     #                lines.append(mpatches.Patch(color=colors[i], label=label))
#                     lines.append(mlines.Line2D([], [], color=clr, label=label))
#             if zand_comp and zand_samp[iy]:
#                 schec = zand_schec[zand_samp[iy]]
#                 for i in range(len(schec['mass'])):
#     #                clr = scalarMap.to_rgba(
#     #                        np.clip((schec['mass'][i] - mmin) / (mmax - mmin), 0, 1))
#                     clr = scalarMap.to_rgba(schec['mass'][i])
#                     ax.errorbar(schec['alpha'][i], schec['mstar'][i],
#                                 xerr=schec['alpha_err'][i],
#                                 yerr=schec['mstar_err'][i],
#                                 fmt='none', color=clr)
#             iy += 1
# #    ax = axes.flat[iy]
# #    pdb.set_trace()
# #    plt.legend(handles=lines, loc='center', frameon=False)
# #    ax = axes.flat[0]
# #    ax.legend(handles=lines, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
# #               ncol=3, handlelength=2, borderaxespad=0.)
# #    ax.legend(handles=lines, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
# #              ncol=6, handlelength=1, borderaxespad=0.)
# #    fig.subplots_adjust(top=0.93)
#     cbar_ax = fig.add_axes([0.15, 0.89, 0.75, 0.02])
# #    cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
# #    fig.colorbar(sc, ax=axes)
# #    cb.set_label('Redshift')
# #    cbar_ax.set_title('Redshift')
# #    cbar_ax.set_title(r'$\lg {\cal M}_h$')
#     cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
#                                    orientation='horizontal')
# #    cb.set_label(r'$\lg {\cal M}_h$')
#     cbar_ax.set_title(r'$\lg {\cal M}_h$')
#     plt.draw()
#     plt.savefig(plot_dir + schec_plot, bbox_inches='tight')
#     plt.show()


# def csmf_plots2(infile='smf_incomp_vmax.pkl', mock_file=None,
#                 schec_fn=SchecMassSq(), gauss_fn=LogNormal(), schecdbl=False,
#                 schec_p0=[[10.5, 9, 12], [-1.2, -2, 1], [-2, -8, 0]],
#                 gauss_p0=[[10.5, 9, 12], [0.5, 0.1, 0.8], [-4, -7, -2]],
#                 Mmin_fit=8.0, Mmax_fit=12.5,
#                 nmin=2, lf_lims=(8.2, 12.2, 5e-8, 0.02),
#                 schec_lims=(-1.7, -0.2, 10.3, 11.05),
#                 gauss_lims=(0.05, 0.45, 10.0, 11.9),
# #                gauss_lims=(-7, -2.7, 10.2, 11.3),
#                 yang_pref='csmf_', tng_file='TNG300-1_84_csmf.pkl',
#                 eagle_file='eagle_csmf.pkl', Lgal_file='smf_Lgal.pkl',
#                 tab_gauss='csmf_gauss.tex', tab_schec='csmf_schec.tex',
#                 colour_plot='csmf_colour.pdf', sersic_plot='csmf_sersic.pdf',
#                 lf_plot='csmf.pdf', sim_plot=None,
#                 gauss_par_plot='csmf_gauss.pdf', schec_par_plot='csmf_schec.pdf',
#                 xlabel=ms_label, ylabel=smf_label,
#                 sigma=[1, ], lc_limits=5, lc_step=32,
#                 plot_size=(6, 7.5), plot_shape=(4, 2),
#                 red_frac_file='red_frac.pkl', main_plot_size=(12, 5.2),
#                 contour_plot_size=(6, 9.5)):
#     """Plot and tabulate galaxy CSMFs by colour and Sersic index,
#     separately for central/satellite."""
#     clf_plots2(infile=infile, mock_file=mock_file,
#                schec_fn=schec_fn, gauss_fn=gauss_fn, schecdbl=schecdbl,
#                schec_p0=schec_p0, gauss_p0=gauss_p0,
#                Mmin_fit=Mmin_fit, Mmax_fit=Mmax_fit,
#                nmin=nmin, lf_lims=lf_lims,
#                schec_lims=schec_lims, gauss_lims=gauss_lims,
#                yang_pref=yang_pref, tng_file=tng_file, eagle_file=eagle_file,
#                Lgal_file=Lgal_file, tab_gauss=tab_gauss, tab_schec=tab_schec,
#                colour_plot=colour_plot, sersic_plot=sersic_plot,
#                lf_plot=lf_plot, sim_plot=sim_plot,
#                gauss_par_plot=gauss_par_plot, schec_par_plot=schec_par_plot,
#                xlabel=xlabel, ylabel=ylabel,
#                sigma=sigma, lc_limits=lc_limits, lc_step=lc_step,
#                plot_size=plot_size,
#                plot_shape=plot_shape, red_frac_file=red_frac_file,
#                main_plot_size=main_plot_size,
#                contour_plot_size=contour_plot_size)


# def clf_plots2(infile='lf_vmax.pkl', mock_file='lf_mockfof.pkl',
#                schec_fn=SchecMagSq(), gauss_fn=LogNormal(), schecdbl=False,
#                schec_p0=[[-20.2, -22, -19], [-1.2, -2, 1], [-2, -8, 0]],
#                gauss_p0=[[-21.5, -23, -20], [0.5, 0.1, 0.8], [-4, -7, -2]],
#                Mmin_fit=-24, Mmax_fit=-16,
#                nmin=2, lf_lims=(-15.5, -23.9, 5e-8, 0.02),
#                schec_lims=(-1.9, -0.2, -20.2, -22.4),
#                gauss_lims=(0.3, 0.68, -20, -22.7),
# #               gauss_lims=(-7, -2.7, -19.9, -22.7),
#                yang_pref='clf_', tng_file=None, eagle_file=None, Lgal_file=None,
#                tab_gauss='clf_gauss.tex', tab_schec='clf_schec.tex',
#                colour_plot='clf_colour.pdf',
#                sersic_plot='clf_sersic.pdf', lf_plot='clf.pdf', sim_plot=None,
#                gauss_par_plot='clf_gauss.pdf', schec_par_plot='clf_schec.pdf',
#                xlabel=mag_label, ylabel=lf_label,
#                sigma=[1, ], lc_limits=5, lc_step=32, alpha=0.8,
# #               mmin=12.4, mmax=14.3,
#                mmin=13.0, mmax=14.1,
#                plot_size=(6, 7.5), plot_shape=(3, 2),
#                red_frac_file=None, main_plot_size=(12, 7),
#                contour_plot_size=(6, 7.5)):
#     """Plot and tabulate galaxy CLFs by colour and Sersic index,
#     separately for central/satellite."""

# #    # For Yang et al. comparison
# #    mbins = np.linspace(min(lf_lims[:2]), max(lf_lims[:2]), 50)
# #    if 'clf' in infile:
# #        Mr_sun = 4.76
# #        lbins = 0.4*(Mr_sun - mbins)
#     lf_dict = pickle.load(open(infile, 'rb'))
#     labels = []
#     descriptions = []
#     if mock_file:
#         mock_dict = pickle.load(open(mock_file, 'rb'))
#         lf_dict = {**lf_dict, **mock_dict}
#         labels.append('mock')
#         descriptions.append('Mock')
#     if Lgal_file:
#         mock_dict = pickle.load(open(Lgal_file, 'rb'))
#         lf_dict = {**lf_dict, **mock_dict}
#         labels.append('Lgal')
#         descriptions.append('L-Galaxies')
#     if tng_file:
#         mock_dict = pickle.load(open(tng_file, 'rb'))
#         lf_dict = {**lf_dict, **mock_dict}
#         labels.append('tng')
#         descriptions.append('TNG')
#     if eagle_file:
#         mock_dict = pickle.load(open(eagle_file, 'rb'))
#         lf_dict = {**lf_dict, **mock_dict}
#         labels.append('eagle')
#         descriptions.append('EAGLE')
#     labels += ['', 'b', 'r', 'nlo', 'nhi']
#     descriptions += ['All', 'Blue', 'Red', 'low-n', 'high-n']
#     if 'smf' in infile:
#         mstar = r'\lg\ {\cal M}^*'
#         mstard = r'$\lg\ {\cal M}^*$'
#     else:
#         mstar = r'M^*'
#         mstard = r'$M^*$'

#     f1 = schec_fn
# #    f1 = SchecMag('schec1')
#     f1.mstar = schec_p0[0][0]
#     f1.Mstar.min = schec_p0[0][1]
#     f1.Mstar.max = schec_p0[0][2]
#     f1.alpha = schec_p0[1][0]
#     f1.alpha.min = schec_p0[1][1]
#     f1.alpha.max = schec_p0[1][2]
#     f1.lgps = schec_p0[2][0]
#     f1.lgps.min = schec_p0[2][1]
#     f1.lgps.max = schec_p0[2][2]
#     if schecdbl:
#         f2 = SchecMag('schec2')
#         sep = Const1D('sep')
#         f2.alpha = f1.alpha + sep.c0
#         sep.c0 = -1.5
#         sep.c0.min = -5
#         sep.c0.max = 0
#         f2.mstar = f1.mstar
#         f2.lgps = schec_p0[2][0]
#         schec_fn = f1 + f2 + 0*sep
#     else:
#         schec_fn = f1

#     gauss_fn.M_c = gauss_p0[0][0]
#     gauss_fn.M_c.min = gauss_p0[0][1]
#     gauss_fn.M_c.max = gauss_p0[0][2]
#     gauss_fn.sigma_c = gauss_p0[1][0]
#     gauss_fn.sigma_c.min = gauss_p0[1][1]
#     gauss_fn.sigma_c.max = gauss_p0[1][2]
#     gauss_fn.lgps = gauss_p0[2][0]
#     gauss_fn.lgps.min = gauss_p0[2][1]
#     gauss_fn.lgps.max = gauss_p0[2][2]

# #    plot_size = (6, 7.5)
# #    plot_size = (5, 20)
#     sa_left = 0.15
#     sa_bot = 0.05
#     nbin = len(mbins_def) - 1
# #    nrow, ncol = util.two_factors(nbin)
#     nrow = len(labels)
#     ncol = 1
# #   Tabulate parameter fit values for centrals and satellites
# #    if mock_file:
# #        labels = ['mock', '', 'b', 'r', 'nlo', 'nhi']
# #        descriptions = ['Mock', 'All', 'Blue', 'Red', 'low-n', 'high-n']
# #    else:
# #        labels = ['', 'b', 'r', 'nlo', 'nhi']
# #        descriptions = ['All', 'Blue', 'Red', 'low-n', 'high-n']
#     pars = ((r'M_c', r'\sigma_c', r'\lg \phi*_c'),
#             (r'M^*', r'\alpha', r'\lg \phi*_s'))
#     for cslbl, tab_file, par in zip(('cen', 'sat'), (tab_gauss, tab_schec), pars):
#         with open(plot_dir + tab_file, 'w') as f:
#             print(r"""
# \begin{math}
# \begin{array}{crcccc}
# \hline
#  & N_{\rm gal} &""", par[0], """ & """, par[1], """ & """, par[2],
#  r"""& \chi^2/\nu \\[0pt]
# """, file=f)
#             for lbl, desc in zip(labels, descriptions):
#                 if desc:
#                     print(r"""
# \hline
# \multicolumn{5}{c}{\mbox{""", desc, r"""}} \\
# """, file=f)
#                     for i in range(nbin):
#                         if lbl in ('mock', 'tng', 'eagle', 'Lgal'):
#                             key = 'M{}'.format(i) + lbl + cslbl
#                         else:
#                             # key = 'M{}'.format(i) + cslbl + lbl
#                             key = 'M{}'.format(i) + '_' + cslbl + '_all'
#                         phi = lf_dict[key]
#                         if cslbl == 'cen':
#                             fn = phi.fn_fit(gauss_fn, Mmin=Mmin_fit, Mmax=Mmax_fit,
#                                             verbose=0)
#                         else:
#                             fn = phi.fn_fit(schec_fn, Mmin=Mmin_fit, Mmax=Mmax_fit,
#                                             verbose=0)
#                         lf_dict[key] = copy.deepcopy(phi)
#                         try:
#                             fit_errs = 0.5*(np.array(phi.errors.parmaxes) -
#                                             np.array(phi.errors.parmins))
#                         except:
#     #                    except AttributeError or TypeError:
#                             fit_errs = (9.99, 9.99, 9.99)
#     #                        print('bad error estimate')
#     #                        pdb.set_trace()
    
#                         print(r'\mass{} & {} & {:5.2f}\pm{:5.2f} & {:5.2f}\pm{:5.2f} & {:5.2f}\pm{:5.2f} & {:5.2f}/{:2d} \\[0pt]'.format(
#                             i+1, int(phi.ngal.sum()),
#                             phi.res.parvals[0], fit_errs[0],
#                             phi.res.parvals[1], fit_errs[1],
#                             phi.res.parvals[2], fit_errs[2],
#                             phi.res.statval, phi.res.dof), file=f)
#             print(r"""
# \hline
# \end{array}
# \end{math}""", file=f)

#     # # Red fraction
#     # if red_frac_file:
#     #     Mfit = np.linspace(6, 12, 100)
#     #     plt.clf()
#     #     fig, axes = plt.subplots(6, 1, sharex=True, sharey=True, num=1)
#     #     fig.set_size_inches(4, 12)
#     #     fig.subplots_adjust(left=sa_left, bottom=sa_bot, top=0.85,
#     #                         hspace=0.0, wspace=0.0)
#     #     for i in range(nbin):
#     #         ax = axes.flat[i]
#     #         ax.text(0.9, 0.9, f'M{i}', transform=ax.transAxes, ha='right')
#     #         # phir = lf_dict[f'M{i}allr']
#     #         # phib = lf_dict[f'M{i}allb']
#     #         phir = lf_dict[f'M{i}_all_red']
#     #         phib = lf_dict[f'M{i}_all_blue']
#     #         sel = phir.comp * phib.comp
#     #         Mbin = phir.Mbin[sel]
#     #         rf = phir.phi[sel]/(phir.phi[sel] + phib.phi[sel])
#     #         rf_err = (rf**2 * ((phir.phi_err[sel]/phir.phi[sel])**2 +
#     #                            (phib.phi_err[sel]/phib.phi[sel])**2))**0.5
#     #         ax.errorbar(Mbin, rf, rf_err, label=f'M{i}')
#     #         # phicenr = lf_dict[f'M{i}cenr']
#     #         # phicenb = lf_dict[f'M{i}cenb']
#     #         # phisatr = lf_dict[f'M{i}satr']
#     #         # phisatb = lf_dict[f'M{i}satb']
#     #         phicenr = lf_dict[f'M{i}_cen_red']
#     #         phicenb = lf_dict[f'M{i}_cen_blue']
#     #         phisatr = lf_dict[f'M{i}_sat_red']
#     #         phisatb = lf_dict[f'M{i}_sat_blue']
#     #         yr = phicenr.fn(Mfit) + phisatr.fn(Mfit)
#     #         yb = phicenb.fn(Mfit) + phisatb.fn(Mfit)
#     #         rf = yr / (yr + yb)
#     #         ax.plot(Mfit, rf)
#     #     pickle.dump((Mfit, rf), open(red_frac_file, 'wb'))
#     # #    plt.legend()
#     #     plt.xlabel(r'log $M^*/M_\odot$')
#     #     plt.ylabel('Red fraction')
#     #     plt.show()

# #   Hydro sim SMFs as a separate plot
#     if sim_plot:
#         plt.clf()
#         fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, num=1)
#         fig.set_size_inches(6, 5)
#         fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
#         fig.text(0.5, 0.0, xlabel, ha='center', va='center')
#         fig.text(0.06, 0.5, csmf_label, ha='center', va='center',
#                  rotation='vertical')
#         plt.semilogy(basey=10, nonposy='clip')
#         ax.set_ylim(1e-7, 1e-2)
#         for i in range(nbin):
#             key = 'M{}'.format(i)
#             mlbl = r'${\cal M}' + r'{}\ [{}, {}]$'.format(
#                     i+1, mbins_def[i], mbins_def[i+1])
# #            ax.text(0.9, 0.9, mlbl, ha='right', transform=ax.transAxes)
#             clr = next(ax._get_lines.prop_cycler)['color']
#             for cs in ('cen', 'sat'):
#                 phi = lf_dict[key + 'tng' + cs]
#                 if cs == 'cen':
#                     phi.plot(ax=ax, nmin=nmin, clr=clr, label=mlbl, alpha=1)
#                 else:
#                     phi.plot(ax=ax, nmin=nmin, ls='--', clr=clr, mfc='w', alpha=1)
#         ax.legend(loc=3)
#         plt.draw()
#         plt.savefig(plot_dir + sim_plot, bbox_inches='tight')
#         plt.show()

# # #   LF in mass bins by type
# #     typess = (('', 'b', 'r'), ('', 'nlo', 'nhi'))
# #     labelss = (('all', 'blue', 'red'), ('all', 'low-n', 'high-n'))
# #     plot_files = (colour_plot, sersic_plot)
# #     for types, lbls, plot_file in zip(typess, labelss, plot_files):
# #         plt.clf()
# #         fig, axes = plt.subplots(*plot_shape, sharex=True, sharey=True, num=1)
# #         fig.set_size_inches(plot_size)
# #         fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
# #         fig.text(0.5, 0.0, xlabel, ha='center', va='center')
# #         fig.text(0.06, 0.5, ylabel, ha='center', va='center', rotation='vertical')
# #         plt.semilogy(basey=10, nonposy='clip')
# #         for i in range(nbin):
# #             key = 'M{}'.format(i)
# #             mlbl = r'${\cal M}' + r'{}\ [{}, {}]$'.format(
# #                     i+1, mbins_def[i], mbins_def[i+1])
# #             ax = axes.flat[i]
# #             ax.axis(lf_lims)
# #             ax.text(0.9, 0.9, mlbl, ha='right', transform=ax.transAxes)
# #             for cs in ('cen', 'sat'):
# #                 if mock_file or tng_file or eagle_file:
# #                     phi = lf_dict[key + labels[0] + cs]
# #                     if cs == 'cen':
# #                         phi.plot(ax=ax, nmin=nmin, clr='g',
# #                                  label=descriptions[0], alpha=alpha)
# #                     else:
# #                         phi.plot(ax=ax, nmin=nmin, ls='--', clr='g', mfc='w', alpha=alpha)
# #                 for typ, clr, lbl in zip(types, 'kbr', lbls):
# #                     phi = lf_dict[key + cs + typ]
# #                     if cs == 'cen':
# #                         phi.plot(ax=ax, nmin=nmin, clr=clr, label=lbl, alpha=alpha)
# #                     else:
# #                         phi.plot(ax=ax, nmin=nmin, ls='--', clr=clr, mfc='w', alpha=alpha)
# #             if i == 4:
# #                 ax.legend(loc=3)
# #         plt.draw()
# #         plt.savefig(plot_dir + plot_file, bbox_inches='tight')
# #         plt.show()

# #   LF in type bins by mass
# #    nrow, ncol = util.two_factors(len(labels))
#     plt.clf()
#     fig, axes = plt.subplots(plot_shape[1], plot_shape[0],
#                              sharex=True, sharey=True, num=1)
#     fig.set_size_inches(main_plot_size)
#     fig.subplots_adjust(left=0.12, bottom=0.08, right=0.9, top=0.9,
#                         hspace=0.0, wspace=0.0)
#     fig.text(0.5, 0.02, xlabel, ha='center', va='center')
#     fig.text(0.06, 0.5, ylabel, ha='center', va='center', rotation='vertical')
#     plt.semilogy(basey=10, nonposy='clip')

#     # Set the colormap and norm to correspond to the data for which
#     # the colorbar will be used.
#     cmap = mpl.cm.viridis_r
#     norm = mpl.colors.Normalize(vmin=mmin, vmax=mmax)
#     scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
#     if len(labels) == 6:
#         type_order = [0, 2, 4, 1, 3, 5]
#     else:
#         type_order = [0, 2, 4, 6, 1, 3, 5, 7]
#     for i in range(len(labels)):
#         ax = axes.flat[i]
#         it = type_order[i]
#         lbl = labels[it]
#         desc = descriptions[it]
#         ax.axis(lf_lims)
#         ax.text(0.9, 0.9, desc, ha='right', transform=ax.transAxes)
#         for im in range(nbin):
#             key = 'M{}'.format(im)
#             clr = scalarMap.to_rgba(lgm_av[im])
#             for cs in ('cen', 'sat'):
#                 if lbl in ('mock', 'tng', 'eagle', 'Lgal'):
#                     phi = lf_dict[key + lbl + cs]
#                 else:
#                     # phi = lf_dict[key + cs + lbl]
#                     phi = lf_dict[key + '_' + cs + '_all']
#                 if cs == 'sat':
#                     phi.plot(ax=ax, nmin=nmin, ls='--', clr=clr, mfc='w',
#                              fmt='s', alpha=alpha)
#                 else:
# #                    phi.plot(ax=ax, nmin=nmin, ls=':', clr=clr, mfc='w', alpha=alpha)
#                     phi.plot(ax=ax, nmin=nmin, clr=clr, fmt='o', alpha=alpha)
#     cbar_ax = fig.add_axes([0.92, 0.08, 0.01, 0.82])
#     cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
#                                    orientation='vertical')
#     cbar_ax.set_title(r'$\lg {\cal M}_h$')
#     plt.draw()
#     plt.savefig(plot_dir + lf_plot, bbox_inches='tight')
#     plt.show()

#     # Read Yang parameters
# #    yang = {'All': yang_par(f'{yang_pref}all.txt', lgm_av),
# #            'Blue': yang_par(f'{yang_pref}blue.txt', lgm_av),
# #            'Red': yang_par(f'{yang_pref}red.txt', lgm_av)}
#     yang = {'All': yang_par(f'{yang_pref}all.txt'),
#             'Blue': yang_par(f'{yang_pref}blue.txt'),
#             'Red': yang_par(f'{yang_pref}red.txt')}
#     #   Likelihood contours for centrals and satellites
#     for cs, lims, plotfile in zip(('cen', 'sat'), (gauss_lims, schec_lims),
#                                    (gauss_par_plot, schec_par_plot)):
#         plt.clf()
#         fig, axes = plt.subplots(*plot_shape, sharex=True, sharey=True, num=1)
#         fig.set_size_inches(contour_plot_size)
#         fig.subplots_adjust(left=sa_left, bottom=sa_bot, top=0.88,
#                             hspace=0.0, wspace=0.0)
#         if cs == 'cen':
#             fig.text(0.52, 0.0, r'$\sigma_c$', ha='center', va='center')
# #            fig.text(0.52, 0.0, r'$\lg \phi^*$', ha='center', va='center')
#             fig.text(0.06, 0.46, r'$M_c$', ha='center', va='center',
#                      rotation='vertical')
#             fig.text(0.52, 0.92, 'Central log-normal parameters',
#                      ha='center', va='center')
#         else:
#             fig.text(0.52, 0.0, r'$\alpha$', ha='center', va='center')
#             fig.text(0.06, 0.46, mstard, ha='center', va='center',
#                      rotation='vertical')
#             fig.text(0.52, 0.92, 'Satellite modified Schechter parameters',
#                      ha='center', va='center')
#         lines = []

#         # Set the colormap and norm to correspond to the data for which
#         # the colorbar will be used.
#         cmap = mpl.cm.viridis_r
#         norm = mpl.colors.Normalize(vmin=mmin, vmax=mmax)
#         scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
#         iy = 0
# #        labels = ['', 'mock', 'b', 'r', 'nlo', 'nhi']
# #        descriptions = ['All', 'Mock', 'Blue', 'Red', 'low-n', 'high-n']
# #        labels = ['', 'b', 'r', 'nlo', 'nhi']
# #        descriptions = ['All', 'Blue', 'Red', 'low-n', 'high-n']
#         for lbl, desc in zip(labels, descriptions):
#             if desc:
#                 ax = axes.flat[iy]
#                 ax.axis(lims)
#                 if mock_file or lbl != 'mock':
#                     ax.text(0.9, 0.9, desc, transform=ax.transAxes, ha='right')
#                     for i in range(nbin):
#                         if lbl in ('mock', 'tng', 'eagle', 'Lgal'):
#                             key = 'M{}'.format(i) + lbl + cs
#                         else:
#                             # key = 'M{}'.format(i) + cs + lbl
#                             key = f'M{i}_{cs}_all'
#                         phi = lf_dict[key]
# #                        nu = len(phi.phi[phi.comp]) - 3
#                         if phi.res.dof > 0:
#                             try:
#                                 label = r'${\cal M}' + r'{}$'.format(i+1)
#                                 clr = scalarMap.to_rgba(lgm_av[i])
#                                 rproj = RegionProjection()
#                                 rproj.prepare(nloop=(lc_step, lc_step), fac=lc_limits,
#                                               sigma=sigma)
#                                 if cs == 'cen':
#                                     rproj.calc(phi.fit, phi.fn.sigma_c, phi.fn.M_c)
#     #                                rproj.calc(phi.fit, phi.fn.lgps, phi.fn.M_c)
#                                 else:
#                                     rproj.calc(phi.fit, phi.fn.alpha, phi.fn.Mstar)
#                     #                rproj_dict[key] = rproj
#                                 ax.plot(rproj.parval0, rproj.parval1, '+', color=clr)
#                                 xmin, xmax = rproj.min[0], rproj.max[0]
#                                 ymin, ymax = rproj.min[1], rproj.max[1]
#                                 nx, ny = rproj.nloop
#                     #                hx = 0.5 * (xmax - xmin) / (nx - 1)
#                     #                hy = 0.5 * (ymax - ymin) / (ny - 1)
#                                 extent = (xmin, xmax, ymin, ymax)
#                                 y = rproj.y.reshape((ny, nx))
#                                 v = rproj.levels
#                     #            pdb.set_trace()
#                                 sc = ax.contour(y, v, origin='lower', extent=extent,
#                                                 colors=(clr,))
#                                 if iy == 0:
#                                     lines.append(mlines.Line2D([], [], color=clr, label=label))
#                             except:
#                                 print('Error determining likelihood contour')
# #                            # Yang comparison
# #                            if desc in ('All', 'Blue', 'Red'):
# #                                if cs == 'cen':
# #                                    ax.plot(yang[desc]['sigma_c'][i],
# #                                            yang[desc]['lgmstar_c'][i], 'o',
# #                                            color=clr)
# #                                else:
# #                                    ax.plot(yang[desc]['alpha'][i],
# #                                            yang[desc]['lgmstar_s'][i], 'o',
# #                                            color=clr)
#                     # Yang comparison
#                     if desc in ('All', 'Blue', 'Red'):
#                         lgmh = yang[desc]['lgmh']
#                         for j in range(len(lgmh)):
#                             if (mmin <= lgmh[j]) and (lgmh[j] < mmax):
#                                 clry = scalarMap.to_rgba(lgmh[j])
#                                 if cs == 'cen':
#                                     ax.plot(yang[desc]['sigma_c'][j],
#                                             yang[desc]['lgmstar_c'][j], 'o',
#                                             color=clry)
#                                 else:
#                                     ax.plot(yang[desc]['alpha'][j],
#                                             yang[desc]['lgmstar_s'][j], 'o',
#                                             color=clry)

#     #                if zand_comp and zand_samp[iy] and cs == 'sat':
#     #                    schec = zand_schec[zand_samp[iy]]
#     #                    for i in range(len(schec['mass'])):
#     #                        clr = scalarMap.to_rgba(schec['mass'][i])
#     #                        ax.errorbar(schec['alpha'][i], schec['mstar'][i],
#     #                                    xerr=schec['alpha_err'][i],
#     #                                    yerr=schec['mstar_err'][i],
#     #                                    fmt='none', color=clr)
#             iy += 1
# #        cbar_ax = fig.add_axes([0.15, 0.89, 0.75, 0.02])
# #        cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
# #                                       orientation='horizontal')
# #        cbar_ax.set_title(r'$\lg {\cal M}_h$')
#         plt.draw()
#         plt.savefig(plot_dir + plotfile, bbox_inches='tight')
#         plt.show()


def test(par=[1, 2, 3]):
    print(fr""""
          \begin{{math}}
          \begin{{array}}{{crcccc}}
          \hline
          & N_{{\rm gal}} & {par[0]} & {par[1]} & {par[2]}
          """)
    fig1, axes1 = plt.subplots(2, 2, sharex=True, sharey=True, num=1)
    plt.figure(1)
    # plt.clf()
    fig1.text(0.5, 0.0, 'x', ha='center', va='center')
    fig1.text(0.06, 0.5, 'y', ha='center', va='center', rotation='vertical')

    fig2, axes2 = plt.subplots(2, 2, sharex=True, sharey=True, num=2)
    plt.figure(2)
    # plt.clf()
    fig2.text(0.52, 0.0, 'x', ha='center', va='center')
    fig2.text(0.06, 0.45, 'y', ha='center', va='center', rotation='vertical')

    ax = axes1[0, 0]
    ax.plot((0, 1), (0, 1))
    ax = axes2[0, 0]
    ax.plot((0, 1), (1, 0))
    # plt.figure(1)
    # # plt.draw()
    # plt.show()
    # plt.figure(2)
    # # plt.draw()
    plt.show()


def csmf_comp_plots():
    csmf_plots(
        gama_file='smf_comp_vmax.pkl',
        cen_table='csmf_comp_gauss.tex', sat_table='csmf_comp_schec.tex',
        plot_files=('csmf_comp.pdf', 'csmf_comp_gauss.pdf', 'csmf_comp_schec.pdf'))


def csmf_comp_schec_plots():
    csmf_plots(
        gama_file='smf_comp_vmax.pkl', schec_fn=SchecMass(),
        cen_table='csmf_comp_gauss.tex', sat_table='csmf_comp_schec.tex',
        plot_files=('csmf_comp.pdf', 'csmf_comp_gauss.pdf', 'csmf_comp_schec.pdf'))


def csmf_incomp_plots():
    csmf_plots(
        gama_file='smf_incomp_vmax.pkl',
        cen_table='csmf_incomp_gauss.tex', sat_table='csmf_incomp_schec.tex',
        plot_files=('csmf_incomp.pdf', 'csmf_incomp_gauss.pdf', 'csmf_incomp_schec.pdf'))


def csmf_plots(
        panels=(('GAMA', 'all', 'GAMA all'), ('GAMA', 'blue', 'GAMA blue'),
                ('GAMA', 'red', 'GAMA red'),
                ('GAMA', 'nlo', 'GAMA low-n'), ('GAMA', 'nhi', 'GAMA high-n'),
                ('TNG', 'all', 'TNG all'), ('TNG', 'blue', 'TNG blue'),
                ('TNG', 'red', 'TNG red'),
                ('LGAL', 'all', 'LGAL all'), ('EAGLE', 'all', 'EAGLE all')),
        gama_file='smf_comp_vmax.pkl', tng_file='TNG300-1_84_csmf.pkl',
        mock_file=None, schec_fn=SchecMassSq(),
        gauss_fn=LogNormal(), schecdbl=False,
        schec_p0=[[10.5, 9, 12], [-1.2, -2.5, 1], [-2, -8, 0]],
        gauss_p0=[[10.5, 9, 12], [0.5, 0.1, 0.8], [-4, -7, -2]],
        Mmin_fit=9.0, Mmax_fit=12.5,
        nmin=2, lf_lims=(8.5, 12.5, 5e-8, 0.02),
        schec_lims=(-2.1, -0.8, 10.25, 11.3),
        gauss_lims=(0.12, 0.38, 10.2, 11.9),
        yang_pref='csmf_',
        eagle_file='eagle_csmf.pkl', Lgal_file='smf_Lgal.pkl',
        cen_table='csmf_comp_gauss.tex', sat_table='csmf_comp_schec.tex',
        plot_files=('csmf_comp.pdf', 'csmf_comp_gauss.pdf', 'csmf_comp_schec.pdf'),
        xlabel=ms_label, ylabel=smf_label,
        sigma=[1, ], lc_limits=8, lc_step=32,
        plot_shape=(5, 2),  # lot_shape=(2, 4),
        red_frac_file='red_frac.pkl', main_plot_size=(5, 13),
        contour_plot_size=(5, 10), transpose=1, xlab_y=0.02):
    """Plot and tabulate galaxy CSMFs by colour and Sersic index,
    separately for central/satellite."""
    clf_plots(panels=panels, gama_file=gama_file, mock_file=mock_file,
              schec_fn=schec_fn, gauss_fn=gauss_fn, schecdbl=schecdbl,
              schec_p0=schec_p0, gauss_p0=gauss_p0,
              Mmin_fit=Mmin_fit, Mmax_fit=Mmax_fit,
              nmin=nmin, lf_lims=lf_lims,
              schec_lims=schec_lims, gauss_lims=gauss_lims,
              yang_pref=yang_pref, tng_file=tng_file, eagle_file=eagle_file,
              Lgal_file=Lgal_file, cen_table=cen_table, sat_table=sat_table,
              plot_files=plot_files,
              xlabel=xlabel, ylabel=ylabel,
              sigma=sigma, lc_limits=lc_limits, lc_step=lc_step,
              plot_shape=plot_shape, red_frac_file=red_frac_file,
              main_plot_size=main_plot_size,
              contour_plot_size=contour_plot_size, transpose=transpose,
              xlab_y=xlab_y)


def clf_plots(
        panels=(('Mock', 'all', 'Mock all'), ('GAMA', 'all', 'GAMA all'),
                ('GAMA', 'blue', 'GAMA blue'), ('GAMA', 'red', 'GAMA red'),
                ('GAMA', 'nlo', 'GAMA low-n'), ('GAMA', 'nhi', 'GAMA high-n')),
        gama_file='lf_vmax.pkl', tng_file='TNG300-1_84_clf.pkl',
        mock_file='lf_mockfof.pkl',
        schec_fn=SchecMagSq(), gauss_fn=LogNormal(), schecdbl=False,
        schec_p0=[[-20.2, -22, -19], [-1.2, -2.5, 1], [-2, -8, 0]],
        gauss_p0=[[-21.5, -23, -20], [0.5, 0.1, 0.8], [-4, -7, -2]],
        Mmin_fit=-24, Mmax_fit=-16,
        nmin=2, lf_lims=(-15.5, -23.9, 5e-8, 0.02),
        schec_lims=(-1.7, -0.5, -20.2, -22.4),
        gauss_lims=(0.3, 0.8, -20.1, -22.7),
        yang_pref='clf_', eagle_file=None, Lgal_file=None,
        cen_table='clf_cen.tex', sat_table='clf_sat.tex',
        plot_files=('clf.pdf', 'clf_gauss.pdf', 'clf_schec.pdf'),
        xlabel=mag_label, ylabel=lf_label,
        sigma=[1, ], lc_limits=5, lc_step=32, alpha=[0.5, 1],
        mmin=13, mmax=14.4, plot_shape=(3, 2), markersize=5,
        red_frac_file=None, main_plot_size=(5, 7),
        contour_plot_size=(5, 6), transpose=0, xlab_y=0.0):
    """Plot and tabulate galaxy CLFs by colour and Sersic index,
    separately for central/satellite."""

    infiles = {'GAMA': gama_file, 'TNG': tng_file, 'LGAL': Lgal_file,
               'EAGLE': eagle_file, 'Mock': mock_file}
    if 'smf' in gama_file:
        mstard = r'$\lg\ {\cal M}^*$'
        func = 'SMF'
    else:
        mstard = r'$M^*$'
        func = 'LF'

    f1 = schec_fn
#    f1 = SchecMag('schec1')
    f1.mstar = schec_p0[0][0]
    f1.Mstar.min = schec_p0[0][1]
    f1.Mstar.max = schec_p0[0][2]
    f1.alpha = schec_p0[1][0]
    f1.alpha.min = schec_p0[1][1]
    f1.alpha.max = schec_p0[1][2]
    f1.lgps = schec_p0[2][0]
    f1.lgps.min = schec_p0[2][1]
    f1.lgps.max = schec_p0[2][2]
    if schecdbl:
        f2 = SchecMag('schec2')
        sep = Const1D('sep')
        f2.alpha = f1.alpha + sep.c0
        sep.c0 = -1.5
        sep.c0.min = -5
        sep.c0.max = 0
        f2.mstar = f1.mstar
        f2.lgps = schec_p0[2][0]
        schec_fn = f1 + f2 + 0*sep
    else:
        schec_fn = f1

    gauss_fn.M_c = gauss_p0[0][0]
    gauss_fn.M_c.min = gauss_p0[0][1]
    gauss_fn.M_c.max = gauss_p0[0][2]
    gauss_fn.sigma_c = gauss_p0[1][0]
    gauss_fn.sigma_c.min = gauss_p0[1][1]
    gauss_fn.sigma_c.max = gauss_p0[1][2]
    gauss_fn.lgps = gauss_p0[2][0]
    gauss_fn.lgps.min = gauss_p0[2][1]
    gauss_fn.lgps.max = gauss_p0[2][2]

    sa_left = 0.12
    sa_bot = 0.05
    nbin = len(mbins_def) - 1

    def tbl_init(filename, sat):
        """Initialise table of fit parameters for centrals (sat=0)
        or satellites (sat=0)."""
        par = ((r'M_c', r'\sigma_c', r'\lg \phi*_c'),
               (r'M^*', r'\alpha', r'\lg \phi*_s'))[sat]
        f = open(plot_dir + filename, 'w')
        print(fr"""\begin{{math}}
             \begin{{array}}{{crcccc}}
             \hline
             & N_{{\rm gal}} & {par[0]} & {par[1]} & {par[2]} &
             \chi^2/\nu \\[0pt]""", file=f)
        return f

    fcen = tbl_init(cen_table, 0)
    fsat = tbl_init(sat_table, 1)

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    cmap = mpl.cm.viridis_r
    norm = mpl.colors.Normalize(vmin=mmin, vmax=mmax)
    scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Plot 1: LF in type bins colour coded by mass
    fig1, axes1 = plt.subplots(*plot_shape,
                               sharex=True, sharey=True, num=1)
    if transpose:
        axes1 = axes1.T
    # plt.clf()
    fig1.set_size_inches(main_plot_size)
    fig1.subplots_adjust(left=sa_left, bottom=sa_bot, right=1.0, top=0.9,
                         hspace=0.0, wspace=0.0)
    fig1.text(0.58, xlab_y, xlabel, ha='center', va='center')
    fig1.text(0.0, 0.48, ylabel, ha='center', va='center', rotation='vertical')
    plt.semilogy(basey=10, nonposy='clip')
    plt.axis(lf_lims)
    # cbar_ax = fig1.add_axes([0.92, 0.08, 0.01, 0.82])
    cbar_ax = fig1.add_axes([0.15, 0.94, 0.82, 0.02])
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                              orientation='horizontal')
    cbar_ax.set_title(r'$\lg {\cal M}_h$')

    # Plot 2: likelihood contours for centrals
    fig2, axes2 = plt.subplots(*plot_shape, sharex=True, sharey=True, num=2)
    if transpose:
        axes2 = axes2.T
    # plt.clf()
    plt.axis(gauss_lims)
    fig2.set_size_inches(contour_plot_size)
    fig2.subplots_adjust(left=sa_left, bottom=sa_bot, top=1.0,
                         hspace=0.0, wspace=0.0)
    fig2.text(0.52, xlab_y, r'$\sigma_c$', ha='center', va='center')
    fig2.text(0.0, 0.52, r'$M_c$', ha='center', va='center',
              rotation='vertical')
    fig2.text(0.52, 1.025, f'Central log-normal {func} parameters',
              ha='center', va='center')

    # Plot 3: likelihood contours for satellites
    fig3, axes3 = plt.subplots(*plot_shape, sharex=True, sharey=True, num=3)
    if transpose:
        axes3 = axes3.T
    # plt.clf()
    plt.axis(schec_lims)
    fig3.set_size_inches(contour_plot_size)
    fig3.subplots_adjust(left=sa_left, bottom=sa_bot, top=1.0,
                         hspace=0.0, wspace=0.0)
    fig3.text(0.52, xlab_y, r'$\alpha$', ha='center', va='center')
    fig3.text(0.0, 0.52, mstard, ha='center', va='center',
              rotation='vertical')
    fig3.text(0.52, 1.025, f'Satellite modified Schechter {func} parameters',
              ha='center', va='center')

    # Read Yang parameters
    yang = {'all': yang_par(f'{yang_pref}all.txt'),
            'blue': yang_par(f'{yang_pref}blue.txt'),
            'red': yang_par(f'{yang_pref}red.txt')}

    for ip, panel in zip(range(len(panels)), panels):
        src = panel[0]
        typ = panel[1]
        desc = panel[2]
        lf_dict = pickle.load(open(infiles[src], 'rb'))
        for f in (fcen, fsat):
            print(fr"""
                  \hline
                  \multicolumn{{5}}{{c}}{{\mbox{{{desc}}}}} \\""", file=f)

        for axes in (axes1, axes2, axes3):
            ax = axes.flat[ip]
            ax.text(0.9, 0.9, desc, ha='right', transform=ax.transAxes)
        # pdb.set_trace()

        for i in range(nbin):
            clr = scalarMap.to_rgba(lgm_av[i])
            for cs, f, num, axc in zip(('cen', 'sat'), (fcen, fsat), (2, 3),
                                       (axes2, axes3)):
                key = 'M{}'.format(i) + '_' + cs + '_' + typ
                phi = lf_dict[key]
                try:
                    clr = scalarMap.to_rgba(phi.info['lgm_av'])
                except:
                    print('No lgm_av info for', infiles[src])
                if cs == 'cen':
                    fn = phi.fn_fit(gauss_fn, Mmin=Mmin_fit, Mmax=Mmax_fit,
                                    verbose=0)
                else:
                    fn = phi.fn_fit(schec_fn, Mmin=Mmin_fit, Mmax=Mmax_fit,
                                    verbose=0)
                try:
                    fit_errs = 0.5*(np.array(phi.errors.parmaxes) -
                                    np.array(phi.errors.parmins))
                except:
    #                    except AttributeError or TypeError:
                    fit_errs = (9.99, 9.99, 9.99)
    #                        print('bad error estimate')
    #                        pdb.set_trace()

                print(fr'''\mass{i+1} & {int(phi.ngal.sum())} &
                      {phi.res.parvals[0]:5.2f}\pm{fit_errs[0]:5.2f} &
                      {phi.res.parvals[1]:5.2f}\pm{fit_errs[1]:5.2f} &
                      {phi.res.parvals[2]:5.2f}\pm{fit_errs[2]:5.2f} &
                      {phi.res.statval:4.1f}/{phi.res.dof:2d} \\[0pt]
                      ''', file=f)

                # LF in type bins by mass
                # plt.figure(1)
                ax = axes1.flat[ip]
                if cs == 'sat':
                    phi.plot(ax=ax, nmin=nmin, ls='--', clr=clr, mfc='w',
                             fmt='o', alpha=alpha, markersize=markersize)
                else:
                    phi.plot(ax=ax, nmin=nmin, clr=clr, fmt='o', alpha=alpha,
                             markersize=markersize)
                # pdb.set_trace()
                ax = axc.flat[ip]
                if phi.res.dof > 0:
                    # try:
                    label = r'${\cal M}' + r'{}$'.format(i+1)
                    # clr = scalarMap.to_rgba(lgm_av[i])
                    clr = scalarMap.to_rgba(phi.info['lgm_av'])
                    rproj = RegionProjection()
                    rproj.prepare(nloop=(lc_step, lc_step), fac=lc_limits,
                                  sigma=sigma)
                    if cs == 'cen':
                        rproj.calc(phi.fit, phi.fn.sigma_c, phi.fn.M_c)
#                                rproj.calc(phi.fit, phi.fn.lgps, phi.fn.M_c)
                    else:
                        rproj.calc(phi.fit, phi.fn.alpha, phi.fn.Mstar)
        #                rproj_dict[key] = rproj
                    ax.plot(rproj.parval0, rproj.parval1, '+', color=clr)
                    xmin, xmax = rproj.min[0], rproj.max[0]
                    ymin, ymax = rproj.min[1], rproj.max[1]
                    nx, ny = rproj.nloop
        #                hx = 0.5 * (xmax - xmin) / (nx - 1)
        #                hy = 0.5 * (ymax - ymin) / (ny - 1)
                    extent = (xmin, xmax, ymin, ymax)
                    y = rproj.y.reshape((ny, nx))
                    v = rproj.levels
        #            pdb.set_trace()
                    sc = ax.contour(y, v, origin='lower', extent=extent,
                                    colors=(clr,))
                    # if iy == 0:
                    #     lines.append(mlines.Line2D([], [], color=clr, label=label))
                    # except:
                    #     print('Error determining likelihood contour')
                    # Yang comparison
                    if src == 'GAMA' and typ in ('all', 'blue', 'red'):
                        lgmh = yang[typ]['lgmh']
                        for j in range(len(lgmh)):
                            if (mmin <= lgmh[j]) and (lgmh[j] < mmax):
                                clry = scalarMap.to_rgba(lgmh[j])
                                if cs == 'cen':
                                    ax.plot(yang[typ]['sigma_c'][j],
                                            yang[typ]['lgmstar_c'][j], 'o',
                                            color=clry)
                                else:
                                    ax.plot(yang[typ]['alpha'][j],
                                            yang[typ]['lgmstar_s'][j], 'o',
                                            color=clry)
           # iy += 1
    for f in (fcen, fsat):
        print(r"""
        \hline
        \end{array}
        \end{math}""", file=f)
        f.close()

    for iplot in range(3):
        plt.figure(iplot+1)
        plt.draw()
        plt.savefig(plot_dir + plot_files[iplot], bbox_inches='tight',
                    pad_inches=0.05)
    plt.show()


def csmf_plots_sep(
        panels=(('GAMA', 'all', 'GAMA all'), ('GAMA', 'blue', 'GAMA blue'),
                ('GAMA', 'red', 'GAMA red'),
                ('GAMA', 'nlo', 'GAMA low-n'), ('GAMA', 'nhi', 'GAMA high-n'),
                ('TNG', 'all', 'TNG all'), ('TNG', 'blue', 'TNG blue'),
                ('TNG', 'red', 'TNG red'),
                ('LGAL', 'all', 'LGAL all'), ('EAGLE', 'all', 'EAGLE all')),
        gama_file='smf_comp_vmax.pkl', tng_file='TNG300-1_84_csmf.pkl',
        mock_file=None,
        sat_fn=SchecMass(), cen_fn=LogNormal(), schecdbl=False,
        # schec_fn=SchecMass(), gauss_fn=LogNormal(), schecdbl=False,
        contour_plots=(
            ('cen', 1, 0, (0.12, 0.38, 10.1, 11.9),
             r'$\sigma_c$', r'$\lg\ {\cal M}_c$', 'csmf_sig_mc.pdf',
             'Central log-normal'),
            ('sat', 1, 0, (-1.9, -0.3, 10.0, 11.19),
             r'$\alpha$', r'$\lg\ {\cal M}^*$', 'csmf_alpha_Mstar.pdf',
             'Satellite Schechter')),
            # ('sat', 1, 2, (-1.7, -0.5, 0.5, 3),
            #  r'$\alpha$', r'$\beta$', 'csmf_alpha_beta.pdf',
            #  'Satellite generalised Schechter'),
            # ('sat', 2, 0, (0.5, 3, 10.2, 11.9),
            #  r'$\beta$', r'$M^*$', 'csmf_beta_Mstar.pdf',
            #  'Satellite generalised Schechter')),
        # schec_p0=[[10.5, 9, 12], [-1.2, -2.5, 1], [-2, -8, 0]],
        # gauss_p0=[[10.5, 9, 12], [0.5, 0.1, 0.8], [-4, -7, -2]],
        Mmin_fit=9.0, Mmax_fit=12.5,
        nmin=2, cen_lf_lims=(9.1, 12.5, 5e-8, 0.02),
        sat_lf_lims=(8.5, 12.1, 5e-8, 0.02),
        # schec_lims=(-2.1, -0.8, 10.25, 11.3),
        # gauss_lims=(0.12, 0.38, 10.2, 11.9),
        yang_pref='csmf_',
        eagle_file='eagle_csmf.pkl', Lgal_file='smf_Lgal.pkl',
        cen_table='csmf_comp_cen.tex', sat_table='csmf_comp_sat.tex',
        plot_files=['csmf_comp_cen.pdf', 'csmf_comp_sat.pdf'],
        xlabel=ms_label, ylabel=smf_label,
        sigma=[1, ], lc_limits=8, lc_step=32,
        plot_shape=(5, 2),  # lot_shape=(2, 4),
        red_frac_file='red_frac.pkl', main_plot_size=(5, 10),
        contour_plot_size=(5, 9), transpose=1, xlab_y=0.02):
    """Plot and tabulate galaxy CSMFs by colour and Sersic index,
    separately for central/satellite."""
    clf_plots_sep(panels=panels, gama_file=gama_file, mock_file=mock_file,
                  sat_fn=sat_fn, cen_fn=cen_fn, schecdbl=schecdbl,
                  # schec_p0=schec_p0, gauss_p0=gauss_p0,
                  contour_plots=contour_plots,
                  Mmin_fit=Mmin_fit, Mmax_fit=Mmax_fit,
                  nmin=nmin, cen_lf_lims=cen_lf_lims, sat_lf_lims=sat_lf_lims,
                  # schec_lims=schec_lims, gauss_lims=gauss_lims,
                  yang_pref=yang_pref, tng_file=tng_file, eagle_file=eagle_file,
                  Lgal_file=Lgal_file, cen_table=cen_table, sat_table=sat_table,
                  plot_files=plot_files,
                  xlabel=xlabel, ylabel=ylabel,
                  sigma=sigma, lc_limits=lc_limits, lc_step=lc_step,
                  plot_shape=plot_shape, red_frac_file=red_frac_file,
                  main_plot_size=main_plot_size,
                  contour_plot_size=contour_plot_size, transpose=transpose,
                  xlab_y=xlab_y)


def clf_plots_sep(
        panels=(('Mock', 'all', 'Mock all'), ('GAMA', 'all', 'GAMA all'),
                ('GAMA', 'blue', 'GAMA blue'), ('GAMA', 'red', 'GAMA red'),
                ('GAMA', 'nlo', 'GAMA low-n'), ('GAMA', 'nhi', 'GAMA high-n')),
        gama_file='lf_vmax.pkl', tng_file='TNG300-1_84_clf.pkl',
        mock_file='lf_mockfof.pkl',
        # schec_fn=SchecMagSq(),
        sat_fn=SchecMag(), cen_fn=LogNormal(), schecdbl=False,
        # schec_p0=[[-20.2, -22, -19], [-1.2, -2.5, 1], [-2, -8, 0]],
        # schec_p0=[[-20.2, -22, -19], [-1.2, -2.5, 1], [1, 0.01, 10], [-2, -8, 0]],
        # gauss_p0=[[-21.5, -23, -20], [0.5, 0.1, 0.8], [-4, -7, -2]],
        Mmin_fit=-24, Mmax_fit=-16,
        nmin=2, cen_lf_lims=(-17.5, -23.9, 5e-8, 0.02),
        sat_lf_lims=(-15.5, -23.9, 5e-8, 0.02),
        # schec_lims=(-1.7, -0.5, -20.2, -22.4),
        contour_plots=(
            ('cen', 1, 0, (0.3, 0.85, -20.1, -22.7),
             r'$\sigma_c$', r'$M_c$', 'clf_sig_mc.pdf', 'Central log-normal'),
            ('sat', 1, 0, (-1.6, 0.1, -19.3, -21.6),
             r'$\alpha$', r'$M^*$', 'clf_alpha_Mstar.pdf',
             'Satellite Schechter')),
            # ('sat', 1, 2, (-1.7, -0.5, 0.5, 3),
            #  r'$\alpha$', r'$\beta$', 'clf_alpha_beta.pdf',
            #  'Satellite generalised Schechter'),
            # ('sat', 2, 0, (0.5, 3, -20.2, -22.4),
            #  r'$\beta$', r'$M^*$', 'clf_beta_Mstar.pdf',
            #  'Satellite generalised Schechter')),
        # schec_lims=(0.5, 3, -20.2, -22.4),
        # gauss_lims=(0.3, 0.8, -20.1, -22.7),
        yang_pref='clf_', eagle_file=None, Lgal_file=None,
        cen_table='clf_cen.tex', sat_table='clf_sat.tex',
        plot_files=['clf_cen.pdf', 'clf_sat.pdf'],  #, 'clf_gauss.pdf', 'clf_schec.pdf'),
        xlabel=mag_label, ylabel=lf_label,
        sigma=[1, ], lc_limits=5, lc_step=32, alpha=[1, 1],
        mmin=13, mmax=14.4, plot_shape=(3, 2), markersize=5,
        red_frac_file=None, main_plot_size=(5, 7),
        contour_plot_size=(5, 6), transpose=0, xlab_y=0.0):
    """Plot and tabulate galaxy CLFs by colour and Sersic index,
    separately for central/satellite."""

    infiles = {'GAMA': gama_file, 'TNG': tng_file, 'LGAL': Lgal_file,
               'EAGLE': eagle_file, 'Mock': mock_file}
    if 'smf' in gama_file:
        mstard = r'$\lg\ {\cal M}^*$'
        func = 'SMF'
        cen_fn.M_c.max = 12
        cen_fn.M_c.min = 9
        cen_fn.M_c = 10.5
    else:
        mstard = r'$M^*$'
        func = 'LF'

#     f1 = schec_fn
# #    f1 = SchecMag('schec1')
#     f1.mstar = schec_p0[0][0]
#     f1.Mstar.min = schec_p0[0][1]
#     f1.Mstar.max = schec_p0[0][2]
#     f1.alpha = schec_p0[1][0]
#     f1.alpha.min = schec_p0[1][1]
#     f1.alpha.max = schec_p0[1][2]
#     f1.lgps = schec_p0[2][0]
#     f1.lgps.min = schec_p0[2][1]
#     f1.lgps.max = schec_p0[2][2]
#     if schecdbl:
#         f2 = SchecMag('schec2')
#         sep = Const1D('sep')
#         f2.alpha = f1.alpha + sep.c0
#         sep.c0 = -1.5
#         sep.c0.min = -5
#         sep.c0.max = 0
#         f2.mstar = f1.mstar
#         f2.lgps = schec_p0[2][0]
#         schec_fn = f1 + f2 + 0*sep
#     else:
#         schec_fn = f1

#     gauss_fn.M_c = gauss_p0[0][0]
#     gauss_fn.M_c.min = gauss_p0[0][1]
#     gauss_fn.M_c.max = gauss_p0[0][2]
#     gauss_fn.sigma_c = gauss_p0[1][0]
#     gauss_fn.sigma_c.min = gauss_p0[1][1]
#     gauss_fn.sigma_c.max = gauss_p0[1][2]
#     gauss_fn.lgps = gauss_p0[2][0]
#     gauss_fn.lgps.min = gauss_p0[2][1]
#     gauss_fn.lgps.max = gauss_p0[2][2]

    sa_left = 0.12
    sa_bot = 0.05
    nbin = len(mbins_def) - 1

    def tbl_init(filename, parstr):
        """Initialise table of fit parameters, where parstr is of the form
        'par1 & par2 & par3'."""
        npar = parstr.count('&') + 1
        colstr = 'crc' + npar*'c'
        f = open(plot_dir + filename, 'w')
        print(fr"""\begin{{math}}
             \begin{{array}}{{{colstr}}}
             \hline
             & N_{{\rm gal}} & {parstr} & \chi^2/\nu \\[0pt]""", file=f)
        return f, len(colstr)

    if 'smf' in gama_file:
        fcen, nccol = tbl_init(cen_table, r'\lg\ {\cal M}_c & \sigma_c & \lg \phi^*_c')
        fsat, nscol = tbl_init(sat_table, r'\lg\ {\cal M}^* & \alpha & \lg \phi^*_s')
    else:
        fcen, nccol = tbl_init(cen_table, r'M_c & \sigma_c & \lg \phi^*_c')
        fsat, nscol = tbl_init(sat_table, r'M^* & \alpha & \lg \phi^*_s')
        
    # if hasattr(sat_fn, 'sigma'):
    #     fsat, nscol = tbl_init(sat_table, r'M^* & \alpha & \sigma_s & \lg \phi^*_s')
    # else:
    #     if hasattr(sat_fn, 'beta'):
    #         fsat, nscol = tbl_init(sat_table, r'M^* & \alpha & \beta & \lg \phi^*_s')
    #     else:
    #         fsat, nscol = tbl_init(sat_table, r'M^* & \alpha & \lg \phi^*_s')

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    cmap = mpl.cm.viridis_r
    norm = mpl.colors.Normalize(vmin=mmin, vmax=mmax)
    scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Plot 0: central LF in type bins colour coded by mass
    fig0, axes0 = plt.subplots(*plot_shape,
                               sharex=True, sharey=True, num=0)
    if transpose:
        axes0 = axes0.T
    # plt.clf()
    fig0.set_size_inches(main_plot_size)
    fig0.subplots_adjust(left=sa_left, bottom=sa_bot, right=1.0, top=0.9,
                         hspace=0.0, wspace=0.0)
    fig0.text(0.58, xlab_y, xlabel, ha='center', va='center')
    fig0.text(0.0, 0.48, ylabel, ha='center', va='center', rotation='vertical')
    plt.semilogy(basey=10, nonposy='clip')
    plt.axis(cen_lf_lims)
    # cbar_ax = fig0.add_axes([0.92, 0.08, 0.01, 0.82])
    cbar_ax = fig0.add_axes([0.15, 0.94, 0.82, 0.02])
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                              orientation='horizontal')
    cbar_ax.set_title(r'$\lg {\cal M}_h$')

    # Plot 1: satellite LF in type bins colour coded by mass
    fig1, axes1 = plt.subplots(*plot_shape,
                               sharex=True, sharey=True, num=1)
    if transpose:
        axes1 = axes1.T
    # plt.clf()
    fig1.set_size_inches(main_plot_size)
    fig1.subplots_adjust(left=sa_left, bottom=sa_bot, right=1.0, top=0.9,
                         hspace=0.0, wspace=0.0)
    fig1.text(0.58, xlab_y, xlabel, ha='center', va='center')
    fig1.text(0.0, 0.48, ylabel, ha='center', va='center', rotation='vertical')
    plt.semilogy(basey=10, nonposy='clip')
    plt.axis(sat_lf_lims)
    # cbar_ax = fig1.add_axes([0.92, 0.08, 0.01, 0.82])
    cbar_ax = fig1.add_axes([0.15, 0.94, 0.82, 0.02])
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                              orientation='horizontal')
    cbar_ax.set_title(r'$\lg {\cal M}_h$')

    # Contour plots
    fig_cont = []
    axes_cont = []
    for num, cplot in zip(range(len(contour_plots)), contour_plots):
        fig, axes = plt.subplots(*plot_shape, sharex=True, sharey=True,
                                 num=num+2)
        if transpose:
            axes = axes.T
        plt.axis(cplot[3])
        fig.set_size_inches(contour_plot_size)
        fig.subplots_adjust(left=sa_left, bottom=sa_bot, top=1.0,
                            hspace=0.0, wspace=0.0)
        fig.text(0.52, xlab_y, cplot[4], ha='center', va='center')
        fig.text(0.0, 0.52, cplot[5], ha='center', va='center',
                 rotation='vertical')
        fig.text(0.52, 1.025, cplot[7] + f' {func} parameters',
                 ha='center', va='center')
        fig_cont.append(fig)
        axes_cont.append(axes)
        plot_files.append(cplot[6])

        for ip, panel in zip(range(len(panels)), panels):
            src = panel[0]
            typ = panel[1]
            desc = panel[2]
            ax = axes.flat[ip]
            if cplot[0] == 'sat':
                cs = ' satellite'
            else:
                cs = ' central'
            ax.text(0.9, 0.9, desc + cs, ha='right', transform=ax.transAxes)

    # Read Yang parameters
    yang = {'all': yang_par(f'{yang_pref}all.txt'),
            'blue': yang_par(f'{yang_pref}blue.txt'),
            'red': yang_par(f'{yang_pref}red.txt')}

    for ip, panel in zip(range(len(panels)), panels):
        src = panel[0]
        typ = panel[1]
        desc = panel[2]
        lf_dict = pickle.load(open(infiles[src], 'rb'))
        for f, ncol in zip((fcen, fsat), (nccol, nscol)):
            print(fr"""
                  \hline
                  \multicolumn{{{ncol}}}{{c}}{{\mbox{{{desc}}}}} \\""", file=f)

        for sat, axes in zip((0, 1), (axes0, axes1)):
            ax = axes.flat[ip]
            if sat:
                cs = ' satellite'
            else:
                cs = ' central'
            ax.text(0.9, 0.9, desc + cs, ha='right', transform=ax.transAxes)

        for i in range(nbin):
            clr = scalarMap.to_rgba(lgm_av[i])
            for cs, f, axb in zip(
                    ('cen', 'sat'), (fcen, fsat), (axes0, axes1)):
                key = 'M{}'.format(i) + '_' + cs + '_' + typ
                phi = lf_dict[key]
                try:
                    clr = scalarMap.to_rgba(phi.info['lgm_av'])
                except:
                    print('No lgm_av info for', infiles[src])
                if cs == 'cen':
                    fn = phi.fn_fit(cen_fn, Mmin=Mmin_fit, Mmax=Mmax_fit,
                                    verbose=0)
                else:
                    fn = phi.fn_fit(sat_fn, Mmin=Mmin_fit, Mmax=Mmax_fit,
                                    verbose=0)
                npar = len(phi.res.parvals)
                # try:
                if hasattr(phi, 'errors'):
                    parmaxes = [9.99 if v is None else v for v in phi.errors.parmaxes]
                    parmins = [-9.99 if v is None else v for v in phi.errors.parmins]
                    fit_errs = 0.5*(np.array(parmaxes) - np.array(parmins))
                else:
                    fit_errs = 9.99*np.ones(len(phi.fn.pars))
                # except:
    #                    except AttributeError or TypeError:
                    # fit_errs = npar*[9.99]
                    # print('bad error estimate')
                    # pdb.set_trace()

                line = fr'\mass{i+1} & {int(phi.ngal.sum())} & '
                for ipar in range(npar):
                    line += fr'{phi.res.parvals[ipar]:5.2f}\pm{fit_errs[ipar]:5.2f} & '
                line += fr'{phi.res.statval:4.1f}/{phi.res.dof:2d} \\[0pt]'
                print(line, file=f)

                # LF in type bins by mass
                phi.plot(ax=axb.flat[ip], nmin=nmin, ls='-', clr=clr,
                         fmt='o', alpha=alpha, markersize=markersize)

                # Contour plots
                if phi.res.dof > 0:
                    for axes, cplot in zip(axes_cont, contour_plots):
                        if cplot[0] == cs:
                            ax = axes.flat[ip]
                            label = r'${\cal M}' + r'{}$'.format(i+1)
                            clr = scalarMap.to_rgba(phi.info['lgm_av'])
                            rproj = RegionProjection()
                            rproj.prepare(nloop=(lc_step, lc_step), fac=lc_limits,
                                          sigma=sigma)
                    # if cs == 'cen':
                            rproj.calc(phi.fit, phi.fn.pars[cplot[1]],
                                       phi.fn.pars[cplot[2]])
#                                rproj.calc(phi.fit, phi.fn.lgps, phi.fn.M_c)
                    # else:
                    #     # rproj.calc(phi.fit, phi.fn.alpha, phi.fn.Mstar)
                    #     # rproj.calc(phi.fit, phi.fn.sigma, phi.fn.Mstar)
                    #     rproj.calc(phi.fit, phi.fn.beta, phi.fn.Mstar)
                            ax.plot(rproj.parval0, rproj.parval1, '+', color=clr)
                            xmin, xmax = rproj.min[0], rproj.max[0]
                            ymin, ymax = rproj.min[1], rproj.max[1]
                            nx, ny = rproj.nloop
                #                hx = 0.5 * (xmax - xmin) / (nx - 1)
                #                hy = 0.5 * (ymax - ymin) / (ny - 1)
                            extent = (xmin, xmax, ymin, ymax)
                            y = rproj.y.reshape((ny, nx))
                            v = rproj.levels
                #            pdb.set_trace()
                            sc = ax.contour(y, v, origin='lower', extent=extent,
                                            colors=(clr,))
                    # if iy == 0:
                    #     lines.append(mlines.Line2D([], [], color=clr, label=label))
                    # except:
                    #     print('Error determining likelihood contour')
                    # Yang comparison
                    if src == 'GAMA' and typ in ('all', 'blue', 'red'):
                        lgmh = yang[typ]['lgmh']
                        for j in range(len(lgmh)):
                            if (mmin <= lgmh[j]) and (lgmh[j] < mmax):
                                clry = scalarMap.to_rgba(lgmh[j])
                                if cs == 'cen':
                                    ax.plot(yang[typ]['sigma_c'][j],
                                            yang[typ]['lgmstar_c'][j], 'o',
                                            color=clry)
                                else:
                                    ax.plot(yang[typ]['alpha'][j],
                                            yang[typ]['lgmstar_s'][j], 'o',
                                            color=clry)
           # iy += 1
    for f in (fcen, fsat):
        print(r"""
        \hline
        \end{array}
        \end{math}""", file=f)
        f.close()

    for iplot in range(2 + len(contour_plots)):
        plt.figure(iplot)
        plt.draw()
        plt.savefig(plot_dir + plot_files[iplot], bbox_inches='tight',
                    pad_inches=0.05)
    plt.show()


def clf_mock_comp(mockfof_file='lf_mockfof.pkl', mockhalo_file='lf_mockhalo.pkl',
                  nmin=8, lf_lims=(-15.5, -23.9, 2e-7, 0.01),
                  schec_lims=(-1.9, 1.1, -21.6, -19.1),
                  schec_fn=SchecMag(), Mmin_fit=-23, Mmax_fit=-16,
                  schec_p0=[[-1.2, -2, 1], [-20.2, -22, -19], [-2, -8, 0]],
                  sigma=[1, 2], lc_limits=5, lc_step=32,
                  plot_file='clf_mock_comp.pdf',
                  xlabel=mag_label, ylabel=lf_label):
    """Compare FOF and halo mocks CLFs."""

    mockfof_dict = pickle.load(open(mockfof_file, 'rb'))
    mockhalo_dict = pickle.load(open(mockhalo_file, 'rb'))

    plot_size = (6, 6)
    sa_left = 0.15
    sa_bot = 0.06
    nbin = 4
    nrow, ncol = util.two_factors(nbin)

    schec_fn.alpha = schec_p0[0][0]
    schec_fn.alpha.min = schec_p0[0][1]
    schec_fn.alpha.max = schec_p0[0][2]
    schec_fn.Mstar = schec_p0[1][0]
    schec_fn.Mstar.min = schec_p0[1][1]
    schec_fn.Mstar.max = schec_p0[1][2]
    schec_fn.lgps = schec_p0[2][0]
    schec_fn.lgps.min = schec_p0[2][1]
    schec_fn.lgps.max = schec_p0[2][2]

    plt.clf()
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
    fig.set_size_inches(plot_size)
    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, xlabel, ha='center', va='center')
    fig.text(0.06, 0.5, ylabel, ha='center', va='center', rotation='vertical')
    plt.semilogy(basey=10, nonposy='clip')
    for i in range(nbin):
        key = 'M{}'.format(i)
        label = r'${\cal M}' + r'{}\ [{}, {}]$'.format(
                i+1, mbins_def[i], mbins_def[i+1])
        ax = axes.flat[i]
        ax.axis(lf_lims)
        ax.text(0.9, 0.9, label, ha='right', transform=ax.transAxes)
        axin = ax.inset_axes([0.25, 0.12, 0.4, 0.4])
#        axin.axis((-1.6, -0.4, -21.5, -20.0))
        axin.xaxis.set_label_position("top")
        axin.yaxis.set_label_position("right")
        axin.set_xlabel(r'$\alpha$')
        axin.set_ylabel(r'$M^*$')
        # rotation='horizontal', rotation_mode='anchor')
        for lf_dict, fmt, colour, label in zip(
                [mockhalo_dict, mockfof_dict], 'os', 'br', ['Halo', 'FoF']):
            phi = lf_dict[key + 'mock']
            if label == 'Halo':
                phih = phi
            else:
                c, nu, p = phi.chi2(phih)
                print('chi2, nu, p =', c, nu, p)
                ax.text(0.9, 0.8, rf'$P_{{\chi^2}} = {p:5.3f}$',
                        ha='right', transform=ax.transAxes)
            fn = phi.fn_fit(schec_fn, Mmin=Mmin_fit, Mmax=Mmax_fit, verbose=0)
            phi.plot(ax=ax, nmin=nmin, clr=colour, fmt=fmt, label=label)
            rproj = RegionProjection()
            rproj.prepare(nloop=(lc_step, lc_step), fac=lc_limits,
                          sigma=sigma)
            rproj.calc(phi.fit, phi.fn.alpha, phi.fn.Mstar)
            axin.plot(rproj.parval0, rproj.parval1, '+', color=colour)
            xmin, xmax = rproj.min[0], rproj.max[0]
            ymin, ymax = rproj.min[1], rproj.max[1]
            nx, ny = rproj.nloop
            extent = (xmin, xmax, ymin, ymax)
            y = rproj.y.reshape((ny, nx))
            v = rproj.levels
#            pdb.set_trace()
            sc = axin.contour(y, v, origin='lower', extent=extent,
                              colors=(colour,))
#            for cs, fmt in zip(['cen', 'sat'], ['--', ':']):
#                phi = lf_dict[key + 'mock' + cs]
#                phi.plot(ax=ax, nmin=nmin, clr=colour, fmt=fmt)
#        if i == 0:
#            ax.legend(loc=4)
    plt.draw()
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


def clf_mock_comp2(mockfof_file='lf_mockfof.pkl',
                   mockhalo_file='lf_mockhalo.pkl',
                   nmin=8, lf_lims=(-15.5, -23.5, 2e-7, 0.005),
                   schec_lims=(-1.9, 1.1, -21.6, -19.1),
                   schec_fn=SchecMag(), gauss_fn=LogNormal(),
                   Mmin_fit=-23, Mmax_fit=-17,
                   schec_p0=[[-20.2, -22, -19], [-1.2, -2, 1], [-2, -8, 0]],
                   gauss_p0=[[-21.5, -23, -20], [0.5, 0.1, 0.8], [-4, -7, -2]],
                   sigma=[1, 2], lc_limits=5, lc_step=32,
                   plot_file='clf_mock_comp.pdf', alpha=[0.5, 1],
                   xlabel=mag_label, ylabel=lf_label):
    """Compare FOF and halo mocks CLFs by central and satellite."""

    mockfof_dict = pickle.load(open(mockfof_file, 'rb'))
    mockhalo_dict = pickle.load(open(mockhalo_file, 'rb'))

    plot_size = (6.5, 6)
    sa_left = 0.15
    sa_bot = 0.06
    nbin = 4
    nrow, ncol = util.two_factors(nbin)

    schec_fn.Mstar = schec_p0[0][0]
    schec_fn.Mstar.min = schec_p0[0][1]
    schec_fn.Mstar.max = schec_p0[0][2]
    schec_fn.alpha = schec_p0[1][0]
    schec_fn.alpha.min = schec_p0[1][1]
    schec_fn.alpha.max = schec_p0[1][2]
    schec_fn.lgps = schec_p0[2][0]
    schec_fn.lgps.min = schec_p0[2][1]
    schec_fn.lgps.max = schec_p0[2][2]
    gauss_fn.M_c = gauss_p0[0][0]
    gauss_fn.M_c.min = gauss_p0[0][1]
    gauss_fn.M_c.max = gauss_p0[0][2]
    gauss_fn.sigma_c = gauss_p0[1][0]
    gauss_fn.sigma_c.min = gauss_p0[1][1]
    gauss_fn.sigma_c.max = gauss_p0[1][2]
    gauss_fn.lgps = gauss_p0[2][0]
    gauss_fn.lgps.min = gauss_p0[2][1]
    gauss_fn.lgps.max = gauss_p0[2][2]

    plt.clf()
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
    fig.set_size_inches(plot_size)
    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, xlabel, ha='center', va='center')
    fig.text(0.06, 0.5, ylabel, ha='center', va='center', rotation='vertical')
    plt.semilogy(basey=10, nonposy='clip')
    for i in range(nbin):
        key = f'M{i}'
        label = r'${\cal M}' + r'{}\ [{}, {}]$'.format(
                i+1, mbins_def[i], mbins_def[i+1])
        ax = axes.flat[i]
        ax.axis(lf_lims)
        ax.text(0.9, 0.9, label, ha='right', transform=ax.transAxes)
        if i == 0:
            ax.text(0.1, 0.7, 'Halo', c='b', transform=ax.transAxes)
            ax.text(0.1, 0.6, 'FoF', c='r', transform=ax.transAxes)

        axin = ax.inset_axes([0.25, 0.12, 0.4, 0.4])
        axin.patch.set_alpha(alpha[0])
#        axin.axis((-1.6, -0.4, -21.5, -20.0))
        axin.xaxis.set_label_position("top")
        axin.yaxis.set_label_position("right")
        axin.set_xlabel(r'$\alpha$')
        axin.set_ylabel(r'$M^*$')
        # rotation='horizontal', rotation_mode='anchor')
        for lf_dict, fmt, colour, label in zip(
                [mockhalo_dict, mockfof_dict], 'os', 'br', ['Halo', 'FoF']):
            phi = lf_dict[key + '_all_all']
            if label == 'Halo':
                phih = phi
            else:
                c, nu, p = phi.chi2(phih)
                print('chi2, nu, p =', c, nu, p)
                ax.text(0.9, 0.8, rf'$P_{{\chi^2}} = {p:3.1e}$',
                        ha='right', transform=ax.transAxes)
            for cs, fn in zip(('cen', 'sat'), (gauss_fn, schec_fn)):
                phi = lf_dict[key + '_' + cs + '_all']
                phi.fn_fit(fn, Mmin=Mmin_fit, Mmax=Mmax_fit, verbose=0)
                if cs == 'cen':
                    phi.plot(ax=ax, nmin=nmin, clr=colour, fmt=fmt,
                             label=label, alpha=alpha, markersize=5)
                else:
                    phi.plot(ax=ax, nmin=nmin, clr=colour, fmt=fmt, ls='--',
                             mfc='w', alpha=alpha, markersize=5)
                    rproj = RegionProjection()
                    rproj.prepare(nloop=(lc_step, lc_step), fac=lc_limits,
                                  sigma=sigma)
                    rproj.calc(phi.fit, phi.fn.alpha, phi.fn.Mstar)
                    axin.plot(rproj.parval0, rproj.parval1, '+', color=colour)
                    xmin, xmax = rproj.min[0], rproj.max[0]
                    ymin, ymax = rproj.min[1], rproj.max[1]
                    nx, ny = rproj.nloop
                    extent = (xmin, xmax, ymin, ymax)
                    y = rproj.y.reshape((ny, nx))
                    v = rproj.levels
        #            pdb.set_trace()
                    axin.contour(y, v, origin='lower', extent=extent,
                                 colors=(colour,), alpha=alpha[0])
    plt.draw()
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


def csmf_trend_plots(infile='csmf_broad.pkl',
                     phi_limits=((1e-4, 2e-2), (4e-4, 5e-3), (1e-5, 5e-4)),
                     plot_file='csmf_non_par.pdf', flipy=False,
                     ylabel=csmf_label,
                     lbl_pref=r'$\lg {\cal M}_* =$'):
    """Plot non-parametric trends in broad bins."""
    clf_trend_plots(infile, phi_limits, plot_file, flipy, ylabel, lbl_pref)


def clf_trend_plots(infile='clf_broad.pkl',
#                    phi_limits=((4e-7, 1e-4), (2e-4, 2e-3), (1e-4, 8e-3)),
                    phi_limits=((2e-3, 5), (0.5, 40), (0.2, 80)),
                    plot_file='clf_non_par.pdf', flipy=True,
                    ylabel=clf_label,
                    lbl_pref=r'$^{0.1}M_r =$'):
    """Plot non-parametric trends in broad bins."""

    sa_left = 0.15
    sa_bot = 0.05
    lf_dict = pickle.load(open(infile, 'rb'))
    plt.clf()
    fig, axes = plt.subplots(3, 5, sharex=True, sharey=False, num=1)
    fig.set_size_inches(11, 6)
    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.05, wspace=0.0)
    fig.text(0.5, -0.02, r'$\log_{10} ({\cal M}_h / {\cal M}_\odot\ h^{-1})$',
             ha='center', va='center')
    fig.text(0.09, 0.5, ylabel, ha='center', va='center', rotation='vertical')
    labels = ['', 'b', 'r', 'nlo', 'nhi']
    descriptions = ['All', 'Blue', 'Red', 'low-n', 'high-n']

    for it in range(5):
        ax = axes[0, it]
        for i in range(6):
            key = 'M{}'.format(i) + labels[it]
            phi = lf_dict[key]
            for im in range(3):
                if flipy:
                    iy = 2-im
                else:
                    iy = im
                ax = axes[iy, it]
                ax.errorbar(halomass[i], phi.phi[im], phi.phi_err[im], fmt='o')
                ax.set_xlim(12.5, 14.5)
                ax.set_ylim(phi_limits[im])
                ax.semilogy(basey=10, nonposy='clip')
                if it > 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
                if iy == 0:
                    ax.text(0.5, 0.9, descriptions[it], ha='center',
                            transform=ax.transAxes)
                if it == 0:
                    ax.text(0.1, 0.1, lbl_pref + ' $[{}, {}]$'.format(
                            phi.bins[im], phi.bins[im+1]),
                            transform=ax.transAxes)

    plt.draw()
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


def csmf_ev_plots(intemp='smf_comp_vmax_z{}_{}.pkl', gal_file='smf_comp_z.pkl',
                  nmin=3, lf_lims=[8.1, 12.0, 1e-6, 3e-3],
                  plot_file='csmf_comp_ev.pdf',
                  plot_schec_file='csmf_comp_ev_schec.pdf',
                  plot_size=(12, 5),
                  xlabel=ms_label, ylabel=smf_label, alpha=0.5, plot_ratio=0):
    """Plot and tabulate galaxy CLFs in redshift bins."""
    clf_ev_plots(intemp=intemp, gal_file=gal_file, nmin=nmin, lf_lims=lf_lims,
                 plot_file=plot_file, plot_schec_file=plot_schec_file,
                 plot_size=plot_size, xlabel=xlabel, ylabel=ylabel,
                 alpha=alpha, plot_ratio=plot_ratio)


def clf_m24_ev_plots(intemp='lf_m24_vmax_z{}_{}.pkl', mbins=(0,),
                     plot_file='clf_m24_ev.pdf', plot_size=(12, 3),):
    """Plot compbined mass bin 2-4 group galaxy LFs in redshift bins."""
    clf_ev_plots(intemp=intemp, mbins=mbins, plot_file=plot_file,
                 plot_size=plot_size)


def clf_m34_ev_plots(intemp='lf_m34_vmax_z{}_{}.pkl', mbins=(0,),
                     plot_file='clf_m34_ev.pdf', plot_size=(12, 3),):
    """Plot compbined mass bin 3-4 group galaxy LFs in redshift bins."""
    clf_ev_plots(intemp=intemp, mbins=mbins, plot_file=plot_file,
                 plot_size=plot_size)


def clf_ev_plots(intemp='lf_vmax_z{}_{}.pkl', gal_file='lfr_z.pkl',
                 mocktemp='lf_mock_z{}_{}.pkl',
                 mock_gal_file='lfr_z_mock.pkl', mbins=range(1, 3),
                 nmin=3, lf_lims=[-15.7, -23.6, 1e-6, 2e-3],
                 plot_file='clf_ev.pdf', plot_schec_file='clf_ev_schec.pdf',
                 plot_size=(12, 5), censat=False,
                 mkey=(3, 4, 56), xlabel=mag_label, ylabel=lf_label,
                 zlabel=[r'$0.0 < z < 0.1$', r'$0.1 < z < 0.2$',
                         r'$0.2 < z < 0.3$'],
                 alpha=0.5, plot_ratio=0):
    """Plot group galaxy LFs in redshift bins."""

    if plot_ratio:
        lf_lims[2:4] = (0.1, 50)
        print(lf_lims)
    lf_dict_list = []
#    mock_dict_list = []
    labels = ['all', 'blue', 'red', 'nlo', 'nhi']
    descriptions = ['All', 'Blue', 'Red', 'low-n', 'high-n']
    for iz in range(len(zbins)-1):
        zlo, zhi = zbins[iz], zbins[iz+1]
        phid = pickle.load(open(intemp.format(zlo, zhi), 'rb'))
        lf_dict_list.append(phid)
#        phid = pickle.load(open(mocktemp.format(zlo, zhi), 'rb'))
#        mock_dict_list.append(phid)

    gal_dict = pickle.load(open(gal_file, 'rb'))
#    pdb.set_trace()
#    lf_dict = {**lf_dict, **mock_dict}
#    mock_gal_dict = pickle.load(open(mock_gal_file, 'rb'))
#    gal_dict = {**gal_dict, **mock_gal_dict}

#    fn = models.ExponentialCutoffPowerLaw1D(
#            amplitude=p0[2], x_0=1.0, alpha=-p0[0], x_cutoff=1.0,
#            tied={'x_cutoff': lambda s: s.x_0})
#            bounds={'amplitude': (1e-5, 1e2), 'x_0': (1e-5, 1e5)})

    sa_left = 0.05
    sa_bot = 0.07
#    mbins = range(2, 6)
    mlabels = (r' ${\cal M}2$', r' ${\cal M}3$', r' ${\cal M}4$')
#    mlabels = (r' ${\cal M}3$', r' ${\cal M}4$', r' ${\cal M}56$')
#    mlabels = (r' ${\cal M}3$', r' ${\cal M}4$', r' ${\cal M}5$', r' ${\cal M}6$')
    nsamp = len(labels)
    nrow, ncol = len(mbins), len(labels)
    colors = 'bgr'
    fmt = 'vo^'

    # Both figs have galaxy type by row, group mass by column
    # Fig 1: LF in redshift bins
    fig1, axes1 = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
    fig1.set_size_inches(plot_size)
    fig1.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
    fig1.text(0.5, 0.0, xlabel, ha='center', va='center')
    fig1.text(0.0, 0.5, ylabel, ha='center', va='center', rotation='vertical')
    plt.semilogy(basey=10, nonposy='clip')

    # Fig 2: (M*, lg phi*) likelihood contours
#    fig2, axes2 = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=2)
#    fig2.set_size_inches(plot_size)
#    fig2.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
#    fig2.text(0.51, 0.0, r'$^{0.1}M^* - 5 \log_{10} h$', ha='center', va='center')
#    fig2.text(0.06, 0.46, r'$\log_{10} \phi^*$', ha='center', va='center',
#              rotation='vertical')

#    rproj_dict = {}
    for it in range(nsamp):
        lbl = labels[it]
        desc = descriptions[it]
        for im in mbins:
            key = f'M{im}'
            label = desc + mlabels[im]
#            label = desc + r' ${\cal M}$' + '{}'.format(im+1)
            if len(mbins) > 1:
                ax1 = axes1[im-mbins[0], it]
            else:
                ax1 = axes1[it]
            ax1.axis(lf_lims)
            ax1.plot(lf_lims[0:2], (1, 1), 'k:')
            ax1.text(0.9, 0.9, label, ha='right', transform=ax1.transAxes)
#            ax2 = axes2[it, im-mbins[0]]
#            ax2.axis(schec_lims)
#            ax2.text(0.9, 0.9, label, ha='right',
#                     transform=ax2.transAxes)
            # Normalise field LF to same total numbers of galaxies in mass+type bin
            ngrouped, nfield = 0, 0
            for iz in range(3):
                phi = lf_dict_list[iz][key + '_all' + '_' + lbl]
                ngrouped += np.sum(phi.ngal)
                phi = gal_dict[f'z{iz}_{lbl}']
                nfield += np.sum(phi.ngal)
            norm = ngrouped/nfield
            for iz in range(3):
                if (im < 2) or (iz > 0):
                    phi = lf_dict_list[iz][key + '_all' + '_' + lbl]
                    gkey = f'z{iz}_{lbl}'
                    gphi = gal_dict[gkey]
                    if it == 0:
                        ax1.text(0.1, 0.28 - 0.11*iz,
                                 r"$\lg\ {\cal M}_h = $" + fr"{phi.info['lgm_av']:5.2f}",
                                 ha='left', transform=ax1.transAxes, c=colors[iz])
                    if plot_ratio:
                        comp = phi.comp * (phi.ngal >= nmin)
                        phir = phi.phi[comp]/(norm*gphi.phi[comp])
                        phir_err = (phir**2 * ((phi.phi_err[comp]/phi.phi[comp])**2 +
                                    (gphi.phi_err[comp]/gphi.phi[comp])**2))**0.5
                        ax1.errorbar(phi.Mbin[comp], phir, phir_err, fmt=fmt[iz],
                                     color=colors[iz], label=zlabel[iz], alpha=alpha)
                    else:
                        phi.plot(ax=ax1, nmin=nmin, clr=colors[iz], fmt=fmt[iz],
                                 show_fit=False, label=zlabel[iz],
                                 markersize=8, alpha=[0.5, 1])
                        gphi.plot(ax=ax1, norm=norm, nmin=nmin, fmt='-', clr=colors[iz],
                                  show_fit=False)

            if not(plot_ratio) and im == mbins[0] and it == 1:
                ax1.legend(loc=3)
    plt.draw()
    plt.figure(1)
    if plot_file:
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
#    plt.figure(2)
#    plt.savefig(plot_dir + plot_schec_file, bbox_inches='tight')
    plt.show()


def clf_ev_comp(infile_all='clf.pkl', infile_ev='clf_ev.pkl',
                nmin=1, lf_lims=(-15.5, -23.9, 1e-7, 0.008),
                plot_file='clf_ev_comp.pdf', plot_size=(6, 8),
                xlabel=mag_label, ylabel=clf_label):
    """Compare ev-corrected LF with low-z slice."""

    lf_dict_all = pickle.load(open(infile_all, 'rb'))
    lf_dict_ev = pickle.load(open(infile_ev, 'rb'))

    pdb.set_trace()
    sa_left = 0.12
    sa_bot = 0.04
    nbin = 6
    nrow, ncol = util.two_factors(nbin)

    plt.clf()
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
    fig.set_size_inches(plot_size)
    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, xlabel, ha='center', va='center')
    fig.text(0.06, 0.5, ylabel, ha='center', va='center', rotation='vertical')
    plt.semilogy(basey=10, nonposy='clip')
    for i in range(nbin):
        key = 'M{}'.format(i)
        label = r'${\cal M}' + r'{}\ [{}, {}]$'.format(
                i+1, mbins_def[i], mbins_def[i+1])
        ax = axes.flat[i]
        ax.axis(lf_lims)
        ax.text(0.9, 0.9, label, ha='right', transform=ax.transAxes)
        phi = lf_dict_all[key]
        phi.plot(ax=ax, nmin=nmin, clr='b', label='All')
        phi = lf_dict_ev[key+'z0']
        phi.plot(ax=ax, nmin=nmin, clr='g', label='z < 0.1')
        if i == 0:
            ax.legend(loc=3)
    plt.draw()
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


def ngrp_vmax_comp_old(infile_ngrp='clf.pkl', infile_vmax='clf_vmax.pkl',
                   infile_vol='clf_vol.pkl', infile_vol_vmax='clf_vol_vmax.pkl',
                   nmin=1, Mrange=(-15.5, -23.9), vmax_range=(1e-7, 0.03),
                   ngrp_range=(0.001, 200),
                   plot_file='clf_ngrp_vmax_comp.pdf', plot_size=(5.5, 7.5),
                   xlabel=mag_label,
                   ngrp_label=clf_label, vmax_label=lf_label):
    """Compare ngrp and vmax normalisation."""

    lf_dict_ngrp = pickle.load(open(infile_ngrp, 'rb'))
    lf_dict_vmax = pickle.load(open(infile_vmax, 'rb'))
#    lf_dict_vol = pickle.load(open(infile_vol, 'rb'))
#    lf_dict_vol_vmax = pickle.load(open(infile_vol_vmax, 'rb'))

    sa_left = 0.11
    sa_right = 0.89
    sa_bot = 0.05
    nbin = 6
    nrow, ncol = util.two_factors(nbin)

    plt.clf()
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
    fig.set_size_inches(plot_size)
    fig.subplots_adjust(left=sa_left, right=sa_right, bottom=sa_bot,
                        hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, xlabel, ha='center', va='center')
    fig.text(0.0, 0.5, ngrp_label, ha='center', va='center', rotation='vertical')
    fig.text(1.0, 0.5, vmax_label, ha='center', va='center', rotation='vertical')
#    plt.semilogy(basey=10, nonposy='clip')
#    plt.xlim(Mrange)
    for i in range(nbin):
        key = 'M{}'.format(i)
        label = r'${\cal M}' + r'{}\ [{}, {}]$'.format(
                i+1, mbins_def[i], mbins_def[i+1])
        ax = axes.flat[i]
        ax.set_ylim(ngrp_range)
        ax.semilogy(basey=10, nonposy='clip')
        ax.text(0.9, 0.9, label, ha='right', transform=ax.transAxes)
        phi = lf_dict_ngrp[key]
        h_ngrp = phi.plot(ax=ax, nmin=nmin, clr='b', label='CLF $\phi_C$')
#        phi = lf_dict_vol[key]
#        phi.plot(ax=ax, nmin=nmin, clr='b', mfc='w', show_fit=0)

        ax = axes.flat[i].twinx()
        ax.set_xlim(Mrange)
        ax.set_ylim(vmax_range)
        ax.semilogy(basey=10, nonposy='clip')
        phi = lf_dict_vmax[key]
        h_vmax = phi.plot(ax=ax, nmin=nmin, fmt='s', clr='g', label='LF $\phi$')
#        phi = lf_dict_vol_vmax[key]
#        phi.plot(ax=ax, nmin=nmin, fmt='s', clr='k', mfc='w', show_fit=0)
        if i in (0, 2, 4):
            ax.axes.yaxis.set_ticklabels([])
        if i == 0:
            plt.legend(handles=[h_ngrp, h_vmax], loc=3)
    plt.draw()
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


def ngrp_vmax_comp(infile_ngrp='clf.pkl', infile_vmax='clf_vmax.pkl',
                   nmin=1, Mrange=(-15.5, -23.9), vmax_range=(1e-7, 0.03),
                   ngrp_range=(0.001, 200),
                   plot_file='clf_ngrp_vmax_comp.pdf', plot_size=(5.5, 7.5),
                   xlabel=mag_label,
                   ngrp_label=clf_label, vmax_label=lf_label):
    """Compare ngrp and vmax normalisation."""

    lf_dict_ngrp = pickle.load(open(infile_ngrp, 'rb'))
    lf_dict_vmax = pickle.load(open(infile_vmax, 'rb'))

    sa_left = 0.11
    sa_right = 0.89
    sa_bot = 0.05
    nbin = 6
    nrow, ncol = util.two_factors(nbin)

    plt.clf()
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
    fig.set_size_inches(plot_size)
    fig.subplots_adjust(left=sa_left, right=sa_right, bottom=sa_bot,
                        hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, xlabel, ha='center', va='center')
    fig.text(0.0, 0.5, ngrp_label, ha='center', va='center', rotation='vertical')
#    fig.text(1.0, 0.5, vmax_label, ha='center', va='center', rotation='vertical')
#    plt.semilogy(basey=10, nonposy='clip')
#    plt.xlim(Mrange)
    for i in range(nbin):
        key = 'M{}'.format(i)
        label = r'${\cal M}' + r'{}\ [{}, {}]$'.format(
                i+1, mbins_def[i], mbins_def[i+1])
        ax = axes.flat[i]
        ax.set_xlim(Mrange)
        ax.set_ylim(ngrp_range)
        ax.semilogy(basey=10, nonposy='clip')
        ax.text(0.9, 0.9, label, ha='right', transform=ax.transAxes)
        phi = lf_dict_ngrp[key]
        comp = phi.comp
        comp *= (phi.ngal >= nmin)
        ax.errorbar(phi.Mbin[comp], phi.phi[comp], phi.phi_err[comp],
                    fmt='o', color='b', label=r'CLF $\phi_C$')
        phisum = np.sum(phi.phi[comp])

        phi = lf_dict_vmax[key]
        comp = phi.comp
        comp *= (phi.ngal >= nmin)
        scale = phisum/np.sum(phi.phi[comp])
        ax.errorbar(phi.Mbin[comp], scale*phi.phi[comp], scale*phi.phi_err[comp],
                    fmt='s', color='g', label=r'LF $\phi$')
        if i == 0:
            ax.legend(loc=3)
    plt.draw()
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


def group_lf_write(infile, outfile):
    """Write out grouped LFs as text files."""

    lf_dict = pickle.load(open(infile, 'rb'))
    f = open(outfile, 'w')
    for key in lf_dict.keys():
        phi = lf_dict[key]
        phi.write(f, key)
    f.close()


def yang(infile, lgmh, lglm):
    """Return Yang et al central and satellite CLF/CSMF at log lum/mass bins
    lglm by interpolating parameters for halo of log mass lgmh."""

    # Read Yang data and sort by increasing halo mass
    dat = np.loadtxt(lf_data + 'Yang/' + infile, delimiter=',')
    idx = np.argsort(dat[:, 2])
    dat = dat[idx, :]
    mav = dat[:, 2]
    assert np.all(np.diff(mav) > 0)

    # Interpolate (in log space for phi*)
    phistar = 10**np.interp(lgmh, mav, np.log10(dat[:, 3]))
    alpha = np.interp(lgmh, mav, dat[:, 5])
    lgmstar_c = np.interp(lgmh, mav, dat[:, 7])
    sigma_c = np.interp(lgmh, mav, dat[:, 9])
    print('Interpolated fit values at {}: {:6.3f} {:6.3f} {:6.3f} {:6.3f}'.format(
            lgmh, phistar, alpha, lgmstar_c, sigma_c))
    A = 1
    phi_c = A/(2*math.pi)**0.5/sigma_c * np.exp(
            -(lglm - lgmstar_c)**2 / (2*sigma_c**2))
    Mr = 10**(lglm - (lgmstar_c - 0.25))
    phi_s = phistar * Mr**(alpha+1) * np.exp(-Mr**2)
    if 'clf' in infile:
        # Convert from per dex lum to per unit mag
        phi_c, phi_s = 0.4*phi_c, 0.4*phi_s
    return phi_c, phi_s


def yang_par(infile):
    """Return Yang et al central and satellite CLF/CSMF parameters
    in bins of log mass."""

    # Read Yang data and sort by increasing halo mass
    dat = np.loadtxt(lf_data + 'Yang/' + infile, delimiter=',')
    lgmh = dat[:, 2]

    # Interpolate (in log space for phi*)
    phistar = dat[:, 3]
    alpha = dat[:, 5]
    lgmstar_c = dat[:, 7]
    lgmstar_s = lgmstar_c - 0.25
    sigma_c = dat[:, 9]
    if 'clf' in infile:
        # Convert log L quanitities to magnitudes
        Mr_sun = 4.76
        lgmstar_c = Mr_sun - 2.5*lgmstar_c
        lgmstar_s = Mr_sun - 2.5*lgmstar_s
        sigma_c *= 2.5
    return {'lgmh': lgmh, 'phistar': phistar, 'alpha': alpha,
            'lgmstar_c': lgmstar_c, 'lgmstar_s': lgmstar_s, 'sigma_c': sigma_c}


def yang_par_interp(infile, lgmh):
    """Return Yang et al central and satellite CLF/CSMF parameters
    by interpolating haloes of log mass lgmh."""

    # Read Yang data and sort by increasing halo mass
    dat = np.loadtxt(lf_data + 'Yang/' + infile, delimiter=',')
    idx = np.argsort(dat[:, 2])
    dat = dat[idx, :]
    mav = dat[:, 2]
    assert np.all(np.diff(mav) > 0)

    # Interpolate (in log space for phi*)
    phistar = 10**np.interp(lgmh, mav, np.log10(dat[:, 3]))
    alpha = np.interp(lgmh, mav, dat[:, 5])
    lgmstar_c = np.interp(lgmh, mav, dat[:, 7])
    lgmstar_s = lgmstar_c - 0.25
    sigma_c = np.interp(lgmh, mav, dat[:, 9])
    if 'clf' in infile:
        # Convert log L quanitities to magnitudes
        Mr_sun = 4.76
        lgmstar_c = Mr_sun - 2.5*lgmstar_c
        lgmstar_s = Mr_sun - 2.5*lgmstar_s
        sigma_c *= 2.5
    return {'phistar': phistar, 'alpha': alpha, 'lgmstar_c': lgmstar_c,
            'lgmstar_s': lgmstar_s, 'sigma_c': sigma_c}


def yang_plot(phistar, alpha, lgmstar_c, sigma_c, A, lgm, ls='--'):
    """Plot Yang+ functional fit (log-normal for centrals, modified Schechter
    for satellites)."""
    phi_c = A/(2*math.pi)**0.5/sigma_c * np.exp(
            -(lgm - lgmstar_c)**2 / (2*sigma_c**2))
    Mr = 10**(lgm - (lgmstar_c - 0.25))
    phi_s = phistar * Mr**(alpha+1) * np.exp(-Mr**2)
    plt.plot(lgm, phi_c)
    plt.plot(lgm, phi_s)


def zandivarez():
    """Return Zandivarez2011 Schechter fit parameters as a dictionary."""

    data = np.loadtxt(lf_data + 'Zandivarez2011.txt')
    zand = {}
    samples = ('all', 'early', 'late', 'red', 'blue')
    for isamp in range(5):
        ilo = 10*isamp
        ihi = 10*(isamp+1)
        mass = data[ilo:ihi, 2]
        mstar = data[ilo:ihi, 4]
        mstar_err = data[ilo:ihi, 5]
        alpha = data[ilo:ihi, 6]
        alpha_err = data[ilo:ihi, 7]
        zand[samples[isamp]] = {'mass': mass,
                                'mstar': mstar, 'mstar_err': mstar_err,
                                'alpha': alpha, 'alpha_err': alpha_err}
    return zand


# Simulation routines to generate and analyse samples with known CLF
def clf_sim_gen10k(nsim=9, nhalo=10000, mlimits=(0, 19.8), poisson=True,
                   grp_file='sim_group_10k.fits', gal_file='sim_gal_10k.fits'):
    """Mock CLF simulations with only 10k groups per sim."""
    clf_sim_gen(nsim=nsim, nhalo=nhalo, mlimits=mlimits, poisson=poisson,
                grp_file=grp_file, gal_file=gal_file, verbose=True)


def smf_sim_gen10k(nsim=9, nhalo=10000, mlimits=(0, 19.8), poisson=True,
                   grp_file='sim_smf_group_10k.fits',
                   gal_file='sim_smf_gal_10k.fits'):
    """Mock CSMF simulations with only 10k groups per sim."""
    clf_sim_gen(nsim=nsim, nhalo=nhalo, mlimits=mlimits, poisson=poisson,
                grp_file=grp_file, gal_file=gal_file, smf=True)


def smf_sim_gen(nsim=9, nhalo=50000, mlimits=(0, 19.8), poisson=True,
                grp_file='sim_smf_group.fits',
                gal_file='sim_smf_gal.fits'):
    """Mock CSMF simulations."""
    clf_sim_gen(nsim=nsim, nhalo=nhalo, mlimits=mlimits, poisson=poisson,
                grp_file=grp_file, gal_file=gal_file, smf=True)


def clf_sim_gen(nsim=9, nhalo=50000, mlimits=(0, 19.8), poisson=True,
                grp_file='sim_group.fits', gal_file='sim_gal.fits',
                smf=False, verbose=False):
    """Mock CLF simulation.  Place haloes with mass drawn from specified HMF
    at random, add galaxies drawn from specified CLF, then add visibility limits."""

    # K- and e-corrections - same as for GAMA group mocks
#    global cosmo, kz0, ez0
    kz0 = 0.2
    ez0 = 0
    Q = 0
    P = 0
    pcoeff = (0.2085, 1.0226, 0.5237, 3.5902, 2.3843)
    zlimits = (0.002, 0.5)
    H0 = 100.0
    omega_l = 0.75
    cosmo = util.CosmoLookup(H0, omega_l, zlimits, P=P)
    kcorr = gs.Kcorr(kz0, pcoeff)
    ecorr = gs.Ecorr(ez0, Q)

    # HMF Schechter function parameters
    hmf = SchecMass()
    hmf.alpha = -1
    hmf.Mstar = 14
    hmf.lgps = 1
    hmf_lgmmin, hmf_lgmmax, nmass = 12, 15, 6
    lgm_bins = np.linspace(hmf_lgmmin, hmf_lgmmax, nmass+1)
    lgms = lgm_bins[:-1] + 0.5*np.diff(lgm_bins)

    # CLF alpha = a0 + a1*dlgm, M* = m0 + m1*dlgm, lg phi* = p0 + p1*dlgm
    # where dlgm = hmf_lgm - hmf_lgmstar
    if smf:
        clff = SchecMass()
        a0, a1, m0, m1, p0, p1 = -1.4, -0.2, 11.2, 0.4, 1, 0.6
        Mmin, Mmax, nmag = 8, 12, 16
    else:
        clff = SchecMag()
        a0, a1, m0, m1, p0, p1 = -1.4, -0.2, -21, -0.5, 1, 0.5
        Mmin, Mmax, nmag = -24, -15, 18

    mag_bins = np.linspace(Mmin, Mmax, nmag+1)
    mags = mag_bins[:-1] + 0.5*np.diff(mag_bins)
    dmag = mags[1] - mags[0]

#    plt.clf()
#    for lgm in range(12, 16):
#        dlgm = hmf.Mstar._val - lgm
#        plt.plot(a0 + a1*dlgm, m0 + m1*dlgm, '+', label='{}'.format(lgm))
#    plt.xlabel('alpha')
#    plt.ylabel('M*')
#    plt.legend()
#    plt.show()

    lgm_av = np.zeros(nmass)
    gal_hist = np.zeros((nmag, nmass, nsim))
    lgm_hist = np.zeros((nmass, nsim))
    grp_table = Table(names=('Volume', 'GroupID', 'Nfof', 'IterCenZ', 'log_mass'),
                      dtype=('i4', 'i4', 'i4', 'f8', 'f8'))
    gal_table = Table(names=('Volume', 'GroupID', 'GalID', 'logmstar',
                             'Rabs', 'Rpetro', 'z'),
                      dtype=('i4', 'i4', 'i4', 'f8', 'f8', 'f8', 'f8'))
    galid = 0
    ngal_tot = 0
    ngrp_tot = 0
    ngrp_vis = 0
    for isim in range(nsim):
        volume = isim + 1
        ngrp_tot += nhalo
        lgm = util.ran_fun(hmf, hmf_lgmmin, hmf_lgmmax, nhalo)
        zz = util.ran_fun(cosmo.vol_ev, zlimits[0], zlimits[1], nhalo)
        dlgm = lgm - hmf.Mstar._val
        lgm_hist[:, isim] = np.histogram(lgm, lgm_bins)[0]

        ngals = np.zeros(nhalo, dtype=np.int)
        for ihalo in range(nhalo):
            groupid = isim*nhalo + ihalo
            clff.alpha = a0 + a1*dlgm[ihalo]
            clff.Mstar = m0 + m1*dlgm[ihalo]
            clff.lgps = p0 + p1*dlgm[ihalo]
            ngal = scipy.integrate.quad(clff, Mmin, Mmax)
            if poisson:
                ngal = np.random.poisson(ngal[0])
            else:
                ngal = round(ngal[0])
            ngals[ihalo] = ngal
            ngal_tot += ngal
            if smf:
                logmstar = util.ran_fun(clff, Mmin, Mmax, ngal)
                M = (np.random.normal(scale=0.43, size=ngal) - 
                     1.88 - 1.79*logmstar)
            else:
                logmstar = np.zeros(ngal)
                M = util.ran_fun(clff, Mmin, Mmax, ngal)
            ibin = int(2*(lgm[ihalo] - hmf_lgmmin))
            gal_hist[:, ibin, isim] += np.histogram(M, mag_bins)[0]
            lgm_av[ibin] += lgm[ihalo]

            # Select visible galaxies to output
            z = zz[ihalo]
            mapp = M + cosmo.dist_mod(z) + kcorr(z) - ecorr(z)
            vis = (mlimits[0] <= mapp) * (mapp < mlimits[1])
            Nfof = len(mapp[vis])
            grp_table.add_row([volume, groupid, Nfof, z, lgm[ihalo]])
            if verbose:
                print(lgm[ihalo], ngal, z, Nfof)
            if Nfof > 0:
                ngrp_vis += 1
                for i in range(Nfof):
                    gal_table.add_row([volume, groupid, galid,
                                       logmstar[vis][i], M[vis][i],
                                       mapp[vis][i], z])
                    galid += 1

    phi = gal_hist/lgm_hist
    phi_ngal = np.mean(gal_hist, axis=2)
    phi_mean = np.mean(phi, axis=2) / dmag
    phi_err = np.std(phi, axis=2) / dmag
    lgm_av /= lgm_hist.sum(axis=1)

    grp_table.meta = {'hmfmstar': hmf.Mstar._val, 'a0': a0, 'a1': a1,
                      'm0': m0, 'm1': m1, 'p0': p0, 'p1': p1,
                      'mlo': mlimits[0], 'mhi': mlimits[1],
                      'zlo': zlimits[0], 'zhi': zlimits[1]}
    grp_table.write(grp_file, overwrite=True)
    gal_table.write(gal_file, overwrite=True)
    print(galid, 'out of', ngal_tot, 'galaxies visible in',
          ngrp_vis, 'out of', ngrp_tot, 'groups')

#    plt.clf()
#    plt.hist(lgm)
#    plt.semilogy(basey=10)
#    plt.xlabel('log M')
#    plt.ylabel('Number')
#    plt.show()

    plt.clf()
    plt.scatter(lgm, ngals, s=0.1)
    plt.semilogy(basey=10)
    plt.xlabel('Halo mass')
    plt.ylabel('Ngal')
    plt.show()
#    pdb.set_trace()

    plt.clf()
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, num=1)
    fig.set_size_inches(8, 8)
    fig.subplots_adjust(left=0.05, bottom=0.05, hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, 'Mag', ha='center', va='center')
    fig.text(0.0, 0.5, 'phi', ha='center', va='center', rotation='vertical')
    for ibin in range(nmass):
        ax = axes.flat[ibin]
        ax.errorbar(mags, phi_mean[:, ibin], phi_err[:, ibin])
        lgm = lgm_av[ibin]
        dlgm = hmf.Mstar._val - lgm
        clff.alpha = a0 + a1*dlgm
        clff.Mstar = m0 + m1*dlgm
        clff.lgps = p0 + p1*dlgm
        ax.plot(mags, clff(mags))
        ax.semilogy(basey=10)
        ax.text(0.1, 0.8, '{:5.2f}'.format(lgm), transform=ax.transAxes)
    plt.show()

    # Schechter fn fit to each mass bin
#    fn = SchecMag()
#    fn.alpha = -1
#    fn.Mstar = -21
#    fn.lgps = 1
#    lff = lf.LF(None, None, mag_bins, error='mock')
#
#    print('Mass   alpha true, est       M* true, est    lgphi* true, est')
#    for ibin in range(nmass):
#        lgm = lgms[ibin]
#        dlgm = hmf.Mstar._val - lgm
#        lff.ngal = phi_ngal[:, ibin]
#        lff.phi = phi_mean[:, ibin]
#        lff.phi_err = phi_err[:, ibin]
#        fn = lff.fn_fit(fn, Mmin=Mmin, Mmax=Mmax, verbose=0)
#        fit_errs = 0.5*(np.array(lff.errors.parmaxes) - np.array(lff.errors.parmins))
#        print(r'{:5.2f}  {:5.2f}  {:5.2f}\pm{:5.2f}   {:5.2f}  {:5.2f}\pm{:5.2f} {:5.2f}  {:5.2f}\pm{:5.2f} '.format(
#                lgm, a0 + a1*dlgm, lff.res.parvals[0], fit_errs[0],
#                m0 + m1*dlgm, lff.res.parvals[1], fit_errs[1],
#                clff.lgps._val, lff.res.parvals[2], fit_errs[2]))


def sim_grp_mass_z():
    """Halo mass-redshift plot."""
    mass_z(grpfile='sim_group.fits', nmin=5, edge_min=0.9,
           vmax=None, zrange=(0, 0.5),
           Mrange=(11.8, 15.4), plot_file=None, plot_size=(5, 4.5))


def sim_gal_mass_z_10k(galfile='sim_smf_gal_10k.fits',
                       grpfile='sim_smf_group_10k.fits', mass_est='sim',
                       plot_file='sim_gal_mass_z_10k.pdf', plot_size=(6, 4)):
    """Simulated lGrouped galaxy mass-redshift scatterplot."""
    sel_dict = {'log_mass': (12, 13.3)}
    gal_mass_z(galfile=galfile, grpfile=grpfile, mass_est=mass_est,
               zlimits=(0.0, 0.4), plot_file=plot_file, plot_size=plot_size,
               colour_by='log_mass', vmin=12, vmax=15,
               cb_label=r'$M_h$', sel_dict=sel_dict)


def sim_gal_mass_z(galfile='sim_smf_gal.fits',
                   grpfile='sim_smf_group.fits', mass_est='sim',
                   plot_file='sim_gal_mass_z.pdf', plot_size=(6, 4)):
    """Simulated lGrouped galaxy mass-redshift scatterplot."""
    gal_mass_z(galfile=galfile, grpfile=grpfile, mass_est=mass_est,
               plot_file=plot_file, plot_size=plot_size, hexbin=True)


def smf_sims(z=[0.002, 0.1, 0.2, 0.3]):
    """Conditional LFs for simulated groups."""
#    clf_sim(nmin=0, gal_file='smf_gal.fits', grp_file='smf_groupfits',
#            colname='logmstar', bins=np.linspace(8, 12, 17),
#            outfile='csmf_sim_nmin0.pkl')
#    clf_sim(nmin=1, gal_file='smf_gal.fits', grp_file='smf_groupfits',
#            colname='logmstar', bins=np.linspace(8, 12, 17),
#            outfile='csmf_sim_nmin1.pkl')
#    clf_sim(nmin=5, gal_file='smf_gal.fits', grp_file='smf_groupfits',
#            colname='logmstar', bins=np.linspace(8, 12, 17),
#            outfile='csmf_sim_nmin5.pkl')
#    clf_sim(nmin=1, gal_file='smf_gal.fits', grp_file='smf_groupfits',
#            colname='logmstar', bins=np.linspace(8, 12, 17),
#            Vmax='Vmax_raw', outfile='csmf_sim_nmin1_vmax.pkl')
    clf_sim(nmin=5, gal_file='sim_smf_gal.fits',
            grp_file='sim_smf_group.fits', masscomp='sim',
            colname='logmstar', bins=np.linspace(8, 12, 17),
            Vmax='Vmax_raw', clff=SchecMass(),
            outfile='csmf_sim_comp_nmin5_vmax.pkl')
    clf_sim(nmin=5, gal_file='sim_smf_gal.fits',
            grp_file='sim_smf_group.fits', masscomp=None,
            colname='logmstar', bins=np.linspace(8, 12, 17),
            Vmax='Vmax_raw', clff=SchecMass(),
            outfile='csmf_sim_incomp_nmin5_vmax.pkl')
    for iz in range(len(z) - 1):
        clf_sim(nmin=5, gal_file='sim_smf_gal.fits',
                grp_file='sim_smf_group.fits', masscomp=None,
                colname='logmstar', bins=np.linspace(8, 12, 17),
                zlimits=(z[iz], z[iz+1]), Vmax='Vmax_raw', clff=SchecMass(),
                outfile=f'csmf_sim_incomp_nmin5_z{z[iz]}_{z[iz+1]}_vmax.pkl')
        clf_sim(nmin=5, gal_file='sim_smf_gal.fits',
                grp_file='sim_smf_group.fits', masscomp='sim',
                colname='logmstar', bins=np.linspace(8, 12, 17),
                zlimits=(z[iz], z[iz+1]), Vmax='Vmax_raw', clff=SchecMass(),
                outfile=f'csmf_sim_comp_nmin5_z{z[iz]}_{z[iz+1]}_vmax.pkl')


def clf_sims():
    """Conditional SMFs for simulated groups."""
    clf_sim(nmin=0, outfile='clf_sim_nmin0.pkl')
    clf_sim(nmin=1, outfile='clf_sim_nmin1.pkl')
    clf_sim(nmin=5, outfile='clf_sim_nmin5.pkl')
    clf_sim(nmin=5, zlimits=[0.002, 0.1], outfile='clf_sim_nmin5_z01.pkl')
    clf_sim(nmin=5, zlimits=[0.002, 0.2], outfile='clf_sim_nmin5_z02.pkl')
    clf_sim(nmin=5, zlimits=[0.002, 0.3], outfile='clf_sim_nmin5_z03.pkl')
    clf_sim(nmin=1, Vmax='Vmax_raw', outfile='clf_sim_nmin1_vmax.pkl')
    clf_sim(nmin=5, Vmax='Vmax_raw', outfile='clf_sim_nmin5_vmax.pkl')


def clf_sims_10k():
    """Conditional LFs for simulated groups."""
    # clf_sim(nmin=0, gal_file='sim_gal_10k.fits', grp_file='sim_group_10k.fits',
    #         outfile='clf_sim_nmin0_10k.pkl')
    # clf_sim(nmin=1, gal_file='sim_gal_10k.fits', grp_file='sim_group_10k.fits',
    #         outfile='clf_sim_nmin1_10k.pkl')
    # clf_sim(nmin=5, gal_file='sim_gal_10k.fits', grp_file='sim_group_10k.fits',
    #         outfile='clf_sim_nmin5_10k.pkl')
    # clf_sim(nmin=1, Vmax='Vmax_raw', gal_file='sim_gal_10k.fits',
    #         grp_file='sim_group_10k.fits', outfile='clf_sim_nmin1_vmax_10k.pkl')
    clf_sim(nmin=5, Vmax='Vmax_raw', gal_file='sim_gal_10k.fits',
            grp_file='sim_group_10k.fits',outfile='clf_sim_nmin5_vmax_10k.pkl')


def clf_sim_z(nmin=5, z=[0.002, 0.1, 0.2, 0.3], Vmax='Vmax_raw'):
    """Conditional LF for simulated groups in redshift slices."""
    for iz in range(len(z) - 1):
        clf_sim(nmin=nmin, zlimits=(z[iz], z[iz+1]), Vmax=Vmax,
                outfile=f'clf_sim_nmin{nmin}_z{z[iz]}_{z[iz+1]}_vmax.pkl')


def clf_sim(mbins=mbins_def, nmin=5, vol_z=None, zlimits=[0.002, 0.5],
            mlimits=[0, 19.8], Vmax='Vmax_grp', bins=np.linspace(-24, -16, 18),
            colname='r_petro', error='mock', Q=0, P=0, nsim=9, masscomp=False,
            gal_file='sim_gal.fits', grp_file='sim_group.fits',
            clff=SchecMag(), outfile='clf_sim.pkl',
            plot_file=None, plot_size=(5, 4)):
    """Conditional LF for simulated groups."""

    if Vmax == 'Vmax_grp':
        find_vis_groups = True
    else:
        find_vis_groups = False
    samp = gs.GalSample(Q=Q, P=P, mlimits=mlimits, zlimits=zlimits)
    samp.read_grouped(gal_file, grp_file, mass_est='sim', nmin=nmin,
                      masscomp=masscomp, find_vis_groups=find_vis_groups)
    # pdb.set_trace()
    if colname == 'r_petro':
        samp.comp_limit_mag()
    if vol_z:
        samp.vol_limit_z(vol_z)
        samp.group_limit(nmin)

    if plot_file and nmin == 1 and zlimits[1] >= 0.5:
        plt.clf()
        plt.scatter(samp.grp['IterCenZ'], samp.grp['log_mass'], s=1,
                    c=samp.grp['Nfof'], vmin=1, vmax=8)
        plt.xlabel('Redshift')
        plt.ylabel(r'$\log_{10}({\cal M}_h/{\cal M}_\odot h^{-1})$')
        cb = plt.colorbar()
        cb.set_label(r'$N_{\rm fof}$')
        plt.draw()
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()

#    samp.vis_calc((samp.sel_mag_lo, samp.sel_mag_hi))
    samp.vmax_calc(denfile=None)
    # dm = samp.t['Rabs'] - samp.abs_mags('r_petro')
#    print('delta M = ', np.mean(dm), ' +- ', np.std(dm))
    V, err = scipy.integrate.quad(samp.cosmo.dV, samp.zlimits[0],
                                  samp.zlimits[1], epsabs=1e-3, epsrel=1e-3)
    V *= samp.area
    # print('area, vol =', samp.area, V)
    lf_dict = {}
    nmbin = len(mbins) - 1
    grps = table.unique(samp.t, keys='GroupID')
    lgm = grps['log_mass']
    grps_int = Table.read(grp_file)
    sel = (np.array(grps_int['IterCenZ'] >= samp.zlimits[0]) *
           np.array(grps_int['IterCenZ'] < samp.zlimits[1]))
    grps_int = grps_int[sel]
    lgm_int = grps_int['log_mass']

    plt.clf()
    plt.scatter(samp.t['z'], samp.abs_mags('r_petro'), s=0.1)
    plt.xlabel('z')
    plt.ylabel('M_r')
    plt.show()

    try:
        plt.clf()
        plt.scatter(samp.t['z'], samp.t['logmstar'], s=0.1)
        plt.xlabel('z')
        plt.ylabel('logmstar')
        plt.show()
    except KeyError:
        pass

    for i in range(nmbin):
        sel_dict = {'log_mass': (mbins[i], mbins[i+1])}
        samp.select(sel_dict)
        # Number of _intrinsic_ groups in this mass bin
        sel = (mbins[i] <= lgm_int) * (lgm_int < mbins[i+1])
        ngroup = len(grps_int[sel])/nsim
#        sel = (mbins[i] <= lgm) * (lgm < mbins[i+1])
#        ngroup = len(grps[sel])/9
        if Vmax == 'Vmax_grp':
            samp.vmax_group(mbins[i], mbins[i+1])
            norm = 1
        else:
#            norm = len(samp.t)/len(samp.tsel())
            norm = 1
        phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax, error=error,
                    sel_dict=sel_dict)

        # Average input LF over groups in this mass bin
        print('zlimits', *zlimits, 'im', i, 'ngroup/V', ngroup/V)
        sel = (mbins[i] <= lgm) * (lgm < mbins[i+1])
        phi.lgm_av = np.mean(lgm[sel])
        phi.lgm_av_gal = np.average(lgm[sel], weights=grps['Nfof'][sel])
        phi.ngroup = ngroup
        phi.vol = V
        meta = samp.meta
        phi_av = np.zeros(len(phi.phi))
        ng = 0
        for mass in lgm[sel]:
            dlgm = mass - meta['HMFMSTAR']
            clff.alpha = meta['A0'] + meta['A1']*dlgm
            clff.Mstar = meta['M0'] + meta['M1']*dlgm
            clff.lgps = meta['P0'] + meta['P1']*dlgm
            if Vmax != 'Vmax_grp':
                clff.lgps = clff.lgps._val + math.log10(phi.ngroup/phi.vol)
            phi_av += clff(phi.Mbin)
            phir = 0
            ng += 1
        phi_av /= ng
        phi.phi_av = phi_av

        # Ratio of recovered to input LF for each group
        idxs = samp.t.groups.indices
        phir_sum = np.zeros((len(phi.phi), nsim))
        ngal = np.zeros((len(phi.phi), nsim))
        if colname == 'logmstar':
            absmag = np.array([samp.t[colname][i]
                               for i in range(len(samp.t))])
            # absval = samp.tsel()[colname]
        else:
            # absval = samp.abs_mags(colname)
            absmag = np.array([samp.t[colname][i].abs
                               for i in range(len(samp.t))])
        wt = samp.t['cweight']/samp.t[Vmax]
        for igrp in range(len(samp.t.groups)):
            ilo = idxs[igrp]
            ihi = idxs[igrp+1]
            mass = samp.t['log_mass'][ilo]
            if (mbins[i] <= mass) and (mass < mbins[i+1]):
                phir, edges = np.histogram(absmag[ilo:ihi], bins,
                                           weights=wt[ilo:ihi])
                ng, edges = np.histogram(absmag[ilo:ihi], bins)
                phir /= np.diff(bins)

                dlgm = mass - meta['HMFMSTAR']
                clff.alpha = meta['A0'] + meta['A1']*dlgm
                clff.Mstar = meta['M0'] + meta['M1']*dlgm
                clff.lgps = meta['P0'] + meta['P1']*dlgm
                if Vmax != 'Vmax_grp':
                    clff.lgps = clff.lgps._val + math.log10(phi.ngroup/phi.vol)
                phir /= clff(phi.Mbin)
                isim = samp.t['Volume'][ilo] - 1
                phir_sum[:, isim] += phir
                ngal[:, isim] += ng
        phir = lf.LF(None, colname, bins, norm=norm, Vmax=Vmax, error=error,
                     sel_dict=sel_dict)
        phir.comp = phi.comp
        phir.Mbin = phi.Mbin
        phir.phi_jack = phir_sum
        phir.ngal = np.mean(ngal, axis=1)
        phir.phi = np.mean(phir.phi_jack, axis=1)
        phir.phi_err = np.std(phir.phi_jack, axis=1)
        # pdb.set_trace()

#        phi.fn_fit(fn=lf.Schechter_mag, Mmin=Mmin_fit, Mmax=Mmax_fit)
        Mkey = 'M{}mock'.format(i)
        lf_dict[Mkey] = phi
        Mkey = 'M{}mockr'.format(i)
        lf_dict[Mkey] = phir
    pickle.dump((samp.meta, lf_dict), open(outfile, 'wb'))


def clf_sim_plots_10k(infile='clf_sim_nmin5_vmax_10k.pkl', yrange=(1e-5, 1e-3)):
    """Plot conditional SMF for simulated groups."""
    clf_sim_plots(infile=infile, yrange=yrange)


def csmf_sim_plots(infile='csmf_sim_comp_nmin5_vmax.pkl', clff=SchecMass(),
                   nmin=2, mags=np.linspace(8, 12, 17), yrange=(1e-5, 0.1)):
    """Plot conditional SMF for simulated groups."""
    clf_sim_plots(infile=infile, clff=clff, nmin=nmin, mags=mags, yrange=yrange)


def clf_sim_plots(infiles=('clf_sim_nmin5_vmax.pkl',), clff=SchecMag(),
                  labels=None,
                  nmin=2, mags=np.linspace(-23, -16, 29), yrange=(1e-5, 1e-1)):
    """Plot conditional LF for simulated groups."""

    nfiles = len(infiles)
    meta_list = []
    lf_list = []
    for i in range(nfiles):
        (meta, lf_dict) = pickle.load(open(infiles[i], 'rb'))
        meta_list.append(meta)
        lf_list.append(lf_dict)
    plt.clf()
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, num=1)
    fig.set_size_inches(8, 8)
    fig.subplots_adjust(left=0.05, bottom=0.05, hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, 'Mag', ha='center', va='center')
    fig.text(0.0, 0.5, 'phi', ha='center', va='center', rotation='vertical')
    for ibin in range(4):
        for i in range(nfiles):
            phi = lf_list[i][f'M{ibin}mock']
            meta = meta_list[i]
            ax = axes.flat[ibin]
            phi.plot(ax=ax, nmin=nmin)
            lgm = phi.lgm_av
            dlgm = lgm - meta['HMFMSTAR']
            clff.alpha = meta['A0'] + meta['A1']*dlgm
            clff.Mstar = meta['M0'] + meta['M1']*dlgm
            clff.lgps = meta['P0'] + meta['P1']*dlgm
            # print(meta)
            # print(lgm, clff.alpha._val, clff.Mstar._val, clff.lgps._val)
            if 'vmax' in infiles[i]:
                clff.lgps = clff.lgps._val + math.log10(phi.ngroup/phi.vol)
            if labels:
                label = labels[i]
            else:
                label = None
            ax.plot(mags, clff(mags), label=label)
        if yrange:
            ax.set_ylim(yrange)
        ax.semilogy(basey=10)
        lbl = r'${\cal M}' + r'{}\ [{}, {}]$'.format(
                ibin+1, mbins_def[ibin], mbins_def[ibin+1])
        ax.text(0.9, 0.9, lbl, ha='right', transform=ax.transAxes)
        if ibin == 0:
            ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncol=len(infiles), handlelength=2, borderaxespad=0.)
    plt.show()


def clf_sim_plotr_10k(infiles=('clf_sim_nmin0_10k.pkl', 'clf_sim_nmin1_10k.pkl',
                           'clf_sim_nmin5_10k.pkl'),
                      npmin=5, lf_lims=(-15.5, -23.9, 0.5, 5),
                      plot_file='clf_sim_10k.pdf', plot_size=(6, 7.5)):
    """Plot ratio of redcovered conditional LF to simulated input."""
    clf_sim_plotr(infiles=infiles,
                  npmin=npmin, lf_lims=lf_lims,
                  plot_file=plot_file, plot_size=plot_size)


def clf_sim_plotr(infiles=('clf_sim_nmin0.pkl', 'clf_sim_nmin1.pkl',
                           'clf_sim_nmin5.pkl'),
                  labels=None, loc=0, npmin=5, lf_lims=(-15.5, -23.9, 0.5, 5),
                  yticks=(0.6, 1, 2, 3, 4), ylegend=0.69, alpha=0.8,
                  plot_file='clf_sim.pdf', plot_size=(6, 5.5)):
    """Plot ratio of recovered conditional LF to simulated input."""

    nfiles = len(infiles)
    meta_list = []
    lf_list = []
    for i in range(nfiles):
        (meta, lf_dict) = pickle.load(open(infiles[i], 'rb'))
        meta_list.append(meta)
        lf_list.append(lf_dict)
    plt.clf()
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, num=1)
    fig.set_size_inches(plot_size)
    fig.subplots_adjust(left=0.13, bottom=0.08, hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, mag_label, ha='center', va='center')
    if 'vmax' in infiles[0]:
        ylabel = r'$\phi(M) / \phi_{\rm sim}(M)$'
    else:
        ylabel = r'$\phi_C(M) / \phi_{C, {\rm sim}}(M)$'
    fig.text(0.06, 0.5, ylabel, ha='center', va='center', rotation='vertical')
    for ibin in range(4):
        ax = axes.flat[ibin]
        ax.plot(lf_lims[:2], (1, 1), 'k:')

        # Recovered LFs normalised by simulated LF
        for i in range(nfiles):
            phi = lf_list[i][f'M{ibin}mockr']
            nmin = meta_list[i]['nmin']
            # pdb.set_trace()
            comp = phi.comp
            comp *= (phi.ngal >= npmin)
            if labels:
                label = labels[i]
            else:
                label = rf'$N_{{\rm fof}} \geq {nmin}$'
            ax.errorbar(phi.Mbin[comp], phi.phi[comp],
                        phi.phi_err[comp], alpha=alpha,
                        label=label)
            print(phi.comp, phi.Mbin[comp], phi.phi[comp])
        if lf_lims:
            ax.axis(lf_lims)
        ax.semilogy(basey=10)
        if yticks:
            ax.set_yticks(yticks)
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        label = r'${\cal M}' + r'{}\ [{}, {}]$'.format(
                ibin+1, mbins_def[ibin], mbins_def[ibin+1])
        ax.text(0.9, 0.9, label, ha='right', transform=ax.transAxes)
        if ibin == 0:
#            ax.legend(loc=loc)
            ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncol=len(infiles), handlelength=2, borderaxespad=0.)
    plt.draw()
    if plot_file:
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


def clf_sim_plotr_vmax(
        infiles=('clf_sim_nmin1_vmax.pkl', 'clf_sim_nmin5_vmax.pkl'),
        plot_file='clf_sim_vmax.pdf'):
    clf_sim_plotr(infiles=infiles, plot_file=plot_file)


def clf_sim_plotr_zlim(
        infiles=('clf_sim_nmin5_z01.pkl', 'clf_sim_nmin5_z02.pkl', 'clf_sim_nmin5.pkl'),
        labels=(r'$z < 0.1$', r'$z < 0.2$', r'$z < 0.5$'),
        plot_file='clf_sim_zlim.pdf'):
    clf_sim_plotr(infiles=infiles, labels=labels, plot_file=plot_file)


def clf_sim_plotr_vmax_10k(
        infiles=('clf_sim_nmin1_vmax_10k.pkl', 'clf_sim_nmin5_vmax_10k.pkl'),
        plot_file='clf_sim_vmax_10k.pdf'):
    clf_sim_plotr(infiles=infiles, plot_file=plot_file)


def clf_sim_plots_z_vmax(
        infiles=('clf_sim_nmin5_z0.002_0.1_vmax.pkl',
                 'clf_sim_nmin5_z0.1_0.2_vmax.pkl',
                 'clf_sim_nmin5_z0.2_0.3_vmax.pkl'),
        labels=(r'$0.0 < z < 0.1$', r'$0.1 < z < 0.2$', r'$0.2 < z < 0.3$')):
    clf_sim_plots(infiles=infiles, labels=labels, yrange=(1e-5, 1e-1))


def clf_sim_plotr_z_vmax(
        infiles=('clf_sim_nmin5_z0.002_0.1_vmax.pkl',
                 'clf_sim_nmin5_z0.1_0.2_vmax.pkl',
                 'clf_sim_nmin5_z0.2_0.3_vmax.pkl'),
        labels=(r'$0.0 < z < 0.1$', r'$0.1 < z < 0.2$', r'$0.2 < z < 0.3$'),
        lf_lims=(-15.5, -23.9, 0.25, 2), yticks=(0.3, 0.5, 1, 1.5),
        plot_file='clf_sim_z_vmax.pdf'):
    clf_sim_plotr(infiles=infiles, labels=labels, lf_lims=lf_lims,
                  yticks=yticks, plot_file=plot_file)


def csmf_sim_plotr(
        infiles=('csmf_sim_incomp_nmin5_vmax.pkl',
                 'csmf_sim_comp_nmin5_vmax.pkl',),
        labels=('incomp', 'comp'),
        lf_lims=(8, 12, 0.5, 5),
        plot_file='csmf_sim_vmax.pdf'):
    clf_sim_plotr(infiles=infiles, labels=labels, lf_lims=lf_lims,
                  plot_file=plot_file)


def csmf_sim_plotr_zlim(
        infiles=('csmf_sim_comp_nmin5_z0.002_0.1_vmax.pkl',
                 'csmf_sim_comp_nmin5_z0.1_0.2_vmax.pkl',
                 'csmf_sim_comp_nmin5_z0.2_0.3_vmax.pkl'),
        labels=(r'$0.0 < z < 0.1$', r'$0.1 < z < 0.2$', r'$0.2 < z < 0.3$'),
        lf_lims=(8, 12, 0.5, 5),
        plot_file='csmf_sim_zlim.pdf'):
    clf_sim_plotr(infiles=infiles, labels=labels, lf_lims=lf_lims,
                  plot_file=plot_file)


# IllustrisTNG LFs and SMFs


def tng_smf(sim='TNG300-1', snap=84, mlim=18,
            mhbins=mbins_sim,
            msbins=np.linspace(8.5, 12.5, 17),
            mrbins=np.linspace(-24, -15, 19)):
    """IllustrisTNG SMF."""

    phot_file = f'/Users/loveday/Data/TNG/{sim}/output/groups_0{snap}/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_0{snap}.hdf5'
    smffile = f'{sim}_{snap}_smf.pkl'
    lffile = f'{sim}_{snap}_lf.pkl'
    csmffile = f'{sim}_{snap}_csmf.pkl'
    clffile = f'{sim}_{snap}_clf.pkl'
    h = 0.6774
    boxsize = 205.0  # Mpc/h
    boxvol = boxsize**3  # (Mpc/h)**3
    smcorr = 1.4  # stellar mass resolution correction factor (Pillepich+18 A1)
    basePath = f'/Users/loveday/Data/TNG/{sim}/output/'
    mscen = msbins[:-1] + np.diff(msbins)
    mrcen = mrbins[:-1] + np.diff(mrbins)
    msbinsize = msbins[1] - msbins[0]
    mrbinsize = mrbins[1] - mrbins[0]

    # Read FoF group catalogue and plot halo mass function
    header = il.groupcat.loadHeader(basePath, snap)
    redshift = header['Redshift']
    fields = ['GroupFirstSub', 'GroupMass', 'Group_M_Mean200', 'GroupPos']
    halos = il.groupcat.loadHalos(basePath, snap, fields=fields)
    lgM = 10 + np.log10(halos['Group_M_Mean200'])
    grp_coords = 0.001 * halos['GroupPos']
    grp_dist = np.sum(grp_coords**2, axis=1)**0.5

    plt.clf()
    plt.hist(lgM, bins=np.linspace(10, 15, 21))
    plt.semilogy(basey=10)
    plt.xlabel(r'$M_{200} [M_\odot/h]$')
    plt.ylabel(r'Frequency')
    plt.title('HMF')
    plt.show()

    # Read subhalo (galaxy) catalogue
    subfields = ['SubhaloFlag', 'SubhaloGrNr', 'SubhaloMassType',
                 'SubhaloMassInRadType', 'SubhaloPos',
                 'SubhaloStellarPhotometrics']
    subhalos = il.groupcat.loadSubhalos(basePath, snap, fields=subfields)

#    lgMs = 10 + np.log10(h*subhalos['SubhaloMassInRadType'][sel, 4])
    lgMs = 10 + np.log10(smcorr*h*subhalos['SubhaloMassInRadType'][:, 4])
    sel = (subhalos['SubhaloFlag'] == 1)  # * (lgMs > msbins[0])
    lgMs = lgMs[sel]
    mags = subhalos['SubhaloStellarPhotometrics'][sel, :]
    Mr = mags[:, 5] - 5*math.log10(h)
    group_id = subhalos['SubhaloGrNr'][sel]
    gal_id = np.arange(subhalos['count'])[sel]
    gal_coords = 0.001 * subhalos['SubhaloPos'][sel, :]
    print(len(lgMs), 'out of', subhalos['count'], 'selected')

    # Field SMF
    hist, edges = np.histogram(lgMs, bins=msbins)
    phi = hist/boxvol/msbinsize
    phierr = hist**0.5/boxvol/msbinsize
    lft = lf.LF(None, None, msbins)
    lft.ngal, lft.phi, lft.phi_err = hist, phi, phierr
    lft.comp_min, lft.comp_max = msbins[0], msbins[-1]
    pickle.dump(lft, open(smffile, 'wb'))
    plt.clf()
    plt.errorbar(mscen, phi, phierr)
    plt.semilogy(basey=10)
    plt.xlabel(r'$M_* [M_\odot/h^2]$')
    plt.ylabel(r'$\phi(M_*)$')
    plt.title('Field SMF')
    plt.show()

    # Field LF
    phot_tab = Table.read(phot_file, path='Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc')
    # print(phot_tab.info())
    # print(phot_tab['col2'][:10, :])
    mag_obs = np.mean(phot_tab['col2'][sel, :], axis=1) - 5*math.log10(h)
    # plt.clf()
    # plt.scatter(Mr, mag_obs - Mr, s=0.01)
    # plt.xlabel(r'$M_{\rm intrinsic} - 5 \log h$')
    # plt.ylabel(r'$M_{\rm obs} - M_{\rm intrinsic}$')
    # plt.show()

    hist, edges = np.histogram(mag_obs, bins=mrbins)
    phi = hist/boxvol/mrbinsize
    phierr = hist**0.5/boxvol/mrbinsize
    lft = lf.LF(None, None, mrbins)
    lft.ngal, lft.phi, lft.phi_err = hist, phi, phierr
    lft.comp_min, lft.comp_max = mrbins[0], mrbins[-1]
    pickle.dump(lft, open(lffile, 'wb'))
    plt.clf()
    plt.errorbar(mrcen, phi, phierr)
    plt.semilogy(basey=10)
    plt.xlabel(r'$M - 5 \log h$')
    plt.ylabel(r'$\phi(M)$')
    plt.title('Field LF')
    plt.show()

    # Assign host halo mass bin to each subhalo
    host_mass = lgM[group_id]

    # Identify centrals and satellites
    cen = halos['GroupFirstSub'][group_id] == gal_id
    sat = halos['GroupFirstSub'][group_id] != gal_id
    allgal = cen + sat

    # Colour cut
    gi = mags[:, 4] - mags[:, 6]
    gi_cut = 0.07*lgMs + 0.24
    blue, red = gi <= gi_cut, gi > gi_cut
    show = (lgMs > 8) * (np.random.random(len(gi)) < 0.05)
    plt.clf()
    plt.scatter(lgMs[blue*show], gi[blue*show], s=0.01, c='b')
    plt.scatter(lgMs[red*show], gi[red*show], s=0.01, c='r')
    plt.xlabel(r'$M_* [M_\odot/h^2]$')
    plt.ylabel(r'$g-i$')
    plt.show()

    # Visible galaxies
    gal_dist = np.sum(gal_coords**2, axis=1)**0.5
    dlim = np.clip(10**(0.2*(mlim - Mr - 25)), 0.0, boxsize)
    vmax = math.pi/6*dlim**3
    mapp = Mr + 5*np.log10(gal_dist) + 25
    obs = (mapp < mlim) * (gal_dist < boxsize)
    print(len(Mr[obs]), 'out of', len(Mr), 'galaxies visible')
#    pdb.set_trace()
    plt.clf()
    hist, edges = np.histogram(lgMs, bins=msbins)
#    plt.plot(mscen, hist/boxvol/h**3)
    plt.plot(mscen, hist/boxvol/msbinsize)
    plt.semilogy(basey=10)
    plt.xlabel(r'$M_* [M_\odot/h^2]$')
    plt.ylabel(r'$\phi(M) [h^3  {\rm Mpc}^{-3} {\rm dex}^{-1}]$')
#    plt.xlabel(r'$M_* [M_\odot]$')
#    plt.ylabel(r'$\phi(M)$')
    plt.title('SMF: all galaxies')
    plt.show()

    # SMF (1/Vol) in halo mass bins
    lf_dict = {}
    plt.clf()
    ax = plt.subplot(111)
    for im in range(len(mhbins)-1):
        Mlo, Mhi = mhbins[im], mhbins[im+1]
        msel = (Mlo <= host_mass) * (Mhi > host_mass)
        nhalo = len(lgM[(Mlo <= lgM) * (Mhi > lgM)])
        ngal = len(host_mass[(Mlo <= host_mass) * (Mhi > host_mass)])
        Mh_mean = np.mean(host_mass[msel])
#        scale = ngal/len(host_mass[sel])
        scale = 1.0

        for censat, cssel in zip(('all', 'cen', 'sat'), (allgal, cen, sat)):
            for clr, clrsel in zip(('all', 'red', 'blue'), (allgal, red, blue)):
                sel = msel*cssel*clrsel
                hist, edges = np.histogram(lgMs[sel], bins=msbins)
                phi = hist/boxvol/msbinsize*scale
                phierr = hist**0.5/boxvol/msbinsize*scale
                color = next(ax._get_lines.prop_cycler)['color']
                plt.errorbar(mscen, phi, phierr, fmt='-', color=color,
                             label=f'{Mh_mean:5.2f}')
                lft = lf.LF(None, None, msbins)
                lft.sel_dict = {'Mlim': [Mlo, Mhi], 'censat': censat,
                                'clr': clr}
                lft.info = {'Ngrp': nhalo, 'Ngal': ngal, 'lgm_av': Mh_mean,
                            'z_av': redshift}
                lft.ngal, lft.phi, lft.phi_err = hist, phi, phierr
                lft.comp_min, lft.comp_max = msbins[0], msbins[-1]
                lf_dict[f'M{im}_{censat}_{clr}'] = lft

    pickle.dump(lf_dict, open(csmffile, 'wb'))
    plt.legend()
    plt.semilogy(basey=10)
#    plt.ylim(1e-6, 8e-3)
    plt.xlabel(r'$\lg M_* [M_\odot/h^2]$')
    plt.ylabel(r'$\phi(M) [h^3  {\rm Mpc}^{-3} {\rm dex}^{-1}]$')
    plt.title('SMF')
    plt.show()

    # LF (1/Vol) in halo mass bins
    lf_dict = {}
    plt.clf()
    ax = plt.subplot(111)
    for im in range(len(mhbins)-1):
        Mlo, Mhi = mhbins[im], mhbins[im+1]
        msel = (Mlo <= host_mass) * (Mhi > host_mass)
        nhalo = len(lgM[(Mlo <= lgM) * (Mhi > lgM)])
        Mh_mean = np.mean(host_mass[msel])
#        scale = ngal/len(host_mass[sel])
        scale = 1.0

        for censat, cssel in zip(('all', 'cen', 'sat'), (allgal, cen, sat)):
            for clr, clrsel in zip(('all', 'red', 'blue'), (allgal, red, blue)):
                sel = msel*cssel*clrsel
                hist, edges = np.histogram(mag_obs[sel], bins=mrbins)
                phi = hist/boxvol/mrbinsize*scale
                phierr = hist**0.5/boxvol/mrbinsize*scale
                color = next(ax._get_lines.prop_cycler)['color']
                plt.errorbar(mrcen, phi, phierr, fmt='-', color=color,
                             label=f'{Mh_mean:5.2f}')
                lft = lf.LF(None, None, mrbins)
                lft.sel_dict = {'Mlim': [Mlo, Mhi], 'censat': censat,
                                'clr': clr, 'Nhalo': nhalo, 'Mav': Mh_mean}
                lft.ngal, lft.phi, lft.phi_err = hist, phi, phierr
                lft.comp_min, lft.comp_max = mrbins[0], mrbins[-1]
                lf_dict[f'M{im}_{censat}_{clr}'] = lft

    pickle.dump(lf_dict, open(clffile, 'wb'))
    plt.legend()
    plt.semilogy(basey=10)
#    plt.ylim(1e-6, 8e-3)
    plt.xlabel(r'$M_r - 5 \log h $')
    plt.ylabel(r'$\phi(M) [h^3  {\rm Mpc}^{-3} {\rm mag}^{-1}]$')
    plt.title('LF')
    plt.show()

    # Conditional SMF
    plt.clf()
    ax = plt.subplot(111)
    for im in range(len(mhbins)-1):
        Mlo, Mhi = mhbins[im], mhbins[im+1]
        sel = (Mlo <= host_mass) * (Mhi > host_mass)
        Mh_mean = np.mean(host_mass[sel])
        ngrp = len(np.unique(group_id[sel]))
#        ngrp2 = len(lgM[(Mlo <= lgM) * (Mhi > lgM)])
        print('Mean log mass =', Mh_mean, len(host_mass[sel]), 'galaxies in',
              ngrp, 'groups')

        hist, edges = np.histogram(lgMs[sel*cen], bins=msbins)
        phi = hist/ngrp/msbinsize
        phierr = hist**0.5/ngrp/msbinsize
        color = next(ax._get_lines.prop_cycler)['color']
        plt.errorbar(mscen, phi, phierr, fmt='--', color=color)

        hist, edges = np.histogram(lgMs[sel*sat], bins=msbins)
        phi = hist/ngrp/msbinsize
        phierr = hist**0.5/ngrp/msbinsize
        plt.errorbar(mscen, phi, phierr, color=color, label=f'{Mh_mean:5.2f}')

    plt.legend()
    plt.semilogy(basey=10)
    plt.ylim(0.1, 1e3)
    plt.xlabel(r'$\lg M_* [M_\odot/h^2]$')
    plt.ylabel(r'N (per group per dex]')
    plt.title('CSMF')
    plt.show()

    # Conditional SMF by colour
    plt.clf()
    ax = plt.subplot(111)
    for im in range(len(mhbins)-1):
        Mlo, Mhi = mhbins[im], mhbins[im+1]
        sel = (Mlo <= host_mass) * (Mhi > host_mass)
        Mh_mean = np.mean(host_mass[sel])
        ngrp = len(np.unique(group_id[sel]))
#        ngrp2 = len(lgM[(Mlo <= lgM) * (Mhi > lgM)])
#        print('Mean log mass =', Mh_mean, len(host_mass[sel]), 'galaxies in',
#              ngrp, 'groups')
        color = next(ax._get_lines.prop_cycler)['color']
        hist, edges = np.histogram(lgMs[sel*blue*cen], bins=msbins)
        phi = hist/ngrp/msbinsize
        phierr = hist**0.5/ngrp/msbinsize
        plt.errorbar(mscen, phi, phierr, fmt='--', color=color)
        hist, edges = np.histogram(lgMs[sel*blue*sat], bins=msbins)
        phi = hist/ngrp/msbinsize
        phierr = hist**0.5/ngrp/msbinsize
        plt.errorbar(mscen, phi, phierr, fmt='--', color=color)
        hist, edges = np.histogram(lgMs[sel*red*cen], bins=msbins)
        phi = hist/ngrp/msbinsize
        phierr = hist**0.5/ngrp/msbinsize
        plt.errorbar(mscen, phi, phierr, color=color)
        hist, edges = np.histogram(lgMs[sel*red*sat], bins=msbins)
        phi = hist/ngrp/msbinsize
        phierr = hist**0.5/ngrp/msbinsize
        plt.errorbar(mscen, phi, phierr, color=color, label=f'{Mh_mean:5.2f}')
    plt.legend()
    plt.semilogy(basey=10)
    plt.ylim(0.001, 1e3)
    plt.xlabel(r'$\lg M_* [M_\odot/h^2]$')
    plt.ylabel(r'N (per group per dex]')
    plt.title('CSMF')
    plt.show()
#
#    # LF
#    plt.clf()
#    ax = plt.subplot(111)
#    for im in range(len(mhbins)-1):
#        Mlo, Mhi = mhbins[im], mhbins[im+1]
#        color = next(ax._get_lines.prop_cycler)['color']
#
#        # All galaxies
#        sel = (Mlo <= host_mass) * (Mhi > host_mass)
#        Mh_mean = np.mean(host_mass[sel])
#        ngrp = len(np.unique(group_id[sel]))
#        hist, edges = np.histogram(Mr[sel], bins=mrbins)
#        phi = hist/boxvol/mrbinsize
#        phierr = hist**0.5/boxvol/mrbinsize
#        plt.errorbar(mrcen, phi, phierr, color=color, label=f'{Mh_mean:5.2f}')
#
#        # Visible galaxies
#        sel = (Mlo <= host_mass) * (Mhi > host_mass) * obs
#        hist, edges = np.histogram(Mr[sel], bins=mrbins, weights=1/vmax[sel])
#        num, edges = np.histogram(Mr[sel], bins=mrbins)
#        phi = hist/mrbinsize
#        phierr = hist/mrbinsize/num**0.5
#        plt.errorbar(mrcen, phi, phierr, color=color, ls='--')
#
#    plt.legend()
#    plt.semilogy(basey=10)
##    plt.ylim(0.1, 1e3)
#    plt.xlabel(r'$M_r - 5 \lg h$')
#    plt.ylabel(r'N (per group per mag]')
#    plt.title('LF')
#    plt.show()
#
#    # Conditional LF: this is very slow for 'observed' LFs
#    plt.clf()
#    ax = plt.subplot(111)
#    for im in range(len(mhbins)-1):
#        Mlo, Mhi = mhbins[im], mhbins[im+1]
#        color = next(ax._get_lines.prop_cycler)['color']
#
#        # All galaxies
#        sel = (Mlo <= host_mass) * (Mhi > host_mass)
#        Mh_mean = np.mean(host_mass[sel])
#        ngrp = len(np.unique(group_id[sel]))
#        hist, edges = np.histogram(Mr[sel], bins=mrbins)
#        phi = hist/ngrp/mrbinsize
#        phierr = hist**0.5/ngrp/mrbinsize
#        plt.errorbar(mrcen, phi, phierr, color=color, label=f'{Mh_mean:5.2f}')
#
#        # Visible galaxies, corrected by all visible groups
#        sel = (Mlo <= host_mass) * (Mhi > host_mass) * obs
#        mass_sel = (Mlo <= lgM) * (Mhi > lgM)
#        ngrpv = np.array([len(lgM[mass_sel * (grp_dist < dlim[sel][i])])
#                         for i in range(len(Mr[sel]))])
#        hist, edges = np.histogram(Mr[sel], bins=mrbins, weights=1/ngrpv)
#        phi = hist/mrbinsize
#        phierr = hist**0.5/mrbinsize
#        plt.errorbar(mrcen, phi, phierr, color=color, ls='--')
#
#        # Visible galaxies, corrected by all visible groups containing vis gal
#        grps = np.unique(group_id[sel])
#        gd = grp_dist[grps]
#        ngrpu = np.array([len(grps[gd < dlim[sel][i]])
#                         for i in range(len(Mr[sel]))])
#        hist, edges = np.histogram(Mr[sel], bins=mrbins, weights=1/ngrpu)
#        phi = hist/mrbinsize
#        phierr = hist**0.5/mrbinsize
#        plt.errorbar(mrcen, phi, phierr, color=color, ls=':')
#        print('ngrp, ngrpv, ngrpu = ', ngrp, ngrpv, ngrpu)
#    plt.legend()
#    plt.semilogy(basey=10)
##    plt.ylim(0.1, 1e3)
#    plt.xlabel(r'$M_r - 5 \lg h$')
#    plt.ylabel(r'N (per group per mag]')
#    plt.title('CLF')
#    plt.show()


def tng_plot(sim='TNG300-1', snaps=(78, 87)):
    """Plot TNG results."""

    sa_left = 0.15
    sa_bot = 0.05
    nbin = 4
    ddict = {}
    for snap in snaps:
        pdict = pickle.load(open(f'{sim}_{snap}_csmf.pkl', 'rb'))
        ddict[snap] = pdict
    plt.clf()
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, num=1)
    fig.set_size_inches(6, 5)
    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, r'$M_*$', ha='center', va='center')
    fig.text(0.06, 0.5, r'$\phi_0/\phi_1$', ha='center', va='center',
             rotation='vertical')
    # plt.semilogy(basey=10, nonposy='clip')
    # plt.ylim(1e-7, 1e-2)
    for i in range(nbin):
        ax = axes.flat[i]
        ax.text(0.9, 0.9, f'M{i}', ha='right', transform=ax.transAxes)
        key = f'M{i}_all_all'
        # for snap in snaps:
        #     phi = ddict[snap][key]
        #     phi.plot(ax=ax, label=snap, alpha=1)
        phi0 = ddict[snaps[0]][key]
        phi1 = ddict[snaps[1]][key]
        ax.errorbar(phi0.Mbin, phi0.phi/phi1.phi,
                    (phi0.phi_err**2 + phi1.phi_err**2)**0.5)
        # if i == 0:
        #     ax.legend(loc=3)
    plt.draw()
    plt.show()


def tng_merged_smf(sim='TNG300-1', mu=1.0,
                   mhbins=(12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0),
                   msbins=np.linspace(7, 12.5, 23)):
    """Illustris TNG SMF after groups randomly merged."""

    h = 0.6774
    boxsize = 205.0
    boxvol = boxsize**3
    basePath = f'/Users/loveday/Data/{sim}/output/'
    mscen = msbins[:-1] + np.diff(msbins)
    msbinsize = msbins[1] - msbins[0]

    """Read FoF halo catalogue and create new merged halos (mhalos):
        0. Select halos with minimum mass corresponding to first mhbin
        1. Generate randomly-ordered list of original halos
        2. Step through list in Poisson-sample sized chunks (mu=1, min=1)
        3. Build pointer from original halo id to selected and merged halo ids
        (shalo_id and mhalo_id)
        4. merged halo has sum of masses or consituent halos
        5. Assign galaxies to mhalos via mhalo_id"""

    fields = ['GroupFirstSub', 'GroupMass', 'Group_M_Mean200', 'GroupPos']
    halos = il.groupcat.loadHalos(basePath, 87, fields=fields)
    nhalos = halos['count']

    # Select haloes above minimum mass; shalo_id points from from original
    # to selected haloes (-1 if not selected)
    sel = halos['Group_M_Mean200'] >= 10**(mhbins[0] - 10)
    Mh = halos['Group_M_Mean200'][sel]
    lgMh = 10 + np.log10(Mh)
    nhsel = len(lgMh)
    shalo_id = -1*np.ones(nhalos, dtype=np.int)
    shalo_id[sel] = np.arange(nhsel)
    print(nhsel, 'out of', nhalos, 'haloes with lgM >=', mhbins[0])

    # Merge randomly-selected haloes
    rand_id = np.arange(nhsel, dtype=np.int)
    mhalo_id = np.zeros(nhsel, dtype=np.int)
    np.random.shuffle(rand_id)
    mhalo = 0
    ihalo = 0
    lgMhm, multiplicity = [], []
    while (ihalo < nhsel):
        mhalo_mass = 0.0
        nsamp = np.clip(np.random.poisson(mu), 1, nhsel - ihalo)
        multiplicity.append(nsamp)
        for ih in range(nsamp):
            mhalo_id[rand_id[ihalo]] = mhalo
            mhalo_mass += Mh[rand_id[ihalo]]
            ihalo += 1
        if mhalo_mass > 0:
            lgMhm.append(10 + math.log10(mhalo_mass))
        else:
            lgMhm.append(0.0)
        mhalo += 1
    lgMhm, multiplicity = np.array(lgMhm), np.array(multiplicity)
    print(mhalo, 'merged halos out of', nhsel)

    plt.clf()
    plt.hist(multiplicity, bins=np.arange(1, multiplicity.max()+1))
    plt.semilogy(basey=10)
    plt.xlabel('Number of haloes merged')
    plt.ylabel('Frequency')
    plt.show()

    hbins = np.linspace(mhbins[0], mhbins[-1], 31)
    plt.clf()
    plt.hist(lgMh, bins=hbins, histtype='step', label='Original')
    plt.hist(lgMhm, bins=hbins, histtype='step', label='Merged')
    plt.legend()
    plt.semilogy(basey=10)
    plt.xlabel(r'$M_{200} [M_\odot/h]$')
    plt.ylabel(r'Frequency')
    plt.show()

    # Read subhalo (galaxy) catalogue
    subfields = ['SubhaloFlag', 'SubhaloGrNr', 'SubhaloMassType',
                 'SubhaloMassInRadType', 'SubhaloPos',
                 'SubhaloStellarPhotometrics']
    subhalos = il.groupcat.loadSubhalos(basePath, 87, fields=subfields)

    sel = ((subhalos['SubhaloFlag'] == 1) *
           (shalo_id[subhalos['SubhaloGrNr']] >= 0))
    lgMs = 10 + np.log10(subhalos['SubhaloMassInRadType'][sel, 4])
    group_id = subhalos['SubhaloGrNr'][sel]
    gal_id = np.arange(subhalos['count'])[sel]
    print(len(lgMs), 'out of', subhalos['count'], 'galaxies selected')

    # Assign true and merged host halo mass bin to each subhalo
    host_mass = lgMh[shalo_id[group_id]]
    mhost_mass = lgMhm[mhalo_id[shalo_id[group_id]]]
    print('mhost_mass - host_mass range:', np.min(mhost_mass - host_mass),
          np.max(mhost_mass - host_mass))
    plt.clf()
    plt.scatter(host_mass, mhost_mass, s=0.1)
    plt.xlabel('True host mass')
    plt.ylabel('Merged host mass')
    plt.show()

    # Identify centrals and satellites
    cen = halos['GroupFirstSub'][group_id] == gal_id
    sat = halos['GroupFirstSub'][group_id] != gal_id

    # True CSMF
    plt.clf()
    ax = plt.subplot(111)
    for im in range(len(mhbins)-1):
        Mlo, Mhi = mhbins[im], mhbins[im+1]
        sel = (Mlo <= host_mass) * (Mhi > host_mass)
        Mh_mean = np.mean(host_mass[sel])
        ngrp = len(np.unique(group_id[sel]))
        print('Mean log mass =', Mh_mean, len(host_mass[sel]), 'galaxies in',
              ngrp, 'true groups')

        color = next(ax._get_lines.prop_cycler)['color']

        hist, edges = np.histogram(lgMs[sel], bins=msbins)
        phi = hist/ngrp/msbinsize
        phierr = hist**0.5/ngrp/msbinsize
        plt.errorbar(mscen, phi, phierr, fmt='-', color=color,
                     label=f'{Mh_mean:5.2f}')

        hist, edges = np.histogram(lgMs[sel*cen], bins=msbins)
        phi = hist/ngrp/msbinsize
        phierr = hist**0.5/ngrp/msbinsize
        plt.errorbar(mscen, phi, phierr, fmt=':', color=color)

        hist, edges = np.histogram(lgMs[sel*sat], bins=msbins)
        phi = hist/ngrp/msbinsize
        phierr = hist**0.5/ngrp/msbinsize
        plt.errorbar(mscen, phi, phierr, fmt='--', color=color)

    plt.legend()
    plt.semilogy(basey=10)
    plt.ylim(0.1, 1e3)
    plt.xlabel(r'$\lg M_* [M_\odot/h]$')
    plt.ylabel(r'N (per group per dex]')
    plt.title('True CSMF')
    plt.show()

    # Merged halo CSMF
    plt.clf()
    ax = plt.subplot(111)
    for im in range(len(mhbins)-1):
        Mlo, Mhi = mhbins[im], mhbins[im+1]
        sel = (Mlo <= mhost_mass) * (Mhi > mhost_mass)
        Mh_mean = np.mean(mhost_mass[sel])
        ngrp = len(np.unique(mhalo_id[shalo_id[group_id[sel]]]))
        print('Mean log mass =', Mh_mean, len(host_mass[sel]), 'galaxies in',
              ngrp, 'merged groups')

        color = next(ax._get_lines.prop_cycler)['color']

        hist, edges = np.histogram(lgMs[sel], bins=msbins)
        phi = hist/ngrp/msbinsize
        phierr = hist**0.5/ngrp/msbinsize
        plt.errorbar(mscen, phi, phierr, fmt='-', color=color,
                     label=f'{Mh_mean:5.2f}')

        hist, edges = np.histogram(lgMs[sel*cen], bins=msbins)
        phi = hist/ngrp/msbinsize
        phierr = hist**0.5/ngrp/msbinsize
        plt.errorbar(mscen, phi, phierr, fmt=':', color=color)

        hist, edges = np.histogram(lgMs[sel*sat], bins=msbins)
        phi = hist/ngrp/msbinsize
        phierr = hist**0.5/ngrp/msbinsize
        plt.errorbar(mscen, phi, phierr, fmt='--', color=color)

    plt.legend()
    plt.semilogy(basey=10)
    plt.ylim(0.1, 1e3)
    plt.xlabel(r'$\lg M_* [M_\odot/h]$')
    plt.ylabel(r'N (per group per dex]')
    plt.title('Merged CSMF')
    plt.show()

    # True and merged halo CSMFs on one plot
    plt.clf()
    ax = plt.subplot(111)
    for im in range(len(mhbins)-1):
        color = next(ax._get_lines.prop_cycler)['color']
        Mlo, Mhi = mhbins[im], mhbins[im+1]

        sel = (Mlo <= host_mass) * (Mhi > host_mass)
        Mh_mean = np.mean(host_mass[sel])
        ngrp = len(np.unique(group_id[sel]))
        print('Mean log mass =', Mh_mean, len(host_mass[sel]), 'galaxies in',
              ngrp, 'true groups')
        hist, edges = np.histogram(lgMs[sel], bins=msbins)
        phi = hist/ngrp/msbinsize
        phierr = hist**0.5/ngrp/msbinsize
        plt.errorbar(mscen, phi, phierr, fmt='-', color=color,
                     label=f'{Mh_mean:5.2f}')

        sel = (Mlo <= mhost_mass) * (Mhi > mhost_mass)
        Mh_mean = np.mean(mhost_mass[sel])
        ngrp = len(np.unique(mhalo_id[shalo_id[group_id[sel]]]))
        print('Mean log mass =', Mh_mean, len(host_mass[sel]), 'galaxies in',
              ngrp, 'merged groups')
        hist, edges = np.histogram(lgMs[sel], bins=msbins)
        phi = hist/ngrp/msbinsize
        phierr = hist**0.5/ngrp/msbinsize
        plt.errorbar(mscen, phi, phierr, fmt='--', color=color)

    plt.legend()
    plt.semilogy(basey=10)
    plt.ylim(0.1, 1e3)
    plt.xlabel(r'$\lg M_* [M_\odot/h]$')
    plt.ylabel(r'N (per group per dex]')
    plt.title('Merged CSMF')
    plt.show()


def grouped_frac(infile=gama_data+'g3cv9/G3CGalv08.fits',
                 bins=np.linspace(0.0, 0.6, 61)):
    """Plot fraction of grouped GAMA galaxies as function of redshift."""
    z = bins[:-1] + 0.5*(bins[1] - bins[0])
    t = Table.read(infile)
    grouped = t['GroupID'] > 0
    hist_all, edges = np.histogram(t['Z'], bins)
    hist_grp, edges = np.histogram(t['Z'][grouped], bins)
    grp_frac = hist_grp/hist_all
    plt.clf()
    plt.plot(z, grp_frac)
    plt.xlabel('Redshift')
    plt.ylabel('Grouped fraction')
    plt.show()


def read_Lgal(infile, counts):
    """Create list of LF instances from tabulated L-galaxies LF/SMF
    from Stephen."""

    # Masses are alreday in Msun/h^2
    # hcorr = math.log10(0.673)

    # Find lines where a new sample starts with comment sign
    iline = 0
    sd_list = []
    startlines = []
    for line in open(infile, 'r'):
        if '#' in line:
            startlines.append(iline)
            sd_next = True
        else:
            if sd_next:
                sd_list.append(line)
                sd_next = False
        iline += 1
    nline = iline
    nsamp = len(startlines)
    lf_dir = {}
    for isamp in range(nsamp):
        nskip = startlines[isamp] + 2
        if isamp < nsamp - 1:
            nrows = startlines[isamp+1] - startlines[isamp] - 2
        else:
            nrows = nline - startlines[isamp]
        data = np.loadtxt(infile, skiprows=nskip, max_rows=nrows)
        sel_dict = eval(sd_list[isamp])
        info = {'Ngrp': int(counts[isamp][2]),
                'Ngal': int(counts[isamp][3] + counts[isamp][4]),
                'lgm_av': counts[isamp][0], 'z_av': counts[isamp][1]}
#        print(sel_dict)
        Mbin = data[:, 0]
        nbin = len(Mbin)
        dm = Mbin[1] - Mbin[0]
        bins = Mbin - 0.5*dm
        bins = bins.tolist()
        bins.append(bins[-1] + dm)
#        print(bins)

        phi = lf.LF(None, None, bins, error='mock', sel_dict=sel_dict,
                    info=info)
        phi.phi = data[:, 1]
        phi.phi_err = data[:, 2]
        sel = phi.phi > 0
        nuse = len(phi.phi[sel])
        phi.ngal = np.zeros(nbin)
        phi.comp = np.zeros(nbin, dtype=np.bool)
        # Distribute galaxy counts equally amingst non-empty bins
        if 'sat' in infile:
            phi.ngal[sel] = counts[isamp][4]/nuse
        else:
             phi.ngal[sel] = counts[isamp][3]/nuse
               
        phi.comp[sel] = True
        phi.comp_min = bins[0]
        phi.comp_max = bins[-1]
        lf_dir[f'M{isamp}'] = phi
#        istart = iend
    return lf_dir


def Lgal_plot(what='smf',
              lgal_dir='/Users/loveday/Documents/Research/LFdata/L-Galaxies/'):
    if what == 'lf':
        outfile = 'lf_Lgal.pkl'
    else:
        outfile = 'smf_Lgal.pkl'
    plt.clf()
    ax = plt.subplot(111)
    plt.xlabel('M*')
    plt.ylabel('Phi')
    plt.semilogy()
    counts = np.loadtxt(lgal_dir + 'MR_MRII_halo_counts.txt')
    cen_smf_dir = read_Lgal(lgal_dir + f'MR_MRII_halo_{what}_centrals.txt',
                            counts)
    sat_smf_dir = read_Lgal(lgal_dir + f'MR_MRII_halo_{what}_satellites.txt',
                            counts)
    smf_dir = {}
    for isamp in range(4):
        smf_dir[f'M{isamp}_sat_all'] = sat_smf_dir[f'M{isamp}']
        smf_dir[f'M{isamp}_cen_all'] = cen_smf_dir[f'M{isamp}']
        cen_smf_dir[f'M{isamp}'].plot(ax, fmt='o')
        sat_smf_dir[f'M{isamp}'].plot(ax, fmt='s')
    plt.show()
    pickle.dump(smf_dir, open(outfile, 'wb'))


def yang_clf(
        mbins=(12, 12.3, 12.6, 12.9, 13.2, 13.5, 13.8, 14.1, 14.4, 15.0),
        magbins=np.linspace(-24, -16, 33), zlimit=0.2):
    """CLF from Yang SDSS DR7 group catalogue."""

    mags = magbins[:-1] + 0.5*np.diff(magbins)
    zcomp = (0.11, 0.13, 0.14, 0.17, 0.18, 0.19, 0.2, 0.2, 0.2)
    yang_dir = '/Users/loveday/Data/Yang_groups/SDSS_DR7/'
    vagc_dir = '/Users/loveday/Data/sdss/VAGC/'
    galfile = yang_dir + 'galaxy_DR7/SDSS7'
    grpfile = yang_dir + 'group_DR7/petroB_group'
    idfile = yang_dir + 'group_DR7/ipetroB_1'
    postfile = vagc_dir + 'post_catalog.dr72all0.fits'
    vmaxfile = vagc_dir + 'vmax-noevol.dr72all0.fits'
    clf_file = yang_dir + f'group_DR7/clf{zlimit}.pkl'
    print(clf_file)
    def neg(z):
        """Returns -1*kernel(z) for scipy to minimize."""
        return -1*kernel(z)

    post = Table.read(postfile, format='fits')
    post.rename_column('OBJECT_POSITION', 'VAGCID')
    post.keep_columns(('VAGCID'))
    post['VAGCID'] = post['VAGCID'] + 1  # Yang VAGC Ids are 1-indexed
    vmax = Table.read(vmaxfile, format='fits')
    vagc = hstack([post, vmax])
    galaxies = Table.read(galfile, format='ascii')
    galaxies.rename_columns(
        ('col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8',
         'col9', 'col10', 'col11', 'col12', 'col13'),
        ('GalID', 'VAGCID', 'ra_gal', 'dec_gal', 'z_gal', 'mag', 'mlim',
         'comp', 'Mr_pet', 'gr_pet', 'Mr_mod', 'gr_mod', 'z_source'))
    groups = Table.read(grpfile, format='ascii')
    groups.rename_columns(
        ('col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8',
         'col9', 'col10', 'col11', 'col12', 'col13'),
        ('GroupID', 'ra_grp', 'dec_grp', 'z_grp', 'lum', 'Mstar', 'Mhalo_l',
         'Mhalo_m', 'mean_sep_bright', 'mean_sep_mass', 'f_edge', 'ID1', 'ID2'))
    ids = Table.read(idfile, format='ascii')
    ids.rename_columns(
        ('col1', 'col2', 'col3', 'col4', 'col5',),
        ('GalID', 'VAGCID', 'GroupID', 'cen_lum', 'cen_mass'))
    ids.remove_column('GalID')

    print(f'{len(galaxies)} in Yang catalogue')
    sel = (galaxies['z_source'] < 4)
    galaxies = galaxies[sel]
    print(f'{len(galaxies)} with z_source < 4')
    gal = join(galaxies, vagc, keys='VAGCID',
               metadata_conflicts=metadata_conflicts)
    print(f'{len(gal)} joined with VAGC')
    gal = join(gal, ids, keys='VAGCID',
               metadata_conflicts=metadata_conflicts)
    gg = join(gal, groups, keys='GroupID',
               metadata_conflicts=metadata_conflicts)
    print(f'{len(gg)} galaxies after joining to groups')
    print(gg.info())

    nbin = len(mbins) - 1
    zbins = np.linspace(0.0, 0.2, 41)
    plt.clf()
    fig1, axes1 = plt.subplots(3, 3, sharex=True, sharey=False, num=1)
    fig1.set_size_inches(8, 8)
    fig1.subplots_adjust(left=0.1, bottom=0.1, top=1.0,
                        hspace=0.0, wspace=0.0)
    fig1.text(0.5, 0.0, 'Redshift', ha='center', va='center')
    fig1.text(0.0, 0.5, 'Frequency', ha='center', va='center', rotation='vertical')

    fig2, axes2 = plt.subplots(3, 3, sharex=True, sharey=False, num=2)
    fig2.set_size_inches(8, 8)
    fig2.subplots_adjust(left=0.1, bottom=0.1, top=1.0,
                        hspace=0.0, wspace=0.0)
    fig2.text(0.5, 0.0, 'Redshift', ha='center', va='center')
    fig2.text(0.0, 0.5, 'Frequency', ha='center', va='center', rotation='vertical')

    fig3, axes3 = plt.subplots(3, 3, sharex=True, sharey=True, num=3)
    fig3.set_size_inches(8, 8)
    fig3.subplots_adjust(left=0.1, bottom=0.1, top=1.0,
                        hspace=0.0, wspace=0.0)
    fig3.text(0.5, 0.0, r'$M_r$', ha='center', va='center')
    fig3.text(0.0, 0.5, r'$\phi(M_r)$', ha='center', va='center', rotation='vertical')

    lf_dict = {}
    for i in range(nbin):
        Mlo = mbins[i]
        Mhi = mbins[i+1]
        zlim = min(zcomp[i], zlimit)
        sel = ((Mlo <= groups['Mhalo_m']) * (groups['Mhalo_m'] < Mhi) *
               (groups['z_grp'] < zlim))
        grpsel = groups[sel]
        nhalo = len(grpsel)
        Mh_mean = np.mean(grpsel['Mhalo_m'])
        ax = axes1.flat[i]
        ax.text(0.9, 0.9, f'{Mlo} < Mh < {Mhi}',
                transform=ax.transAxes, ha='right')
        ax.hist(grpsel['z_grp'], bins=np.linspace(0.0, 0.2, 41))

        kernel = stats.gaussian_kde(grpsel['z_grp'])
        den = kernel(zbins)
        ax = axes2.flat[i]
        ax.text(0.9, 0.9, f'{Mlo} < Mh < {Mhi}',
                transform=ax.transAxes, ha='right')
        ax.plot(zbins, den)
        ipeak = np.argmax(den)
        res = scipy.optimize.fmin(neg, zbins[ipeak], disp=0)
        print(res[0])

        ax = axes3.flat[i]
        ax.semilogy(basey=10)
        ax.set_ylim(1e-2, 1e2)
        ax.text(0.9, 0.9, f'{Mlo} < Mh < {Mhi}',
                transform=ax.transAxes, ha='right')
        sel = (Mlo <= gg['Mhalo_m']) * (gg['Mhalo_m'] < Mhi) * (gg['z_gal'] < zlim)
        ggsel = gg[sel]
        ngal = len(ggsel)
        vmg = np.zeros(ngal)
        for igal in range(ngal):
            vmg[igal] = len(grpsel[grpsel['z_grp'] < ggsel['ZMAX_LOCAL'][igal]])
        clr = 'all'
        for icen in range(2):
            censat = ['cen', 'sat'][icen]
            sel = ggsel['cen_lum'] == icen + 1
            hist, _ = np.histogram(ggsel['Mr_pet'][sel], bins=magbins)
            phi, _ = np.histogram(ggsel['Mr_pet'][sel], bins=magbins,
                                  weights=1.0/vmg[sel])
            phierr = phi/hist**0.5
            bad = np.isnan(phi) + np.isinf(phi)
            good = np.isfinite(phi) * phi > 0
            phi[bad] = 0
            phierr[bad] = 99
            lft = lf.LF(None, None, magbins)
            lft.sel_dict = {'Mlim': [Mlo, Mhi], 'Nhalo': nhalo, 'Mav': Mh_mean,
                            'censat': censat, 'clr': clr}
            lft.ngal, lft.phi, lft.phi_err = hist, phi, phierr
            lft.comp_min, lft.comp_max = magbins[0], magbins[-1]
            lf_dict[f'M{i}_{censat}_{clr}'] = lft

            ax.errorbar(mags[good], phi[good], phierr[good])
            print(phi)
    pickle.dump(lf_dict, open(clf_file, 'wb'))
    plt.show()


def yang_plots(infiles=('clf0.2.pkl', 'clf0.11.pkl')):
    """Plot Yang CLF results."""

    yang_dir = '/Users/loveday/Data/Yang_groups/SDSS_DR7/group_DR7/'
    mbins = (12, 12.3, 12.6, 12.9, 13.2, 13.5, 13.8, 14.1, 14.4, 15.0)

    schec_fn = SchecMagSq()
    gauss_fn = LogNormal()
    schec_p0 = [[-20.5, -23, -19], [-1.2, -3, 1], [1, -2, 2]]
    gauss_p0 = [[-21.5, -23, -20], [0.5, 0.1, 0.8], [1, -2, 2]]
    f1 = schec_fn
    f1.mstar = schec_p0[0][0]
    f1.Mstar.min = schec_p0[0][1]
    f1.Mstar.max = schec_p0[0][2]
    f1.alpha = schec_p0[1][0]
    f1.alpha.min = schec_p0[1][1]
    f1.alpha.max = schec_p0[1][2]
    f1.lgps = schec_p0[2][0]
    f1.lgps.min = schec_p0[2][1]
    f1.lgps.max = schec_p0[2][2]
    schec_fn = f1

    gauss_fn.M_c = gauss_p0[0][0]
    gauss_fn.M_c.min = gauss_p0[0][1]
    gauss_fn.M_c.max = gauss_p0[0][2]
    gauss_fn.sigma_c = gauss_p0[1][0]
    gauss_fn.sigma_c.min = gauss_p0[1][1]
    gauss_fn.sigma_c.max = gauss_p0[1][2]
    gauss_fn.lgps = gauss_p0[2][0]
    gauss_fn.lgps.min = gauss_p0[2][1]
    gauss_fn.lgps.max = gauss_p0[2][2]

    plt.clf()
    fig3, axes3 = plt.subplots(3, 3, sharex=True, sharey=True, num=3)
    fig3.set_size_inches(8, 8)
    fig3.subplots_adjust(left=0.1, bottom=0.1, top=1.0,
                        hspace=0.0, wspace=0.0)
    fig3.text(0.5, 0.0, r'$M_r$', ha='center', va='center')
    fig3.text(0.0, 0.5, r'$\phi(M_r)$', ha='center', va='center', rotation='vertical')
    lf_dicts = []
    for infile in infiles:
        lf_dict = pickle.load(open(yang_dir+infile, 'rb'))
        lf_dicts.append(lf_dict)
    nbin = 9
    clr = 'all'
    alpha = np.zeros(nbin)
    alpha_err = np.zeros(nbin)
    for i in range(nbin):
        Mlo = mbins[i]
        Mhi = mbins[i+1]
        ax = axes3.flat[i]
        ax.semilogy(basey=10)
        ax.set_ylim(1e-2, 1e2)
        ax.text(0.9, 0.9, f'{Mlo} < Mh < {Mhi}',
                transform=ax.transAxes, ha='right')
        for lf_dict in lf_dicts:
            for censat in ('cen', 'sat'):
                phi = lf_dict[f'M{i}_{censat}_{clr}']
                print(phi.phi)
                if censat == 'cen':
                    fn = phi.fn_fit(gauss_fn, verbose=0)
                else:
                    fn = phi.fn_fit(schec_fn, verbose=0)
                try:
                    fit_errs = 0.5*(np.array(phi.errors.parmaxes) -
                                    np.array(phi.errors.parmins))
                except:
                # except AttributeError or TypeError:
                    fit_errs = (9.99, 9.99, 9.99)
                if censat == 'sat' and lf_dict == lf_dicts[0]:
                    alpha[i] = phi.res.parvals[1]
                    alpha_err[i] = fit_errs[1]
                # print(fr'''\mass{i+1} & {int(phi.ngal.sum())} &
                #       {phi.res.parvals[0]:5.2f}\pm{fit_errs[0]:5.2f} &
                #       {phi.res.parvals[1]:5.2f}\pm{fit_errs[1]:5.2f} &
                #       {phi.res.parvals[2]:5.2f}\pm{fit_errs[2]:5.2f} &
                #       {phi.res.statval:4.1f}/{phi.res.dof:2d} \\[0pt]
                #       ''')

                phi.plot(ax=ax)
    print(alpha, alpha_err)
    plt.show()


def plot_saund(alpha=-1.2, Mstar=-20.2, sigma=1, lgps=-2):
    fn = SaundersMag()
    fn.alpha = alpha
    fn.Mstar = Mstar
    fn.sigma = sigma
    fn.lgps = lgps
    mag = np.linspace(-24, -16, 33)
    phi = fn(mag)
    plt.clf()
    plt.plot(mag, phi)
    plt.semilogy(basey=10)
    plt.ylim(1e-8, 1)
    plt.show()
