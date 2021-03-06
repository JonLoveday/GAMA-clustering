# Plots for group LF paper

from array import array
import copy
import math
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker
import mpmath
import numpy as np
import os
import pdb
import pickle
import scipy.optimize
from scipy import stats

from astLib import astSED
from astropy.io import ascii
from astropy.modeling import models, fitting
from astropy import table
from astropy.table import Table, join
import healpy as hp
from sherpa.estmethods import Confidence
from sherpa.plot import RegionProjection

import gal_sample as gs
import lf
from schec import SchecMag, SchecMass
import util

# Global parameters
gama_data = os.environ['GAMA_DATA']
g3cgal = gama_data + 'g3cv9/G3CGalv08.fits'
g3cfof = gama_data + 'g3cv9/G3CFoFGroupv09.fits'
g3cmockfof = gama_data + 'g3cv6/G3CMockFoFGroupv06.fits'
g3cmockhalo = gama_data + 'g3cv6/G3CMockHaloGroupv06.fits'
g3cmockgal = gama_data + 'g3cv6/G3CMockGalv06.fits'
lf_data = os.environ['LF_DATA']
plot_dir = '/Users/loveday/Documents/tex/papers/gama/groupLF/'

mag_label = r'$^{0.1}M_r - 5 \log_{10} h$'
ms_label = r'$\log_{10}\ ({\cal M}_*/{\cal M}_\odot h^{-2})$'

lf_label = r'$\phi(M)\ [h^3\ {\rm Mpc}^{-3}\ {\rm mag}^{-1}]$'
smf_label = r'$\phi({\cal M}_*)\ [h^3\ {\rm Mpc}^{-3}\ {\rm dex}^{-1}]$'
clf_label = r'$\phi_C(M)\ [{\rm group}^{-1}\ {\rm mag}^{-1}]$'
csmf_label = r'$\phi_C({\cal M}_*)\ [{\rm group}^{-1}\ {\rm dex}^{-1}]$'

# Halo mass bin limits and means and cventral red fraction
#mbins = (12.0, 13.1, 13.4, 13.7, 14.0, 14.3, 16)
mbins_def = (12.0, 13.1, 13.3, 13.5, 13.7, 14.0, 15.2)
halomass = (12.91, 13.20, 13.40, 13.59, 13.82, 14.23)
crf = (0.66, 0.69, 0.76, 0.74, 0.74, 0.75)
# Constants
ln10 = math.log(10)

metadata_conflicts = 'silent'  # Alternatives are 'warn', 'error'

# Ticks point inwards on all axes
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['mathtext.fontset'] = 'dejavusans'


def mass_z(grpfile=g3cfof, nmin=5, edge_min=0.9, plot_file='mass_z.pdf',
           zrange=(0, 0.5), Mrange=(11.8, 15.4), plot_size=(5, 4)):
    """Halo mass-redshift plot."""

    # Read and select groups meeting selection criteria
    t = Table.read(grpfile)
    if 'sim' in grpfile:
        sel = np.array(t['Nfof'] >= nmin)
    else:
        t['log_mass'] = 13.98 + 1.16*(np.log10(t['LumBfunc']) - 11.5)
        sel = (np.array(t['GroupEdge'] > edge_min) *
               np.logical_not(t['log_mass'].mask) *
               np.array(t['Nfof'] >= nmin))

    t = t[sel]
    print('mass range of selected groups: {:5.2f} - {:5.2f}'.format(
            np.min(t['log_mass']), np.max(t['log_mass'])))

    plt.clf()
    plt.scatter(t['IterCenZ'], t['log_mass'], s=2, c=t['Nfof'],
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
    cb.set_label(r'$N_{\rm fof}$')
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


def mass_comp(infile=g3cfof, nmin=5, edge_min=0.9, plot_file='mass_comp.pdf',
              plot_size=(5, 4)):
    """Compare dynamical and luminosity-based halo mass estimates."""

    # Read and select groups meeting selection criteria
    t = Table.read(infile)
    t['log_mass_lum'] = 13.98 + 1.16*(np.log10(t['LumBfunc']) - 11.5)
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


def mass_comp_mock(infile=g3cmockhalo, nmin=5, edge_min=0.9,
                   plot_file='mass_comp_mock.pdf', plot_size=(5, 9.6)):
    """Compare dynamical and luminosity-based halo mass estimates for mock groups."""

    # Read and select groups meeting selection criteria
    thalo = Table.read(g3cmockhalo)
    thalo = thalo['HaloMass', 'IterCenRA', 'IterCenDEC']
    tfof = Table.read(g3cmockfof)
    t = join(thalo, tfof, keys=('IterCenRA', 'IterCenDEC'),
             metadata_conflicts=metadata_conflicts)

    t['log_mass_lum'] = 13.98 + 1.16*(np.log10(t['LumBfunc']) - 11.5)
    t['log_mass_dyn'] = np.log10(t['MassAfunc'])
    sel = (np.array(t['GroupEdge'] > edge_min) *
           np.logical_not(t['log_mass_lum'].mask) *
           np.array(t['Nfof'] >= nmin))
    t = t[sel]
    lgMh = np.log10(t['HaloMass'])

    plt.clf()
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, num=1)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.set_size_inches(plot_size)
    ax = axes[0]
#    ax.scatter(lgMh, t['log_mass_dyn'], s=2, c=t['IterCenZ'])
    ax.scatter(lgMh, t['log_mass_dyn'], s=1, c=t['Nfof'], vmax=10,
               norm=mpl.colors.LogNorm())
    ax.set_ylabel(r'$\lg {\cal M}_{\rm dyn}$')
    ax = axes[1]
#    sc = ax.scatter(lgMh, t['log_mass_lum'], s=2, c=t['IterCenZ'])
    sc = ax.scatter(lgMh, t['log_mass_lum'], s=1, c=t['Nfof'], vmax=10,
                    norm=mpl.colors.LogNorm())
    ax.set_ylabel(r'$\lg {\cal M}_{\rm lum}$')
    ax.set_xlabel(r'$\lg {\cal M}_{\rm halo}$')
    ax.axis((11.8, 15, 11.8, 15))
#    ax.set_aspect('equal')
    fig.subplots_adjust(top=0.93)
    cbar_ax = fig.add_axes([0.13, 0.97, 0.75, 0.02])
    cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
#    cb.set_label('Redshift')
#    cbar_ax.set_title('Redshift')
    cbar_ax.set_title('Nfof')

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
        t['log_mass'] = 13.98 + 1.16*(np.log10(t['LumBfunc']) - 11.5)
        sel = (np.array(t['GroupEdge'] > edge_min) *
               np.logical_not(t['log_mass'].mask) *
               np.array(t['Nfof'] >= nmin))
        t = t[sel]
        nmbin = len(mbins) - 1
        ngrp = []
        zmean = []
        for i in range(nmbin):
            sel = (mbins[i] <= t['log_mass']) * (t['log_mass'] < mbins[i+1])
            n = len(t[sel])
            if 'Mock' in infile:
                n /= 9
            ngrp.append('{:d}'.format(int(n)))
            zmean.append('{:3.2f}'.format(np.mean(t['IterCenZ'][sel])))
        return ngrp, zmean

    ngrp_gama, zmean_gama = group_stats(g3cfof)
    ngrp_mock, zmean_mock = group_stats(g3cmockfof)
    names = []
    limits = []
    nmbin = len(mbins) - 1
    for i in range(nmbin):
        names.append(r'${\cal M}' + str(i+1) + r'$')
        limits.append('[{}, {}]'.format(mbins[i], mbins[i+1]))
    t = {'Name': names, r'$\log_{10}({\cal M}_h/{\cal M}_\odot)$': limits,
         r'$N_{\rm GAMA}$': ngrp_gama, r'$\bar{z}_{\rm GAMA}$': zmean_gama,
         r'$N_{\rm Mock}$': ngrp_mock, r'$\bar{z}_{\rm Mock}$': zmean_mock}
    ascii.write(t, output=plot_dir + out_file, format='latex', overwrite=True,
                latexdict={'caption': r'''Group mass bin definitions,
                           number of groups and mean redshift.
                           Note that each mock realisation has 20 per cent
                           smaller volume than the GAMA-II fields.
                           \label{tab:group_mass_def}''',
                           'header_start': r'\hline', 'header_end': r'\hline',
                           'data_end': r'\hline'})


def group_mass_tab_with_mean_mass(nmin=5, edge_min=0.9, mbins=mbins_def,
                                  out_file='group_mass.tex'):
    """Tabulate group mass statistics."""

    def group_stats(infile):
        t = Table.read(infile)
        t['log_mass'] = 13.98 + 1.16*(np.log10(t['LumBfunc']) - 11.5)
        sel = (np.array(t['GroupEdge'] > edge_min) *
               np.logical_not(t['log_mass'].mask) *
               np.array(t['Nfof'] >= nmin))
        t = t[sel]
        nmbin = len(mbins) - 1
        ngrp = []
        zmean = []
        Mmean = []
        for i in range(nmbin):
            sel = (mbins[i] <= t['log_mass']) * (t['log_mass'] < mbins[i+1])
            n = len(t[sel])
            if 'Mock' in infile:
                n /= 9
            ngrp.append('{:d}'.format(int(n)))
            zmean.append('{:3.2f}'.format(np.mean(t['IterCenZ'][sel])))
            Mmean.append('{:3.2f}'.format(np.mean(t['log_mass'][sel])))
        return ngrp, zmean, Mmean

    ngrp_gama, zmean_gama, Mmean_gama = group_stats(g3cfof)
    ngrp_mock, zmean_mock, Mmean_mock = group_stats(g3cmockfof)
    names = []
    limits = []
    nmbin = len(mbins) - 1
#    for i in range(nmbin):
#        names.append(r'${\cal M}' + str(i+1) + r'$')
#        limits.append('[{}, {}]'.format(mbins[i], mbins[i+1]))
#    t = {'Name': names, r'$\lg {\cal M}_{h, {\rm limits}}$': limits,
#         r'$N_G$': ngrp_gama,
#         r'$\overline{\lg \cal M}_G$': Mmean_gama, r'$\overline{z}_G$': zmean_gama,
#         r'$N_M$': ngrp_mock,
#         r'$\overline{\lg \cal M}_M$': Mmean_mock, r'$\overline{z}_M$': zmean_mock,
#         }
#    ascii.write(t, output=plot_dir + out_file, format='latex', overwrite=True,
#                latexdict={'tabletype': 'table',
#                           'caption': r'''Group mass bin names and limits,
#                           with number of groups and mean log-mass and
#                           redshift for GAMA-II data and mocks.
#                           Note that each mock realisation has 20 per cent
#                           smaller volume than the GAMA-II fields.
#                           \label{tab:group_mass_def}''',
#                           'header_start': r'''\hline
#                           & & \multicolumn{3}{c}{GAMA} & 
#                           \multicolumn{3}{c}{Mock} \\
#                           % \cline{3-5} \cline{6-8}
#                           ''',
#                           'header_end': r'\hline',
#                           'data_end': r'\hline'})
    f = open(plot_dir + out_file, 'w')
    print(r'''
\begin{table}
\caption{Group bin names and log-mass limits, with number of groups,
mean log-mass, and mean redshift for GAMA-II data and mocks.
Note that each mock realisation has 20 per cent smaller volume than
the GAMA-II fields.
\label{tab:group_mass_def}}
\begin{tabular}{ccccccccc}
\hline
& & \multicolumn{3}{c}{GAMA} & & \multicolumn{3}{c}{Mocks} \\
\cline{3-5} \cline{7-9}
Name & $\lg {\cal M}_{h, {\rm limits}}$ &
$N$ & $\langle \lg {\cal M} \rangle$ & $\langle z \rangle$ & &
$N$ & $\langle \lg {\cal M} \rangle$ & $\langle z \rangle$ \\
\hline''', file=f)
    for i in range(nmbin):
        print(r'${\cal M}' + str(i+1) + r'$ & ' +
              '[{}, {}] & '.format(mbins[i], mbins[i+1]) +
              ngrp_gama[i] + ' & ' + Mmean_gama[i] + ' & ' + zmean_gama[i] +
              ' & & ' +
              ngrp_mock[i] + ' & ' + Mmean_mock[i] + ' & ' + zmean_mock[i] +
              r'\\', file=f)
    print('''
\hline
\end{tabular}
\end{table}
''', file=f)
    f.close()


def gal_mass_z(nmin=5, edge_min=0.9, fslim=(0.8, 10), Mmin=8, Mmax=12, nz=10,
               pfit_ord=2, nfit=4, nboot=100, z0=0.5,
               Mlim_ord=3, plot_file='gal_mass_z.pdf', plot_size=(6, 4)):
    """Grouped galaxy mass-redshift scatterplot."""

    def neg(M):
        """Returns -1*kernel(M) for scipy to minimize."""
        return -1*kernel(M)

    samp = gs.GalSample()
    samp.read_gama()
    samp.stellar_mass(fslim=fslim)
    samp.group_props()
    t = samp.t
    sel = (np.array(t['GroupEdge'] > edge_min) *
           np.logical_not(t['log_mass'].mask) *
           np.array(t['Nfof'] >= nmin))
    t = t[sel]

#    zlims = np.percentile(t['z'], np.linspace(0, 100, nz+1))
    zlims = np.linspace(0.0, 0.5, nz+1)
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
    plt.scatter(t['z'], t['logmstar'], s=0.1, c=t['gminusi_stars'],
                vmin=0.2, vmax=1.2, cmap='coolwarm')
    cb = plt.colorbar()
    cb.set_label(r'$(g - i)^*$')
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

    plt.clf()
    plt.hist(t['gminusi_stars'], bins=20)
    plt.show()


def gal_mass_lum(nmin=5, edge_min=0.9, fslim=(0.8, 10), pc=95,
                 plot_file='gal_mass_lum.pdf', plot_size=(5, 4)):
    """Galaxy stellar mass-luminosity scatterplot."""

    samp = gs.GalSample()
    samp.read_gama()
    samp.stellar_mass(fslim=fslim)
    samp.group_props()
    t = samp.t
    sel = (np.array(t['GroupEdge'] > edge_min) *
           np.logical_not(t['log_mass'].mask) *
           np.array(t['Nfof'] >= nmin))
    samp.t = t[sel]
    mags = np.linspace(-23, -15, 32)
    masses = samp.smf_comp(mags, pc=pc)
    fitpar = np.polynomial.polynomial.polyfit(mags, masses, 2)
    fit = np.polynomial.polynomial.polyval((mags[0], mags[-1]), fitpar)
    print('95-percentile mass = {} + {} M_r'.format(*fitpar))

    plt.clf()
    plt.scatter(samp.abs_mags('r_petro'), samp.t['logmstar'], s=0.1, label=None)
    plt.plot(mags, masses, 'g-', label='Binned 95-th percentile')
    plt.plot((mags[0], mags[-1]), fit, 'r-', label='Linear fit')
    plt.xlabel(r'$^{0.1}M_r - 5 \log\ h$')
    plt.ylabel(r'$\log_{10}({\cal M}_*/{\cal M}_\odot h^{-2})$')
    plt.axis((-15, -23, 7, 12))
    plt.legend()
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


def sersic_hist(plot_file='sersic_hist.pdf', range=(-1, 1.5), ncut=1.9,
                colour='gi_colour', plot_size=(5, 4)):
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
    plt.hist((logn[blue], logn[red]), bins=20, range=range, color=('b', 'r'))
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
#    plt.draw()
#    fig = plt.gcf()
#    fig.set_size_inches(plot_size)
#    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
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
    phi = lf.Schechter_dbl_mass(logM, wright)
    plt.plot(logM, phi, 'r', label='Wright+2017')
    plt.axis(lf_lims)
    plt.legend()
#    plt.draw()
#    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.show()


def mass_limit(samp):
    """Apply stellar mass completeness limit determined in gal_mass_z()"""
    p = [1.17442222,  29.68880365, -22.58489171]
    a = 1/(1 + samp.t['z'])
    Mt = np.polynomial.polynomial.polyval(a, p)
    sel = samp.t['logmstar'] > Mt
    samp.t = samp.t[sel]
    samp.t['zlo'] = samp.zlimits[0]*np.ones(len(samp.t))
    a = (-p[1] - (p[1]**2 - 4*p[2]*(p[0]-samp.t['logmstar']))**0.5)/2/p[2]
    samp.t['zhi'] = np.minimum(samp.zlimits[1], 1/a - 1)


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


def clf_vmax(Vmax='Vmax_dec', outfile='clf_vmax.pkl'):
    """Vmax LF."""
    clf(Vmax=Vmax, outfile=outfile)


def clf_mock(galfile=g3cmockgal, grpfile=g3cmockfof, Vmax='Vmax_raw',
             error='mock', outfile='clf_mock_vmax.pkl'):
    """Vmax LF."""
    clf(galfile=galfile, grpfile=grpfile, Vmax=Vmax, error=error,
        outfile=outfile)


def clf(galfile=g3cgal, grpfile=g3cfof, mbins=mbins_def, clrname='gi_colour',
        Vmax='Vmax_grp',
        bins=np.linspace(-24, -16, 17), zlimits=(0.002, 0.65), vol_z=None,
        colname='r_petro', error='jackknife', outfile='clf.pkl'):
    """Conditional LF by central/satellite, galaxy colour and Sersic index.
    Normalised to number of groups if Vmax='Vmax_grp', otherwise using
    density-corrected Vmax and normalised to total grouped galaxy sample."""

    samp = gs.GalSample(zlimits=zlimits)
    samp.read_gama_groups(galfile=galfile, grpfile=grpfile)
    if error != 'mock':
        samp.stellar_mass()
        samp.add_sersic_index()
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

#    samp.vis_calc((lf.sel_gama_mag_lo, lf.sel_gama_mag_hi))
    samp.vmax_calc()
    lf_dict = {}
    nmbin = len(mbins) - 1
    for i in range(nmbin):
        sel_dict = {'log_mass': (mbins[i], mbins[i+1])}
        samp.select(sel_dict)
        if Vmax == 'Vmax_grp':
            samp.vmax_group(mbins[i], mbins[i+1])
#            norm = 1.0/len(np.unique(samp.tsel()['GroupID']))
            norm = 1
        else:
            norm = len(samp.t)/len(samp.tsel())
        phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax, error=error,
                    sel_dict=sel_dict)
        Mkey = 'M{}'.format(i)
        lf_dict[Mkey] = phi

        for lbl, rank_lims, p0 in zip(['cen', 'sat'], [[1, 2], [2, 500]],
                                      ((5, -19.5, -2.5), (-1, -20, 1))):
            sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                        'RankIterCen': rank_lims}
            samp.select(sel_dict)
            phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax, error=error,
                        sel_dict=sel_dict)
#            if lbl == 'cen':
#                phi.fn_fit(fn=lf.gaussian, p0=(-22, 1, -2),
#                              Mmin=Mmin_fit, Mmax=Mmax_fit)
#            else:
            lf_dict[Mkey + lbl] = phi

        if error != 'mock':
            for colour in 'br':
                clr_limits = ('a', 'z')
                if (colour == 'b'):
                    clr_limits = ('b', 'c')
                if (colour == 'r'):
                    clr_limits = ('r', 's')
                sel_dict = {'log_mass': (mbins[i], mbins[i+1]), clrname: clr_limits}
                samp.select(sel_dict)
                phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax, error=error,
                            sel_dict=sel_dict)
                lf_dict[Mkey + colour] = phi
    
            for lbl, sersic_lims in zip(['nlo', 'nhi'], [[0, 1.9], [1.9, 30]]):
                sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                            'GALINDEX_r': sersic_lims}
                samp.select(sel_dict)
                phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax, error=error,
                            sel_dict=sel_dict)
                lf_dict[Mkey + lbl] = phi

    pickle.dump(lf_dict, open(outfile, 'wb'))


def csmf_vmax(Vmax='Vmax_dec', outfile='csmf_vmax.pkl'):
    """Vmax SMF."""
    csmf(Vmax=Vmax, outfile=outfile)


def csmf(fslim=(0.8, 10), mbins=mbins_def, masscomp=True,
         Vmax='Vmax_grp', p0=(-1.5, 10.5, 1), bins=np.linspace(7, 12, 20),
         Mmin_fit=8.5, Mmax_fit=12, clrname='gi_colour',
         colname='logmstar', error='jackknife', outfile='csmf.pkl'):
    """Conditional SMF by central/satellite, galaxy colour and Sersic index."""

    samp = gs.GalSample()
    samp.read_gama()
    samp.group_props()
    samp.add_sersic_index()
    samp.stellar_mass(fslim=fslim)

    # Apply stellar mass completeness limit determined in gal_mass_z()
    if masscomp:
        mass_limit(samp)
    else:
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
        else:
            norm = len(samp.t)/len(samp.tsel())
        phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax, error=error,
                    sel_dict=sel_dict)
#        phi.fn_fit(fn=lf.Schechter_mass, p0=p0, Mmin=Mmin_fit, Mmax=Mmax_fit)
#        p0 = phi.fit_par
        Mkey = 'M{}'.format(i)
        lf_dict[Mkey] = phi

        for colour in 'br':
            clr_limits = ('a', 'z')
            if (colour == 'b'):
                clr_limits = ('b', 'c')
            if (colour == 'r'):
                clr_limits = ('r', 's')
            sel_dict = {'log_mass': (mbins[i], mbins[i+1]), clrname: clr_limits}
            samp.select(sel_dict)
            phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax, error=error,
                        sel_dict=sel_dict)
#            phi.fn_fit(fn=lf.Schechter_mass, p0=p0,
#                       Mmin=Mmin_fit, Mmax=Mmax_fit)
            lf_dict[Mkey + colour] = phi

        for lbl, sersic_lims in zip(['nlo', 'nhi'], [[0, 1.9], [1.9, 30]]):
            sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                        'GALINDEX_r': sersic_lims}
            samp.select(sel_dict)
            phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax, error=error,
                        sel_dict=sel_dict)
#            phi.fn_fit(fn=lf.Schechter_mass, p0=p0,
#                       Mmin=Mmin_fit, Mmax=Mmax_fit)
            lf_dict[Mkey + lbl] = phi

        for lbl, rank_lims, p0 in zip(['cen', 'sat'], [[1, 2], [2, 500]],
                                      ((2, 10.5, -2.5), (-1.5, 10.5, -2.5))):
            sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                        'RankIterCen': rank_lims}
            samp.select(sel_dict)
            phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax, error=error,
                        sel_dict=sel_dict)
#            phi.fn_fit(fn=lf.Schechter_mass, p0=p0,
#                       Mmin=Mmin_fit, Mmax=Mmax_fit)
            lf_dict[Mkey + lbl] = phi

    pickle.dump(lf_dict, open(outfile, 'wb'))


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


def lfr_z(nmin=5, edge_min=0.9, colname='r_petro',
          bins=np.linspace(-24, -16, 18),
          Mmin_fit=-24, Mmax_fit=-19.4, zlims=(0.002, 0.1, 0.2, 0.3),
          clrname='gi_colour', error='jackknife', outfile='lfr_z.pkl'):
    """r-band LF in redshift bins (without evolution corrections)."""

    samp = gs.GalSample(Q=0, P=0)
    samp.read_gama()
    samp.stellar_mass()
    samp.add_sersic_index()
    samp.vis_calc((lf.sel_gama_mag_lo, lf.sel_gama_mag_hi))
    samp.group_props()
    t = samp.t

    ngal = len(t)
    sel = (np.array(t['GroupEdge'] > edge_min) *
           np.logical_not(t['log_mass'].mask) *
           np.array(t['Nfof'] >= nmin))
    ngrouped = len(t[sel])
    norm = ngrouped/ngal
    print('norm =', norm)

    lf_dict = {}
    for iz in range(3):
        zlo, zhi = zlims[iz], zlims[iz+1]
        samp.zlimits = (zlo, zhi)
        samp.vmax_calc()
        sel_dict = {'z': (zlo, zhi)}
        samp.select(sel_dict)
        phi = lf.LF(samp, colname, bins,
                    norm=norm, error=error, sel_dict=sel_dict)
        phi.comp_limits(samp, zlo, zhi)
#        phi.fn_fit(fn=lf.Schechter_mag, Mmin=Mmin_fit, Mmax=Mmax_fit)
        Mkey = 'z{}'.format(iz)
        lf_dict[Mkey] = phi

        for colour in 'br':
            clr_limits = ('a', 'z')
            if (colour == 'b'):
                clr_limits = ('b', 'c')
            if (colour == 'r'):
                clr_limits = ('r', 's')
            sel_dict = {'z': (zlo, zhi), clrname: clr_limits}
            samp.select(sel_dict)
            phi = lf.LF(samp, colname, bins,
                        norm=norm, error=error, sel_dict=sel_dict)
            phi.comp_limits(samp, zlo, zhi)
#            phi.fn_fit(fn=lf.Schechter_mag, Mmin=Mmin_fit, Mmax=Mmax_fit)
            lf_dict[Mkey + colour] = phi

        for lbl, sersic_lims in zip(['nlo', 'nhi'], [[0, 1.9], [1.9, 30]]):
            sel_dict = {'z': (zlo, zhi), 'GALINDEX_r': sersic_lims}
            samp.select(sel_dict)
            phi = lf.LF(samp, colname, bins,
                        norm=norm, error=error, sel_dict=sel_dict)
            phi.comp_limits(samp, zlo, zhi)
#            phi.fn_fit(fn=lf.Schechter_mag, Mmin=Mmin_fit, Mmax=Mmax_fit)
            lf_dict[Mkey + lbl] = phi

    pickle.dump(lf_dict, open(outfile, 'wb'))


def lfr_z_mock(nmin=5, edge_min=0.9, colname='r_petro',
               bins=np.linspace(-24, -16, 18),
               Mmin_fit=-24, Mmax_fit=-19.4, zlims=(0.002, 0.1, 0.2, 0.3),
               error='mock', outfile='lfr_z_mock.pkl'):
    """mock r-band LF in redshift bins (without evolution corrections)."""

    samp = gs.GalSample(Q=0, P=0)
    samp.read_gama_group_mocks()
    samp.vis_calc((lf.sel_gama_mag_lo, lf.sel_gama_mag_hi))
    t = samp.t

    ngal = len(t)
    sel = (np.array(t['GroupEdge'] > edge_min) *
           np.logical_not(t['log_mass'].mask) *
           np.array(t['Nfof'] >= nmin))
    ngrouped = len(t[sel])
    norm = ngrouped/ngal
    print('norm =', norm)

    lf_dict = {}
    for iz in range(3):
        zlo, zhi = zlims[iz], zlims[iz+1]
        samp.zlimits = (zlo, zhi)
        samp.vmax_calc(denfile=None)
        sel_dict = {'z': (zlo, zhi)}
        samp.select(sel_dict)
        phi = lf.LF(samp, colname, bins,
                    norm=norm, error=error, sel_dict=sel_dict)
        phi.comp_limits(samp, zlo, zhi)
#        phi.fn_fit(fn=lf.Schechter_mag, Mmin=Mmin_fit, Mmax=Mmax_fit)
        Mkey = 'z{}mock'.format(iz)
        lf_dict[Mkey] = phi

    pickle.dump(lf_dict, open(outfile, 'wb'))


def smf_z(nmin=5, edge_min=0.9, colname='logmstar',
          bins=np.linspace(7, 12, 20),
          Mmin_fit=8.5, Mmax_fit=12, zlims=(0.002, 0.1, 0.2, 0.3),
          fslim=(0.8, 10), pc=95, clrname='gi_colour',
          p0=(-0.5, 10.5, -2.5), error='jackknife', outfile='smf_z.dat'):
    """SMF in redshift bins (without evolution corrections)."""

    samp = gs.GalSample(Q=0, P=0)
    samp.read_gama()
    samp.add_sersic_index()
    samp.stellar_mass(fslim=fslim)
    samp.vis_calc((lf.sel_gama_mag_lo, lf.sel_gama_mag_hi))
#    samp.group_props()
    t = samp.t

    ngal = len(t)
    sel = (np.array(t['GroupEdge'] > edge_min) *
           np.logical_not(t['log_mass'].mask) *
           np.array(t['Nfof'] >= nmin))
    ngrouped = len(t[sel])
    norm = ngrouped/ngal
    print('norm =', norm)

    lf_dict = {}
    for iz in range(3):
        zlo, zhi = zlims[iz], zlims[iz+1]
        samp.zlimits = (zlo, zhi)
        samp.vmax_calc()
        sel_dict = {'z': (zlo, zhi)}
        samp.select(sel_dict)
        phi = lf.LF(samp, colname, bins,
                    norm=norm, error=error, sel_dict=sel_dict)
        phi.comp_limit_mass(samp, zlo)
        phi.fn_fit(fn=lf.Schechter_mass, p0=p0, Mmin=Mmin_fit, Mmax=Mmax_fit)
        Mkey = 'z{}'.format(iz)
        lf_dict[Mkey] = phi

        for colour in 'br':
            clr_limits = ('a', 'z')
            if (colour == 'b'):
                clr_limits = ('b', 'c')
            if (colour == 'r'):
                clr_limits = ('r', 's')
            sel_dict = {'z': (zlo, zhi), clrname: clr_limits}
            samp.select(sel_dict)
            phi = lf.LF(samp, colname, bins,
                        norm=norm, error=error, sel_dict=sel_dict)
            phi.comp_limit_mass(samp, zlo)
            phi.fn_fit(fn=lf.Schechter_mass, p0=p0,
                       Mmin=Mmin_fit, Mmax=Mmax_fit)
            lf_dict[Mkey + colour] = phi

        for lbl, sersic_lims in zip(['nlo', 'nhi'], [[0, 1.9], [1.9, 30]]):
            sel_dict = {'z': (zlo, zhi), 'GALINDEX_r': sersic_lims}
            samp.select(sel_dict)
            phi = lf.LF(samp, colname, bins,
                        norm=norm, error=error, sel_dict=sel_dict)
            phi.comp_limit_mass(samp, zlo)
            phi.fn_fit(fn=lf.Schechter_mass, p0=p0,
                       Mmin=Mmin_fit, Mmax=Mmax_fit)
            lf_dict[Mkey + lbl] = phi

    pickle.dump(lf_dict, open(outfile, 'wb'))


def clf_mock_old(mbins=mbins_def, zlimits=(0.002, 0.65),
             Vmax='Vmax_raw', bins=np.linspace(-24, -16, 18),
             colname='r_petro', error='mock', outfile='clf_mock_vmax.pkl'):
    """Conditional LF for mocks."""

    samp = gs.GalSample(zlimits=zlimits)
    samp.read_gama_groups(galfile=g3cmockgal, grpfile=g3cmockfof)

#    samp.vis_calc((gs.sel_gama_mag_lo, gs.sel_gama_mag_hi))
    samp.vmax_calc(denfile=None)
    lf_dict = {}
    nmbin = len(mbins) - 1
    for i in range(nmbin):
        sel_dict = {'log_mass': (mbins[i], mbins[i+1])}
        samp.select(sel_dict)
        if Vmax == 'Vmax_grp':
            samp.vmax_group(mbins[i], mbins[i+1])
            norm = 1
        else:
            norm = len(samp.t)/len(samp.tsel())
        phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax, error=error,
                    sel_dict=sel_dict)
        Mkey = 'M{}mock'.format(i)
        lf_dict[Mkey] = phi

        for lbl, rank_lims, p0 in zip(['cen', 'sat'], [[1, 2], [2, 500]],
                                      ((3, -20, -2.5), (-1, -20, -2.5))):
            sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                        'RankIterCenF': rank_lims}
            samp.select(sel_dict)
            phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax, error=error,
                        sel_dict=sel_dict)
            lf_dict[Mkey + lbl] = phi

    pickle.dump(lf_dict, open(outfile, 'wb'))


def clf_ev(mbins=(13.3, 13.5, 13.7, 15.2), mkey=(3, 4, 56), Vmax='Vmax_grp',
           zlims=(0.002, 0.1, 0.2, 0.3), clrname='gi_colour',
           bins=np.linspace(-24, -16, 18),
           p0=(-1, -20, 1),  colname='r_petro', error='jackknife',
           outfile='clf_ev.pkl'):
    """Evolution of the conditional LF separated by by central/satellite,
    galaxy colour and Sersic index."""

    samp = gs.GalSample(Q=0, P=0)
    samp.read_gama()
    samp.stellar_mass()
    samp.group_props()
    samp.add_sersic_index()

    samp.vis_calc((lf.sel_gama_mag_lo, lf.sel_gama_mag_hi))
    lf_dict = {}
    nmbin = len(mbins) - 1
    for i in range(nmbin):
        sel_dict = {'log_mass': (mbins[i], mbins[i+1])}
        samp.select(sel_dict)
        if Vmax == 'Vmax_grp':
            norm = 1
        else:
            norm = len(samp.t)/len(samp.tsel())
        for iz in range(3):
            zlo, zhi = zlims[iz], zlims[iz+1]
            samp.zlimits = (zlo, zhi)
            sel_dict = {'log_mass': (mbins[i], mbins[i+1]), 'z': (zlo, zhi)}
            samp.select(sel_dict)
            if Vmax == 'Vmax_grp':
                samp.vmax_group(mbins[i], mbins[i+1])
            else:
                samp.vmax_calc()
            phi = lf.LF(samp, colname, bins,
                        norm=norm, Vmax=Vmax, error=error, sel_dict=sel_dict)
            phi.comp_limits(samp, zlo, zhi)
            Mkey = 'M{}z{}'.format(mkey[i], iz)
            lf_dict[Mkey] = phi

            for colour in 'br':
                clr_limits = ('a', 'z')
                if (colour == 'b'):
                    clr_limits = ('b', 'c')
                if (colour == 'r'):
                    clr_limits = ('r', 's')
                sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                            'z': (zlo, zhi),
                            clrname: clr_limits}
                samp.select(sel_dict)
                phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax,
                            error=error, sel_dict=sel_dict)
                phi.comp_limits(samp, zlo, zhi)
                lf_dict[Mkey + colour] = phi

            for lbl, sersic_lims in zip(['nlo', 'nhi'], [[0, 1.9], [1.9, 30]]):
                sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                            'z': (zlo, zhi),
                            'GALINDEX_r': sersic_lims}
                samp.select(sel_dict)
                phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax,
                            error=error, sel_dict=sel_dict)
                phi.comp_limits(samp, zlo, zhi)
                lf_dict[Mkey + lbl] = phi

            for lbl, rank_lims in zip(['cen', 'sat'], [[1, 2], [2, 500]]):
                sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                            'z': (zlo, zhi),
                            'RankIterCen': rank_lims}
                samp.select(sel_dict)
                phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax,
                            error=error, sel_dict=sel_dict)
                phi.comp_limits(samp, zlo, zhi)
                lf_dict[Mkey + lbl] = phi

    pickle.dump(lf_dict, open(outfile, 'wb'))


def csmf_ev(mbins=(13.3, 13.5, 13.7, 15.2), mkey=(3, 4, 56), Vmax='Vmax_grp',
            zlims=(0.002, 0.1, 0.2, 0.3), clrname='gi_colour', masscomp=1,
            fslim=(0.8, 10), bins=np.linspace(7, 12, 20), colname='logmstar',
            pc=95, error='jackknife', outfile='csmf_ev.pkl'):
    """Evolution of the conditional SMF separated by by central/satellite,
    galaxy colour and Sersic index."""

    samp = gs.GalSample(Q=0, P=0)
    samp.read_gama()
    samp.group_props()
    samp.add_sersic_index()
    samp.stellar_mass(fslim=fslim)

    # Apply stellar mass completeness limit determined in gal_mass_z()
    if masscomp:
        mass_limit(samp)
    else:
        samp.vis_calc((lf.sel_gama_mag_lo, lf.sel_gama_mag_hi))

    lf_dict = {}
    nmbin = len(mbins) - 1
    for i in range(nmbin):
        sel_dict = {'log_mass': (mbins[i], mbins[i+1])}
        samp.select(sel_dict)
        if Vmax == 'Vmax_grp':
            norm = 1
        else:
            norm = len(samp.t)/len(samp.tsel())
        for iz in range(3):
            zlo, zhi = zlims[iz], zlims[iz+1]
            samp.zlimits = (zlo, zhi)
            sel_dict = {'log_mass': (mbins[i], mbins[i+1]), 'z': (zlo, zhi)}
            samp.select(sel_dict)
            if Vmax == 'Vmax_grp':
                samp.vmax_group(mbins[i], mbins[i+1])
            else:
                samp.vmax_calc()
            phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax,
                        error=error, sel_dict=sel_dict)
            phi.comp_limit_mass(samp, zlo)
            Mkey = 'M{}z{}'.format(mkey[i], iz)
            lf_dict[Mkey] = phi

            for colour in 'br':
                clr_limits = ('a', 'z')
                if (colour == 'b'):
                    clr_limits = ('b', 'c')
                if (colour == 'r'):
                    clr_limits = ('r', 's')
                sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                            'z': (zlo, zhi),
                            clrname: clr_limits}
                samp.select(sel_dict)
                phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax,
                            error=error, sel_dict=sel_dict)
                phi.comp_limit_mass(samp, zlo)
                lf_dict[Mkey + colour] = phi

            for lbl, sersic_lims in zip(['nlo', 'nhi'], [[0, 1.9], [1.9, 30]]):
                sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                            'z': (zlo, zhi),
                            'GALINDEX_r': sersic_lims}
                samp.select(sel_dict)
                phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax,
                            error=error, sel_dict=sel_dict)
                phi.comp_limit_mass(samp, zlo)
                lf_dict[Mkey + lbl] = phi

            for lbl, rank_lims in zip(['cen', 'sat'], [[1, 2], [2, 500]]):
                sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                            'z': (zlo, zhi),
                            'RankIterCen': rank_lims}
                samp.select(sel_dict)
                phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax,
                            error=error, sel_dict=sel_dict)
                phi.comp_limit_mass(samp, zlo)
                lf_dict[Mkey + lbl] = phi

    pickle.dump(lf_dict, open(outfile, 'wb'))


def clf_mock_ev(mbins=(13.3, 13.5, 13.7, 15.2), mkey=(3, 4, 56),
                zlims=(0.002, 0.1, 0.2, 0.3), Vmax='Vmax_grp',
                bins=np.linspace(-24, -16, 18),
                colname='r_petro', error='mock', outfile='clf_mock_ev.pkl'):
    """Conditional LF for mocks in redshift bins."""

    samp = gs.GalSample(Q=0, P=0)
    samp.read_gama_group_mocks()

    samp.vis_calc((lf.sel_gama_mag_lo, lf.sel_gama_mag_hi))
    lf_dict = {}
    nmbin = len(mbins) - 1
    for i in range(nmbin):
        sel_dict = {'log_mass': (mbins[i], mbins[i+1])}
        samp.select(sel_dict)
        if Vmax == 'Vmax_grp':
            norm = 1
        else:
            norm = len(samp.t)/len(samp.tsel())
        for iz in range(3):
            zlo, zhi = zlims[iz], zlims[iz+1]
            samp.zlimits = (zlo, zhi)
            sel_dict = {'log_mass': (mbins[i], mbins[i+1]), 'z': (zlo, zhi)}
            samp.select(sel_dict)
            if Vmax == 'Vmax_grp':
                samp.vmax_group(mbins[i], mbins[i+1])
            else:
                samp.vmax_calc(denfile=None)
            phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax, error=error,
                        sel_dict=sel_dict)
            phi.comp_limits(samp, zlo, zhi)
            Mkey = 'M{}z{}mock'.format(mkey[i], iz)
            lf_dict[Mkey] = phi

            for lbl, rank_lims, p0 in zip(['cen', 'sat'], [[1, 2], [2, 500]],
                                          ((3, -20, -2.5), (-1, -20, -2.5))):
                sel_dict = {'log_mass': (mbins[i], mbins[i+1]),
                            'z': (zlo, zhi),
                            'RankIterCenF': rank_lims}
                samp.select(sel_dict)
                phi = lf.LF(samp, colname, bins, norm=norm, Vmax=Vmax,
                            error=error, sel_dict=sel_dict)
                phi.comp_limits(samp, zlo, zhi)
                lf_dict[Mkey + lbl] = phi

    pickle.dump(lf_dict, open(outfile, 'wb'))


def csmf_plots(infile='csmf.pkl', mock_file=None, fn=SchecMass(),
               p0=(-1.5, 10.5, 1), Mmin_fit=8.8, Mmax_fit=12.0,
               nmin=2, lf_lims=(8.5, 12.5, 1e-2, 200),
               schec_lims=(-1.6, 0.1, 9.9, 11.4),
               tab_file='csmf_schec.tex', mock_tab_file=None,
               cen_sat_plot='csmf_cen_sat.pdf',
               colour_plot='csmf_colour.pdf',
               sersic_plot='csmf_sersic.pdf',
               schec_plot='csmf_schec.pdf',
               xlabel=ms_label, ylabel=csmf_label):
    clf_plots(infile, mock_file, fn, p0, Mmin_fit, Mmax_fit, nmin, lf_lims,
              schec_lims,
              tab_file, mock_tab_file, cen_sat_plot, colour_plot, sersic_plot,
              schec_plot, xlabel, ylabel)


def clf_plots(infile='clf_vmax.pkl', mock_file='clf_mock_vmax.pkl',
              fn=SchecMag(), p0=(-1, -20, 1), Mmin_fit=-24, Mmax_fit=-17,
              nmin=2, lf_lims=(-15.5, -23.9, 1e-7, 0.1),
              schec_lims=(-1.6, 0.3, -21.6, -19.1),
              tab_file='clf_schec.tex', mock_tab_file='mock_clf_schec.tex',
              cen_sat_plot='clf_cen_sat.pdf',
              colour_plot='clf_colour.pdf',
              sersic_plot='clf_sersic.pdf',
              schec_plot='clf_schec.pdf', xlabel=mag_label, ylabel=lf_label,
              sigma=[2, ], lc_limits=5, lc_step=32):
    """Plot and tabulate galaxy CLFs by cen/sat, colour and Sersic index."""

    # For Yang et al. comparison
    mbins = np.linspace(min(lf_lims[:2]), max(lf_lims[:2]), 50)
    if 'clf' in infile:
        Mr_sun = 4.76
        lbins = 0.4*(Mr_sun - mbins)
    lf_dict = pickle.load(open(infile, 'rb'))
    if mock_file:
        mock_dict = pickle.load(open(mock_file, 'rb'))
        lf_dict = {**lf_dict, **mock_dict}
        labels = ['mock', '', 'b', 'r', 'nlo', 'nhi']
        descriptions = ['Mock', 'All', 'Blue', 'Red', 'low-n', 'high-n']
    else:
        labels = ['', 'b', 'r', 'nlo', 'nhi']
        descriptions = ['All', 'Blue', 'Red', 'low-n', 'high-n']

    fn.alpha = p0[0]
    fn.Mstar = p0[1]
    fn.lgps = p0[2]

    plot_size = (6, 7.5)
    sa_left = 0.15
    sa_bot = 0.05
    nbin = 6
    nrow, ncol = util.two_factors(nbin)

#   Tabulate Schechter parameter fits
    labels = ['', 'cen', 'sat', 'b', 'r', 'nlo', 'nhi']
    descriptions = ['All', 'Central', 'Satellite', 'Blue', 'Red',
                    'low-n', 'high-n']
    f = open(plot_dir + tab_file, 'w')
    print(r"""
\begin{math}
\begin{array}{crccc}
\hline
 & N_{\rm gal} & \alpha & M^* & \chi^2/\nu \\[0pt]
""", file=f)
    for lbl, desc in zip(labels, descriptions):
        print(r"""
\hline
\multicolumn{5}{c}{\mbox{""", desc, r"""}} \\
""", file=f)
        for i in range(6):
            key = 'M{}'.format(i) + lbl
            phi = lf_dict[key]
            fn = phi.fn_fit(fn, Mmin=Mmin_fit, Mmax=Mmax_fit, verbose=0)
            lf_dict[key] = copy.deepcopy(phi)
#            pdb.set_trace()
            try:
                fit_errs = 0.5*(np.array(phi.errors.parmaxes) -
                                         np.array(phi.errors.parmins))
                print(r'\mass{} & {} & {:5.2f}\pm{:5.2f} & {:5.2f}\pm{:5.2f} & {:5.2f}/{:2d} \\[0pt]'.format(
                        i+1, int(phi.ngal.sum()), phi.res.parvals[0], fit_errs[0],
                        phi.res.parvals[1], fit_errs[1], phi.res.statval, phi.res.dof), file=f)
            except:
#            except TypeError or AttributeError:
                print('bad error estimate')                    
    print(r"""
\hline
\end{array}
\end{math}""", file=f)
    f.close()

    #   Tabulate Schechter parameter fits for mocks
    if mock_file:
#        mlabels = ['mock', 'mockcen', 'mocksat']
        mlabels = ['', 'cen', 'sat']
        mdescriptions = ['Mock', 'Mock central', 'Mock satellite']
        f = open(plot_dir + mock_tab_file, 'w')
        print(r"""
\begin{math}
\begin{array}{crccc}
\hline
 & N_{\rm gal} & \alpha & M^* & \chi^2/\nu \\[0pt]
""", file=f)
        for lbl, desc in zip(mlabels, mdescriptions):
            print(r"""
\hline
\multicolumn{5}{c}{\mbox{""", desc, r"""}} \\
""", file=f)
            for i in range(6):
                key = 'M{}'.format(i) + lbl
                phi = lf_dict[key]
                fn = phi.fn_fit(fn, Mmin=Mmin_fit, Mmax=Mmax_fit, verbose=0)
                lf_dict[key] = copy.deepcopy(phi)
                print(r'\mass{} & {} & {:5.2f}\pm{:5.2f} & {:5.2f}\pm{:5.2f} & {:5.2f}/{:2d} \\[0pt]'.format(
                        i+1, int(phi.ngal.sum()), phi.res.parvals[0], fit_errs[0],
                        phi.res.parvals[1], fit_errs[1], phi.res.statval, phi.res.dof), file=f)
#                print(r'\mass{} & {} & {:5.2f}\pm{:5.2f} & {:5.2f}\pm{:5.2f} & {:5.2f}/{:2d} \\[0pt]'.format(
#                        i+1, int(phi.ngal.sum()), phi.fit_par[0], phi.fit_err[0],
#                        phi.fit_par[1], phi.fit_err[1], phi.chi2, phi.ndof), file=f)
        print(r"""
\hline
\end{array}
\end{math}""", file=f)
        f.close()

#   CLF by central/satellite, including mocks and Yang+ comparison
#   (plot first, so they don't overwrite GAMA results)
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
        if 'clf' in infile:
            yang_file = 'clf_all.txt'
            yang_cen, yang_sat = yang(yang_file, halomass[i], lbins)
        else:
            yang_file = 'csmf_all.txt'
            yang_cen, yang_sat = yang(yang_file, halomass[i], mbins)
        ax.plot(mbins, yang_cen, 'r--')
        ax.plot(mbins, yang_sat, 'b--')
        if mock_file:
#            phi = lf_dict[key + 'mock']
            phi.plot(ax=ax, nmin=nmin, ls=':', clr='k', mfc='w')
        phi = lf_dict[key]
        phi.plot(ax=ax, nmin=nmin, clr='k', label='all')
#        pdb.set_trace()
        for cs, colour, label in zip(['cen', 'sat'], 'rb',
                                     ['central', 'satellite']):
            if mock_file:
#                phi = lf_dict[key + 'mock' + cs]
                phi = lf_dict[key + cs]
                phi.plot(ax=ax, nmin=nmin, ls=':', clr=colour, mfc='w')
            phi = lf_dict[key + cs]
            phi.plot(ax=ax, nmin=nmin, clr=colour, label=label)
        if i == 4:
            ax.legend(loc=3)
    plt.draw()
    plt.savefig(plot_dir + cen_sat_plot, bbox_inches='tight')
    plt.show()

#   LF by mass/colour
    plt.clf()
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
    fig.set_size_inches(plot_size)
    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, xlabel, ha='center', va='center')
    fig.text(0.06, 0.5, ylabel, ha='center', va='center', rotation='vertical')
    plt.semilogy(basey=10, nonposy='clip')
    for i in range(nbin):
        key = 'M{}'.format(i)
        phi = lf_dict[key]
        label = r'${\cal M}' + r'{}\ [{}, {}]$'.format(
                i+1, mbins_def[i], mbins_def[i+1])
        ax = axes.flat[i]
        ax.axis(lf_lims)
        ax.text(0.9, 0.9, label, ha='right', transform=ax.transAxes)
        phi.plot(ax=ax, nmin=nmin, clr='k', label='all')
        for colour, label in zip('br', ['blue', 'red']):
            if 'clf' in infile:
                yang_file = 'clf_{}.txt'.format(label)
                yang_cen, yang_sat = yang(yang_file, halomass[i], lbins)
            else:
                yang_file = 'csmf_{}.txt'.format(label)
                yang_cen, yang_sat = yang(yang_file, halomass[i], mbins)
            # Scale central CLF/CSMF my blue or red fraction
            if colour == 'b':
                yang_cen *= (1 - crf[i])
            else:
                yang_cen *= crf[i]
            ax.plot(mbins, yang_cen, colour+'--')
            ax.plot(mbins, yang_sat, colour+'--')
            phi = lf_dict[key + colour]
            phi.plot(ax=ax, nmin=nmin, clr=colour, label=label)
#        if mock_file:
#            phi = lf_dict[key + 'mock']
#            phi.plot(ax=ax, nmin=nmin, clr='g', label='Mock')
        if i == 4:
            ax.legend(loc=3)
    plt.draw()
    plt.savefig(plot_dir + colour_plot, bbox_inches='tight')
    plt.show()

#   LF by mass/sersic index
    plt.clf()
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
    fig.set_size_inches(plot_size)
    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
    fig.text(0.55, 0.0, xlabel, ha='center', va='center')
    fig.text(0.06, 0.5, ylabel, ha='center', va='center', rotation='vertical')
    plt.semilogy(basey=10, nonposy='clip')
    for i in range(nbin):
        key = 'M{}'.format(i)
        phi = lf_dict[key]
        label = r'${\cal M}' + r'{}\ [{}, {}]$'.format(
                i+1, mbins_def[i], mbins_def[i+1])
        ax = axes.flat[i]
        ax.axis(lf_lims)
        ax.text(0.9, 0.9, label, ha='right', transform=ax.transAxes)
        phi.plot(ax=ax, nmin=nmin, clr='k', label='all')
        for sersic, colour, label in zip(['nlo', 'nhi'], 'br',
                                         ['low-n', 'high-n']):
            phi = lf_dict[key + sersic]
            phi.plot(ax=ax, nmin=nmin, clr=colour, label=label)
        if i == 4:
            ax.legend(loc=3)
    plt.draw()
    plt.savefig(plot_dir + sersic_plot, bbox_inches='tight')
    plt.show()

#   Schechter parameters - mass panels
#    plt.clf()
#    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
#    fig.set_size_inches(plot_size)
#    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
#    fig.text(0.55, 0.0, r'$\alpha$', ha='center', va='center')
#    fig.text(0.06, 0.5, r'$M^*$', ha='center', va='center', rotation='vertical')
#    for i in range(nbin):
#        key = 'M{}'.format(i)
#        phi = lf_dict[key]
#        mlo = phi.sel_dict['log_mass'][0]
#        mhi = phi.sel_dict['log_mass'][1]
#        label = r'$M_h = [{}, {}]$'.format(mlo, mhi)
#        ax = axes.flat[i]
#        ax.text(0.1, 0.9, label, transform=ax.transAxes)
#        phi.like_cont(ax=ax, label=label, c='k')
#        for colour in 'br':
#            phi = lf_dict[key + colour]
#            phi.like_cont(ax=ax, label=label, c=colour)
#        for sersic, colour in zip(['nlo', 'nhi'], 'br'):
#            phi = lf_dict[key + sersic]
#            phi.like_cont(ax=ax, label=label, c=colour, ls='--')
#    plt.axis((-1.7, -0.3, -21.5, -18.5))
##    plt.legend(label_list)
#    plt.show()


#   Schechter parameters - type panels
    plt.clf()
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, num=1)
    fig.set_size_inches(plot_size)
    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
    fig.text(0.55, 0.0, r'$\alpha$', ha='center', va='center')
    lines = []
    colors = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan')
    iy = 0
    if mock_file:
        fig.text(0.06, 0.5, r'$M^*$', ha='center', va='center',
                 rotation='vertical')
        labels = ['', 'mock', 'b', 'r', 'nlo', 'nhi']
        descriptions = ['All', 'Mock', 'Blue', 'Red', 'low-n', 'high-n']
    else:
        fig.text(0.06, 0.5, r'$\lg\ {\cal M}^*$', ha='center', va='center',
                 rotation='vertical')
        labels = ['', 'b', 'r', 'nlo', 'nhi']
        descriptions = ['All', 'Blue', 'Red', 'low-n', 'high-n']
    for lbl, desc in zip(labels, descriptions):
        ax = axes.flat[iy]
        ax.text(0.1, 0.9, desc, transform=ax.transAxes)
        for i in range(nbin):
            key = 'M{}'.format(i) + lbl
            phi = lf_dict[key]
            label = r'${\cal M}' + r'{}$'.format(i+1)
            rproj = RegionProjection()
            rproj.prepare(nloop=(lc_step, lc_step), fac=lc_limits,
                          sigma=sigma)
            rproj.calc(phi.fit, phi.fn.alpha, phi.fn.Mstar)
#                rproj_dict[key] = rproj
            ax.plot(rproj.parval0, rproj.parval1, '+', color=colors[i])
            xmin, xmax = rproj.min[0], rproj.max[0]
            ymin, ymax = rproj.min[1], rproj.max[1]
            nx, ny = rproj.nloop
#                hx = 0.5 * (xmax - xmin) / (nx - 1)
#                hy = 0.5 * (ymax - ymin) / (ny - 1)
            extent = (xmin, xmax, ymin, ymax)
            y = rproj.y.reshape((ny, nx))
            v = rproj.levels
            ax.contour(y, v, aspect='auto', origin='lower', extent=extent,
                       colors=colors[i])
#            pdb.set_trace()
#            phi.like_cont(ax=ax, label=label, lc_limits=schec_lims)
#            phi.like_cont(ax=ax, label=label, lc_limits=lc_limits,
#                          lc_step=lc_step, dchisq=dchisq)
            if iy == 0:
#                lines.append(mpatches.Patch(color=colors[i], label=label))
                lines.append(mlines.Line2D([], [], color=colors[i], label=label))
        iy += 1
#    ax = axes.flat[iy]
#    pdb.set_trace()
#    plt.legend(handles=lines, loc='center', frameon=False)
    ax = axes.flat[0]
#    ax.legend(handles=lines, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#               ncol=3, handlelength=2, borderaxespad=0.)
    ax.legend(handles=lines, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=6, handlelength=1, borderaxespad=0.)
    plt.axis(schec_lims)
    plt.draw()
    plt.savefig(plot_dir + schec_plot, bbox_inches='tight')
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


def csmf_ev_plots(infile='csmf_ev.pkl', gal_file='smf_z.dat', fn=SchecMass(),
#                  mock_file='csmf_mock_ev.pkl',
                  mock_file=None, mock_gal_file=None,
                  p0=(-1.5, 10.5, 1), nmin=3, lf_lims=(8.5, 12.0, 1e-2, 80),
                  Mmin_fit=8.8, Mmax_fit=12, schec_lims=(9.6, 11.0, -0.2, 1.7),
                  plot_file='csmf_ev.pdf', plot_schec_file='csmf_ev_schec.pdf',
                  plot_size=(7, 10),
                  xlabel=ms_label, ylabel=csmf_label):
    """Plot and tabulate galaxy CLFs in redshift bins."""
    clf_ev_plots(infile=infile, gal_file=gal_file, fn=fn,
                 mock_file=mock_file, #mock_gal_file=mock_gal_file,
                 p0=p0, nmin=nmin, lf_lims=lf_lims,
                 Mmin_fit=Mmin_fit, Mmax_fit=Mmax_fit, schec_lims=schec_lims,
                 plot_file=plot_file, plot_schec_file=plot_schec_file,
                 plot_size=plot_size,
                 xlabel=xlabel, ylabel=ylabel)


def clf_ev_plots(infile='clf_ev.pkl', gal_file='lfr_z.pkl',
                 mock_file='clf_mock_ev.pkl', fn=SchecMag(),
                 mock_gal_file=None,
                 p0=(-1, -20, 1), nmin=3, lf_lims=(-15.5, -23.9, 1e-2, 90),
                 Mmin_fit=-24, Mmax_fit=-17, schec_lims=(-19, -21.8, -0.2, 2),
                 plot_file='clf_ev.pdf', plot_schec_file='clf_ev_schec.pdf',
                 plot_size=(7, 12),
                 mkey=(3, 4, 56), xlabel=mag_label, ylabel=clf_label,
                 zlabel=[r'$0.0 < z < 0.1$', r'$0.1 < z < 0.2$',
                         r'$0.2 < z < 0.3$'],
                 sigma=[2, ], lc_limits=5, lc_step=21):
    """Plot and tabulate galaxy CLFs in redshift bins."""

    lf_dict = pickle.load(open(infile, 'rb'))
#    gal_dict = pickle.load(open(gal_file, 'rb'))
    if mock_file:
        mock_dict = pickle.load(open(mock_file, 'rb'))
        lf_dict = {**lf_dict, **mock_dict}
#        mock_gal_dict = pickle.load(open(mock_gal_file, 'rb'))
#        gal_dict = {**gal_dict, **mock_gal_dict}
        labels = ['mock', '', 'b', 'r', 'nlo', 'nhi']
        descriptions = ['Mock', 'All', 'Blue', 'Red', 'low-n', 'high-n']
    else:
        labels = ['', 'b', 'r', 'nlo', 'nhi']
        descriptions = ['All', 'Blue', 'Red', 'low-n', 'high-n']

#    fn = models.ExponentialCutoffPowerLaw1D(
#            amplitude=p0[2], x_0=1.0, alpha=-p0[0], x_cutoff=1.0,
#            tied={'x_cutoff': lambda s: s.x_0})
#            bounds={'amplitude': (1e-5, 1e2), 'x_0': (1e-5, 1e5)})

#    from sherpa.astro.models import Schechter
#    fn = Schechter()
    fn.alpha = p0[0]
    fn.Mstar = p0[1]
    fn.lgps = p0[2]

    sa_left = 0.14
    sa_bot = 0.03
#    mbins = range(3)
    mbins = range(3)
#    mlabels = (r' ${\cal M}3$', r' ${\cal M}4$', r' ${\cal M}56$')
#    mlabels = (r' ${\cal M}3$', r' ${\cal M}4$', r' ${\cal M}5$', r' ${\cal M}6$')
    nsamp = len(labels)
    nrow, ncol = len(labels), len(mbins)
    colors = 'bgr'

    # Both figs have galaxy type by row, group mass by column
    # Fig 1: LF in redshift bins
    fig1, axes1 = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
    fig1.set_size_inches(plot_size)
    fig1.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
    fig1.text(0.51, 0.0, xlabel, ha='center', va='center')
    fig1.text(0.06, 0.46, ylabel, ha='center', va='center', rotation='vertical')
    plt.semilogy(basey=10, nonposy='clip')

    # Fig 2: (M*, lg phi*) likelihood contours
    fig2, axes2 = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=2)
    fig2.set_size_inches(plot_size)
    fig2.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
    fig2.text(0.51, 0.0, r'$^{0.1}M^* - 5 \log_{10} h$', ha='center', va='center')
    fig2.text(0.06, 0.46, r'$\log_{10} \phi^*$', ha='center', va='center',
              rotation='vertical')

#    rproj_dict = {}
    for it in range(nsamp):
        lbl = labels[it]
        desc = descriptions[it]
        for im in mbins:
            label = desc + r' ${\cal M}$' + '{}'.format(mkey[im])
            ax1 = axes1[it, im]
            ax1.axis(lf_lims)
            ax1.text(0.9, 0.9, label, ha='right',
                     transform=ax1.transAxes)
            ax2 = axes2[it, im]
            ax2.axis(schec_lims)
            ax2.text(0.9, 0.9, label, ha='right',
                     transform=ax2.transAxes)
            for iz in range(3):
                key = 'M{}z{}{}'.format(mkey[im], iz, lbl)
#                gkey = 'z{}{}'.format(iz, lbl)
#                if mock_file:
#                    phi = lf_dict[key + 'mock']
#                    phi.plot(ax=ax, nmin=nmin, ls=':', mfc='w', show_fit=False)
                phi = lf_dict[key]
                if iz == 0:
                    getattr(fn, 'alpha').frozen = False
                else:
                    getattr(fn, 'alpha').frozen = True
                fn = phi.fn_fit(fn, Mmin=Mmin_fit, Mmax=Mmax_fit, verbose=0)

                phi.plot(ax=ax1, nmin=nmin, clr=colors[iz], show_fit=1,
                         label=zlabel[iz])

                # Always freeze alpha for (M*, phi*) likelihood plots
                getattr(fn, 'alpha').frozen = True
                rproj = RegionProjection()
                rproj.prepare(nloop=(lc_step, lc_step), fac=lc_limits,
                              sigma=sigma)
                rproj.calc(phi.fit, phi.fn.Mstar, phi.fn.lgps)
                ax2.plot(rproj.parval0, rproj.parval1, '+', color=colors[iz])
                xmin, xmax = rproj.min[0], rproj.max[0]
                ymin, ymax = rproj.min[1], rproj.max[1]
                nx, ny = rproj.nloop
                extent = (xmin, xmax, ymin, ymax)
                y = rproj.y.reshape((ny, nx))
                v = rproj.levels
                ax2.contour(y, v, aspect='auto', origin='lower', extent=extent,
                            colors=colors[iz])
            if im == 0 and it == 0:
                ax1.legend(loc=3)
    plt.draw()
    plt.figure(1)
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    plt.figure(2)
    plt.savefig(plot_dir + plot_schec_file, bbox_inches='tight')
    plt.show()

    # Schecter parameters: galaxy type by row, group mass by column
#    plt.clf()
#    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
#    fig.set_size_inches(plot_size)
#    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.0, wspace=0.0)
#    fig.text(0.5, 0.0, r'$M^*$', ha='center', va='center')
#    fig.text(0.06, 0.5, r'$\log \phi^*$', ha='center', va='center',
#             rotation='vertical')
##    rproj.prepare(nloop=(41, 41))
#    clr = 'bgr'
#    for it in range(nsamp):
#        lbl = labels[it]
#        desc = descriptions[it]
#        for im in mbins:
#            label = desc + r' ${\cal M}$' + '{}'.format(mkey[im])
#            ax = axes[it, im]
##            print(ax)
##            plt.sca(ax)
#            ax.axis(schec_lims)
#            ax.text(0.9, 0.9, label, ha='right',
#                    transform=ax.transAxes)
#            for iz in range(3):
#                key = 'M{}z{}{}'.format(mkey[im], iz, lbl)
#                rproj = rproj_dict[key]
##                print(rproj)
#                xmin, xmax = rproj.x0.min(), rproj.x0.max()
#                ymin, ymax = rproj.x1.min(), rproj.x1.max()
#                nx, ny = rproj.nloop
#                hx = 0.5 * (xmax - xmin) / (nx - 1)
#                hy = 0.5 * (ymax - ymin) / (ny - 1)
#                extent = (xmin - hx, xmax + hx, ymin - hy, ymax + hy)
#                y = rproj.y.reshape((ny, nx))
#                v = rproj.levels
#                ax.contour(y, v, aspect='auto', origin='lower', extent=extent,
#                           colors=clr[iz])
##                pdb.set_trace()
##                ax.imshow(y, aspect='auto',
##                          origin='lower', extent=extent)
##                              linestyles=ls, colors=c, label=label)
##                rproj.contour()
##                gkey = 'z{}{}'.format(iz, lbl)
##                if mock_file:
##                    phi = lf_dict[key + 'mock']
##                    phi.plot(ax=ax, nmin=nmin, ls=':', mfc='w', show_fit=False)
##                phi = lf_dict[key]
##                phi.fit.fit()
##                phi.fit.estmethod = Confidence()
##                phi.fit.estmethod.max_rstat = 10
##                pdb.set_trace()
##                errors = phi.fit.est_errors()
##
##                rproj.prepare(nloop=(lc_step, lc_step))
##                rproj.calc(phi.fit, phi.fn.Mstar, phi.fn.lgps)
###                rproj.contour(overplot=1, clearwindow=0)
##                plt.clf()
##                rproj.contour()
##                plt.show()
##                phi.like_cont(px='schecmag.Mstar', py='schecmag.lgps', ax=ax,
##                              lc_limits=lc_limits,
##                              lc_step=lc_step, dchisq=dchisq)
##                from sherpa.plot import RegionProjection
##                rproj = RegionProjection()
##                rproj.prepare(min=[2.8, 1.75], max=[3.3, 2.1], nloop=[21, 21])
##                if iz == 0:
##                    phi.like_cont(pp=(1, 2), mp=0, ax=ax, lc_limits=lc_limits,
##                                  lc_step=lc_step, dchisq=dchisq)
##                else:
##                    phi.like_cont(pp=(1, 2), mp=None, ax=ax, lc_limits=lc_limits,
##                                  lc_step=lc_step, dchisq=dchisq)
##                print(phi.Mbin, phi.phi)
##                phi = gal_dict[gkey]
##                phi.plot(ax=ax, nmin=nmin, fmt='-', clr=colors[iz],
##                         show_fit=False)
##            if im == 0 and it == 0:
##                ax.legend(loc=3)
#    plt.draw()
##    plt.savefig(plot_dir + plot_schec_file, bbox_inches='tight')
#    plt.show()


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


def yang_plot(phistar, alpha, lgmstar_c, sigma_c, A, lgm, ls='--'):
    """Plot Yang+ functional fit (log-normal for centrals, modified Schechter
    for satellites)."""
    phi_c = A/(2*math.pi)**0.5/sigma_c * np.exp(
            -(lgm - lgmstar_c)**2 / (2*sigma_c**2))
    Mr = 10**(lgm - (lgmstar_c - 0.25))
    phi_s = phistar * Mr**(alpha+1) * np.exp(-Mr**2)
    plt.plot(lgm, phi_c)
    plt.plot(lgm, phi_s)


# Simulation routines to generate and analyse samples with known CLF
def clf_sim_gen10k(nsim=9, nhalo=10000, mlimits=(0, 19.8), poisson=True,
                   grp_file='sim_group_10k.fits', gal_file='sim_gal_10k.fits'):
    """Mock CLF simulations with only 10k groups per sim."""
    clf_sim_gen(nsim=nsim, nhalo=nhalo, mlimits=mlimits, poisson=poisson,
                grp_file=grp_file, gal_file=gal_file)


def clf_sim_gen(nsim=9, nhalo=50000, mlimits=(0, 19.8), poisson=True,
                grp_file='sim_group.fits', gal_file='sim_gal.fits'):
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
    # where dlgm = hmf_lgmstar - hmf_lgm
    clff = SchecMag()
    a0, a1, m0, m1, p0, p1 = -1.4, 0.2, -21, 0.5, 1, -0.5
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
    gal_table = Table(names=('Volume', 'GroupID', 'GalID', 'Rabs', 'Rpetro', 'z'),
                      dtype=('i4', 'i4', 'i4', 'f8', 'f8', 'f8'))
    galid = 0
    ngal_tot = 0
    ngrp_tot = 0
    ngrp_vis = 0
    for isim in range(nsim):
        volume = isim + 1
        ngrp_tot += nhalo
        lgm = util.ran_fun(hmf, hmf_lgmmin, hmf_lgmmax, nhalo)
        zz = util.ran_fun(cosmo.vol_ev, zlimits[0], zlimits[1], nhalo)
        dlgm = hmf.Mstar._val - lgm
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
            if Nfof > 0:
                ngrp_vis += 1
                for i in range(Nfof):
                    gal_table.add_row([volume, groupid, galid, M[vis][i],
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


def clf_sims():
    """Conditional LFs for simulated groups."""
    clf_sim(nmin=0, outfile='clf_sim_nmin0.pkl')
    clf_sim(nmin=1, outfile='clf_sim_nmin1.pkl')
    clf_sim(nmin=5, outfile='clf_sim_nmin5.pkl')
    clf_sim(nmin=1, Vmax='Vmax_raw', outfile='clf_sim_nmin1_vmax.pkl')
    clf_sim(nmin=5, Vmax='Vmax_raw', outfile='clf_sim_nmin5_vmax.pkl')


def clf_sims_10k():
    """Conditional LFs for simulated groups."""
    clf_sim(nmin=0, gal_file='sim_gal_10k.fits', grp_file='sim_group_10k.fits',
            outfile='clf_sim_nmin0_10k.pkl')
    clf_sim(nmin=1, gal_file='sim_gal_10k.fits', grp_file='sim_group_10k.fits',
            outfile='clf_sim_nmin1_10k.pkl')
    clf_sim(nmin=5, gal_file='sim_gal_10k.fits', grp_file='sim_group_10k.fits',
            outfile='clf_sim_nmin5_10k.pkl')
    clf_sim(nmin=1, Vmax='Vmax_raw', gal_file='sim_gal_10k.fits',
            grp_file='sim_group_10k.fits', outfile='clf_sim_nmin1_vmax_10k.pkl')
    clf_sim(nmin=5, Vmax='Vmax_raw', gal_file='sim_gal_10k.fits',
            grp_file='sim_group_10k.fits',outfile='clf_sim_nmin5_vmax_10k.pkl')


def clf_sim_z(nmin=5, z=[0.002, 0.1, 0.2, 0.3], Vmax='Vmax_raw'):
    """Conditional LF for simulated groups in redshift slices."""
    for iz in range(len(z) - 1):
        clf_sim(nmin=nmin, zlimits=(z[iz], z[iz+1]), Vmax=Vmax,
                outfile=f'clf_sim_nmin{nmin}_z{z[iz]}_{z[iz+1]}_vmax.pkl')


def clf_sim(mbins=mbins_def, nmin=5, vol_z=None, zlimits=[0.002, 0.5],
            mlimits=[0, 19.8], Vmax='Vmax_grp', bins=np.linspace(-24, -16, 18),
            colname='r_petro', error='mock', Q=0, P=0, nsim=9,
            gal_file='sim_gal.fits', grp_file='sim_group.fits',
            outfile='clf_sim.pkl', plot_file=None, plot_size=(5, 4)):
    """Conditional LF for simulated groups."""

    samp = gs.GalSample(Q=Q, P=P, mlimits=mlimits, zlimits=zlimits)
    samp.read_gama_groups(gal_file, grp_file, mass_est='sim', nmin=nmin)
#    samp.read_gama_mocks(gal_file)
#    samp.mock_group_props(mass_est='sim', nmin=nmin, grpfile=grp_file)
    if vol_z:
        samp.vol_limit_z(vol_z)
        samp.group_limit(nmin)

    if plot_file and nmin == 1 and zlimits[1] >= 0.5:
        plt.clf()
        plt.scatter(samp.grp['IterCenZ'], samp.grp['log_mass'], s=1,
                    c=samp.grp['Nfof'], vmin=1, vmax=8)  # norm=mpl.colors.LogNorm())
#    poor = grps['Nfof'] < 5
#    rich = grps['Nfof'] >= 5
#    plt.scatter(grps['IterCenZ'][poor], grps['log_mass'][poor], s=2, c='r')
#    plt.scatter(grps['IterCenZ'][rich], grps['log_mass'][rich], s=2, c='b')
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
    dm = samp.t['Rabs'] - samp.abs_mags('r_petro')
#    print('delta M = ', np.mean(dm), ' +- ', np.std(dm))
    V, err = scipy.integrate.quad(samp.cosmo.dV, samp.zlimits[0],
                                  samp.zlimits[1], epsabs=1e-3, epsrel=1e-3)
    V *= samp.area
    print('area, vol =', samp.area, V)
    lf_dict = {}
    nmbin = len(mbins) - 1
    grps = table.unique(samp.t, keys='GroupID')
    lgm = grps['log_mass']
    grps_int = Table.read(grp_file)
    sel = (np.array(grps_int['IterCenZ'] >= samp.zlimits[0]) *
           np.array(grps_int['IterCenZ'] < samp.zlimits[1]))
    grps_int = grps_int[sel]
    lgm_int = grps_int['log_mass']
#    plt.clf()
#    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, num=1)
#    fig.set_size_inches(8, 8)
#    fig.subplots_adjust(left=0.05, bottom=0.05, hspace=0.0, wspace=0.0)
#    fig.text(0.5, 0.0, 'Vmax', ha='center', va='center')
#    fig.text(0.0, 0.5, 'Frequency', ha='center', va='center', rotation='vertical')
    for i in range(nmbin):
        sel_dict = {'log_mass': (mbins[i], mbins[i+1])}
        samp.select(sel_dict)
#        pdb.set_trace()
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
        phi.comp_limits(samp, zlimits[0], zlimits[1])

        # Average input LF over groups in this mass bin
        sel = (mbins[i] <= lgm) * (lgm < mbins[i+1])
        phi.lgm_av = np.mean(lgm[sel])
        phi.lgm_av_gal = np.average(lgm[sel], weights=grps['Nfof'][sel])
        phi.ngroup = ngroup
        phi.vol = V
        meta = samp.meta
        clff = SchecMag()
        phi_av = np.zeros(len(phi.phi))
        ng = 0
        for mass in lgm[sel]:
            dlgm = meta['HMFMSTAR'] - mass
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
        absmag = np.array([samp.t['r_petro'][i].abs for i in range(len(samp.t))])
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

                dlgm = meta['HMFMSTAR'] - mass
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
#        pdb.set_trace()

#        phi.fn_fit(fn=lf.Schechter_mag, Mmin=Mmin_fit, Mmax=Mmax_fit)
        Mkey = 'M{}mock'.format(i)
        lf_dict[Mkey] = phi
        Mkey = 'M{}mockr'.format(i)
        lf_dict[Mkey] = phir
#        ax = axes.flat[i]
#        ax.hist(samp.t[Vmax])
    pickle.dump((samp.meta, lf_dict), open(outfile, 'wb'))
#    plt.show()


def clf_sim_plots(infile='clf_sim_nmin1.pkl',
                  nmin=2, mags=np.linspace(-23, -16, 29), yrange=(1e-3, 1e3)):
    """Plot conditional LF for simulated groups."""

    (meta, lf_dict) = pickle.load(open(infile, 'rb'))
    clff = SchecMag()
    plt.clf()
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, num=1)
    fig.set_size_inches(8, 8)
    fig.subplots_adjust(left=0.05, bottom=0.05, hspace=0.0, wspace=0.0)
    fig.text(0.5, 0.0, 'Mag', ha='center', va='center')
    fig.text(0.0, 0.5, 'phi', ha='center', va='center', rotation='vertical')
    for ibin in range(6):
        phi = lf_dict['M{}mock'.format(ibin)]
#        print(phi.ngal)
        ax = axes.flat[ibin]
        phi.plot(ax=ax, nmin=nmin)
        lgm = phi.lgm_av
        dlgm = meta['HMFMSTAR'] - lgm
        clff.alpha = meta['A0'] + meta['A1']*dlgm
        clff.Mstar = meta['M0'] + meta['M1']*dlgm
        clff.lgps = meta['P0'] + meta['P1']*dlgm
        if 'vmax' in infile:
            clff.lgps = clff.lgps._val + math.log10(phi.ngroup/phi.vol)
        ax.plot(mags, clff(mags))
        if yrange:
            ax.set_ylim(yrange)
        ax.semilogy(basey=10)
        ax.text(0.1, 0.8, '{:5.2f}'.format(lgm), transform=ax.transAxes)
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
                  yticks=(0.6, 1, 2, 3, 4), ylegend=0.69,
                  plot_file='clf_sim.pdf', plot_size=(6, 7.5)):
    """Plot ratio of redcovered conditional LF to simulated input."""

    nfiles = len(infiles)
    meta_list = []
    lf_list = []
    for i in range(nfiles):
        (meta, lf_dict) = pickle.load(open(infiles[i], 'rb'))
        meta_list.append(meta)
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
        ax.plot(lf_lims[:2], (1, 1), 'k:')

        # Simulated LF
#        phi = lf_list[0]['M{}mock'.format(ibin)]
#        lgm = phi.lgm_av
#        meta = meta_list[0]
#        dlgm = meta['HMFMSTAR'] - lgm
#        clff.alpha = meta['A0'] + meta['A1']*dlgm
#        clff.Mstar = meta['M0'] + meta['M1']*dlgm
#        clff.lgps = meta['P0'] + meta['P1']*dlgm
#        if 'vmax' in infiles[0]:
#            clff.lgps = clff.lgps._val + math.log10(phi.ngroup/phi.vol)
#        ax.plot(mags, clff(mags))

        # Recovered LFs normaloised by simulated LF
        for i in range(nfiles):
#            phi = lf_list[i]['M{}mock'.format(ibin)]
            phi = lf_list[i]['M{}mockr'.format(ibin)]
            nmin = meta_list[i]['nmin']
#            phi.plot(ax=ax, nmin=nmin, label=rf'$N_{{\rm fof}} \geq {nmin}$')
            comp = phi.comp
            comp *= (phi.ngal >= npmin)
#            phimod = clff(phi.Mbin[comp])
#            ax.errorbar(phi.Mbin[comp], phi.phi[comp]/phimod,
#                        phi.phi_err[comp]/phimod, # fmt='o',
#                        label=rf'$N_{{\rm fof}} \geq {nmin}$')
#            ax.errorbar(phi.Mbin[comp], phi.phi[comp]/phi.phi_av[comp],
#                        phi.phi_err[comp]/phi.phi_av[comp], # fmt='o',
#                        label=rf'$N_{{\rm fof}} \geq {nmin}$')
            if labels:
                label = labels[i]
            else:
                label = rf'$N_{{\rm fof}} \geq {nmin}$'
            ax.errorbar(phi.Mbin[comp], phi.phi[comp],
                        phi.phi_err[comp], # fmt='o',
                        label=label)
#            pdb.set_trace()
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


def clf_sim_plotr_vmax_10k(
        infiles=('clf_sim_nmin1_vmax_10k.pkl', 'clf_sim_nmin5_vmax_10k.pkl'),
        plot_file='clf_sim_vmax_10k.pdf'):
    clf_sim_plotr(infiles=infiles, plot_file=plot_file)


def clf_sim_plots_z_vmax(
        infiles=('clf_sim_nmin5_z0.002_0.1_vmax.pkl',
                 'clf_sim_nmin5_z0.1_0.2_vmax.pkl',
                 'clf_sim_nmin5_z0.2_0.3_vmax.pkl'),
        labels=(r'$0.0 < z < 0.1$', r'$0.1 < z < 0.2$', r'$0.2 < z < 0.3$'),
        lf_lims=(-15.5, -23.9, 0.2, 2), yticks=(0.3, 0.5, 1, 1.5), 
        plot_file='clf_sim_z_vmax.pdf'):
    clf_sim_plotr(infiles=infiles, labels=labels, lf_lims=lf_lims,
                  yticks=yticks, plot_file=plot_file)
