# Luminosity function estimation and plotting
#
# Revision history
#
# 1.0 24-may-10  Evolving Schechter function, Vmax and SWML fits
# 1.1 29-nov-10  Use polynomial fits to K(z) for each galaxy
# 1.2 01-aug-11  Replace bootstrap with jacknife errors
# 1.3 18-nov-11  Pass selection criteria as a dictionary of keyword:range values
# 1.4 21-feb-12  Allow Saunders et al 1990 LF fit

import glob
import math
import numpy as np
import os
import pickle
from astropy.cosmology import FlatLambdaCDM
import astropy.io.fits as pyfits
#from astLib import astCalc
import pdb
import matplotlib
if not('DISPLAY' in os.environ):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid import AxesGrid
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import scipy.integrate
import scipy.optimize
import time

# Catch invalid values in numpy calls
np.seterr(divide='raise')
np.seterr(invalid='raise')

# Global parameters
par = {}
sel_dict = {'post_class': (0, 4)}
cosmo = None
ln10 = math.log(10)
plot = None

# Solar magnitudes from Blanton et al 2003 for ^{0.1}ugriz bands
Msun_ugriz = [6.80, 5.45, 4.76, 4.58, 4.51]

# Not sure where thse came from
Msun_ugrizYJHK = [6.39, 5.07, 4.62, 4.52, 4.48, 4.06, 3.64, 3.32, 3.28]

# Parameter fits constrained within following limits.
parLimits = {'alpha': (-2.5, 1), 'Mstar': (-22, -17), 'sigma': (0.01, 2.0),
             'beta': (-2.5, 0.5), 'Mt': (-21, -14),
             'Q': (0, 8), 'P': (-10, 5)}

#------------------------------------------------------------------------------
# Driver routines
#------------------------------------------------------------------------------

def lfPCtest(inFile='kcorrz.fits', like=0, jack=0):
    """Test effects of applying post_class cuts on u and r-band LFs"""

    likeFile = None
    for iband in (0,2):
        band = ('u', 'g', 'r', 'i', 'z')[iband]
        appMin = 14
        appMax = (20.3, 19.5, 19.4, 18.7, 18.2)[iband]
        absMin = (-23.0, -24.0, -24.0, -25.0, -25.0)[iband]
        absMinSTY = absMin
        absMax = (-10.0, -10.0, -10.0, -12.0, -12.0)[iband]
        absMaxSTY = absMax
        absStep = 0.25
        zlims = [[0.002, 0.5]]
        zRangeSTY = [0.002, 0.5]
        schec = {'alpha': -1.1, 'Mstar': -18.4, 'Q': 3, 'P': 1}

        colour = 'c'
        outFile = 'lf_pc06_%s%s.dat' % (band, colour)
        lf(inFile, outFile, iband, appMin, appMax,
           absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY, zRangeSTY,
           likeFile, jack=jack, colour=colour, schecInit=schec, pcRange=(0,6))

        outFile = 'lf_pc05_%s%s.dat' % (band, colour)
        lf(inFile, outFile, iband, appMin, appMax,
           absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY, zRangeSTY,
           likeFile, jack=jack, colour=colour, schecInit=schec, pcRange=(0,5))

        outFile = 'lf_pc03_%s%s.dat' % (band, colour)
        lf(inFile, outFile, iband, appMin, appMax,
           absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY, zRangeSTY,
           likeFile, jack=jack, colour=colour, schecInit=schec, pcRange=(0,3))

        outFile = 'lf_pc02_%s%s.dat' % (band, colour)
        lf(inFile, outFile, iband, appMin, appMax,
           absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY, zRangeSTY,
           likeFile, jack=jack, colour=colour, schecInit=schec, pcRange=(0,2))
        
def lfEv(inFile='kcorr.fits', nz=4, like=0, jack=0):
    """LF evolution in redshift slices"""

    likeFile = None
    for iband in range(0,2):
        band = ('u', 'g', 'r', 'i', 'z')[iband]
        appMin = 14
        appMax = (20.3, 19.5, 19.4, 18.7, 18.2)[iband]
        absMin = (-23.0, -24.0, -24.0, -25.0, -25.0)[iband]
        absMinSTY = absMin
        absMax = (-15.0, -16.0, -16.0, -17.0, -17.0)[iband]
        absMaxSTY = absMax
        absStep = 0.25
        if nz == 3:
            zlims = [[0.002, 0.2], [0.2, 0.4], [0.4, 0.6]]
        if nz == 4:
            zlims = [[0.002, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.5]]
        if nz == 8:
            zlims = [[0.002, 0.05], [0.05, 0.1], [0.1, 0.15], [0.15, 0.2],
                     [0.2, 0.25], [0.25, 0.3], [0.3, 0.4], [0.4, 0.5]]
        zRangeSTY = [0.002, 0.5]
        schec = {'alpha': -1.1, 'Mstar': -18.4, 'sigma': 0.2, 'Q': 3, 'P': 1}

        for colour in 'cbr':
             outFile = 'lf_ev%d_%s%s.dat' % (nz, band, colour)
             if like:
                 likeFile = 'like_ev%d_%s%s.dat' % (nz, band, colour)
             lf(inFile, outFile, iband, appMin, appMax,
                absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY, zRangeSTY,
                likeFile, jack=jack, colour=colour, schecInit=schec)
        
def lfEvMorph(inFile='kcorrz.fits', nz=4, like=0, jack=0):
    """LF evolution in redshift slices by morphology (Sersic index)"""

    likeFile = None
    for iband in range(5):
        band = ('u', 'g', 'r', 'i', 'z')[iband]
        appMin = 14
        appMax = (20.3, 19.5, 19.4, 18.7, 18.2)[iband]
        absMin = (-23.0, -24.0, -24.0, -25.0, -25.0)[iband]
        absMinSTY = absMin
        absMax = (-15.0, -16.0, -16.0, -17.0, -17.0)[iband]
        absMaxSTY = absMax
        absStep = 0.25
        if nz == 3:
            zlims = [[0.002, 0.2], [0.2, 0.4], [0.4, 0.6]]
        if nz == 4:
            zlims = [[0.002, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.5]]
        if nz == 8:
            zlims = [[0.002, 0.05], [0.05, 0.1], [0.1, 0.15], [0.15, 0.2],
                     [0.2, 0.25], [0.25, 0.3], [0.3, 0.4], [0.4, 0.5]]
        zRangeSTY = [0.002, 0.5]
        schec = {'alpha': -1.1, 'Mstar': -18.4, 'Q': 3, 'P': 1}

        colour = 'c'
        for morph in 'es':
##        for mass_limits in ((0, 1e13), (1e13, 1e17)):
            outFile = 'lf_ev%d_%s%s.dat' % (nz, band, morph)
            if like:
                likeFile = 'like_ev%d_%s%s.dat' % (nz, band, morph)
            if (morph == 'e'): sersic_limits = (1.9, 15)
            if (morph == 's'): sersic_limits = (0.08, 1.9)
            sel_dict['gal_index_r'] = sersic_limits
##             sel_dict['group_mass'] = (0, 1e13)
            lf(inFile, outFile, iband, appMin, appMax,
               absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY, zRangeSTY,
               likeFile, jack=jack, colour=colour, schecInit=schec)
        
def lfEvBPT(inFile='../kcorrz.fits', nz=4, like=0, jack=0):
    """LF evolution in redshift slices by BPT class"""

    likeFile = None
    for iband in range(5):
        band = ('u', 'g', 'r', 'i', 'z')[iband]
        appMin = 14
        appMax = (20.3, 19.5, 19.4, 18.7, 18.2)[iband]
        absMin = (-23.0, -24.0, -24.0, -25.0, -25.0)[iband]
        absMinSTY = absMin
        absMax = (-15.0, -16.0, -16.0, -17.0, -17.0)[iband]
        absMaxSTY = absMax
        absStep = 0.25
        if nz == 3:
            zlims = [[0.002, 0.2], [0.2, 0.4], [0.4, 0.6]]
        if nz == 4:
            zlims = [[0.002, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.5]]
        if nz == 8:
            zlims = [[0.002, 0.05], [0.05, 0.1], [0.1, 0.15], [0.15, 0.2],
                     [0.2, 0.25], [0.25, 0.3], [0.3, 0.4], [0.4, 0.5]]
        zRangeSTY = [0.002, 0.5]
        schec = {'alpha': -1.1, 'Mstar': -18.4, 'Q': 3, 'P': 1}

        colour = 'c'
        for bpt_code in 'uqsca':
            outFile = 'lf_ev%d_%s%s.dat' % (nz, band, bpt_code)
            if like:
                likeFile = 'like_ev%d_%s%s.dat' % (nz, band, bpt_code)
            if (bpt_code == 'u'): bpt = (' ', ' _')
            if (bpt_code == 'q'): bpt = ('BLANK','BLANK_')
            if (bpt_code == 's'): bpt = ('Star Forming', 'Star Forming_')
            if (bpt_code == 'c'): bpt = ('Composite', 'Composite_')
            if (bpt_code == 'a'): bpt = ('LINER', 'Seyfert_')
            sel_dict['BPT'] = bpt
            lf(inFile, outFile, iband, appMin, appMax,
               absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY, zRangeSTY,
               likeFile, jack=jack, colour=colour, schecInit=schec)
        
def lfEvWHAN(inFile='../kcorrz.fits', nz=4, like=0, jack=0):
    """LF evolution in redshift slices by WHAN class"""

    likeFile = None
    for iband in range(5):
        band = ('u', 'g', 'r', 'i', 'z')[iband]
        appMin = 14
        appMax = (20.3, 19.5, 19.4, 18.7, 18.2)[iband]
        absMin = (-23.0, -24.0, -24.0, -25.0, -25.0)[iband]
        absMinSTY = absMin
        absMax = (-15.0, -16.0, -16.0, -17.0, -17.0)[iband]
        absMaxSTY = absMax
        absStep = 0.25
        if nz == 3:
            zlims = [[0.002, 0.2], [0.2, 0.4], [0.4, 0.6]]
        if nz == 4:
            zlims = [[0.002, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.5]]
        if nz == 8:
            zlims = [[0.002, 0.05], [0.05, 0.1], [0.1, 0.15], [0.15, 0.2],
                     [0.2, 0.25], [0.25, 0.3], [0.3, 0.4], [0.4, 0.5]]
        zRangeSTY = [0.002, 0.5]
        schec = {'alpha': -1.1, 'Mstar': -18.4, 'Q': 3, 'P': 1}

        colour = 'c'
##        for whan_code in ('sf', 'sa', 'wa', 'r', 'p'):
        for whan_code in ('a',):
            outFile = 'lf_ev%d_%s%s.dat' % (nz, band, whan_code)
            if like:
                likeFile = 'like_ev%d_%s%s.dat' % (nz, band, whan_code)
            if (whan_code == 'u'): whan = (' ', ' _')
            if (whan_code == 'sf'): whan = ('Star Forming', 'Star Forming_')
            if (whan_code == 'sa'): whan = ('Strong AGN', 'Strong AGN_')
            if (whan_code == 'wa'): whan = ('Weak AGN', 'Weak AGN_')
            if (whan_code == 'a'): whan = ('Strong AGN', 'Weak AGN_')
            if (whan_code == 'r'): whan = ('Retired', 'Retired_')
            if (whan_code == 'p'): whan = ('Passive', 'Passive_')
            sel_dict['WHAN'] = whan
            lf(inFile, outFile, iband, appMin, appMax,
               absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY, zRangeSTY,
               likeFile, jack=jack, colour=colour, schecInit=schec)
        
def lfSDSSEv(inFile='kcorrz.fits', nz=4, like=0, jack=0):
    """SDSS LF evolution in ugriz"""
    likeFile = None
    for iband in range(5):
        band = ('u', 'g', 'r', 'i', 'z')[iband]
        appMin = 14
        appMax = (19.0, 18.0, 17.7, 17.2, 16.8)[iband]
        absMin = (-23.0, -24.0, -24.0, -25.0, -25.0)[iband]
        absMinSTY = absMin
        absMax = (-15.0, -16.0, -16.0, -17.0, -17.0)[iband]
        absMaxSTY = absMax
        absStep = 0.25
        if nz == 4:
            zlims = [[0.002, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
        if nz == 8:
            zlims = [[0.002, 0.05], [0.05, 0.1], [0.1, 0.15], [0.15, 0.2],
                     [0.2, 0.25], [0.25, 0.3], [0.3, 0.35], [0.35, 0.4]]
        zRangeSTY = [0.002, 0.4]
        schec = {'alpha': -1.1, 'Mstar': -18.4, 'Q': 3, 'P': 1}

        for colour in 'cbr':
            outFile = 'lf_ev%d_%s%s.dat' % (nz, band, colour)
            if like:
                likeFile = 'like_ev%d_%s%s.dat' % (nz, band, colour)
            lf(inFile, outFile, iband, appMin, appMax,
               absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY,
               zRangeSTY, likeFile, jack=jack, colour=colour,
               schecInit=schec, pcRange=None, bpt_type=None)
##             for bpt_type in 'uqsca':
##                 outFile = 'lf_ev%d_%s%s%s.dat' % (nz, band, colour, bpt_type)
##                 if like:
##                     likeFile = 'like_ev%d_%s%s%s.dat' % (nz, band, colour,
##                                                          bpt_type)
##                 lf(inFile, outFile, iband, appMin, appMax,
##                    absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY,
##                    zRangeSTY, likeFile, jack=jack, colour=colour,
##                    schecInit=schec, pcRange=None, bpt_type=bpt_type)

def lfEv2(jack=0):
    """LF evolution in ugriz with quadratic density evolution fitting."""
    inFile = 'kcorrz.fits'
    colour = 'c'
    for iband in range(1):
        band = ('u', 'g', 'r', 'i', 'z')[iband]
        appMin = 14
        appMax = (20.3, 19.5, 19.4, 18.7, 18.2)[iband]
        absMin = (-23.0, -24.0, -24.0, -25.0, -25.0)[iband]
        absMinSTY = absMin
        absMax = (-15.0, -16.0, -16.0, -17.0, -17.0)[iband]
        absMaxSTY = absMax
        absStep = 0.25
        zlims = [[0.002, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.5]]
##         zlims = [[0.002, 0.05], [0.05, 0.1], [0.1, 0.15], [0.15, 0.2],
##                  [0.2, 0.25], [0.25, 0.3], [0.3, 0.4], [0.4, 0.5]]
        zRangeSTY = [0.002, 0.5]
        schec = {'alpha': -1.1, 'Mstar': -18.4, 'Q': (3, 1), 'P': (1, 1)}

        outFile = 'lf_evquad_%s.dat' % band
##         likeFile = 'like_ev2_%s.dat' % band
        likeFile = None
        lf(inFile, outFile, iband, appMin, appMax, absMin, absMax, absStep,
           zlims, absMinSTY, absMaxSTY, zRangeSTY, likeFile, jack=jack,
           colour=colour, schecInit=schec)
        
def lfFaint(inFile='kcorrz.fits', absStep=0.25, bands=range(5), colours='cbr',
            zmax=0.1, like=0, jack=0):
    """Standard Schechter fits at low redshifts"""
    likeFile = None
    for iband in bands:
        band = ('u', 'g', 'r', 'i', 'z')[iband]
        appMin = 14
        appMax = (20.3, 19.5, 19.4, 18.7, 18.2)[iband]
        absMin = (-21.0, -22.0, -23.0, -23.0, -24.0)[iband]
        absMax = (-10.0, -10.0, -10.0, -11.0, -12.0)[iband]
        absMinSTY = absMin
        absMaxSTY = absMax
        zlims = [[0.002, zmax]]
        zRangeSTY = [0.002, zmax]
        schec = {'alpha': -1, 'Mstar': -21, 'sigma': 0.2}

        for colour in colours:
            outFile = 'lf_faint_%s%s.dat' % (band, colour)
            if like:
                likeFile = 'like_faint_%s%s.dat' % (band, colour)
            lf(inFile, outFile, iband, appMin, appMax,
               absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY, zRangeSTY,
               likeFile, jack=jack, colour=colour, schecInit=schec)
        
def lfSDSSFaint(inFile='kcorrz.fits', absStep=0.25, bands=range(5),
                colours='cbr', like=0, jack=0):
    """Standard Schechter fits at low redshifts"""
    likeFile = None
    for iband in bands:
        band = ('u', 'g', 'r', 'i', 'z')[iband]
        appMin = 14
        appMax = (19.0, 18.0, 17.7, 17.2, 16.8)[iband]
        absMin = (-21.0, -22.0, -23.0, -23.0, -24.0)[iband]
        absMax = (-10.0, -10.0, -10.0, -11.0, -12.0)[iband]
        absMinSTY = absMin
        absMaxSTY = absMax
        zlims = [[0.002, 0.1]]
        zRangeSTY = [0.002, 0.1]
        schec = {'alpha': -1, 'Mstar': -21}

        for colour in colours:
            outFile = 'lf_faint_%s%s.dat' % (band, colour)
            if like:
                likeFile = 'like_faint_%s%s.dat' % (band, colour)
            lf(inFile, outFile, iband, appMin, appMax,
               absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY, zRangeSTY,
               likeFile, jack=jack, colour=colour, schecInit=schec, pcRange=None)
        
def lfFaintGroup(inFile='kcorrz.fits', absStep=0.5, bands=5, colours='cbr',
            like=0, jack=0):
    """Standard Schechter fits at low redshifts in groups"""
    likeFile = None
##     for iband in range(5):
    for iband in range(2,3):
        band = ('u', 'g', 'r', 'i', 'z')[iband]
        appMin = 14
        appMax = (20.3, 19.5, 19.4, 18.7, 18.2)[iband]
        absMin = (-21.0, -22.0, -23.0, -23.0, -24.0)[iband]
        absMax = (-14.0, -14.0, -14.0, -15.0, -16.0)[iband]
        absMinSTY = absMin
        absMaxSTY = absMax
        zlims = [[0.01255, 0.1]]
        zRangeSTY = [0.01255, 0.1]
        schec = {'alpha': -1, 'Mstar': -21}

        for colour in colours:
            outFile = 'lf_faint_%s%s.dat' % (band, colour)
            if like:
                likeFile = 'like_faint_%s%s.dat' % (band, colour)
            lf(inFile, outFile, iband, appMin, appMax,
               absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY, zRangeSTY,
               likeFile, jack=jack, colour=colour, schecInit=schec)

def doGroups():
    gama_data = os.environ['GAMA_DATA']
    dirs = (gama_data + '/v16/dr6/tonryz01_noimcorr/nogroup',
            gama_data + '/v16/dr6/tonryz01_noimcorr/ingroup',
            gama_data + '/v16/dr6/tonryz01_noimcorr/ingroup/lomass',
            ## gama_data + '/v16/dr6/tonryz01_noimcorr/ingroup/mimass',
            gama_data + '/v16/dr6/tonryz01_noimcorr/ingroup/himass',
            gama_data + '/v16/dr6/tonryz01_noimcorr/ingroup/lodisp',
            ## gama_data + '/v16/dr6/tonryz01_noimcorr/ingroup/midisp',
            gama_data + '/v16/dr6/tonryz01_noimcorr/ingroup/hidisp',
            gama_data + '/v16/dr6/tonryz01_noimcorr/ingroup/lorich',
            ## gama_data + '/v16/dr6/tonryz01_noimcorr/ingroup/mirich',
            gama_data + '/v16/dr6/tonryz01_noimcorr/ingroup/hirich',
            )
    for dir in dirs:
        os.chdir(dir)
        lfFaintGroup(like=1)
        
def lfDP(inFile='kcorrz.fits', like=0, jack=0):
    """Double power-law Schechter fit to LF faint-end LF at low redshifts"""

    # Initial guess at dp fit parameters [alpha, M*, beta, Mt]
    dpPar = {'c': [[-0.1, -18.1, -1.4, -17.7],
                   [0.1, -19.4, -1.4, -19.3],
                   [0.1, -20.1, -1.4, -20.0],
                   [0.1, -20.5, -1.4, -20.2],
                   [0.1, -20.7, -1.5, -20.3]],
             'b': [[-1.4, -18.5, -0.1, -18],
                   [-1.5, -19.7, -0.1, -15],
                   [-1.5, -20.4, -0.1, -19.5],
                   [-1.0, -20.5, -0.1, -19.9],
                   [-1.3, -21.0, -0.1, -20.5]],
             'r': [[0.1, -17.6, -1.2, -15.7],
                   [0.1, -19.5, -1.2, -17.7],
                   [0.1, -20.1, -1.3, -18.0],
                   [-0.1, -20.6, -1.3, -17.4],
                   [-0.1, -20.9, -1.4, -17.7]]}
             

    likeFile = None
    for iband in range(5):
        band = ('u', 'g', 'r', 'i', 'z')[iband]
        appMin = 14
        appMax = (20.3, 19.5, 19.4, 18.7, 18.2)[iband]
        absMin = (-21.0, -22.0, -23.0, -23.0, -24.0)[iband]
        absMax = (-10.0, -10.0, -10.0, -12.0, -12.0)[iband]
        absMinSTY = absMin
        absMaxSTY = absMax
        absStep = 0.25
        zlims = [[0.002, 0.1]]
        zRangeSTY = [0.002, 0.1]

        for colour in 'cbr':
##         for colour in 'r':
            schec = {}
            schec['alpha'], schec['Mstar'], schec['beta'], schec['Mt'] = dpPar[colour][iband]
            outFile = 'lf_dp_%s%s.dat' % (band, colour)
            if like and colour != 'b':
                likeFile = 'like_dp_%s%s.dat' % (band, colour)
            else:
                likeFile = None
            lf(inFile, outFile, iband, appMin, appMax,
               absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY, zRangeSTY,
               likeFile, jack=jack, colour=colour, schecInit=schec)
                
def lfSDSSDP(inFile='kcorrz.fits', like=0, jack=0):
    """Double power-law Schechter fit to LF faint-end LF at low redshifts"""

    # Initial guess at dp fit parameters [alpha, M*, beta, Mt]
    dpPar = {'c': [[-0.1, -18.1, -1.4, -17.7],
                   [0.1, -19.4, -1.4, -19.3],
                   [0.1, -20.1, -1.4, -20.0],
                   [0.1, -20.5, -1.4, -20.2],
                   [0.1, -20.7, -1.5, -20.3]],
             'b': [[-1.4, -18.5, -0.1, -18],
                   [-1.5, -19.7, -0.1, -15],
                   [-1.5, -20.4, -0.1, -19.5],
                   [-1.0, -20.5, -0.1, -19.9],
                   [-1.3, -21.0, -0.1, -20.5]],
             'r': [[0.1, -17.6, -1.2, -15.7],
                   [0.1, -19.5, -1.2, -17.7],
                   [0.1, -20.1, -1.3, -18.0],
                   [-0.1, -20.6, -1.3, -17.4],
                   [-0.1, -20.9, -1.4, -17.7]]}
             

    likeFile = None
    for iband in range(5):
        band = ('u', 'g', 'r', 'i', 'z')[iband]
        appMin = 14
        appMax = (19.0, 18.0, 17.7, 17.2, 16.8)[iband]
        absMin = (-21.0, -22.0, -23.0, -23.0, -24.0)[iband]
        absMax = (-10.0, -10.0, -10.0, -12.0, -12.0)[iband]
        absMinSTY = absMin
        absMaxSTY = absMax
        absStep = 0.25
        zlims = [[0.002, 0.1]]
        zRangeSTY = [0.002, 0.1]

        for colour in 'cbr':
##         for colour in 'r':
            schec = {}
            schec['alpha'], schec['Mstar'], schec['beta'], schec['Mt'] = dpPar[colour][iband]
            outFile = 'lf_dp_%s%s.dat' % (band, colour)
            if like and colour != 'b':
                likeFile = 'like_dp_%s%s.dat' % (band, colour)
            else:
                likeFile = None
            lf(inFile, outFile, iband, appMin, appMax,
               absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY, zRangeSTY,
               likeFile, jack=jack, colour=colour, schecInit=schec, pcRange=None)
                
def lfSim(nz=4, like=0):
    """LF for sims"""
    iband = 2
    band = ('u', 'g', 'r', 'i', 'z')[iband]
    appMin = 14
    appMax = (20.5, 19.5, 19.4, 18.5, 18.0)[iband]
    absMin = (-22.0, -23.0, -24.0, -25.0, -25.0)[iband]
    absMax = (-10.0, -10.0, -10.0, -12.0, -12.0)[iband]
    absMinSTY = absMin
    absMaxSTY = absMax
    absStep = 0.25
    if nz == 4:
        zlims = [[0.002, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
    if nz == 8:
        zlims = [[0.002, 0.05], [0.05, 0.1], [0.1, 0.15], [0.15, 0.2],
                 [0.2, 0.25], [0.25, 0.3], [0.3, 0.35], [0.35, 0.4]]
    schec = {'alpha': -1.3, 'Mstar': -20.8, 'Q': 0.1, 'P': 0.1}
    for i in range(1, 9):
        inFile = 'spSim%d.fits' % i
        outFile = 'lf_ev{}_{}.dat'.format(nz, i)
        if like:
            likeFile = 'like_spSim%d.dat' % i
        else:
            likeFile = None
        lf(inFile, outFile, iband, appMin, appMax,
           absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY,
           likeFile=likeFile, colour='c', schecInit=schec, pcRange=None)
        
def lfSimFaint():
    """Faint-end LF for non-evolving r-band sims"""
    iband = 2
    band = ('u', 'g', 'r', 'i', 'z')[iband]
    appMin = 14
    appMax = (20.5, 19.5, 19.4, 18.5, 18.0)[iband]
    absMin = (-22.0, -23.0, -24.0, -25.0, -25.0)[iband]
    absMax = (-10.0, -10.0, -10.0, -12.0, -12.0)[iband]
    absMinSTY = (-20.0, -22.0, -25.0, -25.0, -25.0)[iband]
    absMaxSTY = absMax
    absStep = 0.25
    zlims = [[0.002, 0.1]]
    schec = {'alpha': -1.3, 'Mstar': -20.8}
    for i in range(1, 9):
        inFile = 'spSim%d.fits' % i
        outFile = 'lf_faint_spSim%d.dat' % i
##         likeFile = 'like_faint_spSim%d.dat' % i
        likeFile = None
        lf(inFile, outFile, iband, appMin, appMax,
           absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY,
           likeFile, colour='c', schecInit=schec, pcRange=None)
                
def lfSimDP(like=0):
    """Double power-law Schechter fits to non-evolving sims"""
    iband = 2
    band = ('u', 'g', 'r', 'i', 'z')[iband]
    appMin = 14
    appMax = (20.5, 19.5, 19.4, 18.5, 18.0)[iband]
    absMin = (-22.0, -23.0, -24.0, -25.0, -25.0)[iband]
    absMax = (-10.0, -10.0, -10.0, -12.0, -12.0)[iband]
    absMinSTY = (-20.0, -22.0, -25.0, -25.0, -25.0)[iband]
    absMaxSTY = absMax
    absStep = 0.25
    zlims = [[0.002, 0.1]]
    schec = {'alpha': -1.2, 'Mstar': -20.8, 'beta': 0.1, 'Mt': -20.0}
    for i in range(1, 9):
        inFile = 'spSim%d.fits' % i
        outFile = 'lf_dp_spsim%d.dat' % i
        if like:
            likeFile = 'like_dp_spSim%d.dat' % i
        else:
            likeFile = None
        lf(inFile, outFile, iband, appMin, appMax, 
           absMin, absMax, absStep, zlims, absMinSTY, absMaxSTY,
           likeFile, colour='c', schecInit=schec, pcRange=None)
                
def sim_lf(infile):
    "Calculate LF for mlum.simcat simulation."
    schec = {'alpha': -1.3, 'Mstar': -20.8}
    lf(infile, 'sim_lf.dat', 2, 14, 19.8, -23, -12, 22,
       [[0.002, 0.5]], -23, -12, schecInit=schec)
        
def Nltz(inFile, zvals, Mmin, Mmax, colour='c'):
    """Return cumulative N(<z) distribution at given redshifts for generating
    random distributions for clustering analyses."""

    global par, plot
    par = {'idebug': 0, 'iband': 2, 'band': 'r', 'colour': colour, 
           'appMin': 14.0, 'appMax': 19.4,
           'absMin': Mmin, 'absMax': Mmax, 'absStep': 0.5, 
           'absMinSTY': Mmin, 'absMaxSTY': Mmax}
    schec = {'alpha': (-1.23, 0.01), 'Mstar': (-20.70, 0.04), 'Q': 0.7, 'P': 1.8}
    ev = Evol(schec['Q'], schec['P'])
    zRange = (min(zvals), max(zvals))
    plot = plotWindow(2,2)

    samp = readGama(inFile, zRange, colour)
    gamTab = gammaLookup()
    sel = SelFn(schec, ev, gamTab, samp)
    cum = np.zeros(len(zvals))
    ndif = cosmo.dV(zvals)*sel.sel(zvals)
    ncum = np.cumsum(ndif)
    return ncum
        
#------------------------------------------------------------------------------
# Main procedure
#------------------------------------------------------------------------------

def lf(inFile, outFile, iband, appMin, appMax, absMin, absMax, absStep,
       zlims, absMinSTY=None, absMaxSTY=None, zRangeSTY=None, likeFile=None,
       jack=0, colour='c', schecInit=None):
    """Top-level function for LF estimation."""

    global par, plot
    par = {'progName': 'lum.py', 'version': '1.4', 'inFile': inFile,
           'idebug': 1, 'iband': iband,
           'band': ('u', 'g', 'r', 'i', 'z')[iband],
           'colour': colour, 
           'nz': len(zlims), 'appMin': appMin, 'appMax': appMax,
           'nbin': int(math.ceil((absMax - absMin)/absStep)),
           'absMin': absMin, 'absMax': absMax, 'absStep': absStep,
           'absMinSTY': absMinSTY, 'absMaxSTY': absMaxSTY,
           'nNormMin': 10, 'J3fac': 30000.0, 'normType': 'vmax'}
    zRange = [zlims[0][0], zlims[-1][1]]
    if zRangeSTY is None:
        zRangeSTY = zRange
    
    plot = plotWindow(2,2)

    print '\n************************\nlum.py version ', par['version']
    print 'Band ', par['band'], 'colour ', par['colour'], sel_dict
        
    schec = schecInit.copy()
    fullSamp = readGama(inFile, zRange, colour)
    galSamp = fullSamp.sub(zRangeSTY, par['absMinSTY'], par['absMaxSTY'])
    ev, lumdens = parFit(galSamp, zlims, schec, likeFile, errorEst=1)

    phiList = []
    for i in range(len(zlims)):
        phi = BinnedPhi(zlims[i], fullSamp.kmean)
        galSamp = fullSamp.sub(zlims[i], phi.absMin, phi.absMax)
        if galSamp.ngal > 0:
            phi.zmean = np.mean(galSamp.z)
        else:
            phi.zmean = 0.0
        phi.ntot = galSamp.ngal
        phi.kmean = galSamp.kmean.tolist()
        
        if galSamp.ngal > 0:
            vmax(galSamp, phi)
            phi.sty = schecBin(schec, ev, phi)
            swml(galSamp, phi, schec)
        phiList.append(phi)
        
    f = open(outFile, 'w')
    print >> f, par
    print >> f, schec
    print >> f, lumdens
    for phi in phiList:
        phi.save(f)
    f.close()
    plot.save()

    # Jacknife resampling
    if jack:
        for ijack in range(9):
            print 'Jacknife ', ijack
            jackFile = outFile.split('.')[0] + '_j%d' % ijack + '.dat'
            schec = schecInit.copy()
            jackSamp = fullSamp.jacknife(ijack)
            galSamp = jackSamp.sub(zRangeSTY, par['absMinSTY'], par['absMaxSTY'])
            ev, lumdens = parFit(galSamp, zlims, schec, None, errorEst=0)
            
            phiList = []
            for i in range(len(zlims)):
                phi = BinnedPhi(zlims[i], fullSamp.kmean)
                galSamp = jackSamp.sub(zlims[i], phi.absMin, phi.absMax)
                if galSamp.ngal > 0:
                    phi.zmean = np.mean(galSamp.z)
                else:
                    phi.zmean = 0.0
                phi.ntot = galSamp.ngal
                phi.kmean = galSamp.kmean.tolist()
        
                if galSamp.ngal > 0:
                    vmax(galSamp, phi)
                    phi.sty = schecBin(schec, ev, phi)
                    swml(galSamp, phi, schec)
                phiList.append(phi)

            f = open(jackFile, 'w')
            print >> f, par
            print >> f, schec
            print >> f, lumdens
            for phi in phiList:
                phi.save(f)
            f.close()
            plot.save()

        
def readGama(file, zRange, colour):
    """Read GAMA data from fits file."""

    global par, cosmo
    iband = par['iband']
    appMin = par['appMin']
    appMax = par['appMax']
    absMin = par['absMin']
    absMax = par['absMax']
    absStep = par['absStep']
    
##     fname = '/Users/loveday/Data/gama/' + file
    hdulist = pyfits.open(file)
    header = hdulist[1].header
    try:
        par['H0'] = header['H0']
    except:
        par['H0'] = 100.0
    par['omega_l'] = header['OMEGA_L']
    
    par['z0'] = header['Z0']
    par['area'] = header['AREA']
    par['area'] *= (math.pi/180.0)*(math.pi/180.0)

    print 'H0, omega_l, z0, area/Sr = ', par['H0'], par['omega_l'], par['z0'], par['area']

    # Look for any simulation parameters in header
    try:
        par['sim_alpha'] = header['ALPHA']
        par['sim_Mstar'] = header['MSTAR']
        par['sim_phistar'] = header['PHISTAR']
        par['sim_Q'] = header['Q']
        is_sim = 1
    except:
        is_sim = 0
        
    # Look for any group selection parameters in header
    try:
        par['grpsel'] = header['GRPSEL']
    except:
        pass
        
    cosmo = CosmoLookup(par['H0'], par['omega_l'], zRange, is_sim=is_sim)

    tbdata = hdulist[1].data
    ra = tbdata.field('ra')
    z = tbdata.field('z')
    zConf = tbdata.field('zConf')
    appMag = tbdata.field('appMag')[:,iband]
    absMag = tbdata.field('absMag')[:,iband]
    kc = tbdata.field('kcorr')[:,iband]
    weight = tbdata.field('weight')[:,iband]

    idx = (zRange[0] < z) * (z < zRange[1]) * (zConf > 0.8) * \
          (appMin < appMag) * (appMag < appMax) * \
          (absMin < absMag) * (absMag < absMax) * \
          (np.isfinite(weight))

    if colour != 'c':
        grcut = colourCut(tbdata.field('absMag')[:,2])
        gr = (tbdata.field('modelMagCor_g') - tbdata.field('kcorr')[:,1]) - \
             (tbdata.field('modelMagCor_r') - tbdata.field('kcorr')[:,2])
        if colour == 'b':
            idx *= (gr < grcut)
        else:
            idx *= (gr >= grcut)

    for key, limits in sel_dict.iteritems():
        print key, limits
        idx *= (tbdata.field(key) >= limits[0])*(tbdata.field(key) < limits[1])
        par[key] = limits
            
##     # If post_class file specified, select objects within pcRange
##     if par['pcRange']:
##         fin = open(pcFile, 'r')
##         first = 1
##         pcd = {}
##         for line in fin:
##             if first:
##                 first = 0
##             else:
##                 data = line.split()
##                 id = int(data[0])
##                 pcd[id] = int(data[3])
##         fin.close()
##         cataid = tbdata.field('cataid')
##         pca = np.array([pcd[id] for id in cataid], dtype=int)
##         idx *= (pca >= par['pcRange'][0])*(pca <= par['pcRange'][1])

            
##     if bpt_type:
##         bpt = tbdata.field('bpt_type')
##         idx *= (bpt == bpt_type)
    
    ra = ra[idx]
    z = z[idx]
    absMag = absMag[idx]
    kc = kc[idx]
    weight = weight[idx]
    if iband == 0: kcoeff = tbdata.field('kcoeffu')[idx,:]
    if iband == 1: kcoeff = tbdata.field('kcoeffg')[idx,:]
    if iband == 2: kcoeff = tbdata.field('kcoeffr')[idx,:]
    if iband == 3: kcoeff = tbdata.field('kcoeffi')[idx,:]
    if iband == 4: kcoeff = tbdata.field('kcoeffz')[idx,:]
    hdulist.close()
    ngal = z.size
    print ngal, ' galaxies selected'

    # Check K(z) reconstruction
    zbin = np.linspace(zRange[0], zRange[1], 50)
    for igal in range(4):
        print kcoeff[igal,:]
        kcz = np.polyval(kcoeff[igal], zbin-par['z0'])
        plot.window()
        plt.plot(zbin, kcz)
        plt.xlabel('z')
        plt.ylabel('K(z)')
    plt.draw()
    
    nv = 20
    Vhist = np.zeros(nv)
    phiBin = np.floor((absMag - absMin)/absStep).astype(np.int32)
    dm =  25 + 5*np.log10(cosmo.dl(z)) + kc
    Mmin = np.clip(appMin - dm, absMin, absMax)
    Mmax = np.clip(appMax - dm, absMin, absMax)
    
    r = cosmo.dm(z)
    z1 = map(lambda i: zdm(appMin - absMag[i], kcoeff[i], zRange), xrange(ngal))
    r1 = cosmo.dm(z1)
    z2 = map(lambda i: zdm(appMax - absMag[i], kcoeff[i], zRange), xrange(ngal))
    r2 = cosmo.dm(z2)
    vol = r*r*r - r1*r1*r1
    Vm = r2*r2*r2 - r1*r1*r1
        
    VVm = vol/Vm
    plot.window()
    plt.hist(VVm, bins=20, range=(0,1), weights=weight)
    plt.xlabel('V/Vmax')
    plt.ylabel('Frequency')
    plt.draw()

    idx = VVm < 0
    print len(VVm[idx]), ' galaxies with V/Vmax < 0'
    idx = VVm > 1.01
    print len(VVm[idx]), ' galaxies with V/Vmax > 1.01'
    
    fullSamp = Sample(zRange, par['area'], z, weight, absMag,
                      Mmin, Mmax, r1, r2, phiBin, kcoeff, ra)
    return fullSamp


#------------------------------------------------------------------------------
# Parametric fitting  procedures
#------------------------------------------------------------------------------

def parFit(samp, zlims, schec, likeFile, errorEst=0):
    """Parametric fit to LF.  Type of fit (standard Schechter, DP schechter
    or evolving is determined from the keys in the schec dictionary."""
    

    print "Parametric fit..."
    schec['zRangeSty'] = samp.zRange
    schec['ngal'] = samp.ngal

##     if not(schec.has_key('sigma')):
    print "Loading Gamma fn lookup table..."
    gamTab = gammaLookup()
    
    # First we fit for shape and any luminosity evolution parameters
    x0 = [schec['alpha'], schec['Mstar']]
    parSet = ['alpha', 'Mstar']
    lower = [parLimits['alpha'][0], parLimits['Mstar'][0]]
    upper = [parLimits['alpha'][1], parLimits['Mstar'][1]]
    cons = [con_alo, con_ahi, con_mslo, con_mshi]
    if schec.has_key('sigma'):
        x0 += [schec['sigma']]
        parSet += ['sigma']
        lower += [parLimits['sigma'][0]]
        upper += [parLimits['sigma'][1]]
    if schec.has_key('Q'):
        x0 += [schec['Q']]
        parSet += ['Q']
        lower += [parLimits['Q'][0]]
        upper += [parLimits['Q'][1]]
    if schec.has_key('beta'):
        x0 += [schec['beta'], schec['Mt']]
        parSet += ['beta', 'Mt']
        lower += [parLimits['beta'][0], parLimits['Mt'][0]]
        upper += [parLimits['beta'][1], parLimits['Mt'][1]]
        cons += [con_blo, con_bhi, con_mtlo, con_mthi]

    schec['nShapePar'] = len(parSet)

    print "Fitting ", parSet, x0

    # Call fmin repeatedly until value of minimum changes by less than 1e-6
    ftol = 1e-6
    fold = 1
    df = 1
    while df > ftol:
        result = scipy.optimize.fmin(
            parLike, x0, args=(parSet, gamTab, samp), ftol=ftol,
            full_output=1)
        xopt = result[0]
        fopt = result[1]
        print xopt, fopt
        df = abs(fopt - fold)/fopt
        fold = fopt
        x0 = xopt

    
##     xopt = scipy.optimize.fmin_cobyla(parLike, x0, cons,
##                                       args=(parSet, gamTab, samp),
##                                       rhoend=0.00001)
##     print 'fmin_cobyla:', xopt

##     result = scipy.optimize.anneal(parLike, x0, 
##                                          args=(parSet, gamTab, samp),
##                                          full_output=1,
##                                          lower=lower, upper=upper)
##     print 'anneal:', result

    # Errors on each parameter, marginalising over other parameters
    npar = len(parSet)
    if errorEst:
        for i in range(npar):
            imarg = filter(lambda j: j != i, range(npar))
            parOrd = [parSet[j] for j in imarg] + [parSet[i]]
            schec[parSet[i]] = (xopt[i], likeErr(
                lambda marg, x, parOrd, gamTab, samp:
                parLike(np.hstack((marg, x)), parOrd, gamTab, samp),
                xopt[i], cons, limits=parLimits[parSet[i]],
                marg=(xopt[imarg]), args=(parOrd, gamTab, samp)))
            print parSet[i], schec[parSet[i]]
    else:
        for i in range(npar):
            schec[parSet[i]] = (xopt[i], [0, 0])
         
    if schec.has_key('P'):        
        # Now fit for density evolution P
        print "Fitting P ..."
        ev = Evol(schec['Q'][0])
        if par['idebug'] > 1:
            dz = (samp.zRange[1] - samp.zRange[0])/100
            zarr = np.arange(samp.zRange[0], samp.zRange[1], dz)
            rarr = rhofun(zarr, 0.0)
            plot.window()
            plt.plot(zarr, rarr)
            plt.xlabel('z'); plt.ylabel('dV/dz'); plt.draw()

        # Redshift limits for each galaxy
        Mevol = samp.absMag + ev.lum(samp.z)
        z1 = map(lambda i: zdm(par['appMin'] - Mevol[i], samp.kcoeff[i],
                               samp.zRange, ev), xrange(samp.ngal))
        z2 = map(lambda i: zdm(par['appMax'] - Mevol[i], samp.kcoeff[i],
                               samp.zRange, ev), xrange(samp.ngal))

        x0 = [schec['P']]
        xopt = scipy.optimize.fmin(rhoLike, x0, (samp, z1, z2),
                                   xtol=0.001, ftol=0.001)

        PErr = 0
        if len(x0) == 1:
            P = xopt[0]
            if errorEst:
                PErr = likeErr(rhoLike, P, limits=parLimits['P'],
                               args=(samp, z1, z2))
        else:
            P = xopt.tolist()
            if errorEst:
                PErr = []
                PErr.append(likeErr(lambda P1, P0, samp, z1, z2:
                                    rhoLike((P0, P1), samp, z1, z2),
                                    P[0], marg=(P[1]), args=(samp, z1, z2)))
                PErr.append(likeErr(lambda P0, P1, samp, z1, z2:
                                    rhoLike((P0, P1), samp, z1, z2),
                                    P[1], marg=(P[0]), args=(samp, z1, z2)))
        schec['P'] = (P, PErr)
        print 'P', schec['P']
        ev = Evol(schec['Q'][0], schec['P'][0])
    else:
        ev = Evol(0, 0)
        z1 = z2 = None

    # Minimum-variance normalization
    if schec.has_key('sigma'):
        sel = SaundSelFn(schec, ev, samp)
    else:
        sel = SelFn(schec, ev, gamTab, samp)
    selg = sel.sel(samp.z)
    wmean = samp.weight.mean()

    plot.window()
    plt.plot(sel._z, sel._sel)
    plt.xlabel('z')
    plt.ylabel('STY S(z)')
    plt.draw()

    # Simple unweighted normalisation (Lin et al eq 14)
##     idx = selg > 0
##     sum1 = (samp.weight[idx]/selg[idx]/ev.den(samp.z[idx])).sum()
##     rmin = cosmo.dm(np.min(samp.z[idx]))
##     rmax = cosmo.dm(np.max(samp.z[idx]))
##     vol = (rmax**3 - rmin**3)*samp.area/3.0
##     dens = sum1/vol
##     dErr = math.sqrt(dens/vol)
    
    dens = 0.1 # first guess
    tol = 0.001
    err = 1
    niter = 0
    maxiter = 100
    while (err > tol and niter < maxiter):
        niter += 1
        dold = dens
        result = scipy.integrate.quad(dV, samp.zRange[0], samp.zRange[1],
                                      args=(sel, dens, wmean),
                                      epsabs=0.01, epsrel=1e-3, full_output=1)
        if len(result) > 3:
            print result
            if par['idebug'] > 1:
                for i in xrange(len(zarr)):
                    rarr[i] = dV(zarr[i], sel, dens, wmean)
                    plot.window()
                    plt.plot(zarr, rarr)
                    plt.xlabel('z'); plt.ylabel('$dV/dz S(z) w(z)$'); plt.draw()
        vol = result[0]*samp.area
        sum1 = (samp.weight/(1 + par['J3fac']*dens/wmean*selg)/ev.den(samp.z)).sum()
        dens = sum1/vol
        err = abs(dens-dold)/dens
    dErr = math.sqrt(dens/vol)
    print 'dens = ', dens, '+-', dErr, ', vol = ', vol
    if (niter >= maxiter):
        print '''STY normalisation: max number of iterations exceeded.
        Relative error in phi* is ''', err

    phistar = dens/sel.gnorm
    phistarErr = dErr/sel.gnorm
    schec['phistar'] = (phistar, phistarErr)
    print 'phistar', schec['phistar']
##     pdb.set_trace()

    schec['counts'] = parCounts(schec, ev, samp)
    print 'counts', schec['counts']
    
    lumdens = lumDens(schec, zlims, ev, samp, gamTab)

    if likeFile:
        likeGrid(schec, parSet, cons, z1, z2, gamTab, samp, likeFile)

    return ev, lumdens

def parCounts(schec, ev, samp):
    """Predict number of galaxies in sample from parametric LF"""

    def phi_dM(z, schec, ev, kmean):
        """Integral of phi(M,z) dM"""

        dm =  dmodk(z, kmean)
        Mlo = np.clip(par['appMin'] - dm, par['absMin'], par['absMax'])
        Mhi = np.clip(par['appMax'] - dm, par['absMin'], par['absMax'])
        if (Mhi <= Mlo):
            return 0.0
    
        result = scipy.integrate.quad(SchecEv, Mlo, Mhi, args=(z, schec, ev),
                                      epsabs=0.01, epsrel=1e-3, full_output=1)
        return result[0]

    result = scipy.integrate.quad(
        lambda z: phi_dM(z, schec, ev, samp.kmean)*cosmo.dV(z),
        samp.zRange[0], samp.zRange[1],
        epsabs=0.01, epsrel=1e-3, full_output=1)

    return samp.area/samp.weight.mean()*result[0]


    
# Constraint functions for scipy.optimize.fmin_cobyla.
# Must be smooth, such that 0 returned on boundary,
# >0 if inside boundary and <0 if outside.  Step function will not work!

def con_alo(xfit, xfix, parSet, gamTab, samp):
    x = xfit.tolist() + xfix
    a1 = x[parSet.index('alpha')] + 1
    return a1 - gamTab.amin

def con_ahi(xfit, xfix, parSet, gamTab, samp):
    x = xfit.tolist() + xfix
    a1 = x[parSet.index('alpha')] + 1
    return gamTab.amax - a1

def con_mslo(xfit, xfix, parSet, gamTab, samp):
    x = xfit.tolist() + xfix
    Mstar = x[parSet.index('Mstar')]
    return Mstar - parLimits['Mstar'][0]

def con_mshi(xfit, xfix, parSet, gamTab, samp):
    x = xfit.tolist() + xfix
    Mstar = x[parSet.index('Mstar')]
    return limits['Mstar'][1] - Mstar

def con_blo(xfit, xfix, parSet, gamTab, samp):
    x = xfit.tolist() + xfix
    alpha = x[parSet.index('alpha')]
    beta = x[parSet.index('beta')]
    ab1 = alpha + beta + 1
    return ab1 - gamTab.amin

def con_bhi(xfit, xfix, parSet, gamTab, samp):
    x = xfit.tolist() + xfix
    alpha = x[parSet.index('alpha')]
    beta = x[parSet.index('beta')]
    ab1 = alpha + beta + 1
    return gamTab.amax - ab1

def con_mtlo(xfit, xfix, parSet, gamTab, samp):
    x = xfit.tolist() + xfix
    Mstar = x[parSet.index('Mstar')]
    Mt = x[parSet.index('Mt')]
    return Mt - Mstar

def con_mthi(xfit, xfix, parSet, gamTab, samp):
    x = xfit.tolist() + xfix
    Mt = x[parSet.index('Mt')]
    return parLimits['Mt'][1] - Mt

def parLike(x, parSet, gamTab, samp):
    """Returns -ive ln likelihood for parametric fit (vectorized version).
    parSet lists the parameters in the order which they are supplied in x."""

    bad = 1e10

    alpha = x[parSet.index('alpha')]
    a1 = alpha + 1
    if not(gamTab.amin <= a1 < gamTab.amax):
        if par['idebug'] > 1: print x, ' alpha outside range'
        return bad
    
    Mstar = x[parSet.index('Mstar')]
    if not(parLimits['Mstar'][0] <= Mstar < parLimits['Mstar'][1]):
        if par['idebug'] > 1: print x, ' M* outside range'
        return bad

    try:
        sigma = x[parSet.index('sigma')]
        if not(parLimits['sigma'][0] <= sigma < parLimits['sigma'][1]):
            if par['idebug'] > 1: print x, ' sigma outside range'
            return bad
        # Load lookup table for integral of Saunders LF
        saundTab = SaundersIntLookup(alpha, sigma)
    except:
        saundTab = None
        
    try:
        beta = x[parSet.index('beta')]
        ab1 = alpha + beta + 1
        if not(gamTab.amin <= ab1 < gamTab.amax):
            if par['idebug'] > 1: print x, ' alpha + beta outide range'
            return bad
        Mt = x[parSet.index('Mt')]
        if not(Mstar <= Mt < parLimits['Mt'][1]):
            if par['idebug'] > 1: print x, ' Mt outside range'
            return bad
        dp = 1
    except:
        dp = 0

    try:
        Q = x[parSet.index('Q')]
        if not(parLimits['Q'][0] <= Q < parLimits['Q'][1]):
            return bad
        ev = Evol(Q)
    except:
        ev = Evol(0)
        
    Mcorr = Mstar - ev.lum(samp.z)
    L = 10**(0.4*(Mcorr - samp.absMag))
    lgLlo = 0.4*(Mcorr - samp.absMax)
    lgLhi = 0.4*(Mcorr - samp.absMin)

    if saundTab:
        saundInt = saundTab.saund_int(lgLlo) - saundTab.saund_int(lgLhi)
        fc = (samp.weight*(np.log(saundInt) - a1*np.log(L) +
                           np.log10(1+L)**2/(2.0*sigma**2))).sum()
##         print x, fc
        return fc

    gama1 = gamTab.gamma(a1, lgLlo)
    gama2 = gamTab.gamma(a1, lgLhi)
    if min(gama1 - gama2) <= 0:
        if par['idebug'] > 1: print 'min(gama1 - gama2) = ', min(gama1 - gama2)
        return bad
        pdb.set_trace()
    if dp:
        gamb1 = gamTab.gamma(ab1, lgLlo)
        gamb2 = gamTab.gamma(ab1, lgLhi)
        if min(gamb1 - gamb2) <= 0:
            pdb.set_trace()
        Lstb = 10**(0.4*beta*(Mt - Mstar))
        ln_schec = np.log(L**a1 + Lstb*L**ab1) - L
        schecInt = np.log((gama1 - gama2) +
                          Lstb*(gamb1 - gamb2))
        fc = (samp.weight*(schecInt - ln_schec)).sum()
    else:
        schecInt = np.log(gama1 - gama2)
        fc = (samp.weight*(schecInt - (a1*np.log(L) - L))).sum()


    if par['idebug'] > 1:
        print x, fc
    if fc < -1e10:
        pdb.set_trace()
    return fc


def likeErr(fn, xmin, cons=None, limits=None, marg=None, args=(), nsig=1.0):
    """Return one parameter, nsig-sigma lower and upper errors about
    minimum xmin of -log likelihood function fn, marginalising over
    parameters marg.  Since it is called by scipy.optimize.fmin,
    supplied function fn must take parameters in order: marg, x, args.
    """

     # Make args a sequence, if it is not already
    try:
        n = len(args)
    except:
        args = (args,)
        
    if marg is not None:
        fmin = fn(marg, xmin, *args)
    else:
        fmin = fn(xmin, *args)
        
    delta = 0.5*nsig
    fsig = fmin + delta
    tol = 0.001
    
    # Upper bound
    xlo = xmin
    if limits:
        xhi = limits[1]
    else:
        xhi = xmin + 1
        fres = fsig-1
        while fres < fsig:
            xhi += 1.0
            if marg is not None:
                result = scipy.optimize.fmin(fn, marg, args=((xhi,) + args),
                                             xtol=0.001, ftol=0.001,
                                             full_output=1, disp=0)
                fres = result[1]
##                 xopt = scipy.optimize.fmin_cobyla(fn, marg, cons,
##                                                   args=(([xhi],) + args), disp=0)
##                 fres = fn(xopt, [xhi], *args)
            else:
                fres = fn(xhi, *args)

    x = 0.5*(xlo + xhi)
    err = xhi - xlo
    while err > tol:
        if marg is not None:
            result = scipy.optimize.fmin(fn, marg, args=((x,) + args),
                                         xtol=0.001, ftol=0.001,
                                         full_output=1, disp=0)
            fres = result[1]
##             xopt = scipy.optimize.fmin_cobyla(fn, marg, cons,
##                                               args=(([x],) + args), disp=0)
##             fres = fn(xopt, [x], *args)
        else:
            fres = fn(x, *args)
        if fres < fsig:
            xlo = x
        else:
            xhi = x
        x = 0.5*(xlo + xhi)
        err = xhi - xlo
    xupper = x

    # Lower bound
    xhi = xmin
    if limits:
        xlo = limits[0]
    else:
        xlo = xmin - 1
        fres = fsig-1
        while fres < fsig:
            xlo -= 1.0
            if marg is not None:
                result = scipy.optimize.fmin(fn, marg, args=((xlo,) + args),
                                             xtol=0.001, ftol=0.001,
                                             full_output=1, disp=0)
                fres = result[1]
##                 xopt = scipy.optimize.fmin_cobyla(fn, marg, cons,
##                                                   args=(([xlo],) + args), disp=0)
##                 fres = fn(xopt, [xlo], *args)
            else:
                fres = fn(xlo, *args)
            
    x = 0.5*(xlo + xhi)
    err = xhi - xlo
    while err > tol:
        if marg is not None:
            result = scipy.optimize.fmin(fn, marg, args=((x,) + args),
                                         xtol=0.001, ftol=0.001,
                                         full_output=1, disp=0)
            fres = result[1]
##             xopt = scipy.optimize.fmin_cobyla(fn, marg, cons,
##                                               args=(([x],) + args), disp=0)
##             fres = fn(xopt, [x], *args)
        else:
            fres = fn(x, *args)
        if fres < fsig:
            xhi = x
        else:
            xlo = x
        x = 0.5*(xlo + xhi)
        err = xhi - xlo
    xlower = x
    return (xmin-xlower, xupper-xmin)


def rhoLike(P, samp, z1, z2):
    """Calculate -ive ln likelihood for density evolution parameter P.
    samp is the set of galazies, z1[i], z2[i] the min and max redshifts
    at which galaxy i is observable."""

    ev = Evol(P=P)
    dz = 0.001
    ztab = np.arange(samp.zRange[0], samp.zRange[1] + dz, dz)
    nz = len(ztab)
    rhotab = np.zeros(nz)
    rhotab[0] = 0
    for iz in xrange(nz-1):
        zlo = samp.zRange[0] + iz*dz
        zhi = zlo + dz
        result = scipy.integrate.quad(rhofun, zlo, zhi, (ev),
                                      epsabs=0.01, epsrel=1e-3, full_output=1)
        if len(result) > 3:
            pdb.set_trace()
        rhotab[iz+1] = rhotab[iz] + result[0]
    
    result = np.interp(z2, ztab, rhotab) - np.interp(z1, ztab, rhotab)
    fc = -(samp.weight*(ev.logden(samp.z) - np.log(result))).sum()
    if par['idebug'] > 1:
        print P, fc

    return fc


def rhofun(z, ev):
    """10^(0.4Pz) dV/dz for density evolution determination"""
    return ev.den(z)*cosmo.dV(z)

def likeGrid(schec, parSet, cons, z1, z2, gamTab, samp, likeFile):
    """Output likelihood grids for parametric shape and evolutionary
    parameters Q, P.  Currently only works for first-order evolution."""

    print "Likelihood grids ..."
    nsig = {'alpha': 6, 'Mstar': 6, 'beta': 8, 'Mt': 9, 'Q': 5, 'P': 5}
    
    # Peak likelihood
    npar = len(parSet)
    parVal = [schec[parSet[i]][0] for i in range(npar)]
    maxLike = -parLike(parVal, parSet, gamTab, samp)
    prob = 0.05
    nu = 2
    chisq = scipy.special.chdtri(nu, prob)
    Lmin = maxLike - 0.5*chisq
    bad = Lmin - 1
    
    nstep = 31
##     nstep = 21
    like = np.zeros([nstep, nstep])
    nmarg = npar - 2
    f = open(likeFile, 'w')

    root = likeFile.split('.')[0]
    likeFits = root + '.fits'
    header = pyfits.Header()
    header.update('band', par['band'])
    header.update('colour', par['colour'])
    
    hdu = pyfits.PrimaryHDU(header=header)
    hdulist = pyfits.HDUList([hdu])
    
    # Loop over all pairs of shape parameters in parSet
    for i in range(npar-1):
        xpar = parSet[i]
        xopt = schec[xpar][0]
        xmin = max(xopt - nsig[xpar]*schec[xpar][1][0], parLimits[xpar][0])
        xmax = min(xopt + nsig[xpar]*schec[xpar][1][1], parLimits[xpar][1])
        dx = float(xmax - xmin)/nstep
        
        for j in range(i+1, npar):
            ypar = parSet[j]
            yopt = schec[ypar][0]
            ymin = max(yopt - nsig[ypar]*schec[ypar][1][0], parLimits[ypar][0])
            ymax = min(yopt + nsig[ypar]*schec[ypar][1][1], parLimits[ypar][1])
            dy = float(ymax - ymin)/nstep
            
            print xpar, xmin, xmax, ypar, ymin, ymax, maxLike
            print >> f, par['iband'], xpar, ypar, xopt, yopt, maxLike, nstep, xmin, xmax, ymin, ymax

            header = pyfits.Header()
            header.update('maxlike', maxLike)
            header.update('ctype1', xpar)
            header.update('crpix1', 0.5)
            header.update('crval1', xmin)
            header.update('cdelt1', dx)
            header.update('ctype2', ypar)
            header.update('crpix2', 0.5)
            header.update('crval2', ymin)
            header.update('cdelt2', dy)

            # Don't move following line before loop or data overwritten
            data = np.zeros([nmarg + 1, nstep, nstep])

            imarg = filter(lambda k: k != i and k != j, range(npar))
            x0init = []
            parOrd = []
            margstr = ''
            if npar > 2:
                for k in imarg:
                    x0init.append(schec[parSet[k]][0])
                    parOrd.append(parSet[k])
                    margstr += parSet[k] + ' '
            header.update('margpar', margstr)
            parOrd += [parSet[i], parSet[j]]
            print parOrd
            
            for ix in range(nstep):
                x0 = np.array(x0init)
                x = xmin + (ix+0.5)*dx
                for iy in range(nstep):
                    y = ymin + (iy+0.5)*dy
                    if ((parSet[i] == 'Mstar' and parSet[j] == 'Mt' and
                         y < x) or
                        (parSet[i] == 'alpha' and parSet[j] == 'beta' and
                         not(gamTab.amin <= x + y + 1 < gamTab.amax))):
                        like[iy, ix] = bad
                        data[0, iy, ix] = bad
                    else:
                        if npar > 2:
                            # Ensure that inital M* < Mt
                            if parSet[j] == 'Mt':
                                ims = parOrd.index('Mstar')
                                if ims < 2 and y < x0[ims]:
                                    x0[ims] = y - 0.1

                            # Ensure no starting values too close to zero
                            # so that parameter is space fully explored
                            idx = abs(x0) < 0.05
                            x0[idx] = 0.05
                            
                            result = scipy.optimize.fmin(
                                lambda x0, x, y, parOrd, gamTab, samp:
                                parLike(np.hstack((x0, x, y)),
                                        parOrd, gamTab, samp), x0,
                                args=(x, y, parOrd, gamTab, samp),
                                ftol=1e-6, full_output=1, disp=0)
                            x0 = result[0]
##                             result = scipy.optimize.fmin(
##                                 lambda x0, x, y, parOrd, gamTab, samp:
##                                 parLike(np.hstack((x0, x, y)),
##                                         parOrd, gamTab, samp), x0,
##                                 args=(x, y, parOrd, gamTab, samp),
##                                 ftol=1e-6, full_output=1, disp=0)
##                             x0 = result[0]
                            like[iy, ix] = -result[1]
                            data[0, iy, ix] = -result[1]
                            for im in range(nmarg):
                                data[im+1, iy, ix] = result[0][im]
##                             xfit = scipy.optimize.fmin_cobyla(
##                                 parLike, x0, cons,
##                                 args=([x, y], parOrd, gamTab, samp),
##                                 maxfun=2000, disp=0)
##                             x0 = xfit
##                             like[iy, ix] = -parLike(xfit, [x, y], parOrd, gamTab, samp)
##                             print xfit, [x, y], like[iy, ix]
                            if like[iy, ix] < -1e9:
##                                 print x0, [x, y], like[iy, ix]
##                                 pdb.set_trace()
                                like[iy, ix] = bad
                        else:
                            like[iy, ix] = -parLike([x, y], parOrd, gamTab, samp)
                            data[0, iy, ix] = like[iy, ix]

            pickle.dump(like, f)

            hdu = pyfits.ImageHDU(data, header=header)
            hdulist.append(hdu)
            
            plot.window()
            plt.imshow(like, aspect='auto', #cmap=matplotlib.cm.gray,
                       origin='lower', extent=[xmin, xmax, ymin, ymax])
            plt.colorbar()
            plt.contour(like, [Lmin,], aspect='auto',
                        origin='lower', extent=[xmin, xmax, ymin, ymax])
            plt.xlabel(xpar); plt.ylabel(ypar); plt.title('log likelihood')
            plt.draw()

            print 'likelihood range', like.min(), like.max()
    # Special case: Q, P
    if schec.has_key('P'):        

        Q = schec['Q'][0]
        Qmin = max(Q - 5*schec['Q'][1][0], parLimits['Q'][0])
        Qmax = min(Q + 5*schec['Q'][1][1], parLimits['Q'][1])
        dQ = (Qmax - Qmin)/nstep
        P = schec['P'][0]
        Pmin = max(P - 8*schec['P'][1][0], parLimits['P'][0])
        Pmax = min(P + 8*schec['P'][1][1], parLimits['P'][1])
        dP = (Pmax - Pmin)/nstep

        print 'Q', Qmin, Qmax, 'P', Pmin, Pmax
        print >> f, par['iband'], 'Q', 'P', Q, P, maxLike, nstep, \
              Qmin, Qmax, Pmin, Pmax

        header = pyfits.Header()
        header.update('maxlike', maxLike)
        header.update('ctype1', 'Q')
        header.update('crpix1', 0.5)
        header.update('crval1', Qmin)
        header.update('cdelt1', dQ)
        header.update('ctype2', 'P')
        header.update('crpix2', 0.5)
        header.update('crval2', Pmin)
        header.update('cdelt2', dP)

        Qarr = np.zeros(nstep)
        Lsty = np.zeros(nstep)
        Lrho = np.zeros(nstep)
        am_fit = [schec['alpha'][0], schec['Mstar'][0]]
        P_fit = [0.0]
        for iq in range(nstep):
            qt = Qmin + (iq+0.5)*dQ
            Qarr[iq] = qt
            ev = Evol(qt)
        
            # L(Q|alpha, M*)
            result = scipy.optimize.fmin(
                lambda (alpha, Mstar), Q, parSet, gamTab, samp:
                parLike((alpha, Mstar, Q), parSet, gamTab, samp),
                am_fit, (qt, parSet, gamTab, samp),
                xtol=0.001, ftol=0.001, full_output=1, disp=0)
            Lsty[iq] = result[1]

            # L(P|Q)
            Mevol = samp.absMag + qt*(samp.z - par['z0'])
            z1 = map(lambda i: zdm(par['appMin'] - Mevol[i], samp.kcoeff[i],
                                   samp.zRange, ev), xrange(samp.ngal))
            z2 = map(lambda i: zdm(par['appMax'] - Mevol[i], samp.kcoeff[i],
                                   samp.zRange, ev), xrange(samp.ngal))

            result = scipy.optimize.fmin(rhoLike, P_fit, (samp, z1, z2),
                                         xtol=0.001, ftol=0.001,
                                         full_output=1, disp=0)
            Lrho[iq] = result[1]
        
            for ip in range(nstep):
                pt = Pmin + (ip+0.5)*dP
                like[ip, iq] = -(rhoLike(pt, samp, z1, z2) - Lrho[iq]) - Lsty[iq]
        pickle.dump(like, f)
        hdu = pyfits.ImageHDU(like, header=header)
        hdulist.append(hdu)

        plot.window()
        plt.plot(Qarr, Lsty)
        plt.xlabel('Q'); plt.ylabel('-ln L(Q|alpha,M*)')
        plt.draw()
    
        plot.window()
        plt.plot(Qarr, Lrho)
        plt.xlabel('Q'); plt.ylabel('-ln L(Q|P)')
        plt.draw()
    
        plot.window()
        plt.imshow(like, aspect='auto', #cmap=matplotlib.cm.gray,
                   origin='lower', extent=[Qmin, Qmax, Pmin, Pmax])
        plt.xlabel('Q'); plt.ylabel('P'); plt.title('log likelihood')
        plt.colorbar()
        plt.draw()
    f.close()

    hdulist.writeto(likeFits, clobber=True)
    hdulist.close()

## def check_start(xvar, xfix, parOrd, schec):
##     """Check that initial values xvar are valid, given fixed parameters xfix
##     and parameter list parOrd."""

##     try:
##         ia = parOrd.index('alpha')
##         ims = parOrd.index('Mstar')
##         ib = parOrd.index('beta')
##         imt = parOrd.index('Mt')
##         alpha = schec['alpha'][0]
##         Mstar = schec['Mstar'][0]
##         beta = schec['beta'][0]
##         Mt = schec['Mt'][0]
##     except:
##         pass
    
##     nvar = len(xvar)
##     nfix = len(xfix)
##     for ivar in range(nvar):
##         if parOrd[ivar] == 'alpha':
##             if gama.amin > alpha + beta
##                 kk = parOrd.index('beta')
                                        

def likeGridSp(schec, parSet, cons, z1, z2, gamTab, samp, likeFile):
    """Output likelihood grids for parametric shape and evolutionary
    parameters Q, P.  Currently only works for first-order evolution.
    Now spirals out from location of max likelihood until all likelihoods
    on outside edge are below Lmin."""

    def stepsize(parname):
        """Set size of step for this parameter"""
        step = 0.02
        if parname == 'Mt' and par['colour'] != 'c':
            step = 0.1
        return step
    
    def likeSpiral():
        """Spiral out from location of max likelihood until all likelihoods
        on outside edge are below Lmin."""

        Lmax = maxLike
        n = 0
        x0 = x0init
        like_dict = {str(np.array([0,0])): maxLike}
        while Lmax > Lmin:
            Lmax = Lmin
            n += 1
            c = np.array([-n, -n])
            for step in (np.array((1,0)), np.array((0,1)),
                         np.array((-1,0)), np.array((0,-1))):
                c += step
                while max(abs(c)) <= n:
                    x = xopt + c[0]*dx
                    y = yopt + c[1]*dy
                    if ((parSet[i] == 'Mstar' and parSet[j] == 'Mt' and
                         y < x) or
                        (parSet[i] == 'alpha' and parSet[j] == 'beta' and
                         not(gamTab.amin <= x + y + 1 < gamTab.amax))):
                        L = Lmin - 1
                    else:
                        if npar > 2:
                            result = scipy.optimize.fmin(
                                lambda x0, x, y, parOrd, gamTab, samp:
                                parLike(np.hstack((x0, x, y)),
                                        parOrd, gamTab, samp), x0,
                                args=(x, y, parOrd, gamTab, samp),
                                ftol=1e-6, full_output=1, disp=0)
                            x0 = result[0]
                            # Call again starting at previous min
                            result = scipy.optimize.fmin(
                                lambda x0, x, y, parOrd, gamTab, samp:
                                parLike(np.hstack((x0, x, y)),
                                        parOrd, gamTab, samp), x0,
                                args=(x, y, parOrd, gamTab, samp),
                                ftol=1e-6, full_output=1, disp=0)
                            x0 = result[0]
                            L = -result[1]
##                             xfit = scipy.optimize.fmin_cobyla(
##                                 parLike, (x0, x, y),, cons,
##                                 args=(parOrd, gamTab, samp),
##                                 maxfun=2000, disp=0)
##                             x0 = xfit
##                             L = -parLike((xfit, x, y), parOrd, gamTab, samp)
##                             print c, x0, [x, y], L
                        else:
                            L = -parLike((x, y), parOrd, gamTab, samp)
                        Lmax = max(Lmax, L)
                    like_dict[str(c)] = L
                    c += step
                c -= step
            print n, Lmax
            
        # Place likelihoods into an array, pickle & plot
        nstep = 2*n+1
        like = np.zeros([nstep, nstep])
        xmin = xopt - (n+0.5)*dx
        xmax = xopt + (n+0.5)*dx
        ymin = yopt - (n+0.5)*dy
        ymax = yopt + (n+0.5)*dy
        for iy in range(-n, n+1):
            for ix in range(-n, n+1):
                like[iy+n, ix+n] = like_dict[str(np.array([ix, iy]))]

        print >> f, par['iband'], xpar, ypar, xopt, yopt, maxLike, nstep, \
              xmin, xmax, ymin, ymax

        pickle.dump(like, f)

        plot.window()
        plt.imshow(like, aspect='auto', #cmap=matplotlib.cm.gray,
                   origin='lower', extent=[xmin, xmax, ymin, ymax])
        plt.colorbar()
        plt.contour(like, [Lmin,], aspect='auto',
                    origin='lower', extent=[xmin, xmax, ymin, ymax])
        plt.xlabel(xpar); plt.ylabel(ypar); plt.title('log likelihood')
        plt.draw()

        print 'likelihood range', like.min(), like.max()

        return like
    

    print "Likelihood grids ..."
    
    # Peak likelihood
    npar = len(parSet)
    parVal = [schec[parSet[i]][0] for i in range(npar)]
    maxLike = -parLike(parVal, parSet, gamTab, samp)
    prob = 0.05
    nu = 2
    chisq = scipy.special.chdtri(nu, prob)
    Lmin = maxLike - 0.5*chisq
    f = open(likeFile, 'w')

    # Loop over all pairs of shape parameters in parSet
    for i in range(npar-1):
        xpar = parSet[i]
        xopt = schec[xpar][0]
        dx = stepsize(xpar)
        
        for j in range(i+1, npar):
            ypar = parSet[j]
            yopt = schec[ypar][0]
            dy = stepsize(ypar)
            
            print xpar, ypar, maxLike, Lmin

            imarg = filter(lambda k: k != i and k != j, range(npar))
            x0init = []
            parOrd = []
            if npar > 2:
                for k in imarg:
                    x0init.append(schec[parSet[k]][0])
                    parOrd.append(parSet[k])
            parOrd += [parSet[i], parSet[j]]
##             print parOrd

            like = likeSpiral()
            
    # Special case: Q, P
    if schec.has_key('P'):        

        nstep = 20
        Q = schec['Q'][0]
        Qmin = Q - 1.0
        Qmax = Q + 1.0
        dQ = (Qmax - Qmin)/nstep
        P = schec['P'][0]
        Pmin = P - 1.5
        Pmax = P + 1.5
        dP = (Pmax - Pmin)/nstep

        print 'Q', Qmin, Qmax, 'P', Pmin, Pmax
        print >> f, par['iband'], 'Q', 'P', Q, P, maxLike, nstep, \
              Qmin, Qmax, Pmin, Pmax

        Qarr = np.zeros(nstep)
        Lsty = np.zeros(nstep)
        Lrho = np.zeros(nstep)
        like = np.zeros([nstep, nstep])
        parOrd = ['alpha', 'Mstar', 'Q']
        x0 = [schec['alpha'][0], schec['Mstar'][0]]
        P_fit = [0.0]
        for iq in range(nstep):
            qt = Qmin + (iq+0.5)*dQ
            Qarr[iq] = qt
            ev = Evol(qt)
        
            # L(Q), marginalising over alpha, M*
            result = scipy.optimize.fmin(
                lambda x0, qt, parOrd, gamTab, samp:
                parLike(np.hstack((x0, qt)),
                        parOrd, gamTab, samp), x0,
                args=(qt, parOrd, gamTab, samp),
                ftol=1e-6, full_output=1, disp=0)
##             result = scipy.optimize.fmin(
##                 parLike, (x0, qt), args=(parOrd, gamTab, samp),
##                 ftol=1e-6, full_output=1, disp=0)
            x0 = result[0]
            Lsty[iq] = result[1]

            # L(P|Q)
            Mevol = samp.absMag + qt*(samp.z - par['z0'])
            z1 = map(lambda i: zdm(par['appMin'] - Mevol[i], samp.kcoeff[i],
                                   samp.zRange, ev), xrange(samp.ngal))
            z2 = map(lambda i: zdm(par['appMax'] - Mevol[i], samp.kcoeff[i],
                                   samp.zRange, ev), xrange(samp.ngal))

            result = scipy.optimize.fmin(rhoLike, P_fit, (samp, z1, z2),
                                         xtol=0.001, ftol=0.001,
                                         full_output=1, disp=0)
            Lrho[iq] = result[1]
        
            for ip in range(nstep):
                pt = Pmin + (ip+0.5)*dP
                like[ip, iq] = -(rhoLike(pt, samp, z1, z2) - Lrho[iq]) - Lsty[iq]
        pickle.dump(like, f)

        plot.window()
        plt.plot(Qarr, Lsty)
        plt.xlabel('Q'); plt.ylabel('-ln L(Q|alpha,M*)')
        plt.draw()
    
        plot.window()
        plt.plot(Qarr, Lrho)
        plt.xlabel('Q'); plt.ylabel('-ln L(Q|P)')
        plt.draw()
    
        plot.window()
        plt.imshow(like, aspect='auto', #cmap=matplotlib.cm.gray,
                   origin='lower', extent=[Qmin, Qmax, Pmin, Pmax])
        plt.xlabel('Q'); plt.ylabel('P'); plt.title('log likelihood')
        plt.colorbar()
        plt.draw()
    f.close()


def lumDens(schec, zlims, ev, samp, gamTab):
    """Luminosity density in redshift bins (Lin et al eqn 16)."""
        
    Msun = Msun_ugriz[par['iband']]
    
    Lstar = 10.0**(0.4*(Msun - schec['Mstar'][0]))
    ld0 = schec['phistar'][0]*Lstar*scipy.special.gamma(schec['alpha'][0] + 2)
    try:
        beta = schec['beta'][0]
        ab2 = schec['alpha'][0] + beta + 2
        Lt = 10.0**(0.4*(Msun - schec['Mt'][0]))
        Lstb = (Lstar/Lt)**beta
        ld0 += schec['phistar'][0]*Lstar*Lstb*scipy.special.gamma(ab2)
    except:
        pass
    
    r1 = cosmo.dm(samp.zRange[0])
    r2 = cosmo.dm(samp.zRange[1])
    vol = (r2*r2*r2 - r1*r1*r1)*samp.area/3.0
    ld0err = ld0*math.sqrt(par['J3fac']/vol)
    print 'lumDens(z=0) = %e' % ld0, ' +/- %e' % ld0err, ' Lsun h^3 Mpc^-3'

    if schec.has_key('sigma'):
        sel = SaundSelFn(schec, ev, samp, lumWeight=1)
    else:
        sel = SelFn(schec, ev, gamTab, samp, lumWeight=1)

    selg = sel.sel(samp.z)
    zmean = np.zeros(len(zlims))
    ld = np.zeros(len(zlims))
    lderr = np.zeros(len(zlims))
    
    plot.window()
    plt.plot(samp.z, selg, ',')
    plt.xlabel('z'); plt.ylabel(r'$S_L(z)$')
    plt.draw()
    
    for iz in range(len(zlims)):
        zlo = zlims[iz][0]
        zhi = zlims[iz][1]
        r1 = cosmo.dm(zlo)
        r2 = cosmo.dm(zhi)
        vol = (r2*r2*r2 - r1*r1*r1)*samp.area/3.0

        idx = (zlo <= samp.z)*(samp.z < zhi)
        zmean[iz] = (samp.z[idx]).mean()
        ld[iz] = (samp.weight[idx]*10**(0.4*(Msun - samp.absMag[idx]))/
                  selg[idx]).sum()/vol
        lderr[iz] = ld[iz]*math.sqrt(par['J3fac']/vol)
        if not np.isfinite(zmean[iz]): zmean[iz] = 0
        if not np.isfinite(ld[iz]): ld[iz] = 0
        if not np.isfinite(lderr[iz]): lderr[iz] = 0
    plot.window()
    plt.errorbar(zmean, ld, lderr)
    zz = np.linspace(samp.zRange[0], samp.zRange[1], 20)
    lz = ld0*ev.lumden(zz)
    plt.plot(zz, lz)
    plt.xlabel('z'); plt.ylabel('$\rho_L L_\odot h^3$ Mpc$^{-3}$')
    plt.draw()

    lumdens = {'ld0': (ld0, ld0err), 'zRange': samp.zRange,
               'zmean': zmean.tolist(), 'ld': (ld.tolist(), lderr.tolist())}
    return lumdens


#------------------------------------------------------------------------------
# 1/Vmax estimate
#------------------------------------------------------------------------------

def vmax(samp, phi):
    """1/Vmax LF estimate (disjoint z slices)"""

    print "1/Vmax estimate..."
    rRange = cosmo.dm(samp.zRange)
    for i in xrange(samp.ngal):
        ibin = np.floor((samp.absMag[i] - phi.absMin)/phi.absStep).astype(np.int32)
        if 0 <= ibin < phi.nbin:
            r1 = np.clip(samp.rmin[i], rRange[0], rRange[1])
            r2 = np.clip(samp.rmax[i], rRange[0], rRange[1])
            Vm = r2*r2*r2 - r1*r1*r1
            if Vm > 0:
                phi.ngal[ibin] += 1
                phi.wsum[ibin] += samp.weight[i]
                phi.Mav[ibin] += samp.absMag[i]*samp.weight[i]
                phi.Vmax[ibin] += samp.weight[i]/Vm
            else:
                print 'galaxy ', i, ' distance limits ', r1, r2
                
    phi.Vmax *= 3.0/samp.area/par['absStep']
    for ibin in xrange(phi.nbin):
        if phi.wsum[ibin] > 0:
            phi.Mav[ibin] /= phi.wsum[ibin]
        if phi.ngal[ibin] > 0:
            phi.VmaxErr[ibin] = phi.Vmax[ibin]/math.sqrt(phi.ngal[ibin])

    if par['idebug'] > 1:
        print phi.Mav, phi.Vmax

    plot.window()
    phi.VmaxPlot()


#------------------------------------------------------------------------------
# SWML estimate
#------------------------------------------------------------------------------

def swml(samp, phi, schec):
    """SWML LF estimate (disjoint z slices)"""

    print "SWML estimate..."
    phi.swml = np.copy(phi.Vmax)
    maxiter = 100
    tol = 1e-6

    # Bin coverage H[igal, ibin]
    H = np.zeros([samp.ngal, phi.nbin])
    for igal in xrange(samp.ngal):

        # Lower (bright) limit
	ilo, frac = phi.bin(samp.absMin[igal])
        wtlo = 1 - frac
        
        # Upper (faint) limit
	ihi, wthi = phi.bin(samp.absMax[igal])

        H[igal, ilo] = wtlo
        H[igal, ilo+1:ihi] = 1
        H[igal, ihi] = wthi
        
    phisum = (phi.swml).sum()

    # Trap for no galaxies in range
    if phisum <= 0:
        phi.lnLikeRatio = 0
        phi.prob = 0
        phi.swmlErr = np.zeros(len(phi.swmlErr))
        return phi
        
    # Iterate until converged
    niter = 0
    dpm = 1.0
    if par['idebug'] > 1:
        print "Iteration   mean change  rms change  max change sf"
    while (niter < maxiter and dpm > tol):
        phiOld = phi.swml.copy()
        sum0 = np.dot(H, phiOld)
        sum1 = np.dot(samp.weight/sum0, H)
##        print 'sum0: ', sum0
        for ibin in xrange(phi.nbin):
            if sum1[ibin] > 0:
                phi.swml[ibin] = phi.wsum[ibin]/sum1[ibin]
            else:
                phi.swml[ibin] = 0
                
        # Rescale phi and calc fractional mean, rms change
        scale = phisum/(phi.swml).sum()
        phi.swml *= scale
        idx = phiOld > 0
        dfrac = (phi.swml[idx] - phiOld[idx])/phiOld[idx]
        if par['idebug'] > 1: print phi.swml
        dp = dfrac.mean()
        dps = dfrac.std()
        dpm = (np.abs(dfrac)).max()
        if par['idebug'] > 1:
            print niter, dp, dps, dpm, scale
        niter += 1

    # Likelihood ratio test
    sumSTY = 0
    sumSWML = 0
    nbin = 0
    for ibin in xrange(phi.nbin):
        if phi.sty[ibin] > 0 and phi.swml[ibin] > 0:
            sumSTY += phi.wsum[ibin]*math.log(phi.sty[ibin])
            sumSWML += phi.wsum[ibin]*math.log(phi.swml[ibin])
            nbin += 1
    lnLsty = sumSTY - (samp.weight*np.log(np.dot(H, phi.sty))).sum()
    lnLswml = sumSWML - (samp.weight*np.log(np.dot(H, phi.swml))).sum()
    phi.lnLikeRatio = lnLsty - lnLswml
    phi.nu = nbin - schec['nShapePar'] - 1
    if phi.nu > 0:
        phi.prob = scipy.special.chdtrc(phi.nu, -2*phi.lnLikeRatio)
    else:
        phi.prob = 0
    print 'ln likelihood ratio, nu, prob = ', phi.lnLikeRatio, phi.nu, phi.prob
##     if phi.lnLikeRatio > 0: pdb.set_trace()
    
    if par['normType'] == 'dh': swmlNorm(samp, phi)
    if par['normType'] == 'sty': swmlNorm2sty(samp, phi, schec)
    if par['normType'] == 'vmax': phi.counts = swmlNorm2vmax(samp, phi, schec)

    # Estimate errors from covariance matrix
    
    # Find non-zero phi bins and normalisation for constraint
    beta = 1.5
    Mfid = -20
    idx = phi.swml > 0
    nuse = phi.swml[idx].size
    csum = (phi.swml[idx]*10**(-0.4*beta*(phi.Mav[idx] - Mfid))).sum()
    if par['idebug'] > 1:
        print "csum = ", csum

    # Create and fill information matrix
    inf = np.zeros([nuse+1, nuse+1])
    
    for j in xrange(nuse):
        jbin = phi.binno[idx][j]
        for i in xrange(j+1):
            ibin = phi.binno[idx][i]
            zz1 = (samp.weight*H[:, ibin]*H[:, jbin]/(sum0/csum)**2).sum()
            if ibin == jbin:
                zz1 -= phi.wsum[ibin]/(phi.swml[ibin]/csum)**2
            zz1 += 10**(-0.4*beta*(phi.Mav[ibin] + phi.Mav[jbin] - 2*Mfid))
            inf[i,j] = inf[j,i] = zz1

        inf[j, nuse] = inf[nuse, j] = 10**beta*(-0.4*(phi.Mav[jbin] - Mfid))

    covar = scipy.linalg.inv(inf)

    varsum = 0
    g = 0
    for i in xrange(nuse):
        ibin = phi.binno[idx][i]
        var = -covar[i,i]
        if var < 0:
            phi.swmlErr[ibin] = phi.VmaxErr[ibin]
            if par['idebug'] > 0:
                print 'Warning: cov[', ibin, ibin, ') = ', var
        else:
            phi.swmlErr[ibin] = math.sqrt(var)*csum
        g += phi.swml[ibin]
        varsum += (phi.swmlErr[ibin]/phi.swml[ibin])**2
              
    varsum = math.sqrt(varsum)
    if par['idebug'] > 1:
        print 'varsum, g = ', varsum, g
        print phi.Mav, phi.swml, phi.swmlErr
    plot.window()
    phi.swmlPlot()
    
    return phi

def swmlNorm(samp, phi):
    """Normalise SWML estimate for independent z-slice"""
    
    ngal = samp.ngal
    phisum = phi.setNorm()
    phi.selLoad(samp)
    
    plot.window()
    plt.plot(phi._z, phi._sel)
    plt.xlabel('z')
    plt.ylabel('SWML S(z)')
    plt.draw()

    # Select galaxies to use for normalisation
    usegal = [gal[i] for i in xrange(ngal) if phi.use[gal[i].phiBin]]
    nuse = len(usegal)
    weight = np.array([usegal[i].weight for i in xrange(nuse)])
    wmean = weight.mean()
    selg = np.array([phi.sel(usegal[i].z) for i in xrange(nuse)])

    dens = 0.1 # first guess
    err = 1
    niter = 0
    tol = 0.001
    maxiter = 100
    while (err > tol and niter < maxiter):
        niter += 1
        dold = dens
        result = scipy.integrate.quad(dV, samp.zRange[0], samp.zRange[1],
                                      args=(phi, dens, wmean),
                                      epsabs=0.01, epsrel=1e-4, full_output=1)
        if len(result) > 3:
            rarr = np.zeros(len(phi._z))
            for i in xrange(len(phi._z)):
                rarr[i] = dV(phi._z[i], phi, dens, wmean)
            plot.window()
            plt.plot(phi._z, rarr)
            plt.xlabel('z'); plt.ylabel('$dV/dz S(z) w(z)$'); plt.draw()
            pdb.set_trace()
        vol = result[0]*samp.area
	sum1 = (weight/(1 + par['J3fac']*dens/wmean*selg)).sum()
        dens = sum1/vol
        err = abs(dens-dold)/dens
    dErr = math.sqrt(dens/vol)
    print 'dens = ', dens, ' +- ', dErr, ', vol = ', vol
    if (niter >= maxiter):
        print '''LFswml: max number of iterations exceeded.
        Relative error in phi* is ''', err

    # Rescale SWML estimates to give correct normalisation
    phi.swml *= dens/(phisum*phi.absStep)

def swmlNorm2sty(samp, phi, schec):
    """Normalise SWML estimate to STY (Lin et al eqn 20)"""

    swmlVsum = 0.0
    styVsum = 0.0
    for ibin in xrange(phi.nbin):
        r1 = cosmo.dm(zdm(par['appMin'] - phi.Mav[ibin], samp.kmean, samp.zRange))
        r2 = cosmo.dm(zdm(par['appMax'] - phi.Mav[ibin], samp.kmean, samp.zRange))
        Vm = r2*r2*r2 - r1*r1*r1
        swmlVsum += phi.swml[ibin]*Vm
        styVsum += phi.sty[ibin]*Vm
    scale = styVsum/swmlVsum
    phi.swml *= scale
    if par['idebug'] > 1:
        print 'swmlNorm2sty scale = ', scale

def swmlNorm2vmax(samp, phi, schec):
    """Normalise SWML estimate to Vmax (Lin et al eqn 20) and return
    predicted galaxy counts."""

    swmlVsum = 0.0
    vmaxVsum = 0.0
    for ibin in xrange(phi.nbin):
        r1 = cosmo.dm(zdm(par['appMin'] - phi.Mav[ibin], samp.kmean, samp.zRange))
        r2 = cosmo.dm(zdm(par['appMax'] - phi.Mav[ibin], samp.kmean, samp.zRange))
        Vm = r2*r2*r2 - r1*r1*r1
        swmlVsum += phi.swml[ibin]*Vm
        vmaxVsum += phi.Vmax[ibin]*Vm
    scale = vmaxVsum/swmlVsum
    phi.swml *= scale
    if par['idebug'] > 1:
        print 'swmlNorm2vmax scale = ', scale
    return samp.area/3.0/samp.weight.mean()*vmaxVsum*par['absStep']

#------------------------------------------------------------------------------
# Support classes and functions
#------------------------------------------------------------------------------

class Parameters(object):
    """Global parameters for the analysis"""
    
    def __init__(self):
        self.idebug = 0
        self.iband = 0
        self.band = 0
        self.appMin = 0
        self.appMax = 0
        self.nbin = 0
        self.absMin = 0
        self.absMax = 0
        self.absMinSTY = 0
        self.absMaxSTY = 0
        self.absStep = 0
        self.nNormMin = 0
        self.z0 = 0
        self.J3fac = 30000.0
        self.plot = 0
        self.kz = 0
        self.Msun = 0
        self.sim_alpha = None
        self.sim_Mstar = None
        self.sim_phistar = None
        self.sim_Q = None
        
class Sample(object):
    """Stores a sample of galaxies given redshift limits.
    Galaxy attributes are stored in arrays to facilitate vectorisation."""
    
    def __init__(self, zRange, area, z, weight, absMag, Mmin, Mmax, rmin, rmax,
                 phiBin, kcoeff, ra):
        self.zRange = zRange
        self.area = area
        self.ngal = len(z)
        self.z = z
        self.weight = weight
        self.absMag = absMag
        self.absMin = Mmin
        self.absMax = Mmax
        self.rmin = rmin
        self.rmax = rmax
        self.phiBin = phiBin
        self.kcoeff = kcoeff
        self.ra = ra

        # Fit polynomial to median K(z) for this sample
        if self.ngal > 0:
            zbin = np.linspace(zRange[0], zRange[1], 50) - par['z0']
            k_array = [np.polyval(kcoeff[igal], zbin)
                       for igal in xrange(self.ngal)]
            k_median = np.median(k_array, axis=0)
            self.kmean = np.polyfit(zbin, k_median, len(kcoeff[0]))
            k_fit = np.polyval(self.kmean, zbin)
            plot.window()
            plt.plot(zbin + par['z0'], k_median)
            plt.plot(zbin + par['z0'], k_fit, '--')
            plt.xlabel('z')
            plt.ylabel('K(z)')
            plt.draw()
        else:
            self.kmean = np.zeros(5)
        
    def sub(self, zRange, Mmin, Mmax):
        """Select a subset within given redshift and abs mag limits."""
        
        idx = (self.z >= zRange[0]) * (self.z < zRange[1]) * \
              (self.absMag >= Mmin) * (self.absMag < Mmax)
        subset = Sample(zRange, self.area, self.z[idx], self.weight[idx],
                        self.absMag[idx], self.absMin[idx], self.absMax[idx],
                        self.rmin[idx], self.rmax[idx], self.phiBin[idx],
                        self.kcoeff[idx,:], self.ra[idx])
        return subset
    
    def resample(self):
        """Bootstrap resampling"""
        
        idx = np.random.randint(0, self.ngal, self.ngal)
        resamp = Sample(self.zRange, self.area, self.z[idx], self.weight[idx],
                        self.absMag[idx], self.absMin[idx], self.absMax[idx],
                        self.rmin[idx], self.rmax[idx], self.phiBin[idx],
                        self.kcoeff[idx,:], self.ra[idx])
        return resamp
    
    def jacknife(self, jack):
        """Return a subsample with jacknife region jack omitted"""

        # Jacknife regions are 4 deg segments starting at given RA
        ra_jack = (129, 133, 137, 174, 178, 182, 211.5, 215.5, 219.5)
        
        idx = (self.ra < ra_jack[jack]) + (self.ra >= ra_jack[jack] + 4.0)
        resamp = Sample(self.zRange, self.area*8.0/9.0,
                        self.z[idx], self.weight[idx],
                        self.absMag[idx], self.absMin[idx], self.absMax[idx],
                        self.rmin[idx], self.rmax[idx], self.phiBin[idx],
                        self.kcoeff[idx,:], self.ra[idx])
        return resamp
    
class BinnedPhi(object):
    """Binned phi class."""

    def __init__(self, zRange=None, kmean=None, f=None):
        """If file f is specified, read phi instance from that file,
        otherwise create new phi instance."""
        if f:

            line = f.readline()
            self.zRange, self.zmean, self.ntot, self.counts, self.kmean = eval(line)

            data = f.readline().split()
            self.nbin = int(data[0])
            self.absMin = float(data[1])
            self.absMax = float(data[2])
            self.absStep = (self.absMax - self.absMin)/self.nbin
            self.lnLikeRatio = float(data[3])
            self.nu = float(data[4])
            self.prob = float(data[5])
        
        else:
            # Determine range of complete magnitude bins
            # (phiMin <= ibin < phiMax) given redshift range and mean kcoeffs
            Mmin = par['appMin'] - dmodk(zRange[1], kmean)
            phiMin = max(0, int(math.ceil((Mmin - par['absMin'])/par['absStep'])))
            Mmax = par['appMax'] - dmodk(zRange[0], kmean)
            phiMax = min(par['nbin'], int(math.floor((Mmax - par['absMin'])/par['absStep'])))
            self.nbin = phiMax - phiMin
            self.absMin = par['absMin'] + phiMin*par['absStep']
            self.absMax = par['absMin'] + phiMax*par['absStep']
            self.zRange = zRange
            if par['idebug'] > 0:
                print 'phiMin, phiMax, absMin, absMax = ', \
                      phiMin, phiMax, self.absMin, self.absMax
            
            self.absStep = par['absStep']
            self.zmean = 0.5
            self.ntot = 0
            self.counts = 0
            self.lnLikeRatio = 0.0
            self.nu = 0
            self.prob = 0.0

        self.binno = np.arange(self.nbin)
        self.ngal = np.zeros(self.nbin, dtype=np.int32)
        self.wsum = np.zeros(self.nbin)
        self.Mav = np.zeros(self.nbin)
        self.Mcen = np.arange(self.absMin + 0.5*self.absStep, self.absMax, self.absStep)
        self.Vmax = np.zeros(self.nbin)
        self.VmaxErr = np.zeros(self.nbin)
        self.swml = np.zeros(self.nbin)
        self.swmlErr = np.zeros(self.nbin)
        self.sty = np.zeros(self.nbin)
        
        if f:
            for ibin in range(self.nbin):
                data = f.readline().split()
                self.ngal[ibin] = int(data[0])
                self.Mav[ibin] = float(data[1])
                self.Vmax[ibin] = float(data[2])
                self.VmaxErr[ibin] = float(data[3])
                self.swml[ibin] = float(data[4])
                self.swmlErr[ibin] = float(data[5])
                self.sty[ibin] = float(data[6])


    def bin(self, M):
        """Returns bin number and fraction for given magnitude M, such that:
        M = absMin + (ibin+frac)*absStep."""
        M = max(M, self.absMin)
        M = min(M, self.absMax)
	ibin = np.floor((M - self.absMin)/self.absStep).astype(np.int32)
        if ibin >= self.nbin:
            ibin = self.nbin - 1
        frac = (M - (self.absMin + ibin*self.absStep))/self.absStep
        if (frac < 0 or frac > 1):
            print 'frac = ', frac
        return ibin, frac
    
    def VmaxPlot(self, Mrange=None, phirange=[1e-6, 1]):
        if Mrange is None:
            Mrange=[self.absMin, self.absMax]
        plt.semilogy(basey=10, nonposy='clip')
        plt.errorbar(self.Mav, self.Vmax, self.VmaxErr, fmt='o')
        plt.axis(Mrange + phirange)
        plt.xlabel('M')
        plt.ylabel('phi')
        plt.title('Vmax')
        plt.draw()
        
    def swmlPlot(self, Mrange=None, phirange=[1e-6, 1]):
        if Mrange is None:
            Mrange=[self.absMin, self.absMax]
        plt.semilogy(basey=10, nonposy='clip')
        plt.errorbar(self.Mav, self.swml, self.swmlErr, fmt='o')
        plt.axis(Mrange + phirange)
        plt.xlabel('M')
        plt.ylabel('phi')
        plt.title('swml')
        plt.draw()
        
    def styPlot(self, Mrange=None, phirange=[1e-6, 1]):
        if Mrange is None:
            Mrange=[self.absMin, self.absMax]
        plt.semilogy(basey=10, nonposy='clip')
        plt.plot(self.Mav, self.sty)
        plt.axis(Mrange + phirange)
        plt.xlabel('M')
        plt.ylabel('phi')
        plt.title('swml')
        plt.draw()
        
    def save(self, f):
        """Save binned phi estimate to file f"""

        print >> f, [self.zRange, self.zmean, self.ntot, self.counts, self.kmean]
        print >> f, self.nbin, self.absMin, self.absMax, self.lnLikeRatio, self.nu, self.prob
        for i in xrange(self.nbin):
            print >> f, self.ngal[i], self.Mav[i], \
                  self.Vmax[i], self.VmaxErr[i], \
                  self.swml[i], self.swmlErr[i], self.sty[i]
        
        
    def setNorm(self):
        """Set which bins to use for normalisation and find their sum.
        For now, use all bins with > nNormMin galaxies."""

        self.use = np.zeros(self.nbin, dtype=np.int32)
        idx = self.ngal >= par['nNormMin']
        self.use[idx] = 1
        self._phisum = (self.swml[idx]).sum()
        return self._phisum
    
    def selLoad(self, samp):
        """Load selection function lookup table."""
        dz = 0.001
        self._z = np.arange(samp.zRange[0], samp.zRange[1] + dz, dz)
        nz = self._z.size
        self._sel = np.zeros(nz)

        for iz in xrange(nz):
            rmod = dmodk(self._z[iz], samp.kmean)
            # Lower (bright) limit
            ilo, frac = self.bin(par['appMin'] - rmod)
            wtlo = 1 - frac
        
            # Upper (faint) limit
            ihi, wthi = self.bin(par['appMax'] - rmod)

            sumn = self.swml[ilo]*self.use[ilo]*wtlo
            for iphi in xrange(ilo+1, ihi):
                sumn += self.swml[iphi]*self.use[iphi]
            sumn += self.swml[ihi]*self.use[ihi]*wthi
            
            self._sel[iz] = sumn/self._phisum
            if (self._sel[iz] < 0 or self._sel[iz] > 1.001):
                print 'selSWML: bad sel ', self._sel[iz]
    
    def sel(self, z):
        """Lookup selection function."""
        return np.interp(z, self._z, self._sel)

    
class CosmoLookup(object):
    """Distance and volume-element lookup tables.
    NB volume element is per unit solid angle."""

    def __init__(self, H0, omega_l, zRange, dz=0.001, is_sim=0):
        c = 3e5
        cosmo = FlatLambdaCDM(H0=H0, Om0=1-omega_l)
        self._zrange = zRange
        self._z = np.arange(zRange[0], zRange[1] + dz, dz)
        nz = self._z.size
        self._is_sim = is_sim
        if is_sim:
            self._dm = c/H0*self._z
            self._dV = c/H0*self._dm**2
        else:
            self._dm = cosmo.comoving_distance(self._z)
            self._dV = cosmo.comoving_volume(self._z)

    def dm(self, z):
        """Transverse comoving distance."""
        return np.interp(z, self._z, self._dm)

    def dl(self, z):
        """Luminosity distance."""
##         if  self._is_sim:
##             return np.interp(z, self._z, self._dm)
##         else:
        return (1+z)*np.interp(z, self._z, self._dm)

    def dV(self, z):
        """Volume element per unit solid angle."""
        return np.interp(z, self._z, self._dV)

class interpolate(object):
    """Class for 1d interpolation of regularly tabulated data"""
    
    def __init__(self, xmin, xmax, xstep, ytab):
        """Set up a look-up table y(x), where x is regularly spaced"""
        self._xmin = xmin
        self._xmax = xmax
        self._xstep = xstep
        self._ytab = ytab
        self._nx = ytab.size

    def interp(self, x):
        """Return interpolated value"""

        if x < self._xmin or x > self._xmax:
            print 'interpolate.interp: x value ', x, ' outside valid range ', self._xmin, self._xmax
            return 0.0
        
        xscale = (x - self._xmin)/self._xstep
        ix = np.floor(xscale).astype(np.int32)
        if ix > self._nx - 2:
            return self._ytab[self._nx - 1]
        
        wt = ix + 1 - xscale
        if (wt < 0 or wt > 1):
            print 'interpolate.interp: wt = ', wt
        return wt*self._ytab[ix] + (1-wt)*self._ytab[ix+1]

class plotWindow(object):
    """Handles multiple plot windows."""

    def __init__(self, numrows=1, numcols=1, file='lum', format='png'):
        self._numrows = numrows
        self._numcols = numcols
        self._file = file
        self._format = format
        self._fignum = 0
        self._page = 0
        plt.clf()

    def window(self, fignum=0):
        """Select plot window, default is next subplot."""
        if fignum == 0:
            self._fignum += 1
            if self._fignum > self._numrows * self._numcols:
                self.save()
                self._fignum = 1
                plt.clf()
            fignum = self._fignum
            if par['idebug'] > 1:
                print 'plot number ', fignum
        plt.subplot(self._numrows, self._numcols, fignum)
        plt.cla()

    def save(self, savefile=None):
        """Save current figure."""
        if savefile == None:
            self._page += 1
            savefile = self._file + '_%d' % self._page + '.' + self._format
            if par['idebug'] > 1:
                print 'Saving ', savefile
        plt.savefig(savefile)

def test():
    gal = [Galaxy()]*2
    gal[0].z = 0
    gal[1].z = 1
    return gal


class gammaLookup(object):
    """Incomplete Gamma fn lookup"""

    def __init__(self, amin=-2.0, amax=4.5, xmin=-8.0, xmax=2.0):
        self.astep = 0.01
        self.xstep = 0.05
        self.amin = amin
        self.amax = amax
        self.xmin = xmin
        self.xmax = xmax
        a = np.arange(amin, amax+self.astep, self.astep)
        x = np.arange(xmin, xmax+self.xstep, self.xstep)
        xx = 10**np.arange(xmin, xmax+2*self.xstep, self.xstep)
        na = a.size
        nx = x.size
        self.log_gam = np.zeros([na, nx])
        
        # Load the lookup table cumulatively from large to small x at each a
        for i in xrange(na):
            gsum = 0
            for j in xrange(nx-1, -1, -1):
                g = scipy.integrate.quad(gamfun, xx[j], xx[j+1], a[i],
                                         full_output=1)
                if len(g) > 3:
                    pdb.set_trace()
                gsum += g[0]
                self.log_gam[i,j] = np.log10(gsum)

        if par['idebug'] > 1:
            print 'log gamma:', self.log_gam
            plot.window()
            plt.imshow(self.log_gam, aspect='auto',
                       origin='lower', extent=[xmin, xmax, amin, amax])
            plt.xlabel('x'); plt.ylabel('a'); plt.title('log Gamma')
            plt.colorbar()
            plt.draw()

            # Test interpolation
            a = np.arange(amin, amax, self.astep/math.pi)
            x = np.arange(xmin, xmax, self.xstep/math.pi)
            gam_int = np.zeros([a.size, x.size])
            for i in xrange(a.size):
                for j in xrange(x.size):
                    gam_int[i,j] = math.log10(self.gamma(a[i], x[j]))
            plot.window()
            plt.imshow(gam_int, aspect='auto',
                       origin='lower', extent=[xmin, xmax, amin, amax])
            plt.xlabel('x'); plt.ylabel('a'); plt.title('Interpolated log Gamma')
            plt.colorbar()
            plt.draw()

    def gamma(self, a, x):
        """Return interpolated Gamma function for array of x values,
        where a = alpha + 1, x = 0.4(M*-M) = lg(L/L*).
        Returns -2 if a outside range, -1 if x outside range."""

        # Find length of array x.  If scalar, make it a 1-element array
        try:
            gam = np.zeros(len(x))
        except:
            x = np.array([x])
            gam = np.array([0.0])
        
        if a < self.amin or a > self.amax:
            pdb.set_trace()
            gam = -2
            return gam
        
        gam[x < self.xmin] = -1
        if min(gam) < 0: 
            pdb.set_trace()
                        
        idx = (x >= self.xmin) * (x < self.xmax)

        agrid = (a - self.amin)/self.astep
        ia1 = min(np.floor(agrid).astype(np.int32), self.log_gam.shape[0] - 2)
        ia2 = ia1 + 1
        
        xgrid = (x[idx] - self.xmin)/self.xstep
        ix1 = np.floor(xgrid).astype(np.int32)
        ix2 = ix1 + 1
        
        w11 = (ix2-xgrid)*(ia2-agrid)
        w21 = (xgrid-ix1)*(ia2-agrid)
        w12 = (ix2-xgrid)*(agrid-ia1)
        w22 = (xgrid-ix1)*(agrid-ia1)

        ans = 10**(w11*self.log_gam[ia1,ix1] + w21*self.log_gam[ia1,ix2] + 
                   w12*self.log_gam[ia2,ix1] + w22*self.log_gam[ia2,ix2])
        gam[idx] = ans
        
        return gam
        
    
def gamfun(x, a):
    return math.exp(-x)*x**(a-1)


class SaundersIntLookup(object):
    """Lookup table for integral of Saunders et al LF"""

    def __init__(self, alpha, sigma, xmin=-5.0, xmax=3.0):
        self.xstep = 0.05
        self.xmin = xmin
        self.xmax = xmax
        self.x = np.arange(xmin, xmax+self.xstep, self.xstep)
        xx = 10**np.arange(xmin, xmax+2*self.xstep, self.xstep)
        nx = self.x.size
        self.log_int = np.zeros(nx)
        
        # Load the lookup table cumulatively from large to small x
        gsum = 0
        for j in xrange(nx-1, -1, -1):
            g = scipy.integrate.quad(saunders_lf, xx[j], xx[j+1], 
                                     (alpha, sigma), full_output=1)
            if len(g) > 3:
                pdb.set_trace()
            gsum += g[0]
            self.log_int[j] = np.log10(gsum)

        if par['idebug'] > 1:
            print 'log saund:', self.log_int
            plot.window()
            plt.plot(self.x, self.log_int)
            plt.xlabel('x'); plt.ylabel('log int Saunders')
            plt.draw()

    def saund_int(self, x):
        """Return interpolated integral of Saunders function for array
        of x values, where x = 0.4(M*-M) = lg(L/L*).
        Returns -1 if x outside range."""

        # Find length of array x.  If scalar, make it a 1-element array
        try:
            gam = np.zeros(len(x))
        except:
            x = np.array([x])
            gam = np.array([0.0])
        
        idx = (x >= self.xmin) * (x < self.xmax)
        gam[x < self.xmin] = -1
        if min(gam) < 0: 
            pdb.set_trace()

        ans = 10**np.interp(x[idx], self.x, self.log_int)
        gam[idx] = ans
        
        return gam
        
    
def saunders_lf(L, alpha, sigma):
    """Saunders et al. (1990) LF fit"""
    return L**(alpha)*np.exp(-(np.log10(1 + L))**2/(2.0*sigma**2))


class SelFn(object):
    """STY selection function.
    Specify lumWeight = 1 for luminosity-weighted selection."""

    def __init__(self, schec, ev, gamTab, samp, lumWeight=0, dz=0.001):
        a1 = schec['alpha'][0] + 1 + lumWeight
        Mstar = schec['Mstar'][0]
        try:
            beta = schec['beta'][0]
            ab1 = a1 + beta
            Mt = schec['Mt'][0]
            Lstb = 10**(0.4*beta*(Mt - Mstar))
            dp = 1
        except:
            dp = 0
            
        self._z = np.arange(samp.zRange[0], samp.zRange[1] + dz, dz)
        self._nz = self._z.size
        self._sel = np.zeros(self._nz)
        lgLlo = 0.4*(Mstar - par['absMaxSTY'])
        gama1 = gamTab.gamma(a1, lgLlo)
        lgLhi = 0.4*(Mstar - par['absMinSTY'])
        gama2 = gamTab.gamma(a1, lgLhi)
        self.gnorm = gama1[0] - gama2[0]
        if lumWeight:
            self.gnorm = scipy.special.gamma(a1)
            print 'gnorm (M1 - M2): ', gama1[0] - gama2[0], ', extrapolated: ', self.gnorm
        if dp:
            gamb1 = gamTab.gamma(ab1, lgLlo)
            gamb2 = gamTab.gamma(ab1, lgLhi)
            if lumWeight:
                self.gnorm += Lstb*scipy.special.gamma(ab1)
            else:
                self.gnorm += Lstb*(gamb1[0] - gamb2[0])

        for iz in xrange(self._nz):
            rmod = dmodk(self._z[iz], samp.kmean)
            dM = ev.lum(self._z[iz])

            M = np.clip(par['appMax'] - rmod + dM, par['absMinSTY'], par['absMaxSTY'])
            lgLlo = 0.4*(Mstar - M)
            gama1 = gamTab.gamma(a1, lgLlo)

            M = np.clip(par['appMin'] - rmod + dM, par['absMinSTY'], par['absMaxSTY'])
            lgLhi = 0.4*(Mstar - M)
            gama2 = gamTab.gamma(a1, lgLhi)
            gam = gama1[0] - gama2[0]
            if dp:
                gamb1 = gamTab.gamma(ab1, lgLlo)
                gamb2 = gamTab.gamma(ab1, lgLhi)
                gam += Lstb*(gamb1[0] - gamb2[0])
            
            sel = gam/self.gnorm
            if sel < 0 or sel > 1:
                print "selFn: bad sel ", sel
##                 pdb.set_trace()
            self._sel[iz] = sel
        self._sel = np.clip(self._sel, 0.0, 1.0)
        
    def sel(self, z):
        """Linear interpolation of selection function"""
        return np.interp(z, self._z, self._sel)


class SaundSelFn(object):
    """STY selection function for Saunders et al LF.
    Specify lumWeight = 1 for luminosity-weighted selection."""

    def __init__(self, schec, ev, samp, lumWeight=0, dz=0.001):
        alpha = schec['alpha'][0]
        a1 = alpha + 1 + lumWeight
        Mstar = schec['Mstar'][0]
        sigma = schec['sigma'][0]
        saundTab = SaundersIntLookup(alpha, sigma)
            
        self._z = np.arange(samp.zRange[0], samp.zRange[1] + dz, dz)
        self._nz = self._z.size
        self._sel = np.zeros(self._nz)
        lgLlo = 0.4*(Mstar - par['absMaxSTY'])
        lgLhi = 0.4*(Mstar - par['absMinSTY'])
        self.gnorm = saundTab.saund_int(lgLlo)[0] - saundTab.saund_int(lgLhi)[0]
        for iz in xrange(self._nz):
            rmod = dmodk(self._z[iz], samp.kmean)
            dM = ev.lum(self._z[iz])

            M = np.clip(par['appMax'] - rmod + dM, par['absMinSTY'], par['absMaxSTY'])
            lgLlo = 0.4*(Mstar - M)

            M = np.clip(par['appMin'] - rmod + dM, par['absMinSTY'], par['absMaxSTY'])
            lgLhi = 0.4*(Mstar - M)
            gam = saundTab.saund_int(lgLlo) - saundTab.saund_int(lgLhi)
            sel = gam/self.gnorm
            if sel < 0 or sel > 1:
                print "SaundSelFn: bad sel ", sel
##                 pdb.set_trace()
            self._sel[iz] = sel
        self._sel = np.clip(self._sel, 0.0, 1.0)
        
    def sel(self, z):
        """Linear interpolation of selection function"""
        return np.interp(z, self._z, self._sel)


class Evol(object):
    """Class for luminosity and density evolution."""

    def __init__(self, Q=0, P=0):
        try:
            self.Qord = len(Q)
        except:
            self.Qord = 1
            if Q == 0:
                self.Qord = 0
        self.Q = Q

        try:
            self.Pord = len(P)
            if self.Pord == 1: P = P[0]
        except:
            self.Pord = 1
            if P == 0:
                self.Pord = 0
        self.P = P

    def lum(self, z):
        """Luminosity evolution in mags."""
        if self.Qord == 0:
            return np.zeros_like(z)
        if self.Qord == 1:
            return self.Q*(z - par['z0'])
        if self.Qord == 2:
            return self.Q[0]*(z - par['z0']) + self.Q[1]*(z - par['z0'])**2
        
    def den(self, z):
        """Density evolution."""
        if self.Pord == 0:
            return np.ones_like(z)
        if self.Pord == 1:
            return 10**(0.4*self.P*z)
        if self.Pord == 2:
            return 10**(0.4*(self.P[0]*z + self.P[1]*z**2))

    def logden(self, z):
        """Density evolution."""
        if self.Pord == 0:
            return np.zeros_like(z)
        if self.Pord == 1:
            return ln10*0.4*self.P*z
        if self.Pord == 2:
            return ln10*0.4*(self.P[0]*z + self.P[1]*z*z)

    def lumden(self, z):
        """Luminosity density evolution."""
        if self.Qord == 0:
            lum = np.ones_like(z)
        if self.Qord == 1:
            lum = 10**(0.4*self.Q*z)
        if self.Qord == 2:
            lum = 10**(0.4*(self.Q[0]*z + self.Q[1]*z*z))
        if self.Pord == 0:
            den = 1.0
        if self.Pord == 1:
            den = 10**(0.4*self.P*z)
        if self.Pord == 2:
            den = 10**(0.4*(self.P[0]*z + self.P[1]*z*z))
        return lum*den

class lfData(object):
    """Class for LF data"""
    global par
    par = Parameters()
    
    def __init__(self, inFile):
        f = open(inFile, 'r')

        self.par = eval(f.readline())
        self.schec = eval(f.readline())
        self.lumdens = eval(f.readline())

        self.phiList = []
        for iz in range(self.par['nz']):
            phi = BinnedPhi(f=f)
            self.phiList.append(phi)

        f.close()

    def write(self, outFile):
        """Write LF to data file"""

        f = open(outFile, 'w')

        print >> f, self.par
        print >> f, self.schec
        print >> f, self.lumdens
    
        for iz in range(self.par['nz']):
            phi = self.phiList[iz]
            phi.save(f)

        f.close()

class Limits(object):
    """Class for implementing lower (inclusive) and upper (exclusive) limits."""
    
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def within(self, x):
        """Return 1 if x within limits, 0 otherwise."""
        if self.lo <= x < self.hi:
            return 1
        else:
            return 0
            
def dV(z, phi, dens, wmean):
    """Comoving volume element with optimal weighting."""

    sel = phi.sel([z])
    wt = 1.0/(1 + par['J3fac']*dens/wmean*sel)
    return cosmo.dV(z)*sel*wt

def zdm(dmod, kcoeff, zRange, ev=None):
    """Calculate redshift z corresponding to distance modulus dmod, solves
    dmod = m - M = 5*log10(dl(z)) + 25 + K(z) - Q(z-z0),
    ie. including k-correction and optional luminosity evolution Q.
    z is constrained to lie in range zRange."""

    if not ev:
        ev = Evol()
        
    tol = 1e-5

    zlo = zRange[0]
    zhi = zRange[1]
    z = 0.5*(zlo + zhi)
    err = zhi - zlo
    while(err > tol):
        if dmodk(z, kcoeff) - ev.lum(z) > dmod:
            zhi = z
        else:
            zlo = z
        z = 0.5*(zlo + zhi)
        err = zhi - zlo

    return z;

zdm_vec = np.vectorize(zdm)

def dm_resid(z, dmod, Q=0):
    """Returns residual between dm(z) and given dmod.
    Called by scipy.optimize.fsolve to find z corresponding to dmod.
    for fsolve to minimize.
    dmod = m - M = 5*log10(dl(z)) + 25 + K(z) - Q(z-z0),
    ie. including k-correction and optional luminosity evolution Q."""

    ev = Evol(Q)
    return dmodk(z) - ev.lum(z) - dmod

def dmodk(z, kcoeff):
    """Returns the K-corrected distance modulus 25 + 5*lg[dl(z)] + k(z)."""

    dm =  25 + 5*np.log10(cosmo.dl(z)) + np.polyval(kcoeff, z-par['z0'])
    return dm

def schecBin(schec, ev, phi):
    """Integrate Schechter fn over mag and redshift (Lin et al 1999 eqn 19).
    Calls quad twice to do double integral so we can test for warnings.
    NB epsabs and epsrel cannot be any smaller or get lots of warnings about
    rounding errors.  Not serious I think, as typical integrand values are
    ~10000"""

    print "Integrating parametric LF over mag bins"
    
    if par['idebug'] > 1:
        # Plot integrand
        M = np.arange(phi.absMin, phi.absMax, 0.01)
        z = np.arange(phi.zRange[0], phi.zRange[1], 0.001)
        phiv = np.zeros([M.size, z.size])
        for i in xrange(M.size):
            for j in xrange(z.size):
                phiv[i,j] = SchecEv(M[i], z[j], schec, ev)*cosmo.dV(z[j])
        plot.window()
        plt.imshow(phiv, aspect='auto', origin='lower',
                   extent=[phi.zRange[0], phi.zRange[1], phi.absMin, phi.absMax])
        plt.xlabel('z'); plt.ylabel('M'); plt.title('phi(M,z) dV(z)')
        plt.colorbar()
        plt.draw()
            
    schec_bin = np.zeros(phi.nbin)
    for ibin in xrange(phi.nbin):
        Mlo = phi.absMin + ibin*phi.absStep
        Mhi = phi.absMin + (ibin+1)*phi.absStep
        
        if phi.ngal[ibin] > 0:
            result = scipy.integrate.quad(
                phi_int, Mlo, Mhi, args=(schec, ev, phi),
                epsabs=0.01, epsrel=1e-3, full_output=1)
            if len(result) > 3 and par['idebug'] > 0:
                print 'schecBin: phi_int integral over ', Mlo, Mhi, ' gives result ', result

                # Plot integrand
                M = np.arange(Mlo, Mhi, 0.01)
                z = np.arange(phi.zRange[0], phi.zRange[1], 0.001)
                phiv = np.zeros([M.size, z.size])
                for i in xrange(M.size):
                    for j in xrange(z.size):
                        phiv[i,j] = SchecEv(M[i], z[j], schec, ev)*cosmo.dV(z[j])
                plot.window()
                plt.imshow(phiv, aspect='auto', origin='lower',
                           extent=[phi.zRange[0], phi.zRange[1], Mlo, Mhi])
                plt.xlabel('z'); plt.ylabel('M'); plt.title('phi(M,z) dV(z)')
                plt.colorbar()
                plt.draw()

                        
            result2 = scipy.integrate.quad(
                phi2_int, Mlo, Mhi, args=(schec, ev, phi),
                epsabs=0.01, epsrel=1e-3, full_output=1)
            if len(result) > 3 and par['idebug'] > 0:
                print 'schecBin: phi2_int integral over ', Mlo, Mhi, ' gives result ', result
            
            if result[0] > 0:
                schec_bin[ibin] = result2[0]/result[0]
    return schec_bin

def phi_int(M, schec, ev, phi):
    """Integral of phi(M,z) dV/dz dz"""

    zlo = zdm(par['appMin'] - M, phi.kmean, phi.zRange)
    zhi = zdm(par['appMax'] - M, phi.kmean, phi.zRange)
    if (zhi <= zlo):
        return 0.0
    
    result = scipy.integrate.quad(
        lambda z: SchecEv(M, z, schec, ev)*cosmo.dV(z), zlo, zhi,
        epsabs=0.01, epsrel=1e-3, full_output=1)
    ## if len(result) > 3:
    ##     pdb.set_trace()

    return result[0]

def phi2_int(M, schec, ev, phi):
    """Integral of phi^2(M,z) dV/dz dz"""

    zlo = zdm(par['appMin'] - M, phi.kmean, phi.zRange)
    zhi = zdm(par['appMax'] - M, phi.kmean, phi.zRange)
    if (zhi <= zlo):
        return 0.0
    
    result = scipy.integrate.quad(
        lambda z: SchecEv(M, z, schec, ev)**2*cosmo.dV(z), zlo, zhi,
        epsabs=0.01, epsrel=1e-3, full_output=1)
    ## if len(result) > 3:
    ##     pdb.set_trace()

    return result[0]

def SchecEv(M, z, schec, ev):
    """Evolving Schechter or Saunders function."""

    L = 10**(0.4*(schec['Mstar'][0] - ev.lum(z) - M))
    if schec.has_key('sigma'):
        ans = 0.4*ln10*schec['phistar'][0]*ev.den(z)*saunders_lf(L, schec['alpha'][0]+1, schec['sigma'][0])
        return ans

    ans = 0.4*ln10*schec['phistar'][0]*ev.den(z)*L**(schec['alpha'][0]+1)*math.exp(-L)
    if schec.has_key('beta'):
        Lt = 10**(0.4*(schec['Mt'][0] - M))
        ans *= (1 + Lt**schec['beta'][0])
    return ans

#------------------------------------------------------------------------------
# Stellar mass function
#------------------------------------------------------------------------------

def smf(infile='StellarMassesPlus.fits', logM_min = 11.5, logM_max = 12.5,
        fsmin=None, fsmax=None):
    """Number density of high stellar mass galaxies in redshift bins."""

    area = 48.0*(math.pi/180.0)**2  # actually area/3
    zbins = np.arange(0.003, 0.5, 0.06)
    nbins = len(zbins) - 1
    z_mean = np.zeros(nbins)
    nphi = np.zeros(nbins)
    phi = np.zeros(nbins)
    phiErr = np.zeros(nbins)
    
    # Read input file into structure
    hdulist = pyfits.open(infile)
    header = hdulist[1].header
    tbdata = hdulist[1].data
    H0 = 70.0
    omega_l = 0.7
    cosmo = CosmoLookup(H0, omega_l, (zbins[0],zbins[-1]))

    sel = (tbdata.field('nq') > 2)*(tbdata.field('r_petro') < 19.4)
    if fsmin:
        sel *= (tbdata.field('fluxscale') > fsmin)*(tbdata.field('fluxscale') < fsmax)
    tbdata = tbdata[sel]

    logmstar = tbdata.field('logmstar')
    if fsmin:
        logmstar += np.log10(tbdata.field('fluxscale'))
        
    sel = (logmstar >= logM_min)*(logmstar < logM_max)

    logmstar = logmstar[sel]
    z = tbdata.field('z_tonry')[sel]
    zmax = tbdata.field('zmax_19p4')[sel]
    hdulist.close()

    inds = np.digitize(z, zbins)
    for ibin in range(nbins):
        idx = (inds == ibin)
        nphi[ibin] = len(z[idx])
        if nphi[ibin] > 0:
            zlo = zbins[ibin]
            zhi = np.clip(zmax[idx], zlo, zbins[ibin+1])
            sel = zhi > zlo
            phi[ibin] = (1.0/(area*(cosmo.dm(zhi[sel])**3 - cosmo.dm(zlo)**3))).sum()
            phiErr[ibin] = phi[ibin]/np.sqrt(nphi[ibin])
            z_mean[ibin] = np.mean(z[idx][sel])

    idx = nphi > 0
    plt.clf()
    plt.errorbar(z_mean[idx], phi[idx], phiErr[idx], fmt='o')
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\phi(z)$')
    plt.semilogy(basey=10, nonposy='clip')
    plt.draw()
    
#------------------------------------------------------------------------------
# Routines to read, tabulate and plot LF output
#------------------------------------------------------------------------------

def jackErrsAll(root='lf_dp'):
    """Determine jackknife errors for all samples."""

    for band in 'ugriz':
        for colour in 'crb':
            obsFile = root + '_%c%c.dat' % (band, colour)
            fileList = root + '_%c%c_j?.dat' % (band, colour)
            outFile = root + '_%c%c_jackErrs.dat' % (band, colour)
            jackErrs(obsFile, fileList, outFile)
            
def jackErrs(obsFile, jackList, outFile):
    """Determine jackknife errors on LF estimates and parameters"""

    global par

    if len(jackList[0]) == 1:
        jackList = glob.glob(jackList)
    print jackList
    njack = len(jackList)
    estr = range(njack)
    lfList = []
    jfac = njack - 1
    iest = 0
    for inFile in jackList:
        lfList.append(lfData(inFile))

    lf = lfData(obsFile)

    # Schechter shape parameters
    for key in ('alpha', 'Mstar', 'Q', 'P', 'beta', 'Mt', 'phistar'):
        if lf.schec.has_key(key):
            err = np.sqrt(
                jfac*np.var(map(lambda i: lfList[i].schec[key][0], estr)))
            lf.schec[key] = (lf.schec[key][0], err)
    
    key = 'counts'
    if lf.schec.has_key(key):
        err = np.sqrt(
            jfac*np.var(map(lambda i: lfList[i].schec[key], estr)))
        lf.schec[key] = (lf.schec[key], err)
    
    # Luminosity density
    err = np.sqrt(jfac*np.var(map(lambda i: lfList[i].lumdens['ld0'][0],
                                    estr)))
    lf.lumdens['ld0'] = (lf.lumdens['ld0'][0], err)
    err = np.sqrt(jfac*np.var(map(lambda i: lfList[i].lumdens['ld'][0],
                                    estr), 0)).tolist()
    lf.lumdens['ld'] = (lf.lumdens['ld'][0], err)
    
    # Binned estimates
    for iz in range(lf.par['nz']):
        phi = lf.phiList[iz]

        for ibin in range(phi.nbin):
            # Only use bins for which all estimates have one or more galaxies,
            # otherwise average will be over- or under-estimated.
            ngmin = 1
            ngsum = 0
            for i in estr:
                ngmin = min(ngmin, lfList[i].phiList[iz].ngal[ibin])
                ngsum += lfList[i].phiList[iz].ngal[ibin]

            if ngmin > 0:
                phi.VmaxErr[ibin] = np.sqrt(jfac*np.var(
                    map(lambda i: lfList[i].phiList[iz].Vmax[ibin], estr)))
                phi.swmlErr[ibin] = np.sqrt(jfac*np.var(
                    map(lambda i: lfList[i].phiList[iz].swml[ibin], estr)))
            else:
                phi.ngal[ibin] = phi.Mav[ibin] = phi.Vmax[ibin] = phi.VmaxErr[ibin] = phi.swml[ibin] =  phi.swmlErr[ibin] = phi.sty[ibin] = 0
        lf.phiList[iz] = phi

    lf.write(outFile)

def simAv(fileList, outFile):
    """Mean and sd LF estimates and parameters from multiple simulations"""

    global par

    if len(fileList[0]) == 1:
        fileList = glob.glob(fileList)
    print fileList
    nest = len(fileList)
    estr = range(nest)
    lfList = []
    iest = 0
    for inFile in fileList:
        lfList.append(lfData(inFile))

    lf = lfList[0]

    # Schechter shape parameters
    for key in ('alpha', 'Mstar', 'Q', 'P', 'beta', 'Mt', 'phistar'):
        if lf.schec.has_key(key):
            val = np.mean(map(lambda i: lfList[i].schec[key][0], estr))
            err = np.std(map(lambda i: lfList[i].schec[key][0], estr))
            lf.schec[key] = (val, err)
    
    # Luminosity density
    val = np.mean(map(lambda i: lfList[i].lumdens['ld0'][0], estr))
    err = np.std(map(lambda i: lfList[i].lumdens['ld0'][0], estr))
    lf.lumdens['ld0'] = (val, err)

    val = np.mean(map(lambda i: lfList[i].lumdens['ld'][0], estr), 0).tolist()
    err = np.std(map(lambda i: lfList[i].lumdens['ld'][0], estr), 0).tolist()
    lf.lumdens['ld'] = (val, err)
    
    # Binned estimates
    for iz in range(lf.par['nz']):
        phi = lf.phiList[iz]
        phi.lnLikeRatio = np.mean(
            map(lambda i: lfList[i].phiList[iz].lnLikeRatio, estr))
        phi.nu = np.mean(map(lambda i: lfList[i].phiList[iz].nu, estr))
        phi.prob = np.mean(map(lambda i: lfList[i].phiList[iz].prob, estr))

        for ibin in range(phi.nbin):
            # Only use bins for which all estimates have one or more galaxies,
            # otherwise average will be over- or under-estimated.
            ngmin = 1
            ngsum = 0
            for i in estr:
                ngmin = min(ngmin, lfList[i].phiList[iz].ngal[ibin])
                ngsum += lfList[i].phiList[iz].ngal[ibin]

            if ngmin > 0:
                phi.Mav[ibin] = sum(
                    map(lambda i: lfList[i].phiList[iz].ngal[ibin]*
                        lfList[i].phiList[iz].Mav[ibin], estr))/ngsum
                phi.ngal[ibin] = sum(
                    map(lambda i: lfList[i].phiList[iz].ngal[ibin], estr))
                phi.Vmax[ibin] = np.mean(
                    map(lambda i: lfList[i].phiList[iz].Vmax[ibin], estr))
                phi.VmaxErr[ibin] = np.std(
                    map(lambda i: lfList[i].phiList[iz].Vmax[ibin], estr))
                phi.swml[ibin] = np.mean(
                    map(lambda i: lfList[i].phiList[iz].swml[ibin], estr))
                phi.swmlErr[ibin] = np.std(
                    map(lambda i: lfList[i].phiList[iz].swml[ibin], estr))
                phi.sty[ibin] = np.mean(
                    map(lambda i: lfList[i].phiList[iz].sty[ibin], estr))
            else:
                phi.ngal[ibin] = phi.Mav[ibin] = phi.Vmax[ibin] = phi.VmaxErr[ibin] = phi.swml[ibin] =  phi.swmlErr[ibin] = phi.sty[ibin] = 0
        lf.phiList[iz] = phi

    lf.write(outFile)

def evTables():
    """Read data files for evolving Schechter fns and generate tables
    for paper."""

    colour_desc = {'c': 'All', 'b': 'Blue', 'r': 'Red'}

    # Read least-squares fits
    lsqpar = {}
    f = open('lsqEv8Fit.dat', 'r')
    for line in f:
        file, Q, QErr, P, PErr = line.split()
        bc = file[7:9]
        lsqpar[bc] = (Q, QErr, P, PErr)
    f.close()
    
    # Evolving LF parameters
    outFile = 'evTable.tex'
    fout = open(outFile, 'w')
    fout.write(r'''
    \begin{math}
    \begin{array}{cccrrrrrrrrr}
    \hline
     &^{0.1}M_1 & ^{0.1}M_2 & 
    \multicolumn{1}{c}{N_{\rm gal}} & 
    \multicolumn{1}{c}{N_{\rm pred}} & 
    \multicolumn{1}{c}{\alpha} &
    \multicolumn{1}{c}{^{0.1}M^* - 5 \lg h} &
    \multicolumn{1}{c}{Q_{\rm par}} &
    \multicolumn{1}{c}{P_{\rm par}} &
    \multicolumn{1}{c}{Q_{\rm SWML}} &
    \multicolumn{1}{c}{P_{\rm SWML}} &
    \multicolumn{1}{c}{\phi^* \times 100} \\
    & \multicolumn{2}{c}{- 5 \lg h} & & & & & & & & & / \denunit\\
    \hline
    ''')

    for colour in 'cbr':
        fout.write(r'''\mbox{%s}\\
        ''' % colour_desc[colour])
        
        for band in 'ugriz':
            inFile = 'lf_ev8_%s%s_jackErrs.dat' % (band, colour)
##             inFile = 'lf_ev8_%s%s.dat' % (band, colour)
            bc = band + colour
            lf = lfData(inFile)
            schec = lf.schec
##             hdr1, hdr2, band, colour, nz, schec, lumdens, phiList = lf_read(inFile)

            fout.write(r'''%c &
            %5.1f & %5.1f &
            %5d & %5d \pm %5d & 
            %6.2f \pm %6.2f &
            %6.2f \pm %6.2f &
            %5.1f \pm %5.1f &
            %5.1f \pm %5.1f &
            %5.1f \pm %5.1f &
            %5.1f \pm %5.1f &
            %4.2f \pm %4.2f \\
            ''' % (band,
                   lf.par['absMinSTY'], lf.par['absMaxSTY'],
                   schec['ngal'], schec['counts'][0], schec['counts'][1],
                   schec['alpha'][0], schec['alpha'][1],
                   schec['Mstar'][0], schec['Mstar'][1],
                   schec['Q'][0], schec['Q'][1],
                   schec['P'][0], schec['P'][1],
                   float(lsqpar[bc][0]), float(lsqpar[bc][1]),
                   float(lsqpar[bc][2]), float(lsqpar[bc][3]),
                   100*schec['phistar'][0], 100*schec['phistar'][1]))

    fout.write(r'''
    \hline
    \end{array}
    \end{math}
    ''')

    fout.close()
                
    # Luminosity density - 4 zbins
    outFile = 'lumdens4.tex'
    fout = open(outFile, 'w')
    fout.write(r'''
    \begin{math}
    \begin{array}{cccccc}
    \hline
    \mbox{Redshift} & \mbox{Fit} & \mbox{0.0 -- 0.1} & \mbox{0.1 -- 0.2} & \mbox{0.2 -- 0.3} & \mbox{0.3 -- 0.5}\\
    \hline
    ''')

    for colour in 'cbr':
        fout.write(r'''\mbox{%s}\\
        ''' % colour_desc[colour])
        
        for band in 'ugriz':
            inFile = 'lf_ev8_%s%s_jackErrs.dat' % (band, colour)
##             inFile = 'lf_ev8_%s%s.dat' % (band, colour)
            lf = lfData(inFile)
            ld = lf.lumdens

            # Fix for highest z-bin for u-band red galaxies
            if colour == 'r' and band == 'u':
                ld['ld'][0][3] = 0
                ld['ld'][1][3] = 99e8
                
            fout.write(r'''%c &
            %4.2f \pm %4.2f &
            %4.2f \pm %4.2f &
            %4.2f \pm %4.2f &
            %4.2f \pm %4.2f &
            %4.2f \pm %4.2f\\
            ''' % (band, ld['ld0'][0]/1e8, ld['ld0'][1]/1e8,
                   ld['ld'][0][0]/1e8, ld['ld'][1][0]/1e8, 
                   ld['ld'][0][1]/1e8, ld['ld'][1][1]/1e8, 
                   ld['ld'][0][2]/1e8, ld['ld'][1][2]/1e8, 
                   ld['ld'][0][3]/1e8, ld['ld'][1][3]/1e8))

    fout.write(r'''
    \hline
    \end{array}
    \end{math}
    ''')

    fout.close()
                
    # Luminosity density - 8 zbins
    outFile = 'lumdens8.tex'
    fout = open(outFile, 'w')
    fout.write(r'''
    \begin{math}
    \begin{array}{cccccccccc}
    \hline
    \mbox{Redshift} & \mbox{Fit} & \mbox{0.0 -- 0.05} & \mbox{0.05 -- 0.1} &
    \mbox{0.1 -- 0.15} & \mbox{0.15 -- 0.2} &
    \mbox{0.2 -- 0.25} & \mbox{0.25 -- 0.3} &
    \mbox{0.3 -- 0.4} & \mbox{0.4 -- 0.5}\\
    \hline
    ''')

    for colour in 'cbr':
        fout.write(r'''\mbox{%s}\\
        ''' % colour_desc[colour])
        
        for band in 'ugriz':
            inFile = 'lf_ev8_%s%s_jackErrs.dat' % (band, colour)
            lf = lfData(inFile)
            ld = lf.lumdens

            line = r'{} & {:4.2f} \pm {:4.2f}'.format(band,
                                                      ld['ld0'][0]/1e8,
                                                      ld['ld0'][1]/1e8)
            for iz in range(8):
                ldVal = ld['ld'][0][iz]/1e8
                ldErr = ld['ld'][1][iz]/1e8
                if ldVal > 0 and ldErr < 9:
                    line += r' & {:4.2f} \pm {:4.2f}'.format(ldVal, ldErr)
                else:
                    line += r' & \mbox{---}'
            fout.write(line + r'\\')
                    
    fout.write(r'''
    \hline
    \end{array}
    \end{math}
    ''')

    fout.close()
                

def dpTable():
    """Read data files for dp Schechter fns and generate table for paper."""

    colour_desc = {'c': 'All', 'b': 'Blue', 'r': 'Red'}

    # DP Schechter parameters
    outFile = 'dpTable.tex'
    fout = open(outFile, 'w')
    fout.write(r'''
    \begin{math}
    \begin{array}{crrrrrrrrr}
    \hline
     & 
    \multicolumn{1}{c}{N_{\rm pred}} &
    \multicolumn{1}{c}{\alpha} & \multicolumn{1}{c}{\beta} &
    \multicolumn{1}{c}{^{0.1}M^* - 5 \lg h} &
    \multicolumn{1}{c}{^{0.1}M_t - 5 \lg h} &
    \multicolumn{1}{c}{\phi^* \times 100} & \multicolumn{1}{c}{P_{\rm fit}} &
    \multicolumn{1}{c}{{\rho_L}_{\rm fit}} &
    \multicolumn{1}{c}{{\rho_L}_{\rm sum}}\\
    & & & & & & /\denunit & & \multicolumn{2}{c}{/10^8 \ldenunit}\\
    \hline
    ''')

    for colour in 'cbr':
        fout.write(r'''\mbox{%s}\\
        ''' % colour_desc[colour])
        
        for band in 'ugriz':
            inFile = 'lf_dp_%s%s_jackErrs.dat' % (band, colour)
##             inFile = 'lf_dp_%s%s.dat' % (band, colour)
            lf = lfData(inFile)
            schec = lf.schec
            line = r'''%c & %5d \pm %5d & 
            %6.2f \pm %6.2f &
            %6.2f \pm %6.2f &
            %6.2f \pm %6.2f &
            %6.2f \pm %6.2f &
            %4.2f \pm %4.2f &
            %5.3f 
            ''' % (band, int(schec['counts'][0]), int(schec['counts'][1]),
                   schec['alpha'][0], np.mean(schec['alpha'][1]),
                   schec['beta'][0], np.mean(schec['beta'][1]),
                   schec['Mstar'][0], np.mean(schec['Mstar'][1]),
                   schec['Mt'][0], np.mean(schec['Mt'][1]),
                   100*schec['phistar'][0], 100*schec['phistar'][1],
                   lf.phiList[0].prob)
            ldVal = lf.lumdens['ld0'][0]/1e8
            ldErr = lf.lumdens['ld0'][1]/1e8
            if ldVal > 0 and ldErr < 9:
                line += r'& %3.2f \pm %3.2f '% (ldVal, ldErr)
            else:
                line += r' & \multicolumn{1}{c}{\mbox{---}}'
            ldVal = lf.lumdens['ld'][0][0]/1e8
            ldErr = lf.lumdens['ld'][1][0]/1e8
            if ldVal > 0 and ldErr < 9:
                line += r'& %3.2f \pm %3.2f'% (ldVal, ldErr)
            else:
                line += r' & \multicolumn{1}{c}{\mbox{---}}'
            fout.write(line + r'\\')
 
    fout.write(r'''
    \hline
    \end{array}
    \end{math}
    ''')

    fout.close()
                
def faintTable():
    """Read data files for faint low-z Schechter fns and generate table for paper."""

    colour_desc = {'c': 'All', 'b': 'Blue', 'r': 'Red'}

    # Faint Schechter parameters
    outFile = 'faintTable.tex'
    fout = open(outFile, 'w')
    fout.write(r'''
    \begin{math}
    \begin{array}{crrrrrrrrrr}
    \hline
    & ^{0.1}M_1 & ^{0.1}M_2 & 
    \multicolumn{1}{c}{N_{\rm gal}} & \multicolumn{1}{c}{N_{\rm pred}} &
    \multicolumn{1}{c}{\alpha} &
    \multicolumn{1}{c}{^{0.1}M^* - 5 \lg h} & \multicolumn{1}{c}{\phi^* \times 100} &
    \multicolumn{1}{c}{P_{\rm fit}} &
    \multicolumn{1}{c}{{\rho_L}_{\rm fit}} &
    \multicolumn{1}{c}{{\rho_L}_{\rm sum}}\\
    & \multicolumn{2}{c}{- 5 \lg h} & & & & & \multicolumn{1}{c}{/\denunit} & &
    \multicolumn{2}{c}{/10^8 \ldenunit}\\
    \hline
    ''')

    for colour in 'cbr':
        fout.write(r'''\mbox{%s}\\
        ''' % colour_desc[colour])
        
        for band in 'ugriz':
            inFile = 'lf_faint_%s%s_jackErrs.dat' % (band, colour)
##             inFile = 'lf_faint_%s%s.dat' % (band, colour)
            lf = lfData(inFile)
            schec = lf.schec
            fout.write(r'''%c & %5.1f & %5.1f & %5d & %5d \pm %5d & %6.2f \pm %6.2f & %6.2f \pm %6.2f & %4.2f \pm %4.2f & %5.3f &  %3.2f \pm %3.2f &  %3.2f \pm %3.2f\\
            ''' % (band, lf.par['absMinSTY'], lf.par['absMaxSTY'], schec['ngal'], int(schec['counts'][0]), int(schec['counts'][1]), schec['alpha'][0], schec['alpha'][1], schec['Mstar'][0], schec['Mstar'][1], 100*schec['phistar'][0], 100*schec['phistar'][1], lf.phiList[0].prob, lf.lumdens['ld0'][0]/1e8, lf.lumdens['ld0'][1]/1e8, lf.lumdens['ld'][0][0]/1e8, lf.lumdens['ld'][1][0]/1e8))

    fout.write(r'''
    \hline
    \end{array}
    \end{math}
    ''')

    fout.close()
                
# def tabulate(fileList = ('lf_ev4_?c.dat', 'lf_ev4_?b.dat', 'lf_ev4_?r.dat'),
#              outFile = 'lf_ev.txt'):
def tabulate(fileList = ('lf_faint_?c.dat', 'lf_faint_?b.dat', 'lf_faint_?r.dat'),
             outFile = 'lf_faint.txt'):
    """Tabulate Vmax & SWML LF estimates in simple table format."""

    globList = []
    for file in fileList:
        globList += glob.glob(file)
    fileList = globList
    print fileList

    f = open(outFile, 'w')
    
    # Read number of z-bins from first file
    lf = lfData(fileList[0])
    nz = lf.par['nz']
    
    for inFile in fileList:
        lf = lfData(inFile)
        band = lf.par['band']
        colour = lf.par['colour']
        for iz in range(lf.par['nz']):
            phi = lf.phiList[iz]
            print >> f, band, colour, phi.zRange
            print >> f, 'ngal    M     Vmax   VmaxErr     swml  swmlErr     sty'
            for ibin in range(phi.nbin):
                ng = int(phi.ngal[ibin]) # Without int get format error
                if ng > 0:
                    line = '{:4d} {:6.2f} {:4.2e} {:4.2e} {:4.2e} {:4.2e} {:4.2e}'.format(ng, phi.Mav[ibin], phi.Vmax[ibin], phi.VmaxErr[ibin], phi.swml[ibin], phi.swmlErr[ibin], phi.sty[ibin])
                    print >> f, line
            print >> f
    f.close()
    
def lumPlot(inFile, fitSchec=0):
    """Simple LF plot, one panel per window"""
    
    global par

    lf = lfData(inFile)
    par = lf.par

    fig = plt.figure(1)
    plt.clf()

    for iz in range(lf.par['nz']):
        phi = lf.phiList[iz]
        Mmin = phi.absMin
        Mmax = phi.absMax
        idx = phi.ngal > 0

        plt.semilogy(basey=10, nonposy='clip')
        plt.plot(phi.Mav[idx], phi.Vmax[idx], 'wo')
        plt.errorbar(phi.Mav[idx], phi.Vmax[idx], phi.VmaxErr[idx], fmt=None, ecolor='k')
        plt.errorbar(phi.Mav[idx], phi.swml[idx], phi.swmlErr[idx], fmt='ks')
        plt.plot(phi.Mav[idx], phi.sty[idx], 'k-')
        if fitSchec == 1:
            if iz == 0:
                afix = 0
                alpha = -1.0
            else:
                afix = 1
            fit = schecFit(phi.Mav[idx], phi.swml[idx], phi.swmlErr[idx],
                           (alpha, -18.4, -2), afix=afix)
            alpha = fit[0]
            Mstar = fit[2]
            phistar = 10**fit[4]
                
            plotSchec(alpha, Mstar, phistar, Mmin, Mmax)

        if fitSchec == 2:
            fit = saundFit(phi.Mav[idx], phi.swml[idx], phi.swmlErr[idx],
                           -1, -18.4, 0.8, -2)
            alpha = fit[0]
            Mstar = fit[1]
            sigma = fit[2]
            phistar = 10**fit[3]
                
            plotSaund(alpha, Mstar, sigma, phistar, Mmin, Mmax)
            print alpha, Mstar, sigma, phistar
        plt.axis([Mmin, Mmax, 1e-6, 1])
        plt.xlabel('$M_%s$' % lf.par['band'])
        plt.ylabel('$\phi$')
        title = '%5.3f' % phi.zRange[0] + ' < z < %5.3f' % phi.zRange[1]
        xt = Mmin + 0.1*(Mmax - Mmin)
        plt.text(xt, 0.1, title)

    plt.draw()
    
def evPlot(inFile, normSim=0, fitSchec=0):
    """Plot LF in redshift slices to show evolution.
    Specify normSim to normalise by simulated LF,
    fitSchec to fit Schechter fn to SWML points by lesat squares."""
    
    global par, cosmo
    
    inRoot = inFile.split('.')[0]
    plotFile = inRoot + '.png'
    lf = lfData(inFile)
    par = lf.par
    nz = par['nz']
    z0 = par['z0']
    try:
        cosmo = CosmoLookup(par['H0'], par['omega_l'], lf.schec['zRangeSTY'])
    except:
        cosmo = CosmoLookup(100.0, 0.0, [0.002, 0.5])
        
    if fitSchec:
        zmean = []
        alpha = []
        Mstar = []
        MstarErr = []
        lpstar = []
        lpstarErr = []
        chi2 = []
        v = []
        extent = []
        subplot = 121
    else:
        subplot = 111
        
    print lf.schec
##     Mmin = schec['absMinSTY']
##     Mmax = lf.par['absMaxSTY']
    Mmax = -13

    nrows = lf.par['nz']
    ncols = 1

    fig = plt.figure(1)
    plt.clf()
    grid = AxesGrid(fig, subplot, # similar to subplot(111)
                    nrows_ncols = (nrows, ncols), # creates nr*nc grid of axes
                    axes_pad=0.0, # pad between axes in inch.
                    aspect=False)

    # Avoid overlapping mag labels by specifying max of 5 major ticks
    # with 5 minor ticks per major tick
    nmajor = 5
    nminor = 25
    majorLocator = matplotlib.ticker.MaxNLocator(nmajor)
    minorLocator = matplotlib.ticker.MaxNLocator(nminor)

    ix = 0
    iy = 0
    for iz in range(nz):
        ax = grid[iz]
        phi = lf.phiList[iz]
        if phi.ntot > 0:
            idx = phi.ngal > 0
            Mmin = phi.absMin
            if iz == 0:
                zmin = phi.zRange[0]
            else:
                zmax = phi.zRange[1]

            # Normalise by simulated data
            if normSim:
                sim_schec = {'alpha': (lf.par['sim_alpha'], 0.0),
                             'Mstar': (lf.par['sim_Mstar'], 0.0),
                             'phistar': (lf.par['sim_phistar'], 0.0),
                             'Q': (lf.par['sim_Q'], 0.0),
                             'P': (0, 0)}
                ev = Evol(sim_schec['Q'])
                phiSim = schecBin(sim_schec, ev, phi)
                phi.Vmax[idx] /= phiSim[idx]
                phi.VmaxErr[idx] /= phiSim[idx]
                phi.swml[idx] /= phiSim[idx]
                phi.swmlErr[idx] /= phiSim[idx]
                phi.sty[idx] /= phiSim[idx]

                ylimits = [0.1, 1.9]
                xlabel = r'$M - 5 \log h$'
                ylabel = r'$\ \ \ \ \ \phi(M)/\phi_{\rm sim}(M)$'
                ax.plot([Mmin, Mmax], [1, 1], ':')
            else:
                ax.semilogy(basey=10, nonposy='clip')
                ylimits = [1e-7, 0.5]
                xlabel = r'$M_%s - 5 \log h$' % lf.par['band']
                ylabel = r'$\phi(M)/\ h^3$ Mpc$ ^{-3}$'
            
                # Low-z STY fit as dotted line
                if fitSchec == 0:
                    if iz == 0:
                        M0 = phi.Mav[idx].copy()
                        sty0 = phi.sty[idx].copy()
                    else:
                        ax.plot(M0, sty0, 'k:')
                    
            ax.plot(phi.Mav[idx], phi.Vmax[idx], 'wo')
            ax.errorbar(phi.Mav[idx], phi.Vmax[idx], phi.VmaxErr[idx],
                        fmt=None, ecolor='k')
            ax.errorbar(phi.Mav[idx], phi.swml[idx], phi.swmlErr[idx], fmt='bs')
            ax.plot(phi.Mav[idx], phi.sty[idx], 'k-', linewidth=2)

            # Optionally fit Schechter function to SWML estimates
            if fitSchec and len(phi.swmlErr[idx] > 0) > 2:
                if iz == 0:
                    afix = 0
                    alpha = -1.0
                else:
                    afix = 1
                fit = schecFit(phi.Mav[idx], phi.swml[idx], phi.swmlErr[idx],
                               (alpha, -18.4, -2), afix=afix)
                alpha = fit[0]
                zmean.append(phi.zmean)
                Mstar.append(fit[2])
                MstarErr.append(fit[3])
                lpstar.append(fit[4])
                lpstarErr.append(fit[5])
                chi2 = fit[6]
                nu = fit[7]
##                 chi2map.append(fit[8])
##                 v.append(fit[9])
##                 extent.append(fit[10])
            
                plotSchec(alpha, Mstar[-1], 10**lpstar[-1], Mmin, Mmax,
                          lineStyle='r-', axes=ax)

                # Low-z fit as dotted line
                if iz > 0:
                    plotSchec(alpha, Mstar[0], 10**lpstar[0], Mmin, Mmax,
                              lineStyle='k:', axes=ax)

                title1 = r'${:4.2f} < z < {:4.2f}$'.format(phi.zRange[0],
                                                           phi.zRange[1])
                title2 = r'$\chi^2 = {:4.1f}$, $\nu = {:2d}$'.format(chi2, nu)
                ax.text(0.5, 0.3, title1, transform = ax.transAxes)
                ax.text(0.5, 0.1, title2, transform = ax.transAxes)
            else:
                title = r'${:4.2f} < z < {:4.2f}$, $P_L = {:5.3f}$'.format(
                    phi.zRange[0], phi.zRange[1], phi.prob)
                ax.text(0.05, 0.8, title, transform = ax.transAxes)

        ax.axis([Mmin, Mmax - 0.01] + ylimits)

        if iy == nrows - 1:
            ax.set_xlabel(xlabel)
        if iy == nrows/2:
            ax.set_ylabel(ylabel)
        ix += 1
        if ix >= ncols:
            iy += 1
            ix = 0

    plt.draw()
##    plot.save(plotFile)

    if fitSchec:
        zmean = np.array(zmean)
        zz = np.linspace(zmin, zmax)

        ax = plt.subplot(222)
        ax.tick_params(labelleft=0, labelright=1)
        yerr = np.transpose(np.array(MstarErr))
        plt.errorbar(zmean, Mstar, yerr, fmt='o')
        w = 1.0/(0.5*(yerr[0] + yerr[1]))**2
        a, aVar, b, bVar = wtdLineFit(zmean - z0, Mstar, w)
        plt.plot([zmin, zmax], [a + b*(zmin - z0), a + b*(zmax - z0)])
        print 'M* = ', a, ' + ', b, '*z'
        Q = -b
        QErr = math.sqrt(bVar)
        a, cov = wtdPolyFit(zmean - z0, Mstar, w, 2)
        plt.plot(zz, np.polyval(a, zz - z0))
        print a
        plt.ylabel(r'$M^*$')
        plt.xlabel(r'$z$')
        plt.text(0.05, 0.1, r'$Q = {:4.1f} \pm {:4.1f}$'.format(Q, QErr),
                 transform = ax.transAxes)
##         plt.axis([zmin, zmax, -21, -17])
        
        ax = plt.subplot(224)
        ax.tick_params(labelleft=0, labelright=1)
        yerr = np.transpose(np.array(lpstarErr))
        plt.errorbar(zmean, lpstar, yerr, fmt='o')
        w = 1.0/(0.5*(yerr[0] + yerr[1]))**2
        a, aVar, b, bVar = wtdLineFit(zmean, lpstar, w)
        plt.plot([zmin, zmax], [a + b*zmin, a + b*zmax])
        print 'log phi* = ', a, ' + ', b, '*z'
        P = b
        PErr = math.sqrt(bVar)
        a, cov = wtdPolyFit(zmean, lpstar, w, 2)
        plt.plot(zz, np.polyval(a, zz))
        print a
        plt.ylabel(r'$\lg \phi^*$')
        plt.xlabel(r'$z$')
        plt.text(0.05, 0.1, r'$P = {:4.1f} \pm {:4.1f}$'.format(P, PErr),
                 transform = ax.transAxes)
##         plt.axis([zmin, zmax, -5, -1])
        plt.draw()
        return Q, QErr, P, PErr

def swml_sty_residPlot(inFile):
    """Plot SWML - STY residuals in redshift slices."""
    
    inRoot = inFile.split('.')[0]
    plotFile = inRoot + '.png'
    lf = lfData(inFile)
    par = lf.par
##     Mmin = schec['absMinSTY']
    Mmax = lf.par['absMaxSTY']

    # Work out how many plotting panels to use
    nrows = lf.par['nz']
    ncols = 1
    fig = plt.figure(1)
    plt.clf()
    grid = AxesGrid(fig, 111, # similar to subplot(111)
                    nrows_ncols = (nrows, ncols), # creates nr*nc grid of axes
                    axes_pad=0.0, # pad between axes in inch.
                    aspect=False)

    # Avoid overlapping mag labels by specifying max of 5 major ticks
    # with 5 minor ticks per major tick
    nmajor = 5
    nminor = 25
    majorLocator = matplotlib.ticker.MaxNLocator(nmajor)
    minorLocator = matplotlib.ticker.MaxNLocator(nminor)

    ix = 0
    iy = 0
    for iz in range(lf.par['nz']):
        ax = grid[iz]
        phi = lf.phiList[iz]
        Mmin = phi.absMin
        if iz == 0:
            zmin = phi.zRange[0]
        else:
            zmax = phi.zRange[1]

        idx = phi.ngal > 0
        ax.semilogy(basey=10, nonposy='clip')
        ax.errorbar(phi.Mav[idx], phi.swml[idx]/phi.sty[idx],
                    phi.swmlErr[idx]/phi.sty[idx], fmt='ks')

        ax.axis([Mmin, Mmax - 0.01, 0.5, 2])
        title = '%5.3f' % phi.zRange[0] + ' < z < %5.3f' % phi.zRange[1]
        ax.text(0.05, 0.8, title, transform = ax.transAxes)
        ax.text(0.7, 0.2, '%5.3f' % phi.prob, transform = ax.transAxes)
##         ax.xaxis.set_major_locator(majorLocator)
##         ax.xaxis.set_minor_locator(minorLocator)

        if iy == nrows - 1:
            ax.set_xlabel(r'$M_%s - 5 \log h$' % lf.par['band'])
        if ix == 0:
            ax.set_ylabel(r'$\phi(M)\ h^3$ Mpc$ ^{-3}$')
        ix += 1
        if ix >= ncols:
            iy += 1
            ix = 0

    plt.draw()

def lsqEvFit(fileList = ('lf_ev8_??.dat'), outFile='lsqEv8Fit.dat'):
    """Fit evolution parameters Q, P to swml estimates by least-squares."""
    if len(fileList[0]) == 1:
        fileList = glob.glob(fileList)
    print fileList

    fout = open(outFile, 'w')
    
    for inFile in fileList:
        Q, QErr, P, PErr = evPlot(inFile, fitSchec=1)
        print >> fout, inFile, Q, QErr, P, PErr
    fout.close()

def evPlotAll(fileList = ('lf_ev4_?c.dat', 'lf_ev4_?b.dat', 'lf_ev4_?r.dat'),
              fitSchec=0, likeCont=0, phiRange=(1e-7, 0.05)):
    """Plot LF in redshift slices for all five bands on a single plot."""

##     if len(fileList[0]) == 1:
##         fileList = glob.glob(fileList)
##     print fileList
    globList = []
    for file in fileList:
        globList += glob.glob(file)
    fileList = globList
    print fileList

    # M* and lg phi* limits for each band
    mslimit = ((-20.5, -16.8), (-20.5, -18.8), (-21.5, -19.5), (-21.5, -19.99),
               (-22.01, -20.5))
    lplimit = ((-4, -1.1), (-3.01, -1.5), (-2.5, -1.7), (-2.5, -1.7),
               (-2.5, -1.7))
    majorLocator = matplotlib.ticker.MultipleLocator(1.0)
    
    # Read number of z-bins from first file
    lf = lfData(fileList[0])
    nz = lf.par['nz']
    
    fig = plt.figure(1, figsize=(12, 8))
    plt.clf()
    grid = AxesGrid(fig, 111, # similar to subplot(111)
                    nrows_ncols = (nz, 5), # creates nz * nband grid of axes
                    axes_pad=0.0, # pad between axes in inch.
                    share_all=False, aspect=False)
    if likeCont:
        insert = []
        for ax in grid:
            insert.append(inset_axes(ax, width="35%", height="35%", loc=4,
                                     bbox_to_anchor=(0, 0.05, 0.98, 1),
                                     bbox_transform=ax.transAxes))        

    for inFile in fileList:
        lf = lfData(inFile)
    
        iband = lf.par['iband']
        fmt = 'ks'
        fmtv = 'ws'
        contline = 'k-'
        dashline = 'k--'
        dotline = 'k:'
        scale = 1.0
        ytext = 0.3
        textColour = 'k'
        try:
            if lf.par['colour'][0] == 'b' or lf.par['gal_index_r'][0] < 1:
                fmt = 'bo'
                fmtv = 'wo'
                contline = 'b-'
                dashline = 'b--'
                dotline = 'b:'
                scale = 0.1
                ytext = 0.2
                textColour = 'b'
        except:
            pass
        
        try:
            if lf.par['colour'][0] == 'r' or lf.par['gal_index_r'][0] > 1:
                fmt = 'r^'
                fmtv = 'w^'
                contline = 'r-'
                dashline = 'r--'
                dotline = 'r:'
                scale = 0.1
                ytext = 0.1
                textColour = 'r'
        except:
            pass

        Mmin = lf.par['absMinSTY']
        Mmax = lf.par['absMaxSTY']

        for iz in range(lf.par['nz']):
            phi = lf.phiList[iz]
            if phi.ntot > 0:

                idx = phi.ngal > 0
                iplot = 5*iz + iband
                ax = grid[iplot]
                ax.semilogy(basey=10, nonposy='clip')

                ax.errorbar(phi.Mav[idx], scale*phi.Vmax[idx],
                            scale*phi.VmaxErr[idx], fmt=fmtv)
                ax.errorbar(phi.Mav[idx], scale*phi.swml[idx],
                            scale*phi.swmlErr[idx], fmt=fmt)
                ax.plot(phi.Mav[idx], scale*phi.sty[idx], contline)
                # Low-z STY fit as dotted line
                if iz == 0:
                    M0 = phi.Mav[idx].copy()
                    sty0 = scale*phi.sty[idx].copy()
                else:
                    ax.plot(M0, sty0, dotline)
            
                # Optionally fit Schechter function to SWML estimates
                if fitSchec and len(phi.swmlErr[idx] > 0) > 2:
                    if iz == 0:
                        afix = 0
                        alpha = -1.0
                    else:
                        afix = 1
                    fit = schecFit(phi.Mav[idx], phi.swml[idx], phi.swmlErr[idx],
                                   (alpha, -18.4, -2), afix, likeCont)
                    alpha = fit[0]
                    Mstar = fit[2]
                    lpstar = fit[4]
                    plotSchec(alpha, Mstar, scale*10**lpstar, Mmin, Mmax,
                              lineStyle=dashline, axes=ax)

                    # M*, phi* contours as inset
                    if likeCont:
                        axi = insert[iplot]
                        axi.contour(fit[8], fit[9], aspect='auto',
                            colors=textColour, origin='lower', extent=fit[10])
                        axi.axis((mslimit[iband][0], mslimit[iband][1],
                                  lplimit[iband][0], lplimit[iband][1]))
##                         axi.xaxis.set_major_locator(majorLocator)
##                         axi.yaxis.set_major_locator(majorLocator)
                        axi.locator_params(nbins=3)
                        axi.tick_params(labelsize='small')
##                         if lf.par['colour'][0] == 'c':
##                             axi.set_xlabel(r'$M^*$')
##                             axi.set_ylabel(r'$\lg \phi^*$')
                            
#            ax.axis([Mmin, Mmax - 0.01, 1e-7, 0.05])
            ax.axis((Mmin, Mmax - 0.01) + phiRange)
            ax.locator_params('x', nbins=5)

##             ax.text(0.7, ytext, '%5.3f' % phi.prob, transform = ax.transAxes,
##                     color=textColour)
            if iz == 3:
                ax.set_xlabel(r'$^{{{:2.1f}}}M_{} - 5 \log h$'.format(lf.par['z0'], lf.par['band']))
            if iband == 0 and lf.par['colour'] == 'c':
                ax.set_ylabel(r'$\phi(M)/\ h^3$ Mpc$ ^{-3}$')
                title = '%3.1f' % phi.zRange[0] + ' < z < %3.1f' % phi.zRange[1]
                ax.text(0.1, 0.9, title, transform = ax.transAxes)

    plt.draw()
    plt.savefig('lf_ev.pdf', bbox_inches='tight')
    
def ev_comp(fileList = ('lf_ev3_?c.dat')):
    """Compare LF in three redshift slices with Blanton et al 2003,
    VVDS (Ilbert et al 2005) and zCOSMOS (Zucca et al 2009). """

    if len(fileList[0]) == 1:
        fileList = glob.glob(fileList)
    print fileList

    # Read VVDS and zCOSMOS data
    file = os.environ['HOME'] + '/Documents/Research/LFdata/ilbert2005.dat'
    vvds = np.loadtxt(file)
    file = os.environ['HOME'] + '/Documents/Research/LFdata/zucca2009.dat'
    zcos = np.loadtxt(file)
    
    nz = 3
    label = (r'$^{0.1}u, ^{0.0}U + 0.25$', r'$^{0.1}g, ^{0.0}B$',
             r'$^{0.1}r, ^{0.0}V$', r'$^{0.1}i, ^{0.0}R$',
             r'$^{0.1}z, ^{0.0}I$')
    
    fig = plt.figure(1, figsize=(12, 8))
    plt.clf()
    grid = AxesGrid(fig, 111, # similar to subplot(111)
                    nrows_ncols = (nz, 5), # creates nz * nband grid of axes
                    axes_pad=0.0, # pad between axes in inch.
                    share_all=False, aspect=False)

    for inFile in fileList:
        lf = lfData(inFile)
    
        iband = lf.par['iband']
        fmt = 'ks'
        fmtv = 'ws'
        contline = 'k-'
        dotline = 'k:'
        scale = 1.0
        ytext = 0.3
        textColour = 'k'

        Mmin = lf.par['absMinSTY']
        Mmax = lf.par['absMaxSTY']

        for iz in range(lf.par['nz']):
            phi = lf.phiList[iz]
            if phi.ntot > 0:

                idx = phi.ngal > 0
                iplot = 5*iz + iband
                ax = grid[iplot]
                ax.semilogy(basey=10, nonposy='clip')

                ax.errorbar(phi.Mav[idx], scale*phi.Vmax[idx],
                            scale*phi.VmaxErr[idx], fmt=fmtv)
                ax.errorbar(phi.Mav[idx], scale*phi.swml[idx],
                            scale*phi.swmlErr[idx], fmt=fmt)
                ax.plot(phi.Mav[idx], scale*phi.sty[idx], contline)

                # Low-z STY fit as dotted line
##                 if iz == 0:
##                     M0 = phi.Mav[idx].copy()
##                     sty0 = scale*phi.sty[idx].copy()
##                 else:
##                     ax.plot(M0, sty0, dotline)

                # VVDS comparison.  See Ilbert et al sec 5.1 for u dm offset
                # Dashed over whole mag range, solid over fitted
                if iband == 0:
                    dm = 0.25
                else:
                    dm = 0.0
                iv = 3*iband + iz

                plotSchec(vvds[iv][5], vvds[iv][6] + dm, vvds[iv][7],
                          Mmin, Mmax, lineStyle='r--', axes=ax)
                plotSchec(vvds[iv][5], vvds[iv][6] + dm, vvds[iv][7],
                          vvds[iv][3], vvds[iv][4] + dm, lineStyle='r',
                          axes=ax)
                
                # zCOSMOS comparison (B-band only)
                # Dashed over whole mag range, solid over fitted
                if iband == 1:
                    h = 0.7
                    alpha = zcos[iz][6]
                    Mstar = zcos[iz][7] - 5*math.log10(h)
                    phistar = zcos[iz][8]/(1000*h**3)
                    Mlo = zcos[iz][2]
                    Mhi = zcos[iz][3]
                    print alpha, Mstar, phistar
                    
                    plotSchec(alpha, Mstar, phistar, Mmin, Mmax,
                              lineStyle='g--', axes=ax)
                    plotSchec(alpha, Mstar, phistar, Mlo, Mhi, 
                              lineStyle='g', axes=ax)
                
                if iz == 0:
                    # Plot Blanton et al 2003 data
                    file = os.environ['HOME'] + '/Documents/Research/LFdata/blanton2003/{}.dat'.format('ugriz'[iband])
                    blanton = np.loadtxt(file)
                    ax.plot(blanton[:,0], blanton[:,1], 'b', linewidth=1)

            ax.axis([Mmin, Mmax - 0.01, 1e-7, 0.05])
            ax.locator_params('x', nbins=5)

##             ax.text(0.5, 0.1, label[iband], transform = ax.transAxes)

            if iz == nz-1:
                ax.set_xlabel(r'$^{{{:2.1f}}}M_{} - 5 \log h$'.format(lf.par['z0'], lf.par['band']))
            if iband == 0 and lf.par['colour'] == 'c':
                ax.set_ylabel(r'$\phi(M)/\ h^3$ Mpc$ ^{-3}$')
                title = r'$%3.1f' % phi.zRange[0] + ' < z < %3.1f$' % phi.zRange[1]
                ax.text(0.1, 0.9, title, transform = ax.transAxes)

    plt.draw()
    plt.savefig('ev_comp.png', bbox_inches='tight')
    
def evPlotMulti(fileList = ('lf_ev_??.dat')):
    """Plot LF in redshift slices for various samples."""

    mlo = [-23, -24, -24, -25, -25]
    mhi = [-10, -11, -12, -13, -13]
    if len(fileList[0]) == 1:
        fileList = glob.glob(fileList)
    print fileList

    # Read number of z-bins from first file
    lf = lfData(fileList[0])
    nz = lf.par['nz']
    
##     f, axarr = plt.subplots(nz, 1, num=1, axes_pad=0.0)
    fig = plt.figure(1)
    plt.clf()
    grid = AxesGrid(fig, 111, # similar to subplot(111)
                    nrows_ncols = (nz, 1), # creates nz * nband grid of axes
                    axes_pad=0.0, # pad between axes in inch.
                    share_all=False, aspect=False)

    # Avoid overlapping mag labels by only labelling every 2nd mag
    majorLocator = matplotlib.ticker.MultipleLocator(2.0)
    minorLocator = matplotlib.ticker.MultipleLocator(0.2)
##     majorLocator = matplotlib.ticker.MaxNLocator(5)
##     minorLocator = matplotlib.ticker.MaxNLocator(25)

    colour = ('k', 'b', 'g', 'r', 'y', 'm', 'c')
    ifile = 0
    for inFile in fileList:
        lf = lfData(inFile)
    
        iband = lf.par['iband']

        Mmin = lf.par['absMinSTY']
        Mmax = lf.par['absMaxSTY']

        for iz in range(nz):
            phi = lf.phiList[iz]
            if phi.ntot > 0:

                idx = phi.ngal > 0
##                 ax = axarr[iz]
##                 iplot = 5*iz + iband
                ax = grid[iz]
                ax.semilogy(basey=10, nonposy='clip')

                ax.errorbar(phi.Mav[idx], phi.swml[idx], phi.swmlErr[idx],
                            fmt=colour[ifile]+'o')
                ax.plot(phi.Mav[idx], phi.sty[idx], colour[ifile]+'-')
                # Low-z STY fit as dotted line
                if iz == 0:
                    M0 = phi.Mav[idx].copy()
                    sty0 = phi.sty[idx].copy()
                else:
                    ax.plot(M0, sty0, colour[ifile]+':')
            
            ax.axis([mlo[iband], mhi[iband] - 0.01, 1e-7, 0.05])
##             ax.xaxis.set_major_locator(majorLocator)
##             ax.xaxis.set_minor_locator(minorLocator)

##             ax.text(0.7, ytext, '%5.3f' % phi.prob, transform = ax.transAxes,
##                     color=textColour)
            if iz == nz-1:
                ax.set_xlabel(r'$M_%s - 5 \log h$' % lf.par['band'])
            if ifile == 0:
                ax.set_ylabel(r'$\phi(M)\ h^3$ Mpc$ ^{-3}$')
                title = '%3.1f' % phi.zRange[0] + ' < z < %3.1f' % phi.zRange[1]
                ax.text(0.1, 0.9, title, transform = ax.transAxes)
        ifile += 1
    plt.draw()
    plt.savefig('lf_ev.eps', bbox_inches='tight')
    
def faintPlotAll(fileList=None):
    """Plot faint-end LFs for all five bands on a single plot."""

    if fileList == None:
        fileList = glob.glob('lf_faint_??.dat') + glob.glob('lf_dp_??.dat')
    print fileList

    fdir = '/pactusers/loveday/Documents/tex/papers/gama/lf1/'

    fig = plt.figure(1, figsize=(8, 10))
    plt.clf()

    grid = AxesGrid(fig, 111, # similar to subplot(111)
                    nrows_ncols = (5, 1), # creates 2x2 grid of axes
                    axes_pad=0.0, # pad between axes in inch.
                    share_all=False, aspect=False)

    Mmin = -24
    Mmax = -10
    band = 'ugriz'
    
    # Blanton et al 2005 comparison (plot first so doesn't overplot)

    # Correction for bandpass shift
    lf = lfData(fileList[0])
    delta_M = 2.5*math.log10(1 + lf.par['z0'])
    
    for iband in range(5):
        ax = grid[iband]
        blanton = np.loadtxt(fdir + 'Blanton2005' + 'ugriz'[iband] + '.dat')
        ax.plot(blanton[:,0] + delta_M, blanton[:,1], 'wD')
        ax.errorbar(blanton[:,0] + delta_M, blanton[:,1], blanton[:,2], 
                    fmt=None, ecolor='k')
        ax.text(0.05, 0.8, band[iband], transform = ax.transAxes)
        if iband == 2:
            ax.set_ylabel(r'$\phi(M)/\ h^3$ Mpc$ ^{-3}$')
        
    for inFile in fileList:
        lf = lfData(inFile)
    
        fmt = 'ks'
        fmtv = 'ws'
        contline = 'k-'
        dotline = 'k:'
        scale = 1.0
        try:
            if lf.par['colour'][0] == 'b':
                fmt = 'bo'
                fmtv = 'wo'
                contline = 'b-'
                dotline = 'b:'
                scale = 0.1
        except:
            pass
        try:
            if lf.par['colour'][0] == 'r':
                fmt = 'r^'
                fmtv = 'w^'
                contline = 'r-'
                dotline = 'r:'
                scale = 0.1
        except:
            pass

        phi = lf.phiList[0]

        idx = phi.ngal > 0
        ax = grid[lf.par['iband']]
        ax.semilogy(basey=10, nonposy='clip')

        if lf.schec['nShapePar'] > 2:
            # DP estimate
            ax.plot(phi.Mav[idx], scale*phi.Vmax[idx], fmtv)
            ax.errorbar(phi.Mav[idx], scale*phi.swml[idx],
                        scale*phi.swmlErr[idx], fmt=fmt)
            ax.plot(phi.Mav[idx], scale*phi.sty[idx], contline)
        else:
            # Show std Schechter fit as dotted line
            ax.plot(phi.Mav[idx], scale*phi.sty[idx], dotline)
        ax.set_xlabel(r'$^{{{:2.1f}}}M - 5 \log h$'.format(lf.par['z0']))
        ax.axis([Mmin, Mmax, 1e-7, 0.9])


    plt.draw()
    plt.savefig('lf_faint.pdf', bbox_inches='tight')
    
def evCompPlot(fileList):
    """Plot LF in redshift slices for two or more files.  fileList may be a
    sequence of filenames or a single name including wildcards."""
    
    if len(fileList[0]) == 1:
        fileList = glob.glob(fileList)
    print fileList

    fig = plt.figure(1)
    plt.clf()
    grid = AxesGrid(fig, 111, # similar to subplot(111)
                    nrows_ncols = (4, 1), # creates 2x2 grid of axes
                    axes_pad=0.0, # pad between axes in inch.
                    aspect=False)

    # Avoid overlapping mag labels by specifying max of 5 major ticks
    # with 5 minor ticks per major tick
    nmajor = 5
    nminor = 25
    majorLocator = matplotlib.ticker.MaxNLocator(nmajor)
    minorLocator = matplotlib.ticker.MaxNLocator(nminor)

    for inFile in fileList:
        lf = lfData(inFile)
        Mmin = lf.par['absMin']
        Mmax = lf.par['absMax']

        for iz in range(lf.par['nz']):
            phi = lf.phiList[iz]
            ax = grid[iz]

            idx = phi.ngal > 0
            ax.errorbar(phi.Mav[idx], phi.swml[idx], phi.swmlErr[idx])
            ax.plot(phi.Mav[idx], phi.sty[idx], '-')

            ax.semilogy(basey=10, nonposy='clip')
            ax.axis([Mmin, Mmax, 1e-6, 1])
            title = '%5.3f' % phi.zRange[0] + ' < z < %5.3f' % phi.zRange[1]
            ax.text(0.1, 0.9, title, transform = ax.transAxes)
            ax.xaxis.set_major_locator(majorLocator)
            ax.xaxis.set_minor_locator(minorLocator)

            if iz == 3:
                ax.set_xlabel('$M$')
            ax.set_ylabel('$\phi$')
            
    plt.draw()
    
def evColourPlot(fileList):
    """Compare colours, SWML/STY only."""
    
    if len(fileList[0]) == 1:
        fileList = glob.glob(fileList)
    print fileList

    fig = plt.figure(1)
    plt.clf()
    grid = AxesGrid(fig, 111, # similar to subplot(111)
                    nrows_ncols = (2, 2), # creates 2x2 grid of axes
                    axes_pad=0.0, # pad between axes in inch.
                    aspect=False)

    # Avoid overlapping mag labels by specifying max of 5 major ticks
    # with 5 minor ticks per major tick
    nmajor = 5
    nminor = 25
    majorLocator = matplotlib.ticker.MaxNLocator(nmajor)
    minorLocator = matplotlib.ticker.MaxNLocator(nminor)

    for inFile in fileList:
        fmt = 'ks'
        linestyle = 'k-'
        scale = 1.0
        if inFile.find('Blue') > -1:
            fmt = 'bo'
            linestyle = 'b-'
            scale = 0.1
        if inFile.find('Red') > -1:
            fmt = 'r^'
            linestyle = 'r-'
            scale = 0.1
        f = open(inFile, 'r')
        data = f.readline()
        data = f.readline().split()
        band = data[1]
        nz = int(data[2])
        zmean = np.zeros(nz)
    
        schec = f.readline()
        lumdens = f.readline()

        ix = 0
        iy = 0
        for iz in range(nz):
            zRange, zmean[iz] = eval(f.readline())
            data = f.readline().split()
            nbin = int(data[0])
            Mmin = float(data[1])
            Mmax = float(data[2])
            ngal = np.zeros(nbin)
            Mav = np.zeros(nbin)
            Vmax = np.zeros(nbin)
            VmaxErr = np.zeros(nbin)
            swml = np.zeros(nbin)
            swmlErr = np.zeros(nbin)
            sty = np.zeros(nbin)
            for ibin in range(nbin):
                data = f.readline().split()
                ngal[ibin] = int(data[0])
                Mav[ibin] = float(data[1])
                Vmax[ibin] = scale*float(data[2])
                VmaxErr[ibin] = scale*float(data[3])
                swml[ibin] = scale*float(data[4])
                swmlErr[ibin] = scale*float(data[5])
                sty[ibin] = scale*float(data[6])

            idx = ngal > 0
            ax.semilogy(basey=10, nonposy='clip')
            ax.errorbar(Mav[idx], swml[idx], swmlErr[idx], fmt=fmt)
            ax.plot(Mav[idx], sty[idx], linestyle)

            ax.axis([Mmin, Mmax, 1e-7, 0.1])
            title = '%5.3f' % zRange[0] + ' < z < %5.3f' % zRange[1]
            ax.text(0.1, 0.9, title, transform = ax.transAxes)
            ax.xaxis.set_major_locator(majorLocator)
            ax.xaxis.set_minor_locator(minorLocator)

            if iy == 1:
                ax.set_xlabel('$M_%s$' % band)
            if ix == 0:
                ax.set_ylabel('$\phi$')
            ix += 1
            if ix > 1:
                iy += 1
                ix = 0
            
    plt.draw()
    f.close()
    
def faintCompPlot(fileList):
    """Plot LF for lowest redshift slice for a list of files,
    separate panels for Vmax and SWML.  fileList may be a
    sequence of filenames or a single name including wildcards."""

    if len(fileList[0]) == 1:
        fileList = glob.glob(fileList)
    print fileList
    
    fig = plt.figure(1)
    plt.clf()
    grid = AxesGrid(fig, 111, # similar to subplot(111)
                    nrows_ncols = (2, 1), # creates 1x2 grid of axes
                    axes_pad=0.0, # pad between axes in inch.
                    aspect=False)
    colour = 'wk'

    ifile = 0
    for inFile in fileList:
        lf = lfData(inFile)
        Mmin = lf.par['absMin']
        Mmax = lf.par['absMax']
        phi = lf.phiList[0]

        idx = phi.ngal > 0
        grid[0].semilogy(basey=10, nonposy='clip')
        grid[0].plot(phi.Mav[idx], phi.Vmax[idx], colour[ifile]+'o')
        grid[0].errorbar(phi.Mav[idx], phi.Vmax[idx], phi.VmaxErr[idx],
                         fmt=None, ecolor='k')
        grid[0].plot(phi.Mav[idx], phi.sty[idx], 'k-')
        
        grid[1].semilogy(basey=10, nonposy='clip')
        grid[1].plot(phi.Mav[idx], phi.swml[idx], colour[ifile]+'o')
        grid[1].errorbar(phi.Mav[idx], phi.swml[idx], phi.swmlErr[idx],
                         fmt=None, ecolor='k')
        grid[1].plot(phi.Mav[idx], phi.sty[idx], 'k-')
        ifile += 1

    grid[0].axis([Mmin, Mmax, 1e-5, 0.9])
    grid[0].set_ylabel(r'$\phi(M)/\ h^3$ Mpc$ ^{-3}$')
    grid[0].text(0.1, 0.9, '(a) 1/Vmax', transform=grid[0].transAxes)

    grid[1].set_xlabel(r'$^{{{:2.1f}}}M_{} - 5 \log h$'.format(lf.par['z0'],
                                                               lf.par['band']))
    grid[1].set_ylabel(r'$\phi(M)/\ h^3$ Mpc$ ^{-3}$')
    grid[1].axis([Mmin, Mmax, 1e-5, 0.9])
    grid[1].text(0.1, 0.9, '(b) SWML', transform=grid[1].transAxes)

    plt.draw()
    
def faintPlotMulti(fileList, renorm=0, like=0, ylo=1e-6, yhi=0.01):
    """Plot LF for lowest redshift slice for a list of files, SWML only.
    fileList may be a sequence of filenames or a single name including
    wildcards.
    If renorm=1, normalise subsequent LFs to first.
    If like=1, include likelihood contour plot as inset."""

    if len(fileList[0]) == 1:
        fileList = glob.glob(fileList)
    print fileList
    
    colour = 'wkbgrymckbgrymc'
    ls = ('--', '-', '-.', ':')
    yt = (0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)
    fig = plt.figure(1)
    plt.clf()

    ifile = 0
    for inFile in fileList:
        lf = lfData(inFile)
        if ifile == 0:
            lf0 = lf
        else:
            print 'Delta alpha = ', lf.schec['alpha'][0] - lf0.schec['alpha'][0]
            print 'Delta Mstar = ', lf.schec['Mstar'][0] - lf0.schec['Mstar'][0]
            print 'Delta phistar = ', lf.schec['phistar'][0] - lf0.schec['phistar'][0]

        Mmin = lf.par['absMin']
        Mmax = lf.par['absMax']
        phi = lf.phiList[0]
        phisum = 0
        for ibin in xrange(phi.nbin):
            r1 = 10**(0.2*(lf.par['appMin'] - phi.Mav[ibin]))
            r2 = 10**(0.2*(lf.par['appMax'] - phi.Mav[ibin]))
            Vm = r2*r2*r2 - r1*r1*r1
            phisum += phi.swml[ibin]*Vm

        if ifile == 0:
            phisum0 = phisum
        else:
            if renorm:
                scale = phisum0/phisum
                print scale
                phi.swml *= scale
                phi.sty *= scale
        
        idx = phi.ngal > 0
        ax = plt.axes()
        plt.semilogy(basey=10, nonposy='clip')
        plt.plot(phi.Mav[idx], phi.swml[idx], 'o', color=colour[ifile])
        plt.errorbar(phi.Mav[idx], phi.swml[idx], phi.swmlErr[idx], fmt=None,
##                      ecolor=colour[ifile])
                     ecolor='k')
##         plt.plot(phi.Mav[idx], phi.sty[idx], '-', color=colour[ifile])
        plt.plot(phi.Mav[idx], phi.sty[idx], ls[ifile], color='k')
        try:
            plt.text(0.1, yt[ifile], lf.par['grpsel'],
                     transform = ax.transAxes, color=colour[ifile])
        except:
            pass
        plt.axis([Mmin, Mmax, ylo, yhi])
        if ifile == 0:
            plt.xlabel(r'$^{{{:2.1f}}}M_{} - 5 \log h$'.format(lf.par['z0'],
                                                       lf.par['band']))
            plt.ylabel(r'$\phi(M)\ h^3$ Mpc$ ^{-3}$')

        ifile += 1
                
    if like:
        prob = 0.05
        nu = 2
        chisq = scipy.special.chdtri(nu, prob)

        ax = plt.axes([0.5, 0.18, 0.25, 0.25])
        ifile = 0
        for inFile in fileList:
            likefile = inFile.split('lf')[0] + 'like' + inFile.split('lf')[1]

            f = open(likefile, 'r')
            line = f.readline()
            data = line.split()
            iband = int(data[0])
            xpar = data[1]
            ypar = data[2]
            xval = data[3]
            yval = data[4]
            maxLike = float(data[5])
            nbin = int(data[6])
            xmin = float(data[7])
            xmax = float(data[8])
            ymin = float(data[9])
            ymax = float(data[10])
    
            v = [maxLike - 0.5*chisq,]

            like = pickle.load(f)

            plt.contour(like, v, aspect='auto', colors=colour[ifile], 
                        linestyles='solid', origin='lower',
                        extent=[xmin, xmax, ymin, ymax])
            plt.locator_params('x', nbins=5)
            plt.locator_params('y', nbins=8)
            f.close()
            if ifile == 0:
                plt.xlabel(r'$\alpha$')
                plt.ylabel(r'$M*$')

            ifile += 1
    plt.draw()
    
def faintPlotResid(fileList, ylo=0.0, yhi=2.0):
    """Plot comparison LFs normalised by a reference LF (first in list)"""

    if len(fileList[0]) == 1:
        fileList = glob.glob(fileList)
    print fileList
    
    colour = 'kbgrymc'
    fig = plt.figure(1)
    plt.clf()

    lf0 = lfData(fileList[0])
    phi0 = lf0.phiList[0]
    Mmin = lf0.par['absMin']
    Mmax = lf0.par['absMax']
    iplot = 0
    for inFile in fileList[1:]:
        lf = lfData(inFile)
        phi = lf.phiList[0]
        
        idx = phi.ngal > 0
        ax = plt.axes()
##         plt.semilogy(basey=10, nonposy='clip')
        plt.plot(phi.Mav[idx], phi.swml[idx]/phi0.swml[idx], 'o',
                 color=colour[iplot])
        plt.errorbar(phi.Mav[idx], phi.swml[idx]/phi0.swml[idx],
                     phi.swmlErr[idx]/phi0.swml[idx], fmt=None,
                     ecolor=colour[iplot])
        plt.plot(phi.Mav[idx], phi.sty[idx]/phi0.sty[idx], '-',
                 color=colour[iplot])
        plt.axis([Mmin, Mmax, ylo, yhi])

        iplot += 1
                
    plt.xlabel(r'$^{{{:2.1f}}}M_{} - 5 \log h$'.format(lf.par['z0'],
                                                       lf.par['band']))
    plt.ylabel(r'$\phi(M)/\phi_0(M)$')
    plt.draw()
    
def lfAv(fileList):
    """Average LF estimates over a number of simulations.  fileList may be a
    sequence of filenames or a single name including wildcards."""

    if len(fileList[0]) == 1:
        fileList = glob.glob(fileList)
    print fileList
    nest = len(fileList)
    
    alpha = []
    Mstar = []
    Q = []
    P = []
    phistar = []
    ld = []
    ld0 = []

    iest = 0
    for inFile in fileList:
        f = open(inFile, 'r')
        line = f.readline()
        data = f.readline().split()
        band = data[1]
        colour = data[2]
        nz = int(data[3])

        line = f.readline()
        schec = eval(line)
        alpha.append(schec['alpha'][0])
        Mstar.append(schec['Mstar'][0])
        Q.append(schec['Q'][0])
        P.append(schec['P'][0])
        phistar.append(schec['phistar'][0])

        line = f.readline()
        lumDens = eval(line)
        ld.append(lumDens['ld'])[0]
        ld0.append(lumDens['ld0'])[0]
        
        zRange = []
        for iz in range(nz):
            zr, zmean, ngal = eval(f.readline())
            zRange.append(zr)
            data = f.readline().split()
            nbin = int(data[0])
            Mmin = float(data[1])
            Mmax = float(data[2])
            prob = float(data[5])
            if iz == 0 and iest == 0:
                ngal = np.zeros((nbin, nz, nest))
                Mav = np.zeros((nbin, nz, nest))
                Vmax = np.zeros((nbin, nz, nest))
                VmaxErr = np.zeros((nbin, nz, nest))
                swml = np.zeros((nbin, nz, nest))
                swmlErr = np.zeros((nbin, nz, nest))
                sty = np.zeros((nbin, nz, nest))
            for ibin in range(nbin):
                data = f.readline().split()
                ngal[ibin, iz, iest] = int(data[0])
                Mav[ibin, iz, iest] = float(data[1])
                Vmax[ibin, iz, iest] = float(data[2])
                VmaxErr[ibin, iz, iest] = float(data[3])
                swml[ibin, iz, iest] = float(data[4])
                swmlErr[ibin, iz, iest] = float(data[5])
                sty[ibin, iz, iest] = float(data[6])

        f.close()
        iest += 1
        
    print 'alpha ', np.mean(alpha), np.std(alpha)
    print '   M* ', np.mean(Mstar), np.std(Mstar)
    print '    Q ', np.mean(Q), np.std(Q)
    print '    P ', np.mean(P), np.std(P)
    print ' phi* ', np.mean(phistar), np.std(phistar)
    print '   ld ', np.mean(ld), np.std(ld)
    print '  ld0 ', np.mean(ld0), np.std(ld0)
    
    fig = plt.figure(1)
    plt.clf()
    grid = AxesGrid(fig, 111, # similar to subplot(111)
                    nrows_ncols = (2, 2), # creates 2x2 grid of axes
                    axes_pad=0.0, # pad between axes in inch.
                    aspect=False)

    # Avoid overlapping mag labels by specifying max of 5 major ticks
    # with 5 minor ticks per major tick
    nmajor = 5
    nminor = 25
    majorLocator = matplotlib.ticker.MaxNLocator(nmajor)
    minorLocator = matplotlib.ticker.MaxNLocator(nminor)
    ix = 0
    iy = 0
        
    for iz in range(nz):
        ax = grid[iz]
        ngsum = np.zeros(nbin)
        Mp = np.zeros(nbin)
        Vmaxp = np.zeros(nbin)
        VmaxpErr = np.zeros(nbin)
        swmlp = np.zeros(nbin)
        swmlpErr = np.zeros(nbin)
        for ibin in range(nbin):
            ngsum[ibin] = (ngal[ibin, iz,:]).sum()
            if ngsum[ibin] > 0:
                Mp[ibin] = (ngal[ibin, iz,:]*Mav[ibin, iz,:]).sum()/ngsum[ibin]
                Vmaxp[ibin] = (Vmax[ibin, iz,:]).mean()
                VmaxpErr[ibin] = (Vmax[ibin, iz,:]).std()
                swmlp[ibin] = (swml[ibin, iz,:]).mean()
                swmlpErr[ibin] = (swml[ibin, iz,:]).std()
        idx = ngsum > 0
##         pdb.set_trace()
        ax.semilogy(basey=10, nonposy='clip')
        ax.errorbar(Mp[idx], Vmaxp[idx], VmaxpErr[idx], fmt='bo')
        ax.errorbar(Mp[idx], swmlp[idx], swmlpErr[idx], fmt='go')
##         ax.plot(Mav[idx], sty[idx], '-')

        ax.axis([Mmin, Mmax, 1e-6, 1])
        title = '%5.3f' % zRange[iz][0] + ' < z < %5.3f' % zRange[iz][1]
        ax.text(0.1, 0.9, title, transform = ax.transAxes)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator)

        if iy == 1:
            ax.set_xlabel('$M_%s$' % band)
        if ix == 0:
            ax.set_ylabel('$\phi$')
        ix += 1
        if ix > 1:
            iy += 1
            ix = 0
            
    plt.draw()
            
def lfAvFaint(fileList, sim_schec=None):
    """Average LF  over a number of simulations, faintest z-slice only.
    fileList may be a sequence of filenames or a single name including
    wildcards."""

    global par
    par = Parameters()
    par['idebug'] = 1

    if len(fileList[0]) == 1:
        fileList = glob.glob(fileList)
    print fileList
    nest = len(fileList)
    
    alpha = []
    Mstar = []
    Q = []
    P = []
    phistar = []

    iest = 0
    for inFile in fileList:
        f = open(inFile, 'r')
        line = f.readline()
        data = f.readline().split()
        band = data[1]
        nz = int(data[2])
        line = f.readline()
        schec = eval(line)
        alpha.append(schec['alpha'][0])
        Mstar.append(schec['Mstar'][0])
        Q.append(schec['Q'][0])
        P.append(schec['P'][0])
        phistar.append(schec['phistar'][0])

        zRange = []
        for iz in range(nz):
            zRange.append(eval(f.readline()))
            data = f.readline().split()
            nbin = int(data[0])
            Mmin = float(data[1])
            Mmax = float(data[2])
            if iz == 0 and iest == 0:
                ngal = np.zeros((nbin, nz, nest))
                Mav = np.zeros((nbin, nz, nest))
                Vmax = np.zeros((nbin, nz, nest))
                VmaxErr = np.zeros((nbin, nz, nest))
                swml = np.zeros((nbin, nz, nest))
                swmlErr = np.zeros((nbin, nz, nest))
                sty = np.zeros((nbin, nz, nest))
            for ibin in range(nbin):
                data = f.readline().split()
                ngal[ibin, iz, iest] = int(data[0])
                Mav[ibin, iz, iest] = float(data[1])
                Vmax[ibin, iz, iest] = float(data[2])
                VmaxErr[ibin, iz, iest] = float(data[3])
                swml[ibin, iz, iest] = float(data[4])
                swmlErr[ibin, iz, iest] = float(data[5])
                sty[ibin, iz, iest] = float(data[6])

        f.close()
        iest += 1
        
    print 'alpha ', np.mean(alpha), np.std(alpha)
    print '   M* ', np.mean(Mstar), np.std(Mstar)
    print '    Q ', np.mean(Q), np.std(Q)
    print '    P ', np.mean(P), np.std(P)
    print ' phi* ', np.mean(phistar), np.std(phistar)
    
    plot = plotWindow()
    plot.window()

    iz = 0
    ngsum = np.zeros(nbin)
    Mp = np.zeros(nbin)
    Vmaxp = np.zeros(nbin)
    VmaxpErr = np.zeros(nbin)
    swmlp = np.zeros(nbin)
    swmlpErr = np.zeros(nbin)
    for ibin in range(nbin):
        ngsum[ibin] = (ngal[ibin, iz,:]).sum()
        if ngsum[ibin] > 0:
            Mp[ibin] = (ngal[ibin, iz,:]*Mav[ibin, iz,:]).sum()/ngsum[ibin]
            Vmaxp[ibin] = (Vmax[ibin, iz,:]).mean()
            VmaxpErr[ibin] = (Vmax[ibin, iz,:]).std()
            swmlp[ibin] = (swml[ibin, iz,:]).mean()
            swmlpErr[ibin] = (swml[ibin, iz,:]).std()
    idx = ngsum > 0
    plt.semilogy(basey=10, nonposy='clip')
    plt.plot(Mp[idx], Vmaxp[idx], 'wo')
    plt.errorbar(Mp[idx], Vmaxp[idx], VmaxpErr[idx], fmt=None, ecolor='k')
    plt.errorbar(Mp[idx], swmlp[idx], swmlpErr[idx], fmt='ko')
    plt.xlabel(r'$M_%s - 5 \log h$' % band)
    plt.ylabel(r'$\phi(M)\ h^3$ Mpc$ ^{-3}$')
##     title = r'$%5.3f' % zRange[iz][0] + ' < z < %5.3f$' % zRange[iz][1]
##     xt = Mmin + 0.1*(Mmax - Mmin)
##     plt.text(xt, 0.1, title)
    if sim_schec:
        plotSchec(sim_schec[0], sim_schec[1], sim_schec[2], Mmin, Mmax,
                  lineStyle='k-')
    plt.axis([Mmin, Mmax, 1e-6, 1])
    plt.draw()
            
def lumDensPlot(fileList='lf_ev8_??_jackErrs.dat'):
    """Luminosity density plots.  fileList may be a sequence of filenames
    or a single name possibly including wildcards."""
            
    # Lum densities (Mag h Mpc^-3) from Blanton and Montero-Dorta & Prada
    # (Montero-Dorta & Prada 2009, Table 5)
    ld_blanton = ((-14.10, 0.15), (-15.18, 0.03), (-15.90, 0.03),
                  (-16.24, 0.03), (-16.56, 0.02))
    ld_mdp = ((-14.009, 0.014), (-14.386, 0.021), (-15.814, 0.015),
              (-16.355, 0.016), (-16.661, 0.018))
    
    # Read CNOC2 lum densities
    file = os.environ['HOME'] + '/Documents/Research/LFdata/lin1999.dat'
    cnoc = np.loadtxt(file)
    cnocz = (0.185, 0.325, 0.475)
    cnoczerr = (0.065, 0.075, 0.075)
    
    # Read Prescott et al u-band lum densities
    file = os.environ['HOME'] + '/Documents/Research/LFdata/prescott2009.dat'
    prescott = np.loadtxt(file)
    
    if len(fileList[0]) == 1:
        fileList = glob.glob(fileList)
    print fileList
    nest = len(fileList)
##     rhomin = [0.08, 0.08, 0.08, 0.08, 0.08]
##     rhomax = [7, 7, 7, 7, 7]
    rhomin = [0.08, 0.08, 0.4, 0.4, 0.4]
    rhomax = [8, 7, 7, 7, 7]
    if nest > 1:
        nrows = 5
    else:
        nrows = 1
    ncols = 1
    fig = plt.figure(1)
    plt.clf()
    grid = AxesGrid(fig, 111, # similar to subplot(111)
                    nrows_ncols = (nrows, ncols), # creates nr*nc grid of axes
                    axes_pad=0.0, # pad between axes in inch.
                    aspect=False)
    
    for inFile in fileList:
        lf = lfData(inFile)
        par = lf.par
        iband = par['iband']
        Msun = Msun_ugriz[iband]

        try:
            sim_alpha = par['sim_alpha']
            sim_Mstar = par['sim_Mstar']
            sim_phistar = par['sim_phistar']
            sim_Q = par['sim_Q']
            sim_P = 0
        except:
            sim_alpha = None
            
        fmt = 'ks'
        clr = 'black'
        contline = 'k-'
        dotline = 'k:'
        scale = 1e-8
        if par['colour'] == 'b':
            fmt = 'bo'
            clr = 'blue'
            contline = 'b-'
            dotline = 'b:'
            scale = 1e-8
        if par['colour'] == 'r':
            fmt = 'r^'
            clr = 'red'
            contline = 'r-'
            dotline = 'r:'
            scale = 1e-8

        ev = Evol(lf.schec['Q'][0], lf.schec['P'][0])
        if nest > 1:
            ax = grid[iband]
        else:
            ax = grid[0]
        if nest > 1:
            ax.semilogy(basey=10, nonposy='clip')
        ax.errorbar(lf.lumdens['zmean'], scale*np.array(lf.lumdens['ld'][0]),
                    scale*np.array(lf.lumdens['ld'][1]), fmt=fmt)
        zz = np.linspace(lf.lumdens['zRange'][0], lf.lumdens['zRange'][1])
        lz = scale*lf.lumdens['ld0'][0]*ev.lumden(zz)
        ax.plot(zz, lz, contline)
        
        if sim_alpha:
            # Lum dens predicted by simulation input parameters
            sim_Lstar = 10.0**(0.4*(Msun - sim_Mstar))
            ld0 = sim_phistar*sim_Lstar*scipy.special.gamma(sim_alpha + 2)
            ev = Evol(sim_Q, sim_P)
            lz = scale*ld0*ev.lumden(zz)
            ax.plot(zz, lz, 'r--')
        else:
            # Extreme 1-sigma limits
            lzlo = scale*(lf.lumdens['ld0'][0] - lf.lumdens['ld0'][1])* \
                   10**(0.4*(lf.schec['Q'][0] - lf.schec['Q'][1] +
                             lf.schec['P'][0] - lf.schec['P'][1])*zz)
##             ax.plot(zz, lzlo, dotline)
            lzhi = scale*(lf.lumdens['ld0'][0] + lf.lumdens['ld0'][1])* \
                   10**(0.4*(lf.schec['Q'][0] + lf.schec['Q'][1] +
                             lf.schec['P'][0] + lf.schec['P'][1])*zz)
##             ax.plot(zz, lzhi, dotline)
            ax.fill_between(zz, lzlo, lzhi, facecolor=clr, alpha=0.1)
            
            
        if nrows > 1:
            ax.axis([0.0, 0.5, rhomin[iband], rhomax[iband]])

            if par['colour'] == 'c':
                ax.text(0.05, 0.88, par['band'], transform = ax.transAxes)

                # SDSS comparison
                y = scale*10.0**(0.4*(Msun - ld_blanton[iband][0]))
                yerr = 0.5*(
                    scale*10.0**(0.4*(Msun - (ld_blanton[iband][0] -
                                              ld_blanton[iband][1]))) -
                    scale*10.0**(0.4*(Msun - (ld_blanton[iband][0] +
                                              ld_blanton[iband][1]))))
                ax.plot(0.1, y, 'wo')
##                 ax.errorbar(0.1, y, yerr, fmt=None, ecolor='k')
                y = scale*10.0**(0.4*(Msun - ld_mdp[iband][0]))
                yerr = 0.5*(
                    scale*10.0**(0.4*(Msun - (ld_mdp[iband][0] -
                                              ld_mdp[iband][1]))) -
                    scale*10.0**(0.4*(Msun - (ld_mdp[iband][0] +
                                              ld_mdp[iband][1]))))
                ax.plot(0.1, y, 'ws')
##                 ax.errorbar(0.1, y, yerr, fmt=None, ecolor='k')

                # CNOC comparison.  See Ilbert et al sec 5.1 for u dm offset
                if iband <= 1 or iband == 3:
                    if iband == 0:
                        dm = 0.25
                    else:
                        dm = 0.0
                    irow = min(iband, 2)
                    for iz in range(3):
                        rho = cnoc[irow][3+2*iz]
                        rho_err = cnoc[irow][4+2*iz]
                        rho_mag = 34.1 - 2.5*math.log10(rho*1e20) + dm
                        rho_sun = scale*10.0**(0.4*(Msun - rho_mag))
                        rho_sun_err = rho_sun*rho_err/rho
                        ax.plot(cnocz[iz], rho_sun, 'w^')
                        ax.errorbar(cnocz[iz], rho_sun, yerr=rho_sun_err,
                                    xerr=cnoczerr[iz], fmt=None, ecolor='k')
                    
                # Prescott et al comparison
                if iband == 0:
                    # Corrections for h and bandpass shift
                    lgh = math.log10(0.7)
                    delta_M = 2.5*math.log10(1 + lf.par['z0'])
                    for iz in range(5):
                        z = prescott[iz][0]
                        rho = prescott[iz][6]
                        rho_err = prescott[iz][7]
                        rho_mag = 34.1 - 2.5*(rho - lgh) + delta_M
                        rho_sun = scale*10.0**(0.4*(Msun - rho_mag))
                        rho_sun_err = rho_sun*rho_err/rho
                        ax.plot(z, rho_sun, 'wD')
                        ax.errorbar(z, rho_sun, yerr=rho_sun_err,
                                    fmt=None, ecolor='k')
                    
        else:
            ax.axis([0.0, 0.4, 1.8, 4.2])
##             ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        if nrows == 1 or iband == 4:
            ax.set_xlabel(r'$z$')
        if nrows == 1 or iband == 2:
            ax.set_ylabel(r'$\rho_L/10^8 L_\odot h$ Mpc$^{-3}$')

    plt.draw()
    plt.savefig('lumdens.pdf', bbox_inches='tight')


def likePlotEv(iband):
    """alpha, Mstar and Q, P likelihood contour plots one band at a time"""

    def plotContour(f):
        """Read and plot likelihood contour"""

        data = f.readline().split()
        iband = int(data[0])
        xpar = data[1]
        ypar = data[2]
        xval = float(data[3])
        yval = float(data[4])
        maxLike = float(data[5])
        nbin = int(data[6])
        xmin = float(data[7])
        xmax = float(data[8])
        ymin = float(data[9])
        ymax = float(data[10])
        v = maxLike - 0.5*chisq
        like = pickle.load(f)
        plt.contour(like, [v,], aspect='auto', origin='lower',
                   extent=[xmin, xmax, ymin, ymax], colors=col,
                   linestyles='solid')
        plt.plot(xval, yval, symbol[colour])

        
    band = 'ugriz'[iband]
    
    # Blanton et al 2003 alpha, Mstar, Q, P parameters and errors
    blanton = np.array([[-0.92, 0.07, -17.93, 0.03, 4.22, 0.88, 3.20, 3.31],
                        [-0.89, 0.03, -19.39, 0.02, 2.04, 0.51, 0.32, 1.70],
                        [-1.05, 0.01, -20.44, 0.01, 1.62, 0.30, 0.18, 0.57],
                        [-1.00, 0.02, -20.82, 0.02, 1.61, 0.43, 0.58, 1.06],
                        [-1.08, 0.02, -21.18, 0.02, 0.76, 0.29, 2.28, 0.79]])
    
    # Read least-squares fits
    lsqpar4 = {}
    lsqpar8 = {}
    f = open('lsqEv4Fit.dat', 'r')
    for line in f:
        file, Q, QErr, P, PErr = line.split()
        bc = file[7:9]
        lsqpar4[bc] = (Q, QErr, P, PErr)
    f.close()
    f = open('lsqEv8Fit.dat', 'r')
    for line in f:
        file, Q, QErr, P, PErr = line.split()
        bc = file[7:9]
        lsqpar8[bc] = (Q, QErr, P, PErr)
    f.close()
    
    prob = 0.05
    nu = 2
    chisq = scipy.special.chdtri(nu, prob)
    print 'delta chisq = ', chisq

    amlimits = ((-1.5, 0.1, -18.4, -17.1),
                (-1.5, -0.3, -19.8, -19.2),
                (-1.5, -0.3, -20.8, -20.2),
                (-1.5, -0.3, -21.1, -20.6),
                (-1.5, -0.3, -21.4, -20.8))[iband]

    qplimits = ((3, 7, -11, 7),
                (0, 4, -3, 3),
                (0, 3, -2, 3.5),
                (0, 3, -3, 3.5),
                (0, 3, -3, 4))[iband]

    fig = plt.figure(1)
    plt.clf()

    symbol = {'c': 'ks', 'b': 'ob', 'r': 'r^'}
    sym4 = {'c': 'k^', 'b': 'b^', 'r': 'r^'}
    sym8 = {'c': 'kh', 'b': 'bh', 'r': 'rh'}

    for colour in 'cbr':
        col = colour
        if col == 'c': col = 'k'
        inFile = 'like_ev8_%s%s.dat' % (band, colour)
        f = open(inFile, 'r')

        # alpha, M*
        
        plt.subplot(211)
        plotContour(f)

        f.readline()
        like = pickle.load(f)
        f.readline()
        like = pickle.load(f)

        # P, Q
        
        plt.subplot(212)
        plotContour(f)

        # Show lsq estimate as errorbar
        bc = band + colour
##         Q = float(lsqpar4[bc][0])
##         QErr = float(lsqpar4[bc][1])
##         P = float(lsqpar4[bc][2])
##         PErr = float(lsqpar4[bc][3])
##         plt.errorbar(Q, P, QErr, PErr, fmt=sym4[colour], color=col)
        Q = float(lsqpar8[bc][0])
        QErr = float(lsqpar8[bc][1])
        P = float(lsqpar8[bc][2])
        PErr = float(lsqpar8[bc][3])
        plt.errorbar(Q, P, QErr, PErr, fmt=symbol[colour], color=col)
         
        f.close()

    # Show Blanton et al parameters as errorbars
    ax = plt.subplot(211)
    plt.errorbar(blanton[iband,0], blanton[iband,2],
                 xerr=blanton[iband,1], yerr=blanton[iband,3],
                 color='k')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$^{0.1}M^* - 5 \lg h$')
    plt.text(0.05, 0.9, band, transform = ax.transAxes)
    plt.axis(amlimits)
    
    ax = plt.subplot(212)
    plt.errorbar(blanton[iband,4], blanton[iband,6],
                   xerr=blanton[iband,5], yerr=blanton[iband,7],
                   color='k', ls=':')
    plt.xlabel(r'$Q$')
    plt.ylabel(r'$P$')
    plt.text(0.05, 0.9, band, transform = ax.transAxes)
    plt.axis(qplimits)
    
    plt.draw()
    
def likePlotAM():
    """alpha, Mstar Likelihood contour plots for multiple bands and colours."""

    # Blanton et al 2003 Scechter parameters for ugriz bands
    # Columns are alpha, err, Mstar, err, corr, Q, err, P, err, corr
    blanton = np.array([[-0.92, 0.07, -17.93, 0.03, 0.560,
                         4.22, 0.88, 3.20, 3.31, -0.955],
                        [-0.89, 0.03, -19.39, 0.02, 0.760,
                         2.04, 0.51, 0.32, 1.70, -0.949],
                        [-1.05, 0.01, -20.44, 0.01, 0.866,
                         1.62, 0.30, 0.18, 0.57, -0.849],
                        [-1.00, 0.02, -20.82, 0.02, 0.905,
                         1.61, 0.43, 0.58, 1.06, -0.950],
                        [-1.08, 0.02, -21.18, 0.02, 0.885,
                         0.76, 0.29, 2.28, 0.79, -0.908]])
    

    prob = 0.05
    nu = 2
    chisq = scipy.special.chdtri(nu, prob)
    print 'delta chisq = ', chisq

    amlimits = ((-1.58, 0.1, -18.4, -17.1),
                (-1.58, -0.3, -19.8, -19.2),
                (-1.58, -0.3, -20.8, -20.2),
                (-1.58, -0.3, -21.1, -20.6),
                (-1.58, -0.3, -21.4, -20.8))
    nrows = 5
    ncols = 1
    fig = plt.figure(1)
    plt.clf()

    symbol = {'c': 'ks', 'b': 'ob', 'r': 'r^'}
    iband = 0
    for band in 'ugriz':
        ax = plt.subplot(5, 1, iband+1)
        for colour in 'cbr':
            inFile = 'like_ev8_%s%s.dat' % (band, colour)
            f = open(inFile, 'r')
            data = f.readline().split()
            iband = int(data[0])
            xpar = data[1]
            ypar = data[2]
            xval = float(data[3])
            yval = float(data[4])
            maxLike = float(data[5])
            nbin = int(data[6])
            xmin = float(data[7])
            xmax = float(data[8])
            ymin = float(data[9])
            ymax = float(data[10])
            v = maxLike - 0.5*chisq
            like = pickle.load(f)
            f.close()
    
            col = colour
            if col == 'c': col = 'k'
            plt.contour(like, [v,], aspect='auto', origin='lower',
                        extent=[xmin, xmax, ymin, ymax], colors=col,
                        linestyles='solid')
            plt.plot(xval, yval, symbol[colour])

        # Show Blanton et al parameters as error ellipses
        plt.plot(blanton[iband,0], blanton[iband,2], '*k')
        el = error_ellipse(blanton[iband,0], blanton[iband,2],
                           blanton[iband,1], blanton[iband,3], blanton[iband,4])
        ax.add_artist(el)
        
        plt.ylabel(r'$^{0.1}M^* - 5 \lg h$')
        plt.text(0.05, 0.85, band, transform = ax.transAxes)
        ax.locator_params(nbins=8)
        plt.axis(amlimits[iband])
    
        iband += 1

    plt.xlabel(r'$\alpha$')
    plt.draw()
    
def likePlotQP():
    """Q, P Likelihood contour plots for multiple bands and colours."""

    # Blanton et al 2003 Scechter parameters for ugriz bands
    # Columns are alpha, err, Mstar, err, corr, Q, err, P, err, corr
    blanton = np.array([[-0.92, 0.07, -17.93, 0.03, 0.560,
                         4.22, 0.88, 3.20, 3.31, -0.955],
                        [-0.89, 0.03, -19.39, 0.02, 0.760,
                         2.04, 0.51, 0.32, 1.70, -0.949],
                        [-1.05, 0.01, -20.44, 0.01, 0.866,
                         1.62, 0.30, 0.18, 0.57, -0.849],
                        [-1.00, 0.02, -20.82, 0.02, 0.905,
                         1.61, 0.43, 0.58, 1.06, -0.950],
                        [-1.08, 0.02, -21.18, 0.02, 0.885,
                         0.76, 0.29, 2.28, 0.79, -0.908]])
    
    # Read least-squares fits
    lsqpar8 = {}
    f = open('lsqEv8Fit.dat', 'r')
    for line in f:
        file, Q, QErr, P, PErr = line.split()
        bc = file[7:9]
        lsqpar8[bc] = (Q, QErr, P, PErr)
    f.close()
    
    prob = 0.05
    nu = 2
    chisq = scipy.special.chdtri(nu, prob)
    print 'delta chisq = ', chisq
    
    qplimits = ((2, 7.5, -11, 7),
                (-1, 4.5, -5.5, 4),
                (-0.5, 2.5, -2, 4),
                (0, 3, -2.5, 3),
                (-0.5, 3.5, -4, 5.5))
    nrows = 5
    ncols = 1
    fig = plt.figure(1)
    plt.clf()
     
    symbol = {'c': 'ks', 'b': 'ob', 'r': 'r^'}
    iband = 0
    for band in 'ugriz':
        ax = plt.subplot(5, 1, iband+1)
        for colour in 'cbr':
            inFile = 'like_ev8_%s%s.dat' % (band, colour)
            f = open(inFile, 'r')
            f.readline()
            like = pickle.load(f)
            f.readline()
            like = pickle.load(f)
            f.readline()
            like = pickle.load(f)
        
            data = f.readline().split()
            iband = int(data[0])
            xpar = data[1]
            ypar = data[2]
            xval = float(data[3])
            yval = float(data[4])
            maxLike = float(data[5])
            nbin = int(data[6])
            xmin = float(data[7])
            xmax = float(data[8])
            ymin = float(data[9])
            ymax = float(data[10])
            v = maxLike - 0.5*chisq
            like = pickle.load(f)
            f.close()
    
            col = colour
            if col == 'c': col = 'k'
            ax.contour(like, [v,], aspect='auto', origin='lower',
                       extent=[xmin, xmax, ymin, ymax], colors=col,
                       linestyles='solid')
            ax.plot(xval, yval, symbol[colour])

            # Show lsq estimate as errorbar
            bc = band + colour
            Q = float(lsqpar8[bc][0])
            QErr = float(lsqpar8[bc][1])
            P = float(lsqpar8[bc][2])
            PErr = float(lsqpar8[bc][3])
            ax.errorbar(Q, P, 2*QErr, 2*PErr,
                        fmt=symbol[colour], color=col)
            
        # Show Blanton et al estimate as ellipses
        plt.plot(blanton[iband,5], blanton[iband,7], '*k')
        el = error_ellipse(blanton[iband,5], blanton[iband,7],
                           blanton[iband,6], blanton[iband,8], blanton[iband,9])
        ax.add_artist(el)
    
        ax.axis(qplimits[iband])
        plt.ylabel(r'$P$')
        ax.text(0.05, 0.85, band, transform = ax.transAxes)
        ax.locator_params(nbins=8)
        iband += 1
        
    plt.xlabel(r'$Q$')
    plt.draw()
    
def error_ellipse(x, y, sigx, sigy, corr):
    """Plot 2-sigma error ellipse centred on (x,y) with given sigx, sigy, corr.
    See http://www.earth-time.org/projects/upb/public_docs/ErrorEllipses.pdf"""

    scale = 2.4477 # for 2-sigma errors
    sigxy = sigx*sigy*corr
    cov = ((sigx**2, sigxy), (sigxy, sigy**2))
    w, v = np.linalg.eig(cov)
    aa = 2*scale*math.sqrt(w[0]) # twice semi-major
    bb = 2*scale*math.sqrt(w[1]) # twice semi-minor
    try:
        theta = (180.0/math.pi)*0.5*math.atan(2*sigxy/(sigx**2 - sigy**2))
    except:
        theta = 0.0

    el = Ellipse((x,y), aa, bb, theta, fc='none', linestyle='dotted')
    return el

def likePlot(inFile):
    """Plot likelihood contours.
    lnL = lnLmax - 0.5*chi^2(p,m); p = prob, m = degrees freedom = 2."""
    
    global par
    par = {}
    par['idebug'] = 1

    inRoot = inFile.split('.')[0]
    plotFile = inRoot + '.png'
    prob = 0.05
    nu = 2
    chisq = scipy.special.chdtri(nu, prob)
    print 'delta chisq = ', chisq
    fig = plt.figure(1)
    plot = plotWindow(3,2)

    f = open(inFile, 'r')
    line = 'start'
    while len(line) > 0:
        line = f.readline()
        if len(line) > 0:
            data = line.split()
            xpar = data[1]
            ypar = data[2]
            xval = data[3]
            yval = data[4]
            maxLike = float(data[5])
            nbin = int(data[6])
            xmin = float(data[7])
            xmax = float(data[8])
            ymin = float(data[9])
            ymax = float(data[10])
            dx = (xmax-xmin)/nbin
            dy = (ymax-ymin)/nbin
            
            v = [maxLike - 0.5*chisq,]

            like = pickle.load(f)
            ix, iy = np.unravel_index(np.argmax(like), like.shape)
            xopt = xmin + (ix+0.5)*dx
            yopt = ymin + (iy+0.5)*dy
            Lopt = like[ix,iy]
            if Lopt > maxLike:
                print 'higher likelihood like ', Lopt, ' at ', xopt, yopt
            print line
            
            idx = like < -1e9
            like[idx] = None
            plot.window()
            plt.imshow(like, aspect='auto', #cmap=matplotlib.cm.gray,
                       interpolation='nearest', origin='lower',
                       extent=[xmin, xmax, ymin, ymax])
            plt.colorbar()
            plt.contour(like, v, aspect='auto',
                        origin='lower', extent=[xmin, xmax, ymin, ymax])
            plt.xlabel(xpar); plt.ylabel(ypar) #; plt.title('log likelihood')


    plt.draw()
##     plot.save(plotFile)

##     def onclick(event):
##         print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
##             event.button, event.x, event.y, event.xdata, event.ydata)
##         ix = int((event.xdata - xmin)/dx)
##         iy = int((event.ydata - ymin)/dy)
##         print ix, iy, like[ix,iy]
##         if event.button == 3:
##             fig.canvas.mpl_disconnect(cid)

##     cid = fig.canvas.mpl_connect('button_press_event', onclick)

def likePlotMulti(fileList, dp=0, smooth=0):
    """Likelihood contour plots for several bands.  fileList may be a
    sequence of filenames or a single name including wildcards."""

    label = {'alpha': r'$\alpha$', 'Mstar': r'$^{0.1}M^* - 5 \lg h$',
             'beta': r'$\beta$', 'Mt': r'$^{0.1}M_t - 5 \lg h$',
             'Q': r'$Q$', 'P': r'$P$'}
    
    # Plot limits
    if dp:
        limits = {'alpha': (-1.2, 0.5), 'Mstar': (-21.2, -16.8),
                  'beta': (-2.5, 0.0), 'Mt': (-21, -14),
                  'Q': (0, 8), 'P': (-10, 0)}
        nrow = 3
        ncol = 2
    else:
        limits = {'alpha': (-1.5, -0.3), 'Mstar': (-21.5, -17),
                  'beta': (-2, -0.7), 'Mt': (-21, -14),
                  'Q': (0, 7), 'P': (-3, 3)}
        nrow = 2
        ncol = 2
    if len(fileList[0]) == 1:
        fileList = glob.glob(fileList)
    print fileList

    plotFile = 'like_faint.eps'
    prob = 0.05
    nu = 2
    chisq = scipy.special.chdtri(nu, prob)
    print 'delta chisq = ', chisq

    tickxmajor = [0.5, 0.5, 0.5, 1.0, 1.0, 0.5]
    tickxminor = [0.1, 0.1, 0.1, 0.2, 0.2, 0.1]
    tickymajor = [1.0, 0.5, 1.0, 0.5, 1.0, 1.0]
    tickyminor = [0.2, 0.1, 0.2, 0.1, 0.2, 0.2]
    
    band = ((' u', ' g', ' r', ' i', ' z'))
    colour = 'bgrcm'
    plt.clf()
    plt.subplots_adjust(wspace=0.3)

    kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
    kernel /= kernel.sum()
    
    ifile = 0
    for inFile in fileList:
        f = open(inFile, 'r')
        line = 'start'
        iplot = 0
        while len(line) > 0:
            line = f.readline()
            if len(line) > 0:
                data = line.split()
                iband = int(data[0])
                xpar = data[1]
                ypar = data[2]
                xval = data[3]
                yval = data[4]
                maxLike = float(data[5])
                nbin = int(data[6])
                xmin = float(data[7])
                xmax = float(data[8])
                ymin = float(data[9])
                ymax = float(data[10])
    
                v = [maxLike - 0.5*chisq,]

                like = pickle.load(f)
                if smooth:
                    like = scipy.ndimage.filters.convolve(like, kernel,
                                                          mode='nearest')
                iplot += 1
                ax = plt.subplot(nrow, ncol, iplot)
                plt.contour(like, v, aspect='auto', colors=colour[iband], 
                            linestyles='solid', origin='lower',
                            extent=[xmin, xmax, ymin, ymax])
                plt.plot(xval, yval, 'o')
                plt.text(xval, yval, band[iband])
##                 plt.axis(limits[iplot-1])
                plt.axis(list(limits[xpar]) + list(limits[ypar]))
                ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tickxmajor[iplot-1]))
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tickymajor[iplot-1]))
                ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(tickxminor[iplot-1]))
                ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(tickyminor[iplot-1]))
                if ifile == 0:
                    plt.xlabel(label[xpar]); plt.ylabel(label[ypar])

        f.close()
        ifile += 1
        
    plt.draw()
##     plt.savefig(plotFile, bbox_inches='tight')
    
def likePlotComp(fileList):
    """Likelihood contour plots for several files.  fileList may be a
    sequence of filenames or a single name including wildcards."""

    if len(fileList[0]) == 1:
        fileList = glob.glob(fileList)
    print fileList

    prob = 0.05
    nu = 2
    chisq = scipy.special.chdtri(nu, prob)
    print 'delta chisq = ', chisq

    colour = 'kbgrcm'
    plt.clf()
    
    ifile = 0
    for inFile in fileList:
        f = open(inFile, 'r')
        line = 'start'
        iplot = 0
        while len(line) > 0:
            line = f.readline()
            if len(line) > 0:
                data = line.split()
                iband = int(data[0])
                xpar = data[1]
                ypar = data[2]
                xval = data[3]
                yval = data[4]
                maxLike = float(data[5])
                nbin = int(data[6])
                xmin = float(data[7])
                xmax = float(data[8])
                ymin = float(data[9])
                ymax = float(data[10])
    
                v = [maxLike - 0.5*chisq,]

                like = pickle.load(f)

                plt.contour(like, v, aspect='auto', colors=colour[ifile], 
                            linestyles='solid', origin='lower',
                            extent=[xmin, xmax, ymin, ymax])
                if ifile == 0:
                    plt.xlabel(xpar); plt.ylabel(ypar)

        f.close()
        ifile += 1
        
    plt.draw()
    
        
def parPlotDPMulti(fileList='lf_dp_r?_b???.dat'):
    """Plot DP Schechter parameters for data and bootstrap samples."""

    if len(fileList[0]) == 1:
        fileList = glob.glob(fileList)
    print fileList

    plotFile = 'par_dp.eps'


    nrow = 3
    ncol = 2
    plt.clf()
    plt.subplots_adjust(wspace=0.3)

    for inFile in fileList:
        f = open(inFile, 'r')
        data = f.readline()
        data = f.readline().split()
        band = data[1]
        colour = data[2]
        nz = int(data[3])
        iband = 'ugriz'.find(band)
    
        fmt = 'k,'
##         if colour == 'blue':
        if inFile[7] == 'b':
            fmt = 'b,'
##         if colour == 'red':
        if inFile[7] == 'r':
            fmt = 'r,'

        f.readline()
        f.readline()
        schec = eval(f.readline())
        f.readline()
        f.close()

        plt.subplot(nrow, ncol, 1)
        plt.plot(schec['alpha'][0], schec['Mstar'][0], fmt)
        plt.subplot(nrow, ncol, 2)
        plt.plot(schec['alpha'][0], schec['beta'][0], fmt)
        plt.subplot(nrow, ncol, 3)
        plt.plot(schec['alpha'][0], schec['Mt'][0], fmt)
        plt.subplot(nrow, ncol, 4)
        plt.plot(schec['Mstar'][0], schec['beta'][0], fmt)
        plt.subplot(nrow, ncol, 5)
        plt.plot(schec['Mstar'][0], schec['Mt'][0], fmt)
        plt.subplot(nrow, ncol, 6)
        plt.plot(schec['beta'][0], schec['Mt'][0], fmt)

        plt.subplot(nrow, ncol, 1)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$M^*$')
        plt.subplot(nrow, ncol, 2)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\beta$')
        plt.subplot(nrow, ncol, 3)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$M_t$')
        plt.subplot(nrow, ncol, 4)
        plt.xlabel(r'$M^*$')
        plt.ylabel(r'$\beta$')
        plt.subplot(nrow, ncol, 5)
        plt.xlabel(r'$M^*$')
        plt.ylabel(r'$M_t$')
        plt.subplot(nrow, ncol, 6)
        plt.xlabel(r'$\beta$')
        plt.ylabel(r'$M_t$')
        
    plt.draw()
    plt.savefig(plotFile, bbox_inches='tight')
    
    
def schecFit(M, phi, phiErr, (alpha, Mstar, lpstar), afix=False, likeCont=False):
    """Least-squares Schechter fn fit to binned estimate."""

    prob = 0.32
    nbin = len(phiErr > 0)
    if afix:
        nu = nbin - 2
    else:
        nu = nbin - 3
    dchisq = scipy.special.chdtri(nu, prob)
    print nu, dchisq

    if afix:
        x0 = [Mstar, lpstar]
        res = scipy.optimize.fmin(
            lambda (Mstar, lpstar), alpha, M, phi, phiErr:
            schecResid((alpha, Mstar, lpstar), M, phi, phiErr),
            x0, (alpha, M, phi, phiErr),
            xtol=0.001, ftol=0.001, full_output=1, disp=0)
        xopt = res[0]
        chi2 = res[1]
        Mstar = xopt[0]
        lpstar = xopt[1]
        alphaErr = [0, 0]
    else:
        x0 = [alpha, Mstar, lpstar]
        res = scipy.optimize.fmin(schecResid, x0, (M, phi, phiErr),
                                  xtol=0.001, ftol=0.001, full_output=1, disp=0)
        xopt = res[0]
        chi2 = res[1]
        alpha = xopt[0]
        Mstar = xopt[1]
        lpstar = xopt[2]
        alphaErr = likeErr(lambda (Mstar, lpstar), alpha, M, phi, phiErr:
                           schecResid((alpha, Mstar, lpstar), M, phi, phiErr),
                           alpha, limits=(alpha-5, alpha+5),
                           marg=(Mstar, lpstar), args=(M, phi, phiErr),
                           nsig=2*dchisq)
        print '  alpha %6.2f - %6.2f + %6.2f' % (alpha, alphaErr[0], alphaErr[1])
        
    MstarErr = likeErr(lambda (lpstar), Mstar, alpha, M, phi, phiErr:
                       schecResid((alpha, Mstar, lpstar), M, phi, phiErr),
                       Mstar, limits=(Mstar-5, Mstar+5),
                       marg=(lpstar), args=(alpha, M, phi, phiErr),
                       nsig=2*dchisq)
    lpstarErr = likeErr(lambda (Mstar), lpstar, alpha, M, phi, phiErr:
                        schecResid((alpha, Mstar, lpstar), M, phi, phiErr),
                        lpstar, limits=(lpstar-5, lpstar+5),
                        marg=(Mstar), args=(alpha, M, phi, phiErr),
                        nsig=2*dchisq)
    
    print '  Mstar %6.2f - %6.2f + %6.2f' % (Mstar, MstarErr[0], MstarErr[1])
    print 'lpstar %6.4f - %6.4f + %6.4f' % (lpstar, lpstarErr[0], lpstarErr[1])

    if likeCont:
        print "M*, phi* 2-sigma contours ..."
        prob = 0.05
        nu = len(phiErr > 0) # no free parameters
        dchisq = scipy.special.chdtri(nu, prob)
        print nu, dchisq
        nstep = 32
        chi2map = np.zeros([nstep, nstep])

        xmin = Mstar - 3*MstarErr[0]
        xmax = Mstar + 3*MstarErr[1]
        dx = (xmax - xmin)/nstep
        ymin = lpstar - 3*lpstarErr[0]
        ymax = lpstar + 3*lpstarErr[1]
        dy = (ymax - ymin)/nstep

        # chi2 minimum
        chi2min = schecResid((alpha, Mstar, lpstar), M, phi, phiErr)
        v = [chi2min + dchisq,]
        for ix in range(nstep):
            ms = xmin + (ix+0.5)*dx
            for iy in range(nstep):
                ps = ymin + (iy+0.5)*dy
                chi2map[iy, ix] = schecResid((alpha, ms, ps), M, phi, phiErr)

        return (alpha, alphaErr, Mstar, MstarErr, lpstar, lpstarErr, chi2, nu,
                chi2map, v, [xmin, xmax, ymin, ymax])
    else:
        return (alpha, alphaErr, Mstar, MstarErr, lpstar, lpstarErr, chi2, nu)
        
def schecResid((alpha, Mstar, lpstar), M, phi, phiErr):
    """Return chi^2 residual between binned phi estimate and Schechter fit."""

    fc = 0
    for ibin in range(len(M)):
        if phiErr[ibin] > 0:
            diff = phi[ibin] - Schechter(M[ibin], alpha, Mstar, 10**lpstar)
            fc += (diff/phiErr[ibin])**2
    # print 'schecResid ', alpha, Mstar, lpstar, fc
    # pdb.set_trace()
    return fc

def saundFit(M, phi, phiErr, alpha, Mstar, sigma, lpstar):
    """Least-squares Saunders fn fit to binned estimate."""

    x0 = [alpha, Mstar, sigma, lpstar]
    res = scipy.optimize.fmin(saundResid, x0, (M, phi, phiErr),
                              xtol=0.001, ftol=0.001, full_output=1, disp=0)
    xopt = res[0]
    chi2 = res[1]
    alpha = xopt[0]
    Mstar = xopt[1]
    sigma = xopt[2]
    lpstar = xopt[3]
    return (alpha, Mstar, sigma, lpstar)
        
def saundResid((alpha, Mstar, sigma, lpstar), M, phi, phiErr):
    """Return chi^2 residual between binned phi estimate and Saunders fit."""

    fc = 0
    phistar = 10**lpstar
    for ibin in range(len(M)):
        if phiErr[ibin] > 0:
            diff = phi[ibin] - phistar*saunders_lf(10**(0.4*(Mstar - M[ibin])),
                                                   alpha, sigma)
            fc += (diff/phiErr[ibin])**2
##     print 'saundResid ', alpha, Mstar, sigma, lpstar, fc
    return fc

# Plot Schechter function
def plotSchec(alpha, Mstar, phistar, Mmin, Mmax, lineStyle=':', axes=None):
    nstep = 100
    x = np.linspace(Mmin, Mmax, nstep)
    y = Schechter(x, alpha, Mstar, phistar)
    if axes:
        axes.plot(x, y, lineStyle)
    else:
        plt.plot(x, y, lineStyle)
        
# Schechter function
def Schechter(M, alpha, Mstar, phistar):
    L = 10**(0.4*(Mstar-M))
    schec = 0.4*ln10*phistar*L**(alpha+1)*np.exp(-L)
    return schec

# Plot Saunders function
def plotSaund(alpha, Mstar, sigma, phistar, Mmin, Mmax,
              lineStyle=':', axes=None):
    nstep = 100
    M = np.linspace(Mmin, Mmax, nstep)
    L = 10**(0.4*(Mstar - M))
    phi = phistar*saunders_lf(L, alpha, sigma)
    if axes:
        axes.plot(M, phi, lineStyle)
    else:
        plt.plot(M, phi, lineStyle)

##     plt.clf()
##     plt.plot(M, phi)
##     plt.xlabel('M'); plt.ylabel('Saunders LF')
##     plt.semilogy(basey=10, nonposy='clip')
##     plt.draw()
        
def wtdLineFit(x, y, w):
    """Weighted straight line fit y = a + bx.  w is inverse variance of y.
    See Bevington & Robinson, sec 6.3."""
    
    xsum = (x*w).sum()
    ysum = (y*w).sum()
    xxsum = (x*x*w).sum()
    xysum = (x*y*w).sum()
    wsum = w.sum()
            
    delta = wsum*xxsum - xsum*xsum
    a = (xxsum*ysum - xsum*xysum)/delta
    aVar = xxsum/delta
    b = (wsum*xysum - xsum*ysum)/delta
    bVar = wsum/delta

    return a, aVar, b, bVar

def wtdPolyFit(x, y, w, n):
    """Weighted polynomial fit of degree n.  w is inverse variance of y.
    See Bevington & Robinson, sec 7.2.  NB coefficents are returned 
    highest-order first for compatability with numpy polyfit and polyval."""

    beta = np.zeros(n+1)
    alpha = np.zeros((n+1, n+1))
    for i in range(n+1):
        beta[i] = (w*y*x**(n-i)).sum()
        for j in range(n+1):
            alpha[i,j] = (w*x**(n-i)*x**(n-j)).sum()
    cov = np.linalg.inv(alpha)
    a = np.dot(beta, cov)
    
    return a, cov

def test():
    """Function to test bits of code"""
        
    a = np.random.randint(0, 10, 10)
    print a
    
#------------------------------------------------------------------------------
# Misc plots for paper
#------------------------------------------------------------------------------

def colourCut(mag):
    """Return g-r colour cut corresponding to r-band absolute mag"""
##     return 0.85 - 0.033*mag
##     return 2.06 - 0.244*np.tanh((mag + 20.07)/1.09)
##     return 0.62 + z0 - 0.026*(mag + 20.0)
    return 0.15 - 0.03*mag

def colourMag(file='kcorrz.fits', imag=2, icol=(1,2),
              limits=(-23, -15.5, 0.1, 1.2)):
    """Plot colour-mag diagrams"""

    bands = 'ugriz'
    
    hdulist = pyfits.open(file)
    header = hdulist[1].header
    H0 = header['H0']
    omega_l = header['OMEGA_L']
    z0 = header['Z0']

    maglbl = r'$^{{{:2.1f}}}M_{}$'.format(z0, bands[imag])
    collbl = r'$^{{{:2.1f}}}({} - {})$'.format(z0, bands[icol[0]], bands[icol[1]])

    tbdata = hdulist[1].data
    z = tbdata.field('z')
    cataid = tbdata.field('cataid')
    ra = tbdata.field('ra')
    dec = tbdata.field('dec')
    zConf = tbdata.field('zConf')
    kcorr = tbdata.field('kcorr')
    mag = tbdata.field('absMag')[:,imag]
##     col = (tbdata.field('modelMagCor_{}'.format(bands[icol[0]])) -
##            kcorr[:,icol[0]]) - \
##           (tbdata.field('modelMagCor_{}'.format(bands[icol[1]])) -
##            kcorr[:,icol[1]])
    col = (tbdata.field('modelMagCor_g') - kcorr[:,1]) - \
          (tbdata.field('modelMagCor_r') - kcorr[:,2])
    hdulist.close()

    
    mags = np.linspace(limits[0], limits[1], 50)
    cutline = colourCut(mags)
    cut = colourCut(mag)
    nb = (col < cut).sum()
    nr = (col > cut).sum()
    print 'nblue, nred = ', nb, nr

    zlims = [[0.002, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.5]]
    fig = plt.figure(1)
    plt.clf()
    grid = AxesGrid(fig, 111, # similar to subplot(111)
                    nrows_ncols = (2,2), # creates nr*nc grid of axes
                    axes_pad=0.0, # pad between axes in inch.
                    aspect=False)

    for iz in range(4):
        ax = grid[iz]
        idx = (zlims[iz][0] <= z)*(z < zlims[iz][1])
        hist, xedges, yedges = np.histogram2d(col[idx], mag[idx], 50, 
                                              [limits[2:4], limits[0:2]])
        ax.contour(hist, 10, extent=limits)
##         ax.scatter(mag[idx], col[idx], 0.01)
        ax.plot(mags, cutline, '-')
        ax.axis(limits)
        ax.set_ylabel(collbl); ax.set_xlabel(maglbl)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
        title = r'${:2.1f} < z < {:2.1f}$'.format(zlims[iz][0], zlims[iz][1])
        ax.text(0.6, 0.9, title, transform = ax.transAxes)
        nb = (col[idx] < cut[idx]).sum()
        nr = (col[idx] > cut[idx]).sum()
        print zlims[iz], ' nblue, nred = ', nb, nr
    plt.draw()

    # Output faint-end red galaxies for eyeballing
    idx = (col > cut)*(mag > -16)
    cataid = cataid[idx]
    ra = ra[idx]
    dec = dec[idx]

    file = open('faint_red.txt', 'w')
    for i in xrange(len(ra)):
        print >> file, cataid[i], ra[i], dec[i]
    file.close()
    
def completeness(file='kcorr.fits', dZ=0.1, dM=0.1, nbin=50, iplot=100):
    """Test completeness using Tc & Tv statistics of Johnston, Teodoro & Hendry
    """

    # Nominal mag limits
    mb = 14
    mf = (20.3, 19.5, 19.4, 18.7, 18.2)
    Mmin, Mmax, Zmin, Zmax = -25, -10, 30, 45
    
    hdulist = pyfits.open(file)
    header = hdulist[1].header
    zRange = (0.002, 0.5)
    cosmo = CosmoLookup(header['H0'], header['omega_l'], zRange)

    tbdata = hdulist[1].data
    z = tbdata.field('z')
    nQ = tbdata.field('nQ')
    appMag = tbdata.field('appMag')[:,2]
    absMag = tbdata.field('absMag')[:,2]
    kcorr = tbdata.field('kcorr')[:,2]
    hdulist.close()

    idx = (nQ > 2) * (appMag >= mb) * (appMag < mf[2])
    
##     M = absMag[idx,:]
    Z = 5*np.log10(cosmo.dl(z[idx])) + 25
    M = appMag[idx] - Z
    m = appMag[idx]
    
    ngal = len(Z)
    galit = xrange(ngal)
    
    fig = plt.figure(1)
    plt.clf()
    plt.scatter(M, Z, s=0.1)
    plt.plot((Mmin, Mmax), (mf[2] - Mmin, mf[2] - Mmax))
    plt.plot((Mmin, Mmax), (mb - Mmin, mb - Mmax))
    plt.xlabel('M')
    plt.ylabel('Z')
    plt.axis((Mmin, Mmax, Zmin, Zmax))

    # M, Z lower limits for regions S1 - S4
    Zlo1 = np.where(Z - dZ > mb - M, Z - dZ, mb - M)
    Mlo1 = mb - Zlo1

    Mhi2 = mf[2] - Z
    Zlo2 = Zlo1

    Mlo3 = np.where(M - dM > mb - Z, M - dM, mb - Z)
    Zlo3 = mb - Mlo3

    Zhi4 = mf[2] - M
    Mlo4 = Mlo3

    plt.plot((Mlo1[iplot], M[iplot], M[iplot], Mlo1[iplot], Mlo1[iplot]),
             (Zlo1[iplot], Zlo1[iplot], Z[iplot], Z[iplot], Zlo1[iplot]))
    plt.plot((M[iplot], Mhi2[iplot], Mhi2[iplot], M[iplot], M[iplot]),
             (Zlo2[iplot], Zlo2[iplot], Z[iplot], Z[iplot], Zlo2[iplot]))
    plt.plot((Mlo3[iplot], M[iplot], M[iplot], Mlo3[iplot], Mlo3[iplot]),
             (Zlo3[iplot], Zlo3[iplot], Z[iplot], Z[iplot], Zlo3[iplot]))
    plt.plot((Mlo4[iplot], M[iplot], M[iplot], Mlo4[iplot], Mlo4[iplot]),
             (Z[iplot], Z[iplot], Zhi4[iplot], Zhi4[iplot], Z[iplot]))

    plt.draw()

    r = np.array(map(lambda i:
                     float(len(np.where((Mlo1[i] <= M) * (M <= M[i]) *
                                  (Zlo1[i] <= Z) * (Z <= Z[i]))[0])), galit))

    nbin = 20
    Tc = np.zeros(nbin)
    mfarray = np.linspace(mb, mf[2] + 0.5, nbin)
    ibin = 0
    for mfs in mfarray:
        idx = np.where(m < mfs)[0]
        ntest = len(idx)
        Mt = M[idx]
        Zt = Z[idx]
        galit = xrange(ntest)
        
        # M, Z limits for regions S2 & S4
        Mhi2 = mfs - Zt
        Zlo2 = Zlo1[idx]

        Zhi4 = mfs - Mt
        Mlo4 = Mlo3[idx]

        n = np.array(map(lambda i:
                         float(len(np.where((Mlo1[idx[i]] <= Mt) *
                                            (Mt <= Mhi2[i]) *
                                            (Zlo2[i] <= Zt) *
                                            (Zt <= Zt[i]))[0])),
                         galit))

        zeta = r[idx]/(n + 1)
        var_zeta = (n-1)/(n+1)/12
        ## idx = np.where(std_zeta > 0)
        ## Tc[ibin] = ((zeta[idx] - 0.5)/std_zeta[idx]).sum()
        Tc[ibin] = ((zeta - 0.5)).sum()/np.sqrt((var_zeta).sum())
        ## pdb.set_trace()
        ibin += 1
        
    plt.clf()
    plt.plot(mfarray, Tc)
    plt.xlabel(r'$m_f$')
    plt.ylabel(r'$Tc$')
    plt.axis((mfarray[0], mfarray[-1], -9, 9))
    plt.draw()
    
