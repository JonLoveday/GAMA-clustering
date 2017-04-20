# Routines for clustering analysis

from __future__ import division
from __future__ import print_function

import copy
import glob
import math
import numpy as np
import os
import os.path
import pdb
import pickle
#from astLib import astCalc
import matplotlib as mpl
if not('DISPLAY' in os.environ):
    mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from mayavi import mlab
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import scipy.spatial
import scipy.stats
import subprocess
#import triangle
import warnings

from astropy.io import fits
import emcee
import h5py

import jswml
import lum
import util

# Catch invalid values in numpy calls
np.seterr(divide='warn')
np.seterr(invalid='warn')
#np.seterr(invalid='raise')
#warnings.simplefilter('error')

# Global parameters
gama_data = os.environ['GAMA_DATA']
H0 = 100
omega_l = 0.7
ln10 = math.log(10)
root2 = math.sqrt(2)
root2pi = math.sqrt(2*math.pi)
# Default evolution parameters
Qdef, Pdef = 1.0, 1.0
def_mass_limits = (8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5)
def_mag_limits = (-23, -22, -21, -20, -19, -18, -17, -16, -15)
def_binning = (-2, 2, 20, 0, 100, 100)
def_theta_max = 12
def_J3_pars = (1.84, 5.59, 30)
qsub_xi_cmd = 'qsub /research/astro/gama/loveday/Documents/Research/python/apollo_job.sh $BIN/xi {} {} {}'
qsub_xia_cmd = 'qsub /research/astro/gama/loveday/Documents/Research/python/apollo_job.sh $BIN/xi {} {}'
qsub_xix_cmd = 'qsub /research/astro/gama/loveday/Documents/Research/python/apollo_job.sh $BIN/xi {} {} {}'

# Default jackknife regions
njack = 9
ra_limits = [[129.0, 133.0], [133.0, 137.0], [137.0, 141.0],
             [174.0, 178.0], [178.0, 182.0], [182.0, 186.0],
             [211.5, 215.5], [215.5, 219.5], [219.5, 223.5]]

# Standard symbol and colour order for plots
#symb_list = 'os^<v>p*os^<v>p*os^<v>p*'
def_plot_size = (5, 3.5)
clr_list = 'bgrmyckbgrmyckbgrmyck'
symb_list = ('ko', 'bs', 'g^', 'r<', 'mv', 'y>', 'cp')
line_list = ('k-', 'b--', 'g:', 'r-.', 'm-', 'y--', 'c:')
sl_list = ('ko-', 'bs--', 'g^:', 'r<-.', 'mv-', 'y>--', 'cp:')
mpl.rcParams['image.cmap'] = 'viridis'
xlabel = {'xis': r'$s\ [h^{-1}{\rm Mpc}]$',
          'xi2': '',
          'w_p': r'$r_p\ [h^{-1}{\rm Mpc}]$',
          'xir': r'$r\ [h^{-1}{\rm Mpc}]$', 'bias': r'$M_r$'}
ylabel = {'xis': r'$\xi(s)$', 'xi2': '',
          'w_p': r'$w_p(r_p)$', 'xir': r'$\xi(r)$',
          'bias': r'$b(M) / b(M^*)$'}

# Directory to save plots
plot_dir = os.environ['HOME'] + '/Documents/tex/papers/gama/pvd/'


# -----
# Tests
# -----

def test(infile=gama_data+'/jswml/auto/kcorrz01.fits', ran_dist='vol',
         Q=Qdef, P=Pdef, key='w_p', xlimits=(0.01, 100)):
    """Test basic functionality of sample selection, correlation function
    calculation and plotting on a small data sample."""

    Mlimits = (-22, -21)
    zlimits = util.vol_limits(infile, Q=Q, Mlims=Mlimits)
    z_range = [0.002, zlimits[1]]
    galout = 'gal_test.dat'
    ranout = 'ran_test.dat'
    xiout = 'xi_test.dat'
    xi_select(infile, galout, ranout, xiout,
              z_range=z_range, nz=20, app_range=(14, 19.8),
              abs_range=Mlimits,
              Q=Q, P=P, ran_dist=ran_dist, ran_fac=1)

    # Run the clustering code executable in $BIN/xi, compiled from xi.c
#    cmd = '$BIN/xi {} {} {}'.format(galout, ranout, xiout)
#    subprocess.call(cmd, shell=True)
    cmd = '$BIN/xi {} {}'.format(galout, 'gg_test.dat')
    subprocess.call(cmd, shell=True)
    cmd = '$BIN/xi {} {} {}'.format(galout, ranout, 'gr_test.dat')
    subprocess.call(cmd, shell=True)
    cmd = '$BIN/xi {} {}'.format(ranout, 'rr_test.dat')
    subprocess.call(cmd, shell=True)

    # Plot the results
    panels = []
    comps = []
    label = 'Test'
    panels.append({'files': (xiout, ), 'comps': comps, 'label': label})
    xi_plot(key, panels, xlimits=xlimits)
    plt.show()
    xi2d_plot(xiout, binning=0, mirror=0)
    plt.show()
    xi2d_plot(xiout, binning=1, mirror=0)
    plt.show()
    xi2d_plot(xiout, binning=2, mirror=0)
    plt.show()
#    xi_plot('xi2', panels, binning=0, xlimits=xlimits)
#    xi_plot('xi2', panels, binning=1, xlimits=xlimits)
#    xi_plot('xi2', panels, binning=2, xlimits=xlimits)


def xtest(infile=gama_data+'/jswml/auto/kcorrz01.fits', ran_dist='vol',
          Q=Qdef, P=Pdef, key='w_p', xlimits=(0.01, 100), run=1,
          pi_lim=100, rp_lim=100, onevol=0):
    """Test cross-correlation using two luminsoity-selected samples
    within a volume-limited sample if onevol is True."""

    if run > 0:
        Mlimits = (-21, -20, -19)
        if onevol:
            zlimits = util.vol_limits(infile, Q=Q, Mlims=(Mlimits[-1],))
        else:
            zlimits = util.vol_limits(infile, Q=Q, Mlims=Mlimits[1:3])
        for ilim in xrange(2):
            if onevol:
                z_range = [0.002, zlimits[0]]
            else:
                z_range = [0.002, zlimits[ilim]]
            Mrange = Mlimits[ilim:ilim+2]
            galout = 'gal_test_{}.dat'.format(ilim)
            ranout = 'ran_test_{}.dat'.format(ilim)
            xiout = 'xi_test_{}.dat'.format(ilim)
            xi_select(infile, galout, ranout, xiout,
                      z_range=z_range, nz=20, app_range=(14, 19.8),
                      abs_range=Mrange,
                      Q=Q, P=P, ran_dist=ran_dist, ran_fac=5, run=run)

    # Cross counts
    if run == 1:
        cmd = '$BIN/xi {} {} {}'.format('gal_test_0.dat', 'gal_test_1.dat',
                                        'gg_test_x.dat')
        subprocess.call(cmd, shell=True)
        cmd = '$BIN/xi {} {} {}'.format('gal_test_0.dat', 'ran_test_1.dat',
                                        'gr_test_x.dat')
        subprocess.call(cmd, shell=True)
    if run == 2:
        cmd = qsub_xix_cmd.format('gal_test_0.dat', 'gal_test_1.dat',
                                  'gg_test_x.dat')
        subprocess.call(cmd, shell=True)
        cmd = qsub_xix_cmd.format('gal_test_0.dat', 'ran_test_1.dat',
                                  'gr_test_x.dat')
        subprocess.call(cmd, shell=True)

    # Plot the results
    panels = []
    comps = []
    label = 'Test'
    panels.append({'files': ('xi_test_0.dat', 'xi_test_1.dat'),
                   'comps': comps, 'label': label})
    xi_plot(key, panels, xlimits=xlimits)
    plt.show()
#    xi2d_plot(xiout, binning=0, mirror=0)
#    plt.show()
#    xi2d_plot(xiout, binning=1, mirror=0)
#    plt.show()
#    xi2d_plot(xiout, binning=2, mirror=0)
#    plt.show()
    Gg = PairCounts('gg_test_x.dat')
    Gr = PairCounts('gr_test_x.dat')
    gr = PairCounts('gr_test_1.dat')
    rr = PairCounts('rr_test_1.dat')
    counts = {'Gg': Gg, 'Gr': Gr, 'gr': gr, 'rr': rr}
    xi = Xi()
    w_p_dpx = xi.est(counts, dpx, key=key, pi_lim=pi_lim, rp_lim=rp_lim)
    w_p_lsx = xi.est(counts, lsx, key=key, pi_lim=pi_lim, rp_lim=rp_lim)
    plt.clf()
    ax = plt.subplot(111)
    w_p_dpx.plot(ax, label='DPX')
    w_p_lsx.plot(ax, label='LSX')
    ax.loglog(basex=10, basey=10, nonposy='clip')
    ax.set_xlabel(r'$r_p\ [h^{-1}{\rm Mpc}]$')
    ax.set_ylabel(r'$w_p(r_p)$')
    ax.legend()
    plt.show()


# -------
# Classes
# -------

class Cat(object):
    """Galaxy or random catalogue."""

    def __init__(self, ra, dec, r, weight=None, den=None, Vmax=None, info=""):
        # Trim tail of high-redshift objects
        idx = r < info['rcut']
        ra, dec, r = ra[idx], dec[idx], r[idx]
        try:
            weight, den, Vmax = weight[idx], den[idx], Vmax[idx]
        except:
            pass
        self.nobj = len(ra)
        self.ra = ra
        self.dec = dec
        self.x = r*np.cos(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
        self.y = r*np.sin(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
        self.z = r*np.sin(np.deg2rad(dec))
        if weight is None: weight = np.ones(self.nobj)
        self.weight = weight
        if den is None: den = np.ones(self.nobj)
        self.den = den
        if Vmax is None: Vmax = np.ones(self.nobj)
        self.Vmax = Vmax
        self.info = info
           
    def output(self, outfile, binning=def_binning, theta_max=def_theta_max,
               J3_pars=def_J3_pars):
        """Output the galaxy or random data for xi.c v 2.1."""
        
        #  3 jackknife regions per GAMA field (each 4x4 deg).  Single cell.
        ncell = 1
        ix = 0
        iy = 0
        iz = 0
        cellsize = 100.0
        njack = self.info['njack']

        print('Writing out ', outfile)
        fout = open(outfile, 'w')
        print(self.info, file=fout)
        print(self.nobj, ncell, ncell, njack, cellsize,
              binning[0], binning[1], binning[2],
              binning[3], binning[4], binning[5],
              theta_max, J3_pars[0], J3_pars[1], J3_pars[2], file=fout)
        print(ix, iy, iz, self.nobj, file=fout)
        for i in xrange(self.nobj):
            if njack > 1:
                for ireg in range(njack):
                    if (ra_limits[ireg][0] <= self.ra[i] <= ra_limits[ireg][1]):
                        ijack = ireg
            else:
                ijack = 0
            print(self.x[i], self.y[i], self.z[i], self.weight[i],
                  self.den[i], self.Vmax[i], ijack, file=fout)
        fout.close()


class PairCounts(object):
    """Class to hold pair counts."""

    def __init__(self, infile=None, pi_rebin=1, rp_rebin=1):
        """Read pair counts from file if specified with optional rebinning."""

        if infile is None:
            return

        f = open(infile, 'r')
        f.readline()
        self.info = eval(f.readline())

        args = f.readline().split()
        self.na = float(args[0])
        self.nb = float(args[1])
        self.njack = int(args[2])
        self.n2d = int(args[3])

        # Read direction-averaged counts
        args = f.readline().split()
        self.ns = int(args[0])
        self.smin = float(args[1])
        self.smax = float(args[2])
        self.sep = np.zeros(self.ns)
        self.pc = np.zeros((self.ns, self.njack+1))
        for i in range(self.ns):
            data = f.readline().split()
            self.sep[i] = float(data[0])
            self.pc[i, :] = map(float, data[1:])
        if self.nb > 0:
            self.pcn = self.pc/self.na/self.nb
        else:
            self.pcn = 2*self.pc/self.na/(self.na - 1)

        # Read counts for 2d binnings
        self.pc2_list = []
        for i2d in range(self.n2d):
            args = f.readline().split()
            nrp = int(args[0])
            rpmin = float(args[1])
            rpmax = float(args[2])
            npi = int(args[3])
            pimin = float(args[4])
            pimax = float(args[5])
            pi = np.zeros((npi, nrp))
            rp = np.zeros((npi, nrp))
            pc = np.zeros((npi, nrp, self.njack+1))
            for i in range(nrp):
                for j in range(npi):
                    data = f.readline().split()
                    pi[j, i] = float(data[0])
                    rp[j, i] = float(data[1])
                    pc[j, i, :] = map(float, data[2:])

            # Rebin counts
            if rp_rebin * pi_rebin > 1:
                npibin = npi//pi_rebin
                nrpbin = nrp//rp_rebin
                pibin = np.zeros((npibin, nrpbin))
                rpbin = np.zeros((npibin, nrpbin))
                pcbin = np.zeros((npibin, nrpbin, self.njack+1))
                for i in range(0, nrp, rp_rebin):
                    ib = i//rp_rebin
                    for j in range(0, npi, pi_rebin):
                        jb = j//pi_rebin
                        for ii in range(i, min(nrp, i + rp_rebin)):
                            for jj in range(j, min(npi, j + pi_rebin)):
                                pibin[jb, ib] += pc[jj, ii, 0] * pi[jj, ii]
                                rpbin[jb, ib] += pc[jj, ii, 0] * rp[jj, ii]
                                pcbin[jb, ib, :] += pc[jj, ii, :]
                                if pcbin[jb, ib, 0] > 0:
                                    pibin[jb, ib] /= pcbin[jb, ib, 0]
                                    rpbin[jb, ib] /= pcbin[jb, ib, 0]
                npi = npibin
                nrp = nrpbin
                pi = pibin
                rp = rpbin
                pc = pcbin

            if self.nb > 0:
                pcn = pc/self.na/self.nb
            else:
                pcn = 2*pc/self.na/(self.na - 1)
            self.pc2_list.append(
                {'npi': npi, 'pimin': pimin, 'pimax': pimax, 'pi': pi,
                 'nrp': nrp, 'rpmin': rpmin, 'rpmax': rpmax, 'rp': rp,
                 'pc': pc, 'pcn': pcn})
        f.close()

    def sum(self, pcs):
        """Sum over GAMA regions."""
        nest = len(pcs)
        ests = xrange(nest)
        self.na = np.sum([pcs[i].na for i in ests])
        self.nb = np.sum([pcs[i].nb for i in ests])
        self.njack = pcs[0].njack
        self.n2d = pcs[0].n2d
        self.info = pcs[0].info

        # Direction-averaged counts
        self.ns = pcs[0].ns
        self.smin = pcs[0].smin
        self.smax = pcs[0].smax
        self.sep = np.ma.average(
            [pcs[i].sep for i in ests], axis=0,
            weights=[pcs[i].pc[:, 0] for i in ests]).filled(0)
        self.pc = np.zeros((self.ns, self.njack+1))
        self.pc[:, 0] = np.sum([pcs[i].pc[:, 0] for i in ests], axis=0)

        # Counts for 2d binnings
        self.pc2_list = []
        for i2d in range(self.n2d):
            nrp = pcs[0].pc2_list[i2d]['nrp']
            rpmin = pcs[0].pc2_list[i2d]['rpmin']
            rpmax = pcs[0].pc2_list[i2d]['rpmax']
            npi = pcs[0].pc2_list[i2d]['npi']
            pimin = pcs[0].pc2_list[i2d]['pimin']
            pimax = pcs[0].pc2_list[i2d]['pimax']
            pi = np.ma.average(
                [pcs[i].pc2_list[i2d]['pi'] for i in ests], axis=0,
                weights=[pcs[i].pc2_list[i2d]['pc'][:, :, 0]
                         for i in ests]).filled(0)
            rp = np.ma.average(
                [pcs[i].pc2_list[i2d]['rp'] for i in ests], axis=0,
                weights=[pcs[i].pc2_list[i2d]['pc'][:, :, 0]
                         for i in ests]).filled(0)
            pc = np.zeros((npi, nrp, self.njack+1))
            pc[:, :, 0] = np.sum([pcs[i].pc2_list[i2d]['pc'][:, :, 0]
                                 for i in ests], axis=0)
            self.pc2_list.append(
                {'npi': npi, 'pimin': pimin, 'pimax': pimax, 'pi': pi,
                 'nrp': nrp, 'rpmin': rpmin, 'rpmax': rpmax, 'rp': rp,
                 'pc': pc})

    def average(self, pcs):
        """Average over different estimates."""
        nest = len(pcs)
        ests = xrange(nest)
        self.na = np.mean([pcs[i].na for i in ests])
        self.nb = np.mean([pcs[i].nb for i in ests])
        self.njack = nest
        self.n2d = pcs[0].n2d
        self.info = pcs[0].info

        # Direction-averaged counts
        self.ns = pcs[0].ns
        self.smin = pcs[0].smin
        self.smax = pcs[0].smax
        self.sep = np.ma.average(
            [pcs[i].sep for i in ests], axis=0,
            weights=[pcs[i].pc[:, 0] for i in ests]).filled(0)
        self.pc = np.zeros((self.ns, self.njack+1))
        self.pc[:, 0] = np.mean([pcs[i].pc[:, 0] for i in ests], axis=0)
        self.pc[:, 1:] = np.array([pcs[i].pc[:, 0] for i in ests]).T
#        pdb.set_trace()
        # Counts for 2d binnings
        self.pc2_list = []
        for i2d in range(self.n2d):
            nrp = pcs[0].pc2_list[i2d]['nrp']
            rpmin = pcs[0].pc2_list[i2d]['rpmin']
            rpmax = pcs[0].pc2_list[i2d]['rpmax']
            npi = pcs[0].pc2_list[i2d]['npi']
            pimin = pcs[0].pc2_list[i2d]['pimin']
            pimax = pcs[0].pc2_list[i2d]['pimax']
            pi = np.ma.average(
                [pcs[i].pc2_list[i2d]['pi'] for i in ests], axis=0,
                weights=[pcs[i].pc2_list[i2d]['pc'][:, :, 0]
                         for i in ests]).filled(0)
            rp = np.ma.average(
                [pcs[i].pc2_list[i2d]['rp'] for i in ests], axis=0,
                weights=[pcs[i].pc2_list[i2d]['pc'][:, :, 0]
                         for i in ests]).filled(0)
            pc = np.zeros((npi, nrp, self.njack+1))
            pc[:, :, 0] = np.mean([pcs[i].pc2_list[i2d]['pc'][:, :, 0]
                                  for i in ests], axis=0)
            pc[:, :, 1:] = np.transpose(np.array(
                [pcs[i].pc2_list[i2d]['pc'][:, :, 0] for i in ests]),
                (1, 2, 0))
            self.pc2_list.append(
                {'npi': npi, 'pimin': pimin, 'pimax': pimax, 'pi': pi,
                 'nrp': nrp, 'rpmin': rpmin, 'rpmax': rpmax, 'rp': rp,
                 'pc': pc})

    def write(self, outfile):
        """Write pair counts to file."""

        f = open(outfile, 'w')
        print('PairCounts.write() output', file=f)
        print(self.info, file=f)
        print(self.na, self.nb, self.njack, self.n2d, file=f)

        print(self.ns, self.smin, self.smax, file=f)
        for i in range(self.ns):
            print(self.sep[i], ' '.join(map(str, self.pc[i, :])), file=f)

        for i2d in range(self.n2d):
            pc2 = self.pc2_list[i2d]
            print(pc2['nrp'], pc2['rpmin'], pc2['rpmax'],
                  pc2['npi'], pc2['pimin'], pc2['pimax'], file=f)
            for i in range(pc2['nrp']):
                for j in range(pc2['npi']):
                    print(pc2['pi'][j, i], pc2['rp'][j, i],
                          ' '.join(map(str, pc2['pc'][j, i, :])), file=f)
        f.close()


class Xi(object):
    """Class to hold clustering estimates."""

    def __init__(self):
        """Placeholder initialiser."""

    def est(self, counts, estimator, key='w_p', binning=1,
            pi_lim=100, rp_lim=100):
        """Calculate xi(s) and xi(rp,pi) from pair counts
        using specified estimator."""

        if 'Gg' in counts:
            galpairs = counts['Gg']
        else:
            galpairs = counts['gg']
        if 'rr' in counts:
            ranpairs = counts['rr']
        else:
            ranpairs = counts['Gr']
        self.info = galpairs.info
        self.njack = galpairs.njack
        self.n2d = galpairs.n2d
        self.err_type = self.info['err_type']

        # Direction-averaged xi(s)
        ns = galpairs.ns
        smin = galpairs.smin
        smax = galpairs.smax
        xis = Xi1d(ns, self.njack, smin, smax, 'xis', self.err_type)
        xis.sep = galpairs.sep
        xis.galpairs = galpairs.pc[:, 0]
        xis.ranpairs = ranpairs.pc[:, 0]
        xis.est = estimator(counts, -1)

        # xi(r_p, pi) and w_p(r_p) for 2d binnings
        xi2_list = []
        for i2d in range(self.n2d):
            gal2 = galpairs.pc2_list[i2d]
            ran2 = ranpairs.pc2_list[i2d]
            nrp = gal2['nrp']
            rpmin = gal2['rpmin']
            rpmax = gal2['rpmax']
            npi = gal2['npi']
            pimin = gal2['pimin']
            pimax = gal2['pimax']

            rpstep = (rpmax - rpmin)/nrp
            pistep = (pimax - pimin)/npi

            if pimin < 0:
                pilim = min(pimax, math.log10(pi_lim))
            else:
                pilim = min(pimax, pi_lim)
            npi_use = int((pilim - pimin)/pistep)
            pilim = pimin + npi_use*pistep

            if rpmin < 0:
                rplim = min(rpmax, math.log10(rp_lim))
            else:
                rplim = min(rpmax, rp_lim)
            nrp_use = int((rplim - rpmin)/rpstep)
            rplim = rpmin + nrp_use*rpstep

            xi2 = Xi2d(nrp_use, rpmin, rplim, npi_use, pimin, pilim,
                       self.njack, self.err_type)
            xi2.pi = gal2['pi'][:npi_use, :nrp_use]
            xi2.rp = gal2['rp'][:npi_use, :nrp_use]
            xi2.galpairs = gal2['pc'][:npi_use, :nrp_use]
            xi2.ranpairs = ran2['pc'][:npi_use, :nrp_use]
            xi2.est = estimator(counts, i2d)[:npi_use, :nrp_use, :]
            xi2_list.append(xi2)
        self.xis = xis
        self.xi2_list = xi2_list

        if key == 'xis':
            xis = self.xis
            xis.clear_empties()
            xis.cov = Cov(xis.est[:, 1:], self.err_type)
            return xis

        xi2 = self.xi2_list[binning]
        if key == 'xi2':
#            xi2.cov = Cov(xi2.est[:, :, 1:], self.err_type)
            return xi2
        w_p = xi2.w_p(rp_lim, pi_lim)
        if key == 'w_p':
            w_p.cov = Cov(w_p.est[:, 1:], self.err_type)
            return w_p
        xir = w_p.xir()
        xir.cov = Cov(xir.est[:, 1:], self.err_type)
        return xir


class Xi1d(object):
    """1d clustering estimate, including jackknife sub-estimates."""

    def __init__(self, nbin, njack, rmin, rmax, xi_type, err_type):

        self.nbin = nbin
        self.njack = njack
        self.rmin = rmin
        self.rmax = rmax
        self.rstep = (rmax - rmin)/nbin
        self.sep = np.zeros(nbin)
        self.galpairs = np.zeros(nbin)
        self.ranpairs = np.zeros(nbin)
        self.est = np.zeros((nbin, njack+1))
        self.ic = 0.0
        self.xi_type = xi_type
        self.err_type = err_type

    def clear_empties(self):
        """Remove any empty bins with zero galaxy-galaxy pairs."""
        keep = self.galpairs > 0
        self.sep, self.galpairs, self.ranpairs, self.est = \
            self.sep[keep], self.galpairs[keep], self.ranpairs[keep],\
            self.est[keep]
        self.nbin = len(self.sep)
        return self.nbin

    def xir(self):
        """Inversion of w_p(r_p) to xi(r) - Saunders et al 1992, eq 26.
        Assumes log binning."""

        def invert(rp, wp, njack):
            nbin = len(rp)
            xi = np.zeros((nbin-1, njack+1))
            for i in range(nbin-1):
                sum = 0.0
                for j in range(i, nbin-1):
                    try:
                        sum += ((wp[j+1, :] - wp[j, :])/(rp[j+1] - rp[j]) *
                                math.log((rp[j+1] +
                                          math.sqrt(rp[j+1]**2 - rp[i]**2)) /
                                         (rp[j] + math.sqrt(rp[j]**2 - rp[i]**2))))
                    except:
                        pass
                xi[i, :] = -sum/math.pi
            return xi

        xir = Xi1d(self.nbin-1, self.njack, self.rmin, self.rmax,
                   'xir', self.err_type)
        xir.sep = self.sep[:-1]
        xir.galpairs = self.galpairs[:-1]
        xir.ranpairs = self.ranpairs[:-1]
        xir.est = invert(self.sep, self.est, self.njack)
#        for ijack in range(self.njack):
#            xir.jack[:, ijack] = invert(self.sep, self.jack[:, ijack])
        xir.cov = Cov(xir.est[:, 1:], xir.err_type)
        return xir

    def ic_calc(self, gamma, r0, ic_rmax):
        """Returns estimated integral constraint for power law xi(r)
        truncated at ic_rmax."""
        xi_mod = np.zeros(len(self.sep))
        pos = (self.sep > 0) * (self.sep < ic_rmax)
        xi_mod[pos] = (self.sep[pos]/r0)**-gamma
        self.ic = (self.ranpairs * xi_mod).sum() / (self.ranpairs).sum()

    def plot(self, ax, jack=0, color=None, fout=None, label=None, pl_div=None):
        if pl_div:
            pl_fit = (self.sep/pl_div[0])**(- pl_div[1])
        else:
            pl_fit = 1
#        if color:
        ax.errorbar(self.sep, self.est[:, jack]/pl_fit + self.ic,
                    self.cov.sig/pl_fit,
                    fmt='o', color=color, label=label, capthick=1)
#        else:
#            ax.errorbar(self.sep, self.est[:, jack] + self.ic, self.cov.sig,
#                        fmt='o', label=label, capthick=1)

        if fout:
            print(label, file=fout)
            for i in range(self.nbin):
                print(self.sep[i], self.est[i, jack] + self.ic,
                      self.cov.sig[i], file=fout)

    def fit(self, fit_range, jack=0, logfit=0, ic_rmax=0, neig=0,
            verbose=1, ax=None, cov_ax=None, covn_ax=None, color=None):
        """Fit a power law to main and jackknife estimates."""

        def dofit(x, y, cov, neig=neig):
            """Do the fit."""

            def fit_chi2(p, x, y, cov, neig):
                # returns chi^2 for given power-law parameters p
                if logfit:
                    fit = x*p[1] + p[0]
                else:
                    fit = (x/p[0])**-p[1]
                return cov.chi2(y, fit, neig)

            pinit = [5.0, 1.8]
            out = scipy.optimize.fmin(fit_chi2, pinit, args=(x, y, cov, neig),
                                      full_output=1, disp=0)
            p = out[0]
            chisq = out[1]
            if neig in(0, 'all', 'full'):
                nu = len(x) - 2
            else:
                nu = neig - 2

            if logfit:
                gamma = -p[1]
                r0 = math.exp(-p[0]/p[1])
            else:
                gamma = p[1]
                r0 = p[0]
            if self.xi_type == 'w_p':
                gamma += 1
                r0 = (r0**(gamma-1)/scipy.special.gamma(0.5) /
                      scipy.special.gamma(0.5*(gamma-1)) *
                      scipy.special.gamma(0.5*gamma))**(1.0/gamma)
            return gamma, r0, p, chisq, nu

        idx = ((fit_range[0] < self.sep) * (self.sep < fit_range[1]) *
               (self.galpairs > 0) * np.all(self.est > 0, axis=1))
        if len(self.sep[idx]) < 2:
            print('Insufficient valid bins for fit')
#            pdb.set_trace()
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if logfit:
            sep = np.log(self.sep[idx])
            est = np.log(self.est[idx, :])
        else:
            sep = self.sep[idx]
            est = self.est[idx, :]
        cov = Cov(est[:, 1:], self.err_type)
        if cov_ax:
            cov.plot(ax=cov_ax)
        if covn_ax:
            cov.plot(norm=1, ax=covn_ax)

        # Main estimate
        if ic_rmax:
            dic = 1
            niter = 0
            while dic > 0.01 and niter < 10:
                ic_old = self.ic
                y = est[:, jack] + self.ic
                gamma, r0, p, chisq, nu = dofit(sep, y, cov, neig)

                self.ic_calc(p[1], p[0], ic_rmax)
                dic = math.fabs(self.ic - ic_old)
                niter += 1
            if dic > 0.01:
                print('IC failed to converge', self.ic, ic_old)
        else:
            y = est[:, jack]
            gamma, r0, p, chisq, nu = dofit(sep, y, cov, neig)

        fra = np.array(fit_range)
        if logfit:
            yfit = np.exp(p[1]*np.log(fra) + p[0])
        else:
            yfit = (fra/p[0])**-p[1]
        if ax:
            if color:
                ax.plot(fra, yfit, color=color)
            else:
                ax.plot(fra, yfit)

        # Jackknife estimates
        r0_jack = []
        gamma_jack = []
        for ijack in xrange(self.njack):
            y = est[:, ijack+1] + self.ic
            gamma_j, r0_j, p, chisq_j, nu_j = dofit(sep, y, cov, neig)
            if not(math.isnan(gamma_j)) and not(math.isnan(r0_j)):
                gamma_jack.append(gamma_j)
                r0_jack.append(r0_j)
        gamma_err = jack_err(gamma_jack, self.err_type)
        r0_err = jack_err(r0_jack, self.err_type)

        if verbose:
            print('gamma {:4.2f}+/-{:4.2f} r_0 {:4.2f}+/-{:4.2f} chi^2/nu {:4.2f}/{:2d} IC {:4.2f}'.format(
                gamma, gamma_err, r0, r0_err, chisq, nu, self.ic))
        return gamma, gamma_err, r0, r0_err, self.ic, gamma_jack, r0_jack

    def interp(self, r, jack=0, log=False):
        """Returns interpolated value and error (zero for r > r_max).
        Interpolates in log-log space if log=True."""
        if log:
            return np.expm1(np.interp(np.log(r), np.log(self.sep),
                                      np.log1p(self.est[:, jack]), right=0)), \
                   np.expm1(np.interp(np.log(r), np.log(self.sep),
                                      np.log1p(self.cov.sig)))
        else:
            return np.interp(r, self.sep, self.est[:, jack], right=0), \
                   np.interp(r, self.sep, self.cov.sig)


class Xi2d(object):
    """2d clustering estimate."""

    def __init__(self, nrp, rpmin, rpmax, npi, pimin, pimax, njack, err_type):
        self.nrp = nrp
        self.rpmin = rpmin
        self.rpmax = rpmax
        self.rpstep = (rpmax - rpmin)/nrp
        self.rpc = rpmin + (np.arange(nrp) + 0.5) * self.rpstep
        if rpmin < 0:
            self.rpc = 10**self.rpc
        self.npi = npi
        self.pimin = pimin
        self.pimax = pimax
        self.pistep = (pimax - pimin)/npi
        self.pic = pimin + (np.arange(npi) + 0.5) * self.pistep
        if pimin < 0:
            self.pic = 10**self.pic
        self.rp, self.pi = np.meshgrid(self.rpc, self.pic)
        self.njack = njack
        self.est = np.zeros((npi, nrp, njack+1))
        self.err_type = err_type
        self.galpairs = np.zeros((npi, nrp, njack+1))
        self.ranpairs = np.zeros((npi, nrp, njack+1))

    def reflect(self, axes=(0, 1)):
        """Reflect 2d correlation function about specified axes."""

        # Ensure that axes is a tuple
        try:
            n = len(axes)
        except:
            axes = (axes,)

        npi = self.npi
        pi0 = 0
        nrp = self.nrp
        rp0 = 0
        pimin = self.pimin
        pimax = self.pimax
        rpmin = self.rpmin
        rpmax = self.rpmax
        if 0 in axes:
            pi0 = npi
            npi *= 2
            pimin = -pimax
        if 1 in axes:
            rp0 = nrp
            nrp *= 2
            rpmin = -rpmax
        xir = Xi2d(nrp, rpmin, rpmax, npi, pimin, pimax,
                   self.njack, self.err_type)
        xir.est[pi0:, rp0:, :] = self.est
        xir.pi[pi0:, rp0:] = self.pi
        xir.rp[pi0:, rp0:] = self.rp
        xir.pic[pi0:] = self.pic
        xir.rpc[rp0:] = self.rpc
        if 1 in axes:
            xir.est[pi0:, :rp0, :] = np.fliplr(self.est)
            xir.pi[pi0:, :rp0] = np.fliplr(self.pi)
            xir.rp[pi0:, :rp0] = np.fliplr(self.rp)
            xir.rpc[:rp0] = -self.rpc[::-1]
        if 0 in axes:
            xir.est[:pi0, rp0:, :] = np.flipud(self.est)
            xir.pi[:pi0, rp0:] = np.flipud(self.pi)
            xir.rp[:pi0, rp0:] = np.flipud(self.rp)
            xir.pic[:pi0] = -self.pic[::-1]
        if 0 in axes and 1 in axes:
            xir.est[:pi0, :rp0, :] = np.flipud(np.fliplr(self.est))
            xir.pi[:pi0, :rp0] = np.flipud(np.fliplr(self.pi))
            xir.rp[:pi0, :rp0] = np.flipud(np.fliplr(self.rp))
#        xir.cov = Cov(xir.est[:, :, 1:], self.err_type)
        return xir

    def beta_model(self, beta, xir=None, r0=None, gamma=None, meansep=0,
                   interplog=0, epsabs=1e-5, epsrel=1e-5):
        """Kaiser/Hamilton model of 2d correlation function."""
        fac0 = 1 + 2*beta/3 + beta**2/5
        fac2 = 4*beta/3 + 4*beta**2/7
        fac4 = 8*beta**2/35
        if meansep:
            # Use mean separation rather than bin centres
            rpgrid = self.rp
            pigrid = self.pi
        else:
            rpgrid, pigrid = np.meshgrid(self.rpc, self.pic)
        s = (rpgrid**2 + pigrid**2)**0.5
        mu = pigrid / s
        P2 = 0.5*(3*mu**2 - 1)
        P4 = 0.125*(35*mu**4 - 30*mu**2 + 3)
        if xir:
            xi0 = np.interp(s, xir.r, xir.xi0, right=0)
            xi2 = np.interp(s, xir.r, xir.xi2, right=0)
            xi4 = np.interp(s, xir.r, xir.xi4, right=0)
            self.est[:, :, 0] = xi0*fac0 + xi2*fac2*P2 + xi4*fac4*P4
#            pdb.set_trace()
        else:
            xi = (s/r0)**-gamma
            self.est[:, :, 0] = xi*(fac0 + fac2*(gamma/(gamma-3))*P2 +
                                    fac4*gamma*(2+gamma)/(3-gamma)/(5-gamma)*P4)

    def plot(self, ax, what='logxi', jack=0, prange=(-2, 2), mirror=True,
             cbar=True, cmap=None, aspect='auto'):
        nrp = self.nrp
        npi = self.npi
        if what == 'logxi':
            label = r'$\log\ \xi$'
            dat = self.est[:npi, :nrp, jack]
        if what == 'log1xi':
            label = r'$\log\ (1 + \xi)$'
            dat = self.est[:npi, :nrp, jack] + 1
        if what == 'logxierr':
            label = r'$\log\ \epsilon_\xi$'
            dat = self.cov.sig[:npi, :nrp]
        if what == 'sn':
            label = r'$\log\ (s/n)$'
            dat = self.est[:npi, :nrp, jack] / self.cov.sig[:npi, :nrp]
        logdat = np.zeros((npi, nrp)) + prange[0]
        pos = dat > 0
        logdat[pos] = np.log10(dat[pos])

        if mirror:
            # Reflect about axes
            ximap = np.zeros((2*npi, 2*nrp))
            ximap[npi:, nrp:] = logdat
            ximap[npi:, :nrp] = np.fliplr(logdat)
            ximap[:npi, nrp:] = np.flipud(logdat)
            ximap[:npi, :nrp] = np.flipud(np.fliplr(logdat))
            extent = (-self.rpmax, self.rpmax, -self.pimax, self.pimax)
        else:
            ximap = np.flipud(logdat)
            extent = (self.rpmin, self.rpmax, self.pimin, self.pimax)

        # aspect = self.pimax/self.rpmax
        # print aspect, extent
        # aspect = 1
        # if self.rpmin * self.pimin > 0:
        im = ax.imshow(ximap, cmap, aspect=aspect, interpolation='none',
                       vmin=prange[0], vmax=prange[1],
                       extent=extent)
        ax.set_xlabel(r'$r_\perp\ [h^{-1} {{\rm Mpc}}]$')
        ax.set_ylabel(r'$r_\parallel\ [h^{-1} {{\rm Mpc}}]$')
#        divider = make_axes_locatable(ax)
#        cax = divider.append_axes("top", size="5%", pad=0.5)

        if what == 'logxi':
            Li_cont = [0.1875]
            while Li_cont[-1] < 48:
                Li_cont.append(2*Li_cont[-1])
            Li_cont = np.log10(Li_cont)
            cont = ax.contour(np.flipud(ximap), Li_cont, aspect=aspect,
                              extent=extent)

#        cb = plt.colorbar(im, cax=cax, orientation='horizontal')
#        cb = plt.colorbar(im, cax=ax)
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            cb.set_label(label)

    def vdist(self, lgximin=-2, hsmooth=0, neig=0, plots=1):
        """Velocity distribution function via Fourier transform of
        2d correlation function."""

        def vdist_samp(ximap, plots):
            """Velocity distribution for single xi."""

            pk = np.fft.fftshift(np.fft.fft2(ximap))
            freq = np.fft.fftshift(np.fft.fftfreq(nrp, 2*self.rpmax/nrp))
            kextent = (freq[0], freq[-1], freq[0], freq[-1])
            ratio = np.zeros(npi)
            pklim = 0.001*np.max(pk)
            use = np.abs(pk[nrp//2, :]) > pklim
            ratio[use] = pk[use, npi//2] / pk[nrp//2, use]
#            ratio = np.ma.masked_invalid(pk[:, npi//2] / pk[nrp//2, :])
            fv = np.abs(np.fft.fftshift(np.fft.ifft(ratio)))
            v = np.fft.fftshift(np.fft.fftfreq(nrp, (freq[-1]-freq[0])/(nrp)))
#            pdb.set_trace()

            if plots > 0:
                plt.clf()
                im = plt.imshow(np.abs(pk), aspect=aspect,
                                interpolation='none', extent=kextent)
                plt.xlabel(r'$k_\bot\ [h\ {\rm Mpc}^{-1}]$')
                plt.ylabel(r'$k_\parallel\ [h\ {\rm Mpc}^{-1}]$')
                plt.title('FT(Xi)')
                plt.colorbar()
                plt.show()

                plt.clf()
                fig, axes = plt.subplots(3, 1, sharex=True, num=1)
                fig.set_size_inches(3, 6)
                fig.subplots_adjust(hspace=0, wspace=0)
                ax = axes[0]
                ax.plot(freq, np.abs(pk[nrp//2, :]))
                ax.set_ylabel(r'$\hat\xi(k_\bot)$')
                ax = axes[1]
                ax.plot(freq, np.abs(pk[:, npi//2]))
                ax.set_ylabel(r'$\hat\xi(k_\parallel)$')
                ax = axes[2]
                ax.plot(freq, np.abs(ratio))
                ax.set_ylabel(r'$F(k)$')
                ax.set_xlabel(r'$k\ [h\ {\rm Mpc}^{-1}]$')
                plt.show()
                plt.clf()
                plt.plot(v, fv)
                plt.xlabel(r'$v\ [100\ \mathrm{km\ s}^{-1}]$')
                plt.ylabel(r'$f(v)$')
            return freq[nrp//2:], ratio[nrp//2:], v[nrp//2:], fv[nrp//2:]

        nrp = self.nrp
        npi = self.npi
        aspect = self.pimax/self.rpmax
        if hsmooth:
            rpgrid, pigrid = np.meshgrid(np.arange(nrp), np.arange(npi))
            hann = (np.sin(math.pi * rpgrid / (nrp-1)) *
                    np.sin(math.pi * pigrid / (npi-1)))**2
        else:
            hann = 1
        freq, ratio, v, fv = vdist_samp(hann*self.est[:, :, 0], plots)

        if self.njack > 0:
            ratio_jack = np.zeros((len(fv), self.njack))
            fv_jack = np.zeros((len(fv), self.njack))
            for ijack in xrange(self.njack):
                freq, ratio_jack[:, ijack], v, fv_jack[:, ijack] = vdist_samp(
                    hann*self.est[:, :, ijack+1], 0)
            # Exclude zeroth data point from ratio covariance calculation,
            # since it is always unity
            ratio_cov = Cov(ratio_jack[1:, :], self.err_type)
            fv_cov = Cov(fv_jack, self.err_type)
        else:
            ratio_cov = None
            fv_cov = None

        return freq, ratio, ratio_cov, v, fv, fv_cov

    def w_p(self, rp_lim, pi_lim):
        """Form projected corr fn w_p(r_p) from xi(r_p, pi)."""
        if self.rpmin < 0:
            rplim = min(self.rpmax, math.log10(rp_lim))
        else:
            rplim = min(self.rpmax, rp_lim)
        nrp = int((rplim - self.rpmin)/self.rpstep)
        rplim = self.rpmin + nrp*self.rpstep
        if self.pimin < 0:
            pilim = min(self.pimax, math.log10(pi_lim))
        else:
            pilim = min(self.pimax, pi_lim)
        npi = int((pilim - self.pimin)/self.pistep)
        pilim = self.pimin + npi*self.pistep
        w_p = Xi1d(self.nrp, self.njack, self.rpmin, rplim, 'w_p',
                   self.err_type)
        w_p.sep = self.rpc
        use = np.sum(self.galpairs[:npi, :nrp, 0], axis=0) > 0
        w_p.sep[use] = np.average(
            self.rp[:npi, use], weights=self.galpairs[:npi, use, 0], axis=0)
        w_p.galpairs = self.galpairs[:npi, :nrp, 0].sum(axis=0)
        w_p.ranpairs = self.ranpairs[:npi, :nrp, 0].sum(axis=0)
        if self.pimin < 0:
            if hasattr(self, 'pi'):
                w_p.est = 2*ln10*self.pistep*(
                    self.pi[:npi, :nrp, np.newaxis] *
                    self.est[:npi, :nrp, :]).sum(axis=0)
            else:
                w_p.est = 2*ln10*self.pistep*(
                    self.pic[:npi, np.newaxis, np.newaxis] *
                    self.est[:npi, :nrp, :]).sum(axis=0)
        else:
            w_p.est = 2*self.pistep*self.est[:npi, :nrp, :].sum(axis=0)
#        pdb.set_trace()
#        for i in range(nrp):
#            est = 0
#            jack = np.zeros(self.njack)
#            ggsum = 0.0
#            w_p.sep[i] = 0.0
#            for j in range(npi):
#                pi = self.pi[j, i]
#                w_p.galpairs[i] += self.galpairs[j, i, 0]
#                w_p.ranpairs[i] += self.ranpairs[j, i, 0]
#                ggsum += self.galpairs[j, i, 0]
#                w_p.sep[i] += self.galpairs[j, i, 0] * self.rp[j, i]
#                if self.pimin < 0:
#                    est += 2*ln10*pi*self.pistep*self.est[j, i]
#                    jack += 2*ln10*pi*self.pistep*self.jack[j, i, :]
#                else:
#                    est += 2*self.pistep*self.est[j, i]
#                    jack += 2*self.pistep*self.jack[j, i, :]
##                    pdb.set_trace()
#            w_p.est[i] = est
#            w_p.jack[i, :] = jack
#            if ggsum > 0:
#                w_p.sep[i] /= ggsum
        nrp = w_p.clear_empties()
        w_p.cov = Cov(w_p.est[:, 1:], self.err_type)
        return w_p


class P2d(object):
    """2d clustering P(k, mu) estimate."""

    def __init__(self, xi2, pimax=40, rpmin=0.1, rpmax=50, nsub=1,
                 smooth=20, err_type='jack'):
        """Calculate P(k, mu) from xi(pi, rp) using Li+2006 eqn (6)."""

        def gsmooth(pisep, rpsep, s):
            """Gaussian smoothing (Li+2006 eqn 8)."""
            return np.exp(-(pisep**2 + rpsep**2)/(2*s**2))

        # We assume log-binning in rp, linear in pi
        assert xi2.rpmin < 0 and xi2.pimin >= 0

#        lg_k_min, lg_k_max, nk = -1.0, 0.8, 10
        lg_k_min, lg_k_max, nk = -1.0, 2, 15
        kvals = 10**np.linspace(lg_k_min, lg_k_max, nk)
        mu_min, mu_max, nmu = 0.0, 0.9, 10
        muvals = np.linspace(mu_min, mu_max, nmu)
        Pjack = np.zeros((nk, nmu, xi2.njack+1))

        # Use bilinear cubic interpolation to obtain Gaussian-tapered
        # xi(pi, rp) on nsub times finer grid
        npi = min(xi2.npi, int((pimax - xi2.pimin) / xi2.pistep))
        pibins = xi2.pimin + (np.arange(npi)+0.5) * xi2.pistep
        pistep = pimax/(nsub*npi)
        pivals = np.linspace(0.5*pistep, pimax - 0.5*pistep, nsub*npi)
        # print 'pibins ', pibins
        # print 'pivals ', pivals

        nrplo = max(0, int((math.log10(rpmin) - xi2.rpmin) / xi2.rpstep))
        nrphi = min(xi2.nrp, int((math.log10(rpmax)-xi2.rpmin) / xi2.rpstep)+1)
        nrp = nrphi - nrplo
        rpbins = xi2.rpmin + (np.arange(nrplo, nrphi)+0.5) * xi2.rpstep
        lgrpmin = xi2.rpmin + nrplo*xi2.rpstep
        lgrpmax = xi2.rpmin + nrphi*xi2.rpstep
        lgrpstep = (lgrpmax - lgrpmin)/(nsub*nrp)
        rpvals = np.linspace(lgrpmin + 0.5*lgrpstep, lgrpmax - 0.5*lgrpstep,
                             nsub*nrp)
        # print 'rpbins ', rpbins
        # print 'rpvals ', rpvals
        print('rprange ', 10**lgrpmin, 10**lgrpmax)
#        pdb.set_trace()

        # Replaced masked xi vales by zero, else scipy gives NaNs
        xi2_est = np.ma.filled(xi2.est, 0.0)
        xit = np.zeros((nsub*npi, nsub*nrp, xi2.njack+1))
        rpgrid, pigrid = np.meshgrid(10**rpvals, pivals)
        smooth = gsmooth(pigrid, rpgrid, smooth)
        spline = scipy.interpolate.RectBivariateSpline(
            pibins, rpbins, xi2_est[:npi, nrplo:nrphi, 0],
            bbox=[0.0, pimax, lgrpmin, lgrpmax])
        xit[:, :, 0] = spline(pivals, rpvals) * smooth
        for ijack in xrange(xi2.njack):
            spline = scipy.interpolate.RectBivariateSpline(
                pibins, rpbins, xi2_est[:npi, nrplo:nrphi, ijack+1],
                bbox=[0.0, pimax, lgrpmin, lgrpmax])
            xit[:, :, ijack+1] = spline(pivals, rpvals) * smooth

        for ik in xrange(nk):
            k = kvals[ik]
            for imu in xrange(nmu):
                mu = muvals[imu]
                kp = k*(1 - mu**2)**0.5
                for ijack in xrange(xi2.njack+1):
                    Pjack[ik, imu, ijack] = np.sum(
                        rpgrid**2 * xit[:, :, ijack] * np.cos(k*mu*pigrid) *
                        scipy.special.jn(0, kp*rpgrid))
        Pjack *= 4*math.pi * math.log(10) * pistep * lgrpstep
        self.err_type = err_type
        self.P = Pjack
        self.cov = Cov(Pjack[:, :, 1:], err_type)
        self.k = kvals
        self.mu = muvals

    def plot(self, ax, Prange=(0.1, 2e4)):
        nk, nmu, njp1 = self.P.shape
        for ik in xrange(nk):
            ax.errorbar(self.mu, self.P[ik, :, 0], yerr=self.cov.sig[ik, :],
                        fmt='o', capthick=1)
            if Prange[0] <= self.P[ik, -1, 0] <= Prange[1]:
                ax.text(0.95, self.P[ik, -1, 0], '{:.2f}'.format(self.k[ik]))
        ax.set_xlabel(r'$\mu$')
        ax.set_ylabel(r'$P(k, \mu)$')
        ax.set_xlim(-0.1, 1.0)
        ax.set_ylim(Prange)
        ax.semilogy(basey=10, nonposy='clip')


class Cov(object):
    """Covariance matrix and eigenvalue decomposition."""

    def __init__(self, ests, err_type):
        """Generate covariance matrix from jackknife or mock estimates."""

        dims = ests.shape[:-1]
        ndat = np.prod(dims)
        nest = ests.shape[-1]
        self.cov = np.ma.cov(ests.reshape((ndat, nest), order='F'))
        if err_type == 'jack':
            self.cov *= (nest-1)
        try:
            self.icov = np.linalg.inv(self.cov)
        except:
            print('Unable to invert covariance matrix')
#            pdb.set_trace()
        try:
            self.sig = np.sqrt(np.diag(self.cov)).reshape(dims, order='F')
            self.siginv = np.diag(1.0/np.sqrt(np.diag(self.cov)))
#            pdb.set_trace()
            cnorm = np.nan_to_num(self.siginv.dot(self.cov).dot(self.siginv))
            self.cnorm = np.clip(cnorm, -1, 1)
            eig_val, eig_vec = np.linalg.eigh(self.cnorm)
            idx = eig_val.argsort()[::-1]
            self.eig_val = eig_val[idx]
            self.eig_vec = eig_vec[:, idx]
        except:
            self.sig = np.sqrt(self.cov)
            self.siginv = 1.0/self.sig

    def add(self, cov):
        """Add second covariance matrix to self."""
        self.cov += cov.cov
        dims = self.cov.shape[:-1]
        self.sig = np.sqrt(np.diag(self.cov)).reshape(dims, order='F')
        self.siginv = np.diag(1.0/np.sqrt(np.diag(self.cov)))
        cnorm = np.nan_to_num(self.siginv.dot(self.cov).dot(self.siginv))
        self.cnorm = np.clip(cnorm, -1, 1)
        eig_val, eig_vec = np.linalg.eig(self.cnorm)
        idx = eig_val.argsort()[::-1]
        self.eig_val = eig_val[idx].real
        self.eig_vec = eig_vec[:, idx].real

    def chi2(self, obs, model, neig=0):
        """
        Chi^2 residual between obs and model, using first neig eigenvectors
        (Norberg+2009, eqn 12).  By default (neig=0), use diagonal elements
        only.  Set neig='full' for full covariance matrix,
        'all' for all e-vectors.  For chi2 calcs using mean of mock catalogues,
        multiply returned chi2 by nest to convert from standard deviation
        to standard error."""

        if neig == 0:
            if len(obs) > 1:
                diag = np.diag(self.cov)
                nonz = diag > 0
                return np.sum((obs[nonz] - model[nonz])**2 / diag[nonz])
            else:
                return (obs - model)**2 / self.cov
        if neig == 'full':
            return (obs-model).T.dot(self.icov).dot(obs-model)
        yobs = self.eig_vec.T.dot(self.siginv).dot(obs)
        ymod = self.eig_vec.T.dot(self.siginv).dot(model)
        if neig == 'all':
            return np.sum((yobs - ymod)**2 / self.eig_val)
        else:
            return np.sum((yobs[:neig] - ymod[:neig])**2 / self.eig_val[:neig])

    def plot(self, norm=False, ax=None, label=None):
        """Plot (normalised) covariance matrix."""
        try:
            ndat = self.cov.shape[0]
            extent = (0, ndat, 0, ndat)
            aspect = 1

            if ax is None:
                plt.clf()
                ax = plt.subplot(111)
            if norm:
                val = self.cnorm
                xlabel = 'Normalized Covariance'
            else:
                val = self.cov
                xlabel = 'Covariance'

            im = ax.imshow(val, aspect=aspect, interpolation='none',
                           extent=extent, origin='lower')
            ax.set_xlabel(xlabel)
            if label:
                ax.set_title(label)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
        except:
            print('Error plottong covariance matrix')

    def plot_eig(self):
        """Plot eigenvalues & eigenvectors."""
        if hasattr(self, 'eig_val'):
            plt.clf()
            ax = plt.subplot(121)
            ax.plot(self.eig_val/self.eig_val.sum())
            ax.plot(np.cumsum(self.eig_val/self.eig_val.sum()))
            # ax.semilogy(basey=10, nonposy='clip')
            ax.set_xlabel('eigen number')
            ax.set_ylabel(r'$\lambda_i / \sum \lambda$')

            ax = plt.subplot(122)
            for i in range(len(self.eig_val)):
                ax.plot(self.eig_vec[i, :]/(self.eig_vec**2).sum(axis=0)**0.5)
            # ax.semilogy(basey=10, nonposy='clip')
            ax.set_xlabel('separation bin')
            ax.set_ylabel(r'$E_i / (\sum E_i^2)^{0.5}$')
            plt.show()


# ---------------------------------------
# Run pair-count code on selected samples
# ---------------------------------------

def w_select(infile, outroot, selfile,
             appMin=-99, appMax=99, absMin=-99, absMax=99):
    """
    Select GAMA galaxies or random points for wcorr.c.
    Projects onto unit sphere, one file for each GAMA region.
    """

    zmin = 0.002
    zmax = 0.5
    nz = 100
    iband = 2
    
    # Read input file into structure
    if '.fits' in infile:
        outfile = [outroot + 'g09.dat', outroot + 'g12.dat', outroot + 'g15.dat']
        hdulist = pyfits.open(infile)
        tbdata = hdulist[1].data
        ra = tbdata.field('ra')
        dec = tbdata.field('dec')
        z = tbdata.field('z')
        nq = tbdata.field('nq')
        appMag = tbdata.field('appMag')[:,iband]
        absMag = tbdata.field('absMag')[:,iband]
        idx = (zmin < z) * (z < zmax) * (nq > 2) * \
          (appMin < appMag) * (appMag < appMax) * \
          (absMin < absMag) * (absMag < absMax)
        ra = ra[idx]
        dec = dec[idx]
        z = z[idx]
    else:
        outfile = [outroot + 'r09.dat', outroot + 'r12.dat', outroot + 'r15.dat']
        data = np.loadtxt(infile, skiprows=1)
        ra = data[:,0]
        dec = data[:,1]

    nsamp = len(ra)
    print(nsamp, ' objects selected')

    # Project points onto unit sphere
    xc = np.cos(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
    yc = np.sin(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
    zc = np.sin(np.deg2rad(dec))

    # No jacknife regions - treat each GAMA region separately
    njack = 0
    ncell = 1
    ix = 0
    iy = 0
    cellsize = 12.0

    ra_limits = [[129.0, 141.0], [174.0, 186.0], [211.5, 223.5]]
##     pdb.set_trace()
    for ireg in range(3):
        print('Writing out ', outfile[ireg])
        idx = (ra >= ra_limits[ireg][0])*(ra <= ra_limits[ireg][1])
        nsamp = len(ra[idx])
        fout = open(outfile[ireg], 'w')
        print(nsamp, ncell, cellsize, njack, file=fout)
        weight = 1
        print(ix, iy, nsamp, file=fout)
        for i in xrange(nsamp):
            print(xc[idx][i], yc[idx][i], zc[idx][i], weight, njack, file=fout)
        fout.close()

    # Output N(z) distribution for this sample
    if selfile:
        zhist = np.histogram(z, bins=nz, range=(zmin, zmax))
        fout = open(selfile, 'w')
        print(nz, zmin, zmax, file=fout)
        for h in zhist[0]:
            fout.write(str(h) + ' ')
        fout.write('\n')
        fout.close()

def w_mag_select():
    """App mag selected samples."""

    w_select('../kcorr.fits', 'gal_14_17_', 'sel_14_17.dat', appMin=14, appMax=17)
    w_select('../kcorr.fits', 'gal_17_18_', 'sel_17_18.dat', appMin=17, appMax=18)
    w_select('../kcorr.fits', 'gal_18_19_', 'sel_18_19.dat', appMin=18, appMax=19)
    w_select('../kcorr.fits', 'gal_19_194_', 'sel_19_194.dat', appMin=19, appMax=19.4)


def xi_select(infile, galout, ranout, xiout,
              mask=gama_data+'/mask/zcomp.ply', param='lum',
              z_range=(0.002, 0.65), nz=65, app_range=(14, 19.8),
              abs_range=(-99, 99), mass_range=(-99, 99), colour='c',
              Q=Qdef, P=Pdef, ran_dist='fit', ran_fac=10, weighting='unif',
              set_vmax=False, ax=None, run=0, ranzfile='ranz.dat',
              survey_codes=None, zpctrim=99):
    """
    Select GAMA galaxies from stellar masses catalogue for xi.c.
    Ignore fluxscale parameter for now as only ~half galaxies have valid value.
    Meaning of run paramater: 0: select samples only, 1: run interactively,
    2: submit as job using qsub.
    """

    def ecorr(z, Q):
        """e-correction."""
        return Q * (z-z0)

    def den_evol(z, P):
        """Density evolution at redshift z."""
        print('den_evol, P =', P)
        return 10**(0.4*P*(z-z0))

    def vol_ev(z, P):
        """Volume element multiplied by density evolution."""
        pz = samp.cosmo.dV(z) * den_evol(z, P)
        return pz

    if ax is None:
        plt.clf()
        ax = plt.axes()

    zvals = np.linspace(z_range[0], z_range[1], nz)

    # Read input file into structure
    par = {}
    par['clean_photom'] = False
    par['kc_use_poly'] = False
    par['param'] = 'r_petro'
    par['ev_model'] = 'z'
    par['zmin'] = z_range[0]
    par['zmax'] = z_range[1]
    par['nz'] = 65
    par['Mmin'] = abs_range[0]
    par['Mmax'] = abs_range[1]
    par['Mbin'] = 1
    par['mlims'] = app_range
    par['idebug'] = 0
    clr_limits = ('a', 'z')
    if (colour == 'b'):
        clr_limits = ('b', 'c')
    if (colour == 'r'):
        clr_limits = ('r', 's')
    sel_dict = {}
    sel_dict['colour'] = clr_limits
    samp = jswml.Sample(infile, par, sel_dict)
    gala = samp.calc_limits(Q, vis=True)
    z0 = samp.par['z0']
    area = samp.par['area']

#    hdulist = pyfits.open(infile)
#    header = hdulist[1].header
#    tbdata = hdulist[1].data
#    z0 = header['z0']
#    area = header['area']*(math.pi/180.0)*(math.pi/180.0)
#    cosmo = util.CosmoLookup(H0, omega_l, z_range)
#
#    z = tbdata.field('z_tonry')
#    appMag = tbdata.field('r_petro')
#    idx = ((tbdata.field('survey_class') >= 3) *
#           (app_range[0] <= appMag) * (appMag < app_range[1]) *
#           (z_range[0] <= z) * (z < z_range[1]) * (tbdata.field('nq') > 2))
##    pdb.set_trace()
#    tbdata = tbdata[idx]
#    appMag = appMag[idx]
#    z = z[idx]
#
#    absMag = appMag - cosmo.dist_mod(z) - tbdata.field('kcorr_r') + ecorr(z, Q)
#    idx = (abs_range[0] <= absMag) * (absMag < abs_range[1])
#
#    logmstar = tbdata.field('logmstar') - 2*math.log10(1.0/0.7)
#    if mass_range[0] > 0:
#        idx *= (mass_range[0] <= logmstar) * (logmstar < mass_range[1])
#
#    if colour != 'c':
#        grcut = lum.colourCut(absMag)
#        gr = ((tbdata.field('g_model') - tbdata.field('kcorr_g')) -
#              (tbdata.field('r_model') - tbdata.field('kcorr_r')))
#        if colour == 'b':
#            idx *= (gr < grcut)
#        else:
#            idx *= (gr >= grcut)
#
#    if survey_codes:
#        idx *= np.in1d(tbdata.field('survey_code'), survey_codes)
#
#    tbdata = tbdata[idx]
#    ra = tbdata.field('ra')
#    dec = tbdata.field('dec')
#    ngal = len(ra)
#    z = z[idx]
#    absMag = absMag[idx]
#    logmstar = logmstar[idx]
#    rgal = cosmo.dm(z)
#    nran = ran_fac*ngal

    z = gala['z']
    ra = gala['ra']
    dec = gala['dec']
    ngal = len(z)
    rgal = samp.cosmo.dm(z)
    nran = ran_fac*ngal

    # Redshift distribution
    zhist, bin_edges = np.histogram(z, bins=nz, range=z_range)
    zstep = bin_edges[1] - bin_edges[0]
    zcen = bin_edges[:-1] + 0.5*zstep
    ax.step(zcen, zhist)

    # Trim redshift distribution to avoid long tail of galaxies or randoms
    if zpctrim:
        zcut = scipy.stats.scoreatpercentile(z, zpctrim)
    else:
        zcut = z_range[1]
    rcut = samp.cosmo.dm(zcut)
    print('zcut, rcut = ', zcut, rcut)

    # Field to field variance for fit to N(z)
    zh = np.zeros((njack, nz))
    for ij in xrange(njack):
        idx = (ra_limits[ij][0] <= ra) * (ra <= ra_limits[ij][1])
        zh[ij, :], bin_edges = np.histogram(z[idx], bins=nz, range=z_range)
    zhvar = njack*np.var(zh, axis=0)
#    ax.errorbar(zcen, zhist, np.sqrt(zhvar), capthick=1)

    if ran_dist == 'jswml':
        if param == 'lum':
            limits = abs_range
        if param == 'mass':
            limits = mass_range
#        jswml.ran_gen_sample(infile=infile, Q=Q, P=P,
#                             outfile=ranzfile, param=param, limits=limits,
#                             colour=colour, zmin=z_range[0], zmax=z_range[1],
#                             nz=65, nfac=ran_fac)
        jswml.ran_gen(gala, ranzfile, ran_fac, Q=Q, P=P, vol=0)
        data = np.loadtxt(ranzfile, skiprows=1)
        zran = data[:, 0]
        # Select only reshifts within z_range
        idx = (z_range[0] <= zran) * (zran < z_range[1])
        zran = zran[idx]
        nran = len(zran)
        rran = samp.cosmo.dm(zran)
        if set_vmax:
            Vran = data[:, 1]
        else:
            Vran = np.ones(nran)
        if weighting == 'mass':
            wran = data[:, 2]
        else:
            wran = np.ones(nran)

    if ran_dist == 'gal':
        # Generate distances and Vmax according to galaxy distribution
        # Only suitable for very large volumes
        ir = (ngal*np.random.random(nran)).astype(np.int32)
        rran = rgal[ir]
        wran = wgal[ir]
        Vran = Vgal[ir]

    if ran_dist == 'lf':
        # Generate distances from LF-derived selection function
        ncum = lum.Nltz(infile, zvals, absMin, absMax)
        pcum = ncum/max(ncum)
        p = np.random.random(nran)
        zran = np.interp(p, pcum, zvals)
        rran = cosmo.dm(zran)

    if ran_dist == 'vol':
        zran = util.ran_fun(vol_ev, z_range[0], z_range[1], nran, args=(P,))
        rran = samp.cosmo.dm(zran)
        wran = np.ones(nran)
        Vran = np.ones(nran)

    if ran_dist == 'smooth':
        # Smoothed N(z) distribution for this sample
        zsmooth = util.smooth(zhist, 15)
        zran = util.ran_dist(zcen, zsmooth, nran)
        # hmax = max(zhist)
        # for i in xrange(nran):
        #     accept = 0
        #     while accept == 0:
        #         ztry = z_range[0] + (z_range[1] - z_range[0])*np.random.random()
        #         p = np.interp(ztry, zcen, zsmooth)/hmax
        #         if np.random.random() < p: accept = 1
        #         z[i] = ztry
        rran = samp.cosmo.dm(zran)
        ax.plot(zcen, zsmooth)

    if ran_dist == 'fit':
        # Fitting function of Blake et al. 2013

        def fitfunc(z, p):
            """ Blake et al 2013 eqn (8) """
            fit = p[3] * (z / p[0])**p[1] * np.exp(-(z / p[0])**p[2])
#            fit =  p[3] * z**p[1] * np.exp(-(z / p[0])**p[2])
            return fit

        def resid(p, z, hist, var):
            """Residual normalised by variance (or fit if variance is zero)."""
            if p[0] <= 0 or p[3] <= 0:
                return 1e10
            chi2 = 0
            for zi, hi, vi in zip(z, hist, var):
                fit = fitfunc(zi, p)
                if vi <= 0: vi = fit
                if vi > 0:
                    chi2 += (hi - fit)**2 / vi
#            print p, chi2
            return chi2

        p0 = (0.2, 1, 3, 5000)
        out = scipy.optimize.fmin(resid, p0, args=(zcen, zhist, zhvar),
                                  maxfun=10000, maxiter=10000, full_output=1)
        p = out[0]

        # Renormalise to give same total number of galaxies
        fitnum = fitfunc(zcen, p).sum()
        rfac = float(ngal)/fitnum
        p[3] *= rfac
        print('renormalised by factor ', rfac)
        print(p)
        zran = util.ran_fun(fitfunc, z_range[0], z_range[1], nran, args=(p,))
        rran = samp.cosmo.dm(zran)
        ax.plot(zcen, fitfunc(zcen, p))

    zran_hist, bin_edges = np.histogram(zran, bins=nz, range=z_range)
    zran_hist = zran_hist*float(ngal)/nran  # Note cannot use *= due int->float
    V_int = area/3.0 * samp.cosmo.dm(bin_edges)**3
    Vbin = np.diff(V_int)
    denbin = zran_hist/Vbin
    ax.plot(zcen, zran_hist)

    # Density for minimum variance weighting
    dengal = np.interp(z, zcen, denbin)
    denran = np.interp(zran, zcen, denbin)

    wgal = np.ones(ngal)
    if weighting == 'mass':
        wgal *= 10**logmstar

    # Vmax
    if set_vmax:
        Vgal = tbdata.field('Vmax_dec')
    else:
        Vgal = np.ones(ngal)

    M_mean = np.mean(gala['absval_sel'])
#    logm_mean = np.mean(logmstar[np.isfinite(logmstar)])
    logm_mean = None
    z_mean = np.mean(z)
    info = {'file': infile, 'weighting': weighting, 'M_mean': M_mean,
            'logm_mean': logm_mean, 'z_mean': z_mean, 'njack': njack,
            'zcut': zcut, 'rcut': rcut, 'set_vmax': set_vmax, 'Q': Q, 'P': P,
            'err_type': 'jack', 'abs_range': abs_range, 'z_range': z_range}

    galcat = Cat(ra, dec, rgal, weight=wgal, den=dengal, Vmax=Vgal, info=info)
    if ran_dist == 'vol':
        J3_pars = (0, 0, 0)
    else:
        J3_pars = def_J3_pars
    galcat.output(galout, J3_pars=J3_pars)
    print(ngal, ' galaxies selected')

    # For random distributions other than 'gal' and 'jswml',
    # obtain weight and Vmax from linear interpolation of galaxy data
    # if ran_dist not in ('gal', 'jswml'):
    #     isort = np.argsort(rgal)
    #     wran = np.interp(rran, rgal[isort], wgal[isort])
    #     Vran = np.interp(rran, rgal[isort], Vgal[isort])

    # Generate random coords using ransack
    ranc_file = 'ransack.dat'
    # cmd = "$MANGLE_DIR/bin/ransack -r{} $GAMA_DATA/mask/zcomp.ply {}".format(
    #     nran, ranc_file)
    cmd = "$MANGLE_DIR/bin/ransack -r{} {} {}".format(nran, mask, ranc_file)
    subprocess.call(cmd, shell=True)
    data = np.loadtxt(ranc_file, skiprows=1)
    ra = data[:, 0]
    dec = data[:, 1]

    rancat = Cat(ra, dec, rran, weight=wran, den=denran, Vmax=Vran, info=info)
    rancat.output(ranout, J3_pars=J3_pars)

    # Plot labeling
    if mass_range[0] > 0:
        label = r'${:5.1f} < \log_{{10}}(M/M_\odot) < {:5.1f}$'.format(*mass_range)
    else:
        label = r'${:5.1f} < M_r < {:5.1f}$'.format(*abs_range)
    ax.text(0.5, 0.9, label, transform=ax.transAxes)
    label = r'{} galaxies'.format(ngal)
    ax.text(0.5, 0.8, label, transform=ax.transAxes)
    # label = r'$z_0 = {0[0]:5.3f}$, $\alpha = {0[1]:5.3f}$, $\beta = {0[2]:5.3f}$'.format(p)
    # ax.text(0.5, 0.7, label, transform = ax.transAxes)
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'$N(z)$')

    # if ax:
    #     ax.hist((rgal, rran), bins=20, normed=True)
    # else:
    #     plt.clf()
##     plt.hist((rgal, rran), bins=20, normed=True, histtype='step')
# Specifying both normed and histtype='step' doesn't work with matplotlib 1.1
        # plt.hist((rgal, rran), bins=20, normed=True)
        # plt.draw()

    if run == 1:
        cmd = '$BIN/xi {} {}'.format(galout, xiout.replace('xi', 'gg', 1))
        subprocess.call(cmd, shell=True)
        cmd = '$BIN/xi {} {}'.format(ranout, xiout.replace('xi', 'rr', 1))
        subprocess.call(cmd, shell=True)
        cmd = '$BIN/xi {} {} {}'.format(galout, ranout,
                                        xiout.replace('xi', 'gr', 1))
        subprocess.call(cmd, shell=True)
    if run == 2:
        cmd = qsub_xia_cmd.format(galout, xiout.replace('xi', 'gg', 1))
        subprocess.call(cmd, shell=True)
        cmd = qsub_xia_cmd.format(ranout, xiout.replace('xi', 'rr', 1))
        subprocess.call(cmd, shell=True)
        cmd = qsub_xix_cmd.format(galout, ranout, xiout.replace('xi', 'gr', 1))
        subprocess.call(cmd, shell=True)


def xi_g3c_mock_select(infile=gama_data+'g3cv6/G3CMockGalv06.fits',
                   mask=gama_data+'../gama1/mask/gama_rect.ply',
                   ranfac=10, mlim=19.8, zrange=(0.002, 0.5), 
                   vol_limit=False, set_vmax=False, weighting=None, qsub=False):
    """
    Lum-selected samples from G3C mock catalogues.
    """

    Mlimits = (-23, -22, -21, -20, -19, -18, -17, -16)
    if vol_limit:
        zlimits = vol_limits_mock(Mlimits, mlim, zrange)
        galroot = 'gal_V_{}_{}_{}.dat'
        ranroot = 'ran_V_{}_{}_{}.dat'
        xiroot = 'xi_V_{}_{}_{}.dat'
    else:
        galroot = 'gal_M_{}_{}_{}.dat'
        ranroot = 'ran_M_{}_{}_{}.dat'
        xiroot = 'xi_M_{}_{}_{}.dat'
        zlo = zrange[0]
        zhi = zrange[-1]

    # Read input file into structure
    hdulist = pyfits.open(infile)
    header = hdulist[1].header
    tbdata = hdulist[1].data
    cosmo = util.CosmoLookup(H0, omega_l, zrange)

    ra = tbdata.field('ra')
    dec = tbdata.field('dec')
    z = tbdata.field('Z')
    appMag = tbdata.field('Rpetro')
    dm = tbdata.field('DM_100_25_75')
    volume = tbdata.field('Volume')
    absMag = appMag - dm - util.mock_ke_corr(z) + 1.75*z + 0.14*(z - 0.2) # undo e-corr
    rgal = cosmo.dm(z)
    hdulist.close()

    # plt.clf()
    # plt.scatter(z, dm - cosmo.dist_mod(z), 0.1)
    # plt.xlabel('Redshift')
    # plt.ylabel('dm (mock file) - dm (astUltils)')
    # plt.draw()

    for ivol in range(1,10):
        for i in range(len(Mlimits)-1):
            Mlo = Mlimits[i]
            Mhi = Mlimits[i+1]
            if vol_limit:
                zlo = 0.002
                zhi = zlimits[i+1]
            else:
                ranfile = gama_data+'g3cv6/jswml/ranz_{}_{}_{}.dat'.format(ivol, Mlo, Mhi)
            idx = ((ivol == volume) * (zlo <= z) * (z < zhi) *
                   (appMag < mlim) * (Mlo <= absMag) * (absMag < Mhi))
            ngal = len(ra[idx])
            wgal = np.ones(ngal)
            Vgal = np.ones(ngal)
            M_mean = (absMag[idx]).mean()
            logm_mean = 0.0
            z_mean = z[idx].mean()
            info = {'file': infile, 'weighting': 'unif', 'M_mean': M_mean,
            'logm_mean': logm_mean, 'z_mean': z_mean}
    
            galcat = Cat(ra[idx], dec[idx], rgal[idx], weight=wgal, Vmax=Vgal,
                         info=info)
            galout = galroot.format(ivol, Mlo, Mhi)
            galcat.output(galout)
            print(ngal, ' galaxies selected')

            if vol_limit:
                # Uniform density randoms for vol-ltd samples
                nran = ranfac*ngal
                rmin = cosmo.dm(zlo)
                rmax = cosmo.dm(zhi)
                p = np.random.random(nran)
                rran = (p*rmax**3 + (1-p)*rmin**3)**(1.0/3.0)
                wran = np.ones(nran)
                Vran = np.ones(nran)
            else:
                # Read distances from jswml.ran_gen
                f = open(ranfile, 'r')
                rz_info = eval(f.readline())
                f.close()
                data = np.loadtxt(ranfile, skiprows=1)
                zran = data[:,0]
                nran = len(zran)
                rran = cosmo.dm(zran)
                if set_vmax:
                    Vran = data[:,1]
                else:
                    Vran = np.ones(nran)
                if weighting == 'mass':
                    wran = data[:,2]
                else:
                    wran = np.ones(nran)

            galhist, zbins = np.histogram(z[idx], rz_info['zbins'])
            ranhist, zbins = np.histogram(data[:,0], rz_info['zbins'])
            ranhist *= float(rz_info['ngal'])/rz_info['nran']
            plt.clf()
            plt.step(rz_info['zcen'], galhist, where='mid')
            plt.step(rz_info['zcen'], rz_info['galhist'], where='mid')
            plt.step(rz_info['zcen'], ranhist, where='mid')
            plt.xlabel('Redshift')
            plt.ylabel('Frequency')
            plt.draw()

            # Generate random coords using ransack
            ranc_file = 'ransack.dat'
            cmd = "$MANGLE_DIR/bin/ransack -r{} {} {}".format(nran, mask, ranc_file)
            subprocess.call(cmd, shell=True)
            data = np.loadtxt(ranc_file, skiprows=1)
            ra_ran = data[:,0]
            dec_ran = data[:,1]

            rancat = Cat(ra_ran, dec_ran, rran, weight=wran, Vmax=Vran,
                         info=info)
            ranout = ranroot.format(ivol, Mlo, Mhi)
            rancat.output(ranout)

            if qsub:
                xiout = xiroot.format(ivol, Mlo, Mhi)
                cmd = qsub_xi_cmd.format(galout, ranout, xiout)
                subprocess.call(cmd, shell=True)


def xi_mock_vol(infile='Gonzalez.fits', Mlims=(-25, -20), z_type='obs',
                maxran=1000000, qsub=True):
    """
    Correlation function from volume-limited mock samples.
    """

    njack = 0
    region = ('09', '12', '15', '02', '23')
    ranc_file = 'ransack.dat'
    zcol = 'redshift_{}'.format(z_type)
    Mlo = Mlims[0]
    Mhi = Mlims[1]
    zmax = util.vol_limits(infile, Q=Qdef, Mlims=(Mlims[1],))[0]
    zrange = (0.002, zmax)
    cosmo = util.CosmoLookup(H0, omega_l, zrange)
    rcut = cosmo.dm(zrange[1])

    hdulist = pyfits.open(infile)
    tbdata = hdulist[1].data
    info = {'file': infile, 'z_type': z_type, 'weighting': 'unif',
            'njack': njack, 'err_type': 'mock', 'Mlo': Mlo, 'Mhi': Mhi,
            'zrange': zrange, 'rcut': rcut, 'Q': Qdef, 'P': Pdef}

    samp = ((tbdata.field(zcol) >= zrange[0]) *
            (tbdata.field(zcol) < zrange[1]) *
            (tbdata.field('SDSS_r_rest_abs') >= Mlo) *
            (tbdata.field('SDSS_r_rest_abs') < Mhi))
    tbsamp = tbdata[samp]
    z_ran = tbsamp.field(zcol)

    for field in range(1, 4):
        for ireal in range(26):
            idx = ((tbsamp.field('ireal') == ireal) *
                   (tbsamp.field('field') == field))

            z = tbsamp[idx].field(zcol)
            ra = tbsamp[idx].field('ra')
            dec = tbsamp[idx].field('dec')
            rgal = cosmo.dm(z)

            ngal = len(z)
            wgal = np.ones(ngal)
            Vgal = np.ones(ngal)

            galcat = Cat(ra, dec, rgal, weight=wgal, Vmax=Vgal, info=info)
            galcat.output('gal_vol_{}_{}_{:02d}.dat'.format(
                z_type, field, ireal), J3_pars=(0, 0, 0))
            print(ngal, ' galaxies selected')

    # Generate random coords for this sample using summed N(z)
    if maxran > 0:
        if maxran < len(z_ran):
            z_ran = np.random.choice(z_ran, maxran, replace=0)
        nran = len(z_ran)
        rran = cosmo.dm(z_ran[:nran])
        wran = np.ones(nran)
        Vran = np.ones(nran)
        for field in range(1, 4):
            # Generate random coords using ransack
            mask_file = '$GAMA_DATA/mask/G{}_rect.ply'.format(region[field-1])
            cmd = "$MANGLE_DIR/bin/ransack -r{} {} {}".format(
                nran, mask_file, ranc_file)
            subprocess.call(cmd, shell=True)
            data = np.loadtxt(ranc_file, skiprows=1)
            ra_ran = data[:, 0]
            dec_ran = data[:, 1]
            rancat = Cat(ra_ran, dec_ran, rran, weight=wran, Vmax=Vran, info=info)
            rancat.output('ran_vol_{}.dat'.format(field), J3_pars=(0, 0, 0))

    # Submit the jobs
    if qsub:
        for field in range(1, 4):
            ranout = 'ran_vol_{}.dat'.format(field)
            if maxran > 0:
                xiout = 'rr_vol_{}.dat'.format(field)
                cmd = qsub_xia_cmd.format(ranout, xiout)
                subprocess.call(cmd, shell=True)
            for ireal in range(26):
                galout = 'gal_vol_{}_{}_{:02d}.dat'.format(z_type, field, ireal)
                xiout = 'gg_vol_{}_{}_{:02d}.dat'.format(z_type, field, ireal)
                cmd = qsub_xia_cmd.format(galout, xiout)
                subprocess.call(cmd, shell=True)
                xiout = 'gr_vol_{}_{}_{:02d}.dat'.format(z_type, field, ireal)
                cmd = qsub_xix_cmd.format(galout, ranout, xiout)
                subprocess.call(cmd, shell=True)


def xi_mock(infile='Gonzalez.fits', param='lum_c', zrange=(0.002, 0.5), nz=50,
            z_type='obs', maxran=1000000, qsub=True):
    """
    GAMA v1 mock catalogues (from one FITS file).
    """

    def outname(prefix):
        """Return output file name."""
        if prefix == 'gal':
            outfile = 'gal_{}_{}_{}_{}_{:02d}.dat'.format(
                param, bins[isamp], bins[isamp+1], field, ireal)

        if prefix == 'ran':
            outfile = 'ran_{}_{}_{}_{}.dat'.format(
                param, bins[isamp], bins[isamp+1], field)

        if prefix in ('gg', 'gr'):
            outfile = '{}_{}_{}_{}_{}_{:02d}.dat'.format(
                prefix, param, bins[isamp], bins[isamp+1], field, ireal)

        if prefix == 'rr':
            outfile = '{}_{}_{}_{}_{}.dat'.format(
                prefix, param, bins[isamp], bins[isamp+1], field)

        return outfile

    area = math.pi**2/180.0
    njack = 0
    cosmo = util.CosmoLookup(H0, omega_l, zrange)
    region = ('09', '12', '15', '02', '23')
    ranc_file = 'ransack.dat'
    zcol = 'redshift_{}'.format(z_type)
    rcut = cosmo.dm(zrange[1])

    nsamp = 1
    if 'lum' in param:
        bins = def_mag_limits
        nsamp = len(bins) - 1
    if 'mass' in param:
        bins = def_mass_limits
        nsamp = len(bins) - 1

    hdulist = pyfits.open(infile)
    tbdata = hdulist[1].data
    info = {'file': infile, 'z_type': z_type, 'weighting': 'unif',
            'rcut': rcut, 'njack': njack, 'err_type': 'mock'}

    for isamp in range(nsamp):
        samp = ((tbdata.field(zcol) >= zrange[0]) *
                (tbdata.field(zcol) < zrange[1]))
        if 'lum' in param:
            Mlo = bins[isamp]
            Mhi = bins[isamp+1]
            info.update({'Mlo': Mlo, 'Mhi': Mhi})
            samp *= ((tbdata.field('SDSS_r_rest_abs') >= Mlo) *
                     (tbdata.field('SDSS_r_rest_abs') < Mhi))
        if 'mass' in param:
            Mlo = bins[isamp]
            Mhi = bins[isamp+1]
            info.update({'M*lo': Mlo, 'M*hi': Mhi})
            logmass = np.log10(tbdata.field('bulgemass') +
                               tbdata.field('diskstellarmass'))
            samp *= ((logmass >= Mlo) * (logmass < Mhi))
        tbsamp = tbdata[samp]

        # Density for minimum variance weighting
        z_ran = tbsamp.field(zcol)
        zran_hist, bin_edges = np.histogram(z_ran, bins=nz, range=zrange)
        zstep = bin_edges[1] - bin_edges[0]
        zcen = bin_edges[:-1] + 0.5*zstep
        V_int = area/3.0 * cosmo.dm(bin_edges)**3
        Vbin = np.diff(V_int)
        denbin = zran_hist/Vbin/(3*26)

        for field in range(1, 4):
            for ireal in range(26):
                idx = ((tbsamp.field('ireal') == ireal) *
                       (tbsamp.field('field') == field))

                z = tbsamp[idx].field(zcol)
                ra = tbsamp[idx].field('ra')
                dec = tbsamp[idx].field('dec')
                rgal = cosmo.dm(z)

                ngal = len(z)
                wgal = np.ones(ngal)
                Vgal = np.ones(ngal)
                dengal = np.interp(z, zcen, denbin)

                galcat = Cat(ra, dec, rgal, weight=wgal, den=dengal, Vmax=Vgal,
                             info=info)
                galcat.output(outname('gal'))
                print(ngal, ' galaxies selected')

        # Generate random coords for this sample using summed N(z)
        if maxran < len(z_ran):
            z_ran = np.random.choice(z_ran, maxran, replace=0)
        nran = len(z_ran)
        rran = cosmo.dm(z_ran[:nran])
        wran = np.ones(nran)
        Vran = np.ones(nran)
        for field in range(1, 4):
            # Generate random coords using ransack
            mask_file = '$GAMA_DATA/mask/G{}_rect.ply'.format(region[field-1])
            cmd = "$MANGLE_DIR/bin/ransack -r{} {} {}".format(
                nran, mask_file, ranc_file)
            subprocess.call(cmd, shell=True)
            data = np.loadtxt(ranc_file, skiprows=1)
            ra_ran = data[:, 0]
            dec_ran = data[:, 1]
            denran = np.interp(z_ran, zcen, denbin)

            rancat = Cat(ra_ran, dec_ran, rran, weight=wran, den=denran,
                         Vmax=Vran, info=info)
            rancat.output(outname('ran'))

        # Submit the jobs
        if qsub:
            for field in range(1, 4):
                ranout = outname('ran')
                xiout = outname('rr')
                cmd = qsub_xia_cmd.format(ranout, xiout)
                subprocess.call(cmd, shell=True)
                for ireal in range(26):
                    galout = outname('gal')
                    xiout = outname('gg')
                    cmd = qsub_xia_cmd.format(galout, xiout)
                    subprocess.call(cmd, shell=True)
                    xiout = outname('gr')
                    cmd = qsub_xix_cmd.format(galout, ranout, xiout)
                    subprocess.call(cmd, shell=True)


def xi_mock_cube(infile='Gonzalez_r21.txt', galfile='mock_cube.dat',
                 ranfile='ran_cube.dat', xitemp='xi_mock_{}.dat',
                 logmin=-2, logmax=2, nlog=20,
                 linmin=0, linmax=100, nlin=50, theta_max=12, z_type='z_obs',
                 ranfac=1, qsub=False):
    """
    Clustering in mock data cube.
    Divide cube into 100 Mpc cells which also act as jackknife regions.
    """

    xifile = xitemp.format(z_type)
    cubesize = 500.0
    nc = 5
    cellsize = cubesize/nc
    ncell = nc**3
    data = np.loadtxt(infile, skiprows=1, delimiter=',')
    x, y, z, vx, vy, vz = data[:, 0], data[:, 1], data[:, 2], \
        data[:, 3], data[:, 4], data[:, 5]

    if z_type == 'z_obs':
        # Apply LOS peculiar velocity
        d = np.sqrt(x**2 + y**2 + z**2)
        r = np.array((x, y, z))
        v = np.array((vx, vy, vz))
        v_los = ((v*r).sum(0))/d
        d_obs = d + v_los/H0
        dr = d_obs/d
        x, y, z = dr*x, dr*y, dr*z

        # Apply periodic boundary conditios
        r = np.array((x, y, z))
        fix = (r < 0)
        r[fix] += cubesize
        fix = (r > cubesize)
        r[fix] -= cubesize
        x, y, z = r[0], r[1], r[2]

    ngal = len(x)
    idx = np.floor(x/cellsize).astype(np.int32)
    idy = np.floor(y/cellsize).astype(np.int32)
    idz = np.floor(z/cellsize).astype(np.int32)
    galidx = idz*nc*nc + idy*nc + idx
    galhist, edges = np.histogram(galidx, bins=np.arange(ncell+1)-0.5)

    fout = open(galfile, 'w')
    info = {'file': infile, 'err_type': 'jack', 'z_type': z_type}
    print(info, file=fout)
    print(ngal, nc, ncell, ncell, cellsize, logmin, logmax, nlog,
          linmin, linmax, nlin, theta_max, 0, 0, 0, file=fout)

    if ranfac > 0:
        nrcell = np.random.poisson(ranfac*ngal/ncell, ncell)
        nran = nrcell.sum()
        fran = open(ranfile, 'w')
        print(info, file=fran)
        print(nran, nc, ncell, ncell, cellsize, file=fran)
        progname = 'xi'
    else:
        progname = 'xi_cube'

    icell = 0
    ncuse = 0
    ngsum = 0
    nrsum = 0
    for iz in xrange(nc):
        for iy in xrange(nc):
            for ix in xrange(nc):
                ng = galhist[icell]
                ncuse += 1
                ngsum += ng
                print(ix, iy, iz, ng, file=fout)
                idx = (galidx == icell)
                for i in xrange(ng):
                    print(x[idx][i], y[idx][i], z[idx][i],
                          1, 1, 1, icell, file=fout)
                if ranfac > 0:
                    nr = nrcell[icell]
                    nrsum += nr
                    print(ix, iy, iz, nr, file=fran)
                    xr = np.random.uniform(ix*cellsize, (ix+1)*cellsize, nr)
                    yr = np.random.uniform(iy*cellsize, (iy+1)*cellsize, nr)
                    zr = np.random.uniform(iz*cellsize, (iz+1)*cellsize, nr)
                    for i in xrange(nr):
                        print(xr[i], yr[i], zr[i], 1, 1, 1, icell, file=fran)
                icell += 1
    fout.close()
    if ranfac > 0:
        fran.close()

    print('{} galaxies read, {} written out in {} out of {} cells'.format(
        ngal, ngsum, ncuse, ncell))
    print('{} out of {} randoms written'.format(nrsum, nran))
    if qsub:
        script = '/research/astro/gama/loveday/Documents/Research/python/apollo_job.sh'
        if ranfac > 0:
            qsubcmd = 'qsub {} $BIN/xi {} {} {}'
            cmd = qsubcmd.format(script, galfile, ranfile, xifile)
        else:
            qsubcmd = 'qsub {} $BIN/xi_cube {} {}'
            cmd = qsubcmd.format(script, galfile, xifile)
    else:
        if ranfac > 0:
            cmd = '$BIN/xi {} {} {}'.format(galfile, ranfile, xifile)
        else:
            cmd = '$BIN/xi_cube {} {}'.format(galfile, xifile)
    istat = subprocess.call(cmd, shell=True)
    if istat:
        print('Error {} from C code'.format(istat, progname))


def xi_mass_prep(infile='mass_file', ranfile='ran_file',
                 galout='gal_mass.dat', ranout = 'ran_mass.dat',
                 zrange=(0.002, 0.5), lgmMin=5, lgmMax=15,
                 appMin=14, appMax=19.4, absMin=-30, absMax=12):
    """
    Create input files for xi.c weighted by stellar mass.
    Ignore fluxscale parameter for now as only ~half galaxies have valid value.
    """

    # Read input file into structure
    hdulist = pyfits.open(infile)
    header = hdulist[1].header
    cosmo = util.CosmoLookup(H0, omega_l, zrange)

    tbdata = hdulist[1].data
    ra = tbdata.field('ra')
    dec = tbdata.field('dec')
    z = tbdata.field('z_tonry')
    nq = tbdata.field('nq')
    appMag = tbdata.field('r_petro')
    absMag = tbdata.field('absmag_r') + 5*math.log10(1.0/0.7)
    fluxscale = tbdata.field('fluxscale')
    logmstar = tbdata.field('logmstar') - 2*math.log10(1.0/0.7)
    idx = (zrange[0] < z) * (z < zrange[1]) * (nq > 2) * \
          (appMin < appMag) * (appMag < appMax) * \
          (absMin < absMag) * (absMag < absMax) * \
          (lgmMin <= logmstar)*(logmstar < lgmMax)
        
    ra = ra[idx]
    dec = dec[idx]
    z = z[idx]
    rgal = cosmo.dm(z)
    weight = 10**logmstar[idx]
    zmax = tbdata.field('zmax_19p4')[idx]
    zmax = np.clip(zmax, zrange[0], zrange[1])
    Vmax = cosmo.dm(zmax)
    M_mean = (absMag[idx]).mean()
    logm_mean = (logmstar[idx]).mean()
    info = {'M_mean': M_mean, 'logm_mean': logm_mean}
    
    nsamp = len(ra)
    print(nsamp, ' objects selected')

    galcat = Cat(ra, dec, rgal, weight=weight, Vmax=Vmax, info=info)
    galcat.output(galout)
    
    # Read randoms generated by ransack
    data = np.loadtxt(ranfile, skiprows=1)
    ra = data[:,0]
    dec = data[:,1]
    nran = len(dec)
    
    # Generate distances, weights and Vmax according to galaxy distribution
    ir = (nsamp*np.random.random(nran)).astype(np.int32)
    rran = rgal[ir]
    wran = weight[ir]
    Vran = Vmax[ir]

    plt.clf()
##     plt.hist((rgal, rran), bins=20, normed=True, histtype='step')
    plt.hist((rgal, rran), bins=20, normed=True)
    plt.draw()

    rancat = Cat(ra, dec, rran, weight=wran, Vmax=Vran, info=info)
    rancat.output(ranout)
    

def xi_lum_prep(infile='mass_file', ranfile='ran_file',
                zrange=(0.002, 0.5), lgmMin=5, lgmMax=15,
                appMin=14, appMax=19.4, absMin=-30, absMax=12):
    """
    Create input files for xi.c weighted by luminosity.
    Ignore fluxscale parameter for now as only ~half galaxies have valid value.
    """

    # Read input file into structure
    hdulist = pyfits.open(infile)
    header = hdulist[1].header
    cosmo = util.CosmoLookup(H0, omega_l, zrange)
    tbdata = hdulist[1].data

    # Read randoms generated by ransack
    data = np.loadtxt(ranfile, skiprows=1)
    ra_ran = data[:,0]
    dec_ran = data[:,1]
    nran = len(dec_ran)
    
    for band in 'ugriz':
        galout = 'gal_lum_' + band + '.dat'
        ranout = 'ran_lum_' + band + '.dat'
        absmag_col = 'absmag_' + band
        
        ra = tbdata.field('ra')
        dec = tbdata.field('dec')
        z = tbdata.field('z_tonry')
        nq = tbdata.field('nq')
        appMag = tbdata.field('r_petro')
        absMag = tbdata.field(absmag_col) + 5*math.log10(1.0/0.7)
        fluxscale = tbdata.field('fluxscale')
        logmstar = tbdata.field('logmstar') - 2*math.log10(1.0/0.7)
        idx = (zrange[0] < z) * (z < zrange[1]) * (nq > 2) * \
              (appMin < appMag) * (appMag < appMax) * \
              (absMin < absMag) * (absMag < absMax) * \
              (lgmMin <= logmstar)*(logmstar < lgmMax)
        
        ra = ra[idx]
        dec = dec[idx]
        z = z[idx]
        rgal = cosmo.dm(z)
        weight = 10**(-0.4*(absMag[idx] + 20))
        M_mean = (absMag[idx]).mean()
        logm_mean = (logmstar[idx]).mean()
        info = {'M_mean': M_mean, 'logm_mean': logm_mean}
    
        nsamp = len(ra)
        print(nsamp, ' objects selected')

        galcat = Cat(ra, dec, rgal, weight=weight, info=info)
        galcat.output(galout)
    
        # Generate distances, weights and Vmax according to galaxy distribution
        ir = (nsamp*np.random.random(nran)).astype(np.int32)
        rran = rgal[ir]
        wran = weight[ir]

        plt.clf()
        plt.hist((rgal, rran), bins=20, normed=True, histtype='step')
##         plt.hist((rgal, rran), bins=20, normed=True)
        plt.draw()

        rancat = Cat(ra_ran, dec_ran, rran, weight=wran, info=info)
        rancat.output(ranout)


def xi_vol(infile=gama_data+'kcorr_auto_z01.fits', Mlims=(-25, -20),
           Q=Qdef, P=Pdef, ran_dist='vol', run=2):
    """
    Calculate spatial correlation function for volume-limited sample.
    """

    zmax = util.vol_limits(infile, Q=Q, Mlims=(Mlims[1],), kplot=1)[0]
    z_range = (0.002, zmax)

    plt.clf()
    ax = plt.subplot(111)
    galout = 'gal_vol.dat'
    ranout = 'ran_vol.dat'
    xiout = 'xi_vol.dat'

    xi_select(infile, galout, ranout, xiout,
              z_range=z_range, nz=65, abs_range=Mlims, Q=Q, P=P,
              ran_dist=ran_dist, ax=ax, run=run, zpctrim=None)
    plt.show()


def xi_spec(infile=gama_data+'kcorr_auto_z01.fits', Mlims=(-25, -20),
           Q=Qdef, P=Pdef, ran_dist='vol', run=2, survey_codes=(1, 5),
           mask=gama_data+'/mask/zcomp_gama_sdss.ply'):
    """
    Calculate spatial correlation function for volume-limited
    'spectroscopic' sample (GAMA and SDSS spectra only).
    """

    zmax = util.vol_limits(infile, Q=Qdef, Mlims=(Mlims[1],))[0]
    z_range = (0.002, zmax)

    plt.clf()
    ax = plt.subplot(111)
#    galout = 'gal_spec.dat'
#    ranout = 'ran_spec.dat'
#    xiout = 'xi_spec_.dat'
    galout = 'gal_spec_gsmask.dat'
    ranout = 'ran_spec_gsmask.dat'
    xiout = 'xi_spec_gsmask.dat'

    xi_select(infile, galout, ranout, xiout, mask=mask,
              z_range=z_range, nz=65, abs_range=Mlims, Q=Q, P=P,
              ran_dist=ran_dist, ax=ax, run=run, survey_codes=survey_codes)
    plt.show()


def xi_samples(infile=gama_data+'kcorr_auto_z01.fits', param='lum',
               thresh=0, vol_lim=None, Q=Qdef, P=Pdef, ran_dist='jswml',
               evsamp=False, run=2):
    """
    Calculate spatial correlation function for specified subsamples.
    If vol_lim is specified, all samples will be volume-limited to the
    specified luminosity limit.
    """

    z_range = [0.002, 0.65]
    if param in ('lum', 'vlum'):
        bins = np.array(def_mag_limits)
        if param == 'vlum':
            ran_dist = 'vol'
            zlimits = util.vol_limits(infile, Q=Q, Mlims=bins)
    if vol_lim:
        bins = bins[bins <= vol_lim]
        z_range = [0.002, util.vol_limits(infile, Q=Q, Mlims=(vol_lim,))[0]]
        ran_dist = 'vol'
    if param == 'mass':
        bins = def_mass_limits
    nsamp = len(bins)
    label = param
    if thresh == 0:
        nsamp -= 1
    else:
        label += 't'
    if vol_lim:
        label += '{}'.format(vol_lim)
    nrow, ncol = util.two_factors(nsamp)
    plt.clf()
    app_range = [14, 19.8]
    abs_range = [-99, 99]
    mass_range = [-99, 99]
    for i in range(nsamp):
        ax = plt.subplot(nrow, ncol, i+1)
        if thresh:
            limit = bins[i]
            if param in ('lum', 'vlum'):
                abs_range[1] = limit
            if param == 'mass':
                mass_range[1] = limit
        else:
            limits = bins[i], bins[i+1]
            if param in ('lum', 'vlum'):
                abs_range = limits
            if param == 'mass':
                mass_range = limits

        if param == 'vlum':
            z_range[1] = zlimits[i+1]
        for colour in 'c':
#        for colour in 'cbr':
#            if ran_dist != 'vol':
#                ranfile = gama_data + 'jswml/auto/ranz_{}_{}_{}_{}.dat'.format(
#                    label, colour, *limits)
            # If evsamp specified, read individual evolution parameters

            if evsamp:
                evfile = gama_data + 'jswml/auto/ranz_{}_{}_{}_{}.dat'.format(
                        param, colour, abs_range[0], abs_range[1])
                f = open(evfile, 'r')
                info = eval(f.readline())
                Q, P = info['Q'], info['P']
                f.close()
                print('Q, P = {}, {}'.format(Q, P))
            if thresh:
                galout = 'gal_{}_{}_{}.dat'.format(label, colour, limit)
                ranout = 'ran_{}_{}_{}.dat'.format(label, colour, limit)
                ranzfile = 'ranz_{}_{}_{}.dat'.format(label, colour, limit)
                xiout = 'xi_{}_{}_{}.dat'.format(label, colour, limit)
            else:
                galout = 'gal_{}_{}_{}_{}.dat'.format(label, colour, *limits)
                ranout = 'ran_{}_{}_{}_{}.dat'.format(label, colour, *limits)
                ranzfile = 'ranz_{}_{}_{}_{}.dat'.format(label, colour, *limits)
                xiout = 'xi_{}_{}_{}_{}.dat'.format(label, colour, *limits)

            xi_select(infile, galout, ranout, xiout, ranzfile=ranzfile,
                      param=param, z_range=z_range, nz=65,
                      app_range=app_range, abs_range=abs_range, Q=Q, P=P,
                      mass_range=mass_range, colour=colour,
                      ran_dist=ran_dist, ax=ax, run=run)
    plt.draw()


def xi_farrow_comp(infile=gama_data+'jswml/auto/kcorrz00.fits',
                   ran_dist='jswml', run=2):
    """
    Calculate spatial correlation function for subsamples that match
    Farrow+2015 selection (largest of first 5 samples, Table 2).
    """

    Mlims = (-22, -21, -20, -19, -18, -17)
    zlims = (0.5, 0.35, 0.24, 0.14, 0.02)
    z_range = [0.002, 0.65]
    nsamp = len(Mlims) - 1
    nrow, ncol = util.two_factors(nsamp)
    plt.clf()
    app_range = [14, 19.8]
    mass_range = [-99, 99]
    for i in range(nsamp):
        ax = plt.subplot(nrow, ncol, i+1)
        abs_range = (Mlims[i], Mlims[i+1])
        j = min(i, nsamp-2)  # Faintest (last) two samples share same z limits
        z_range = (zlims[j+1], zlims[j])

        ranfile = gama_data + 'jswml/auto/ranz_lum_c_{}_{}.dat'.format(
                abs_range[0], abs_range[1])
        galout = 'gal_f_M{}_{}_z{}_{}.dat'.format(abs_range[0], abs_range[1],
                                                  z_range[0], z_range[1])
        ranout = 'ran_f_M{}_{}_z{}_{}.dat'.format(abs_range[0], abs_range[1],
                                                  z_range[0], z_range[1])
        xiout = 'xi_f_M{}_{}_z{}_{}.dat'.format(abs_range[0], abs_range[1],
                                                z_range[0], z_range[1])

        xi_select(infile, ranfile, galout, ranout, xiout,
                  z_range=z_range, nz=65,
                  app_range=app_range, abs_range=abs_range,
                  mass_range=mass_range, colour='c',
                  ran_dist=ran_dist, ax=ax, run=run)
    plt.draw()


def xi_colour_samples(infile=gama_data+'kcorr_auto_z01.fits', param='lum',
                      ran_dist='jswml', Q=Qdef, P=Pdef, run=0):
    """
    Create colour sub-samples for xi(s).
    """
    nrow = 1
    ncol = 3
    plt.clf()
    iplot = 1
    for colour in 'cbr':
        ax = plt.subplot(nrow, ncol, iplot)
        galout = 'gal_colour_{}.dat'.format(colour)
        ranout = 'ran_colour_{}.dat'.format(colour)
        ranzfile = 'ranz_colour_{}.dat'.format(colour)
        xiout = 'xi_colour_{}.dat'.format(colour)
        xi_select(infile, galout, ranout, xiout, ranzfile=ranzfile,
                  param=param, colour=colour, Q=Q, P=P,
                  ran_dist=ran_dist, ax=ax, run=run)
        iplot += 1
    plt.draw()


def xi_mag_samples(infile='mass_file', ranfile='ran_file', cshfile='xi.csh'):
    """
    Create app mag selected sub-samples for xi(s).
    """

    fout = open(cshfile, 'w')
    mlimits = (16, 17, 18, 18.5, 19, 19.4)
    for i in range(len(mlimits)-1):
        mlo = mlimits[i]
        mhi = mlimits[i+1]
        galout = 'gal_' + str(mlo) + '_' + str(mhi) + '.dat'
        ranout = 'ran_' + str(mlo) + '_' + str(mhi) + '.dat'
        xiout = 'xi_{}_{}.dat'.format(mlo, mhi)
        xi_select(infile, ranfile, galout, ranout, xiout,
                  zrange=(0.002, 0.5), nz=50,
                  appMin=mlo, appMax=mhi, absMin=-99, absMax=99,
                  ran_dist='smooth')
        print('nice xi {} {} {} -1 2 15 -1 2 15  0 60 30'.format(
            galout, ranout, xiout), file=fout)

    # Spawn xi subprocess
    fout.close()
    subprocess.Popen(['csh', cshfile])


def xi_vthresh_ev(infile=gama_data + 'jswml/auto/kcorrz01.fits', ran_dist='vol',
                  qsub=True):
    """
    Evolution within volume-limited luminosity threshold samples.
    """

    galroot = 'gal_Vt_{}_{}_z_{:4.2f}_{:4.2f}.dat'
    ranroot = 'ran_Vt_{}_{}_z_{:4.2f}_{:4.2f}.dat'
    xiroot = 'xi_Vt_{}_{}_z_{:4.2f}_{:4.2f}.dat'

    Mlimits = (-23, -22, -21, -20, -19, -18)
    zlimits = util.vol_limits(infile, Q=Qdef, Mlims=Mlimits)
    zmin = 0.002
    zbins = ((zmin, 0.4, zlimits[0]),
             (zmin, 0.2, 0.3, 0.4, zlimits[1]),
             (zmin, 0.1, 0.2, 0.3, zlimits[2]),
             (zmin, 0.1, 0.2, zlimits[3]),
             (zmin, 0.1, zlimits[4]),
             (zmin, 0.06, zlimits[5]))

    nrow = 2
    ncol = 3
    plt.clf()
    ranfile = None
    for colour in 'cbr':
        for i in range(len(Mlimits)):
            ax = plt.subplot(nrow, ncol, i+1)
            Mlo = -25
            Mhi = Mlimits[i]
            for iz in range(len(zbins[i])-1):
                zlo = zbins[i][iz]
                zhi = zbins[i][iz+1]
                galout = galroot.format(colour, Mhi, zlo, zhi)
                ranout = ranroot.format(colour, Mhi, zlo, zhi)
                xiout = xiroot.format(colour, Mhi, zlo, zhi)
                xi_select(infile, ranfile, galout, ranout, xiout,
                          zrange=(zlo, zhi), nz=int(100*(zhi-zlo)),
                          appMin=14, appMax=19.8, absMin=Mlo, absMax=Mhi,
                          colour=colour, ran_dist=ran_dist, ax=ax)
                cmd = qsub_xi_cmd.format(galout, ranout, xiout)
                if qsub:
                    subprocess.call(cmd, shell=True)
    plt.draw()


def xi_z_samples(infile='mass_file', ranfile='ran_file', cshfile='xi.csh'):
    """
    Create redshift sub-samples to same lum limit (-21) for xi(s).
    """
    zlimits = (0.002, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
    for i in range(len(zlimits)-1):
        zlo = zlimits[i]
        zhi = zlimits[i+1]
        galout = 'gal_z_' + str(zlo) + '_' + str(zhi) + '.dat'
        ranout = 'ran_z_' + str(zlo) + '_' + str(zhi) + '.dat'
        xi_select(infile, ranfile, galout, ranout, None,
                  zrange=(zlo, zhi), nz=50, appMin=14, appMax=19.4,
                  absMin=-30, absMax=-21, ran_dist='vol')


def xi_group_prep(infile=gama_data+'/groups/G3Cv04/G3CFoFGroup194v04.dat',
                  ranfile='ran_file',
                  galout='group.dat', ranout='ran_group.dat',
                  zrange=(0.002, 0.5), lgmMin=5, lgmMax=15,
                  appMin=14, appMax=19.4, absMin=-30, absMax=12):
    """
    Create input files for xi.c for GAMA groups.
    """

    cosmo = util.CosmoLookup(H0, omega_l, zrange)

    # Read group data (see Robotham et al 2011 sec 4.3 re 'A' factor)
    file = gama_data+'/groups/G3Cv04/G3CFoFGroup194v04.dat'
    data = np.loadtxt(file, skiprows=1)
    ra = data[:,3]
    dec = data[:,4]
    z = data[:,6]
    totrmag = data[:,16]
    A = -1.2 + 20.7/np.sqrt(data[:,1]) + 2.3/np.sqrt(data[:,6])
    mass = A*data[:,18]
    
    rgal = cosmo.dm(z)
    weight = 10**mass
    M_mean = totrmag.mean()
    logm_mean = np.log10(mass.mean())
    info = {'M_mean': M_mean, 'logm_mean': logm_mean}
    
    nsamp = len(ra)
    print(nsamp, ' objects selected')

    galcat = Cat(ra, dec, rgal, info=info)
    galcat.output(galout)
    
    # Read randoms generated by ransack
    data = np.loadtxt(ranfile, skiprows=1)
    ra = data[:,0]
    dec = data[:,1]
    nran = len(dec)
    
    # Generate distances, weights and Vmax according to galaxy distribution
    ir = (nsamp*np.random.random(nran)).astype(np.int32)
    rran = rgal[ir]
##     wran = weight[ir]

    plt.clf()
##     plt.hist((rgal, rran), bins=20, normed=True, histtype='step')
    plt.hist((rgal, rran), bins=20, normed=True)
    plt.draw()

    rancat = Cat(ra, dec, rran, info=info)
    rancat.output(ranout)
    

def xi_hubble_mock_select(infile='1.2df_ngp', 
              galout='gal.dat', ranout = 'ran.dat', zcol=4,
              mlim=19.0, zrange=(0.002, 0.5), cellsize=100.0):
    """
    Select Hubble Volume mock galaxies for xi.c.
    zcol = 4 for observed redshift, = 5 for Hubble redshift.
    """

    def output(cat, outfile):
        """Output the galaxy or random data."""
        
        #  Jackknife regions correspond to five 15 deg regions in RA.
        ijack = np.floor((cat.ra - 147.5)/15.0).astype(np.int32)
        njack = max(ijack) + 1
        cellsize = 100.0

        # Assign objects to cells
        idx = np.floor((cat.x - xmin)/cellsize).astype(np.int32)
        idy = np.floor((cat.y - ymin)/cellsize).astype(np.int32)
        idz = np.floor((cat.z - zmin)/cellsize).astype(np.int32)
        idxcell = idz*nx*ny + idy*nx + idx
        cellHist, edges = np.histogram(idxcell, bins=np.arange(nx*ny*nz+1)-0.5)
##         cellHist = np.bincount(idxcell, minlength=nx*ny*nz)
##         u, inv = np.unique(idxcell, return_inverse=True)
##         nused = np.count_nonzero(cellHist)
        nused = sum(c > 0 for c in cellHist)
        print(nused, ' out of', nx*ny*nz,' cells occupied')

        print('Writing out ', outfile)
        nsamp = len(cat.ra)
        fout = open(outfile, 'w')
        print(nsamp, nused, cellsize, njack, file=fout)
        weight = 1
        icell = 0
        for iz in xrange(nz):
            for iy in xrange(ny):
                for ix in xrange(nx):
                    nc = cellHist[icell]
                    if nc > 0:
                        print(ix, iy, iz, nc, file=fout)
                        idx = (idxcell == icell)
                        for i in xrange(nc):
                            print(cat.x[idx][i], cat.y[idx][i], cat.z[idx][i],
                                  weight, ijack[idx][i], file=fout)
                    icell += 1
        fout.close()

    # Read input file into array  NB ra, dec in radians
    data = np.loadtxt(infile)
    cosmo = util.CosmoLookup(H0, omega_l, zrange)

    mag = data[:,3]
    z = data[:,zcol]
    idx = (zrange[0] < z) * (z < zrange[1]) * (mag < mlim)
    data = data[idx,:]
    ra = np.rad2deg(data[:,1])
    dec = np.rad2deg(data[:,2])
    z = data[:,zcol]
    rgal = cosmo.dm(z)
    gal = Cat(ra, dec, rgal)
    
    nsamp = len(ra)
    print(nsamp, 'objects selected')

    # Generate random points
    nran = 5*nsamp
    ra_range = (147.5, 222.5)
    dec_range = (-7.5, 2.5)
    dec = dec_range[0] + (dec_range[1] - dec_range[0])*np.random.random(nran)
    print(nran, 'initial randoms')

    # Select according to cos(dec)
    idx = np.random.random(nran) < np.cos(np.deg2rad(dec))
    dec = dec[idx]
    nran = len(dec)
    ra = ra_range[0] + (ra_range[1] - ra_range[0])*np.random.random(nran)
    print(nran, 'randoms after dec selection')
    
    # Generate distances according to galaxy distribution
    ir = (nsamp*np.random.random(nran)).astype(np.int32)
    rran = rgal[ir]
    ran = Cat(ra, dec, rran)
    
    xmin = min(np.concatenate((gal.x, ran.x)))
    xmax = max(np.concatenate((gal.x, ran.x)))
    ymin = min(np.concatenate((gal.y, ran.y)))
    ymax = max(np.concatenate((gal.y, ran.y)))
    zmin = min(np.concatenate((gal.z, ran.z)))
    zmax = max(np.concatenate((gal.z, ran.z)))
    nx = int(math.ceil((xmax-xmin)/cellsize))
    ny = int(math.ceil((ymax-ymin)/cellsize))
    nz = int(math.ceil((zmax-zmin)/cellsize))
    print('Cells: ', nx, ny, nz)
    
    plt.clf()
    plt.hist((rgal, rran), bins=20, normed=True, histtype='step')
    plt.draw()

    output(gal, galout)
    output(ran, ranout)


def vol_limits_mock(Mlims, mlim=19.8, zrange=(0.002, 0.5)):
    """Return redshift limits for given absolute magnitude limits for mocks."""

    cosmo = util.CosmoLookup(H0, omega_l, zrange)

    z_list = []
    for Mlim in Mlims:
        z = scipy.optimize.bisect(
            lambda z: Mlim - (mlim - cosmo.dist_mod(z) - util.mock_ke_corr(z)),
            zrange[0], zrange[-1])
        z_list.append(z)
        print(Mlim, z)
    return z_list


def cat_stats(infile):
    """Summary stats for galaxy or random catalogue."""
    dat = np.loadtxt(infile, skiprows=3)
    r = np.mean(np.sum(dat[:, 0:3]**2, axis=1)**0.5)
    w = np.mean(dat[:, 3])
    d = np.mean(dat[:, 4])
    V = np.mean(dat[:, 5])
    print('mean r {}, wt {}, den {}, Vmax {}'.format(r, w, d, V))


# --------------------------------
# Direct PVD calculation for mocks
# --------------------------------

def pvd_mocks(infile='Gonzalez.fits', param='lum_c', zrange=(0.002, 0.5),
              rpmin=-2, rpmax=2, nrp=20, pimin=-2, pimax=2, npi=20,
              vmax=5000, nv=50, qsub=False):
    """Direct PVD calculation for mocks."""

    c = 3e5
    cosmo = util.CosmoLookup(H0, omega_l, zrange)
    hdulist = pyfits.open(infile)
    tbdata = hdulist[1].data
    idx = ((tbdata.field('field') >= 1) *
           (tbdata.field('field') <= 3) *
           (tbdata.field('redshift_obs') >= zrange[0]) *
           (tbdata.field('redshift_obs') < zrange[1]))
    tbdata = tbdata[idx]

    if 'lum' in param:
        bins = def_mag_limits
    else:
        bins = def_mass_limits
    nsamp = len(bins) - 1
    for ireal in range(26):
        for isamp in range(nsamp):
            Mlo = bins[isamp]
            Mhi = bins[isamp+1]
            galfile = 'mock_{}_{}_{:02d}.dat'.format(Mlo, Mhi, ireal)
            pvdfile = 'pvd_{}_{}_{:02d}.dat'.format(Mlo, Mhi, ireal)
            if 'lum' in param:
                idx = ((tbdata.field('ireal') == ireal) *
                       (tbdata.field('SDSS_r_rest_abs') >= Mlo) *
                       (tbdata.field('SDSS_r_rest_abs') < Mhi))
            else:
                logmass = np.log10(tbdata.field('bulgemass') +
                                   tbdata.field('diskstellarmass'))
                idx = ((tbdata.field('ireal') == ireal) *
                       (logmass >= Mlo) * (logmass < Mhi))

            z_obs = tbdata[idx].field('redshift_obs')
            z_cos = tbdata[idx].field('redshift_cos')
            ra = tbdata[idx].field('ra')
            dec = tbdata[idx].field('dec')
            ngal = len(ra)

            rgal = cosmo.dm(z_cos)
            x_c = rgal*np.cos(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
            y_c = rgal*np.sin(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
            z_c = rgal*np.sin(np.deg2rad(dec))
            rgal = cosmo.dm(z_obs)
            x_o = rgal*np.cos(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
            y_o = rgal*np.sin(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
            z_o = rgal*np.sin(np.deg2rad(dec))
            vpec = c * ((1+z_obs)/(1+z_cos) - 1)
            coords = np.array((x_c, y_c, z_c, x_o, y_o, z_o, vpec)).T

            # sort on z_cos so that vpl is negative for infall
            isort = np.argsort(z_cos)
            coords = coords[isort]
            vpec = vpec[isort]

            ncell = 1
            ix = 0
            iy = 0
            iz = 0
            cellsize = 100.0
            fout = open(galfile, 'w')
            print(infile, file=fout)
            print(ngal, ncell, cellsize, rpmin, rpmax, nrp,
                  pimin, pimax, npi, vmax, nv, def_theta_max, file=fout)
            print(ix, iy, iz, ngal, file=fout)
            for i in xrange(ngal):
                print(coords[i, 0], coords[i, 1], coords[i, 2],
                      coords[i, 3], coords[i, 4], coords[i, 5],
                      coords[i, 6], file=fout)
            fout.close()

            if qsub:
                qsubcmd = 'qsub /research/astro/gama/loveday/Documents/Research/python/apollo_job.sh $BIN/pvd_mock {} {}'
                cmd = qsubcmd.format(galfile, pvdfile)
            else:
                cmd = '$BIN/pvd_mock {} {}'.format(galfile, pvdfile)
            istat = subprocess.call(cmd, shell=True)
            if istat:
                print('Error {} from pvd_mock.c'.format(istat))

    # Find pairs using scipy.spatial.KDTree
#    tree = scipy.spatial.KDTree(coords)
#    pairs = np.array(list(tree.query_pairs(rmax)))
#    npair = len(pairs)
#    sep = np.zeros((npair, 2))
#    ipair = 0
#    for p in pairs:
#        s = coords[p[1]] - coords[p[0]]
#        sep[ipair, 0] = math.sqrt((s**2).sum())
#        l = 0.5*(coords[p[1]] + coords[p[0]])
#        sep_par = abs(np.dot(s, l)/math.sqrt(np.dot(l, l)))
#        sep[ipair, 1] = math.sqrt(np.dot(s, s) - sep_par**2)
#        ipair += 1
#
#    vmean = np.zeros((nbin, 2))
#    vvar = np.zeros((nbin, 2))
#    sig_exp = np.zeros((nbin, 2))
#    if plot:
#        plt.clf()
#        ttlstr = r'r = {:5.1f}, {} pairs, $\mu = {:5.1f} \pm {:5.1f}$, $\sigma_G = {:5.1f}$, $\sigma_e = {:5.1f}$'
#    for isep in xrange(2):
#        for ir in xrange(nbin):
#            sel = ((rbin_lim[ir] <= sep[:, isep]) *
#                   (rbin_lim[ir+1] > sep[:, isep]))
#            vpl = vpec[pairs[sel, 1]] - vpec[pairs[sel, 0]]
#            vmean[ir, isep] = np.mean(vpl)
#            vvar[ir, isep] = np.var(vpl)
#            sig_exp[ir, isep] = (root2 *
#                                 np.sum(np.fabs(vpl - vmean[ir, isep])) /
#                                 len(vpl))
#            if plot:
#                vmean_err = scipy.stats.sem(vpl)
#                plt.hist(vpl, 25)
#                plt.semilogy(basey=10, nonposy='clip')
#                plt.xlabel(r'$v_{12}$ (km/s)')
#                plt.ylabel('Frequency')
#                plt.title(ttlstr.format(rbin_cen[ir], len(vpl),
#                                        vmean[ir, isep], vmean_err,
#                                        math.sqrt(vvar[ir, isep]),
#                                        sig_exp[ir, isep]))
#                plt.show()
#
##    np.savetxt(outfile, (rbin_cen, vmean, vvar, sig_exp))
#    np.savez(outfile, rbin_cen=rbin_cen, vmean=vmean, vvar=vvar,
#             sig_exp=sig_exp)


def pvd_mocks_vol(infile='Gonzalez.fits', Mlims=(-25, -20),
                  rpmin=-2, rpmax=2, nrp=20, pimin=-2, pimax=2, npi=20,
                  vmax=5000, nv=50, qsub=False):
    """Direct PVD calculation for mock volume-limited sample."""

    c = 3e5
    Mlo = Mlims[0]
    Mhi = Mlims[1]
    zmax = util.vol_limits(infile, Q=Qdef, Mlims=(Mlims[1],))[0]
    zrange = (0.002, zmax)
    cosmo = util.CosmoLookup(H0, omega_l, zrange)
    hdulist = pyfits.open(infile)
    tbdata = hdulist[1].data
    idx = ((tbdata.field('field') >= 1) *
           (tbdata.field('field') <= 3) *
           (tbdata.field('redshift_obs') >= zrange[0]) *
           (tbdata.field('redshift_obs') < zrange[1]))
    tbdata = tbdata[idx]

    for ireal in range(26):
        galfile = 'mock_vol_{:02d}.dat'.format(ireal)
        pvdfile = 'pvd_vol_ll_{:02d}.dat'.format(ireal)
        idx = ((tbdata.field('ireal') == ireal) *
               (tbdata.field('SDSS_r_rest_abs') >= Mlo) *
               (tbdata.field('SDSS_r_rest_abs') < Mhi))
        z_obs = tbdata[idx].field('redshift_obs')
        z_cos = tbdata[idx].field('redshift_cos')
        ra = tbdata[idx].field('ra')
        dec = tbdata[idx].field('dec')
        ngal = len(ra)

        rgal = cosmo.dm(z_cos)
        x_c = rgal*np.cos(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
        y_c = rgal*np.sin(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
        z_c = rgal*np.sin(np.deg2rad(dec))
        rgal = cosmo.dm(z_obs)
        x_o = rgal*np.cos(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
        y_o = rgal*np.sin(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
        z_o = rgal*np.sin(np.deg2rad(dec))
        vpec = c * ((1+z_obs)/(1+z_cos) - 1)
        coords = np.array((x_c, y_c, z_c, x_o, y_o, z_o, vpec)).T

        # sort on z_cos so that vpl is negative for infall
        isort = np.argsort(z_cos)
        coords = coords[isort]
        vpec = vpec[isort]

        ncell = 1
        ix = 0
        iy = 0
        iz = 0
        cellsize = 100.0
        fout = open(galfile, 'w')
        print(infile, file=fout)
        print(ngal, ncell, cellsize, rpmin, rpmax, nrp,
              pimin, pimax, npi, vmax, nv, def_theta_max, file=fout)
        print(ix, iy, iz, ngal, file=fout)
        for i in xrange(ngal):
            print(coords[i, 0], coords[i, 1], coords[i, 2],
                  coords[i, 3], coords[i, 4], coords[i, 5],
                  coords[i, 6], file=fout)
        fout.close()

        if qsub:
            qsubcmd = 'qsub /research/astro/gama/loveday/Documents/Research/python/apollo_job.sh $BIN/pvd_mock {} {}'
            cmd = qsubcmd.format(galfile, pvdfile)
        else:
            cmd = '$BIN/pvd_mock {} {}'.format(galfile, pvdfile)
        istat = subprocess.call(cmd, shell=True)
        if istat:
            print('Error {} from pvd_mock.c'.format(istat))


def pvd_mock_av_samples(param='lum_c'):
    """
    Average PVD from several mock samples.
    """
    if 'lum' in param:
        bins = def_mag_limits
    else:
        bins = def_mass_limits
    nsamp = len(bins) - 1
    for isamp in range(nsamp):
        Mlo = bins[isamp]
        Mhi = bins[isamp+1]
        intemp = 'pvd_{}_{}'.format(Mlo, Mhi) + '_{:02d}.dat'
        outfile = 'pvd_{}_{}.npz'.format(Mlo, Mhi)
        pvd_mock_av(intemp, outfile, vhist_plot_file=None,
                    twod_plot_file=None, plot_file=None)


def pvd_mock_av(filetemp='pvd_vol_ll_{:02d}.dat', outfile='pvd_vol_ll.npz',
                pilim=50, gamma=1.64, r0=6.16, beta=0.5, nest=26,
                vhist_rp_bins=(3, 8, 13, 18), neig=0, vlims=(-5000, 5000),
                vmean_lims=(-350, 0), vsig_lims=(0, 1500),
                vhist_plot_file='pvd_mock_hist_ll.pdf',
                twod_plot_file='pvd_mock_twod_ll.pdf',
                plot_file_old='pvd_mock_ll_old.pdf',
                plot_file='pvd_mock_ll.pdf',
                size=(5, 8)):
    """
    Average PVD from mocks.
    """

    def chi2(p, fn, vbins, vhist, cov):
        """chi2 from exponential or Gaussian fit."""
        fit = fn(p, vbins)
        return cov.chi2(vhist, fit, neig=neig)

    def exp((mu, sigma, norm), vbins):
        """Exponential."""
        return norm * np.exp(-root2*np.abs(vbins - mu)/sigma)

    def gauss((mu, sigma, norm), vbins):
        """Gaussian."""
        return norm * np.exp(-(vbins - mu)**2/(2*sigma**2))

    fmt = ['k-', 'b--', 'g-.', 'r:']
    n_tot_arr = []
    n_red_arr = []
    n_proj_arr = []
    r_tot_arr = []
    vmean_tot_arr = []
    sigma_tot_arr = []
    sigma_exp_tot_arr = []
    vmean_red_arr = []
    sigma_red_arr = []
    sigma_exp_red_arr = []
    vmean_proj_arr = []
    sigma_proj_arr = []
    sigma_exp_proj_arr = []
    vbin_arr = []
    vhist_arr = []
    for ireal in range(nest):
        infile = filetemp.format(ireal)
        with open(infile, 'r') as f:
            f.readline()
            f.readline()
            args = f.readline().split()
        nrp = int(args[1])
        lgrpmin = float(args[2])
        lgrpmax = float(args[3])
        lgrpstep = (lgrpmax - lgrpmin)/nrp
        npi = int(args[4])
        pimin = float(args[5])
        pimax = float(args[6])

        data = np.genfromtxt(infile, skip_header=3, skip_footer=nrp*npi)
        n_tot_arr.append(data[:, 0])
        r_tot_arr.append(data[:, 1])
        vmean_tot_arr.append(data[:, 2])
        sigma_tot_arr.append(data[:, 3])
        sigma_exp_tot_arr.append(data[:, 4])
        vbin_arr.append(data[:, 5::2])
        vhist_arr.append(data[:, 6::2])

        data = np.genfromtxt(infile, skip_header=nrp+3)
        n_red = data[:, 0].reshape((nrp, npi)).T
        vmean_red = data[:, 3].reshape((nrp, npi)).T
        sigma_red = data[:, 4].reshape((nrp, npi)).T
        sigma_exp_red = data[:, 5].reshape((nrp, npi)).T
        n_red_arr.append(n_red)
        vmean_red_arr.append(vmean_red)
        sigma_red_arr.append(sigma_red)
        sigma_exp_red_arr.append(sigma_exp_red)

        # Avearge over line-of-sight direction up to pilim
        if ireal == 0:
            if pimin < 0:
                pilim = math.log10(pilim)
            pistep = (pimax-pimin)/npi
            ipimax = min(npi, int((pilim-pimin)/pistep))
        n_proj_arr.append(np.sum(n_red[:ipimax, :], axis=0))
        vmean_proj_arr.append(np.ma.average(vmean_red[:ipimax, :], axis=0,
                                            weights=n_red[:ipimax, :]))
        sigma_proj_arr.append(np.ma.average(sigma_red[:ipimax, :], axis=0,
                                            weights=n_red[:ipimax, :]))
        sigma_exp_proj_arr.append(
            np.ma.average(sigma_exp_red[:ipimax, :], axis=0,
                          weights=n_red[:ipimax, :]))

    # Weighted avearge and standard deviation from individual estimates
    # Need weighted mean so that empty bins are not included
    sep = np.ma.average(r_tot_arr, axis=0, weights=n_tot_arr)
    vlt = np.ma.average(vmean_tot_arr, axis=0, weights=n_tot_arr)
    vlt_err = np.sqrt(np.ma.average((vmean_tot_arr - vlt)**2, axis=0,
                      weights=n_tot_arr))
    slt = np.ma.average(sigma_tot_arr, axis=0, weights=n_tot_arr)
    slt_err = np.sqrt(np.ma.average((sigma_tot_arr - slt)**2, axis=0,
                      weights=n_tot_arr))
    slt_exp = np.ma.average(sigma_exp_tot_arr, axis=0, weights=n_tot_arr)
    slt_exp_err = np.sqrt(np.ma.average((sigma_exp_tot_arr - slt_exp)**2,
                                        axis=0, weights=n_tot_arr))

    n_red = np.mean(n_red_arr, axis=0)
    vmean_red = np.ma.average(vmean_red_arr, axis=0, weights=n_red_arr)
    vlp = np.ma.average(vmean_proj_arr, axis=0, weights=n_proj_arr)
    vlp_err = np.sqrt(np.ma.average((vmean_proj_arr - vlp)**2, axis=0,
                                    weights=n_proj_arr))
    sigma_red = np.ma.average(sigma_red_arr, axis=0, weights=n_red_arr)
    slp = np.ma.average(sigma_proj_arr, axis=0, weights=n_proj_arr)
    slp_err = np.sqrt(np.ma.average((sigma_proj_arr - slp)**2, axis=0,
                                    weights=n_proj_arr))
    slp_exp = np.ma.average(sigma_exp_proj_arr, axis=0, weights=n_proj_arr)
    slp_exp_err = np.sqrt(np.ma.average((sigma_exp_proj_arr - slp_exp)**2,
                                        axis=0, weights=n_proj_arr))
    vhist_arr = np.array(vhist_arr)
    vbins = np.ma.average(vbin_arr, axis=0, weights=vhist_arr)
    vhist = np.mean(vhist_arr, axis=0)
    vhist_err = np.std(vhist_arr, axis=0)

    np.savez(outfile, sep=np.ma.filled(sep, 0),
             vlt=np.ma.filled(vlt, 0), vlt_err=np.ma.filled(vlt_err, 0),
             slt=np.ma.filled(slt, 0), slt_err=np.ma.filled(slt_err, 0),
             slt_exp=np.ma.filled(slt_exp, 0),
             slt_exp_err=np.ma.filled(slt_exp_err, 0),
             vlp=np.ma.filled(vlp, 0), vlp_err=np.ma.filled(vlp_err, 0),
             slp=np.ma.filled(slp, 0), slp_err=np.ma.filled(slp_err, 0),
             slp_exp=np.ma.filled(slp_exp, 0),
             slp_exp_err=np.ma.filled(slp_exp_err, 0))

    # velocity histograms
    nrow = len(vhist_rp_bins)
    plt.clf()
    fig, axes = plt.subplots(nrow, 1, sharex=True, num=1)
    axes = axes.ravel()
    fig.subplots_adjust(left=0.18, bottom=0.05, hspace=0.0, wspace=0.0)
    fig.text(0.55, 0.0, r'$v_{12}\ ({{\rm km/s}})$',
             ha='center', va='center')
    fig.text(0.06, 0.47, 'Frequency', ha='center', va='center',
             rotation='vertical')
    irow = 0
    for i in vhist_rp_bins:
        ax = axes[irow]
        vb = vbins[i, :]
        vh = vhist[i, :]
        use = vh > 0
        vb = vb[use]
        vh = vh[use]
        if len(vh) > 0:
            vhe = vhist_err[i, use]
            vhc = Cov(vhist_arr[:, i, use].T, 'mock')
    #        vhc.plot(norm=True, label='V hist')
    #        plt.show()
    #        print vb
    #        print vh
    #        pdb.set_trace()
            ax.step(vb, vh, where='mid')
    #        ax.errorbar(vb, vh, vhe, fmt='none', ecolor='b', capthick=1)
    
            p0 = (0.0, 500.0, np.amax(vh))
            ndof = len(vb)
            if neig > 0 and neig < ndof:
                ndof = neig
            ndof -= len(p0)
            out = scipy.optimize.fmin(chi2, p0, args=(exp, vb, vh, vhc),
                                      maxfun=10000, maxiter=10000, full_output=1,
                                      disp=0)
            pe = out[0]
            chi_e = out[1]*nest/ndof
            efit = exp(pe, vb)
            ax.plot(vb, efit)
            out = scipy.optimize.fmin(chi2, p0, args=(gauss, vb, vh, vhc),
                                      maxfun=10000, maxiter=10000, full_output=1,
                                      disp=0)
            pg = out[0]
            chi_g = out[1]*nest/ndof
            gfit = gauss(pg, vb)
            ax.plot(vb, gfit, '--')
            ax.set_xlim(vlims)
            ax.set_ylim(10, 3*np.amax(vh))
            ax.semilogy(basey=10, nonposy='clip')
    #        ax.set_ylabel('Frequency')
            r = 10**(lgrpmin + (i+0.5)*lgrpstep)
            label = r'$r_\bot = {:5.1f}\ h^{{-1}} {{\rm Mpc}}$'.format(r)
            ax.text(0.05, 0.85, label, transform=ax.transAxes)
            ax.text(0.05, 0.74, r'$\mu_e = {:5.0f}\ {{\rm km/s}}$'.format(pe[0]),
                    transform=ax.transAxes)
            ax.text(0.05, 0.63, r'$\sigma_e = {:5.0f}\ {{\rm km/s}}$'.format(pe[1]),
                    transform=ax.transAxes)
            ax.text(0.05, 0.52, r'$\chi^2_e = {:5.1f}$'.format(chi_e),
                    transform=ax.transAxes)
            ax.text(0.95, 0.74, r'$\mu_G = {:5.0f}\ {{\rm km/s}}$'.format(pg[0]),
                    ha='right', transform=ax.transAxes)
            ax.text(0.95, 0.63, r'$\sigma_G = {:5.0f}\ {{\rm km/s}}$'.format(pg[1]),
                    ha='right', transform=ax.transAxes)
            ax.text(0.95, 0.52, r'$\chi^2_G = {:5.1f}$'.format(chi_g),
                    ha='right', transform=ax.transAxes)
        irow += 1
#    ax.set_xlabel(r'$v_\parallel\ ({{\rm km/s}})$')
#    plt.show()
    plt.draw()

    if vhist_plot_file:
        fig = plt.gcf()
        fig.set_size_inches((5, 10))
        plt.savefig(plot_dir + vhist_plot_file, bbox_inches='tight')

    # 2d distributions of v_mean and sigma
    extent = (lgrpmin, lgrpmax, pimin, pimax)
    contours = (1e2, 1e3, 1e4, 1e5, 1e6)
    plt.clf()
    fig, axes = plt.subplots(2, 1, sharex=True, num=1)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    if pimin < 0:
        ylabel = r'$\log_{10} (r_\parallel\ [h^{-1}\ {{\rm Mpc}}])$'
    else:
        ylabel = r'$r_\parallel\ [h^{-1}\ {{\rm Mpc}}]$'
    ax = axes[0]
    im = ax.imshow(np.flipud(vmean_red), aspect='auto', cmap='viridis_r',
                   interpolation='none',
                   vmin=vmean_lims[0], vmax=vmean_lims[1], extent=extent)
    cs = ax.contour(n_red, contours, colors='w', aspect='auto', extent=extent)
    ax.clabel(cs, inline=1, fontsize=10, fmt='%1.0e')
#    ax.semilogx(basex=10, nonposy='clip')
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(r'$\overline{v_{12}}\ [{{\rm km\ s}}^{-1}]$')
    ax.set_ylabel(ylabel)

    ax = axes[1]
    im = ax.imshow(np.flipud(sigma_exp_red), aspect='auto',
                   interpolation='none',
                   vmin=vsig_lims[0], vmax=vsig_lims[1], extent=extent)
    cs = ax.contour(n_red, contours, colors='w', aspect='auto', extent=extent)
    ax.clabel(cs, inline=1, fontsize=10, fmt='%1.0e')
#    ax.semilogx(basex=10, nonposy='clip')
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(r'$\sigma^{\rm exp}_{12}\ [{{\rm km\ s}}^{-1}]$')
    ax.set_ylabel(ylabel)

    ax.set_xlabel(r'$\log_{10} (r_\perp\ [h^{-1}\ {{\rm Mpc}}])$')
#    pdb.set_trace()
    plt.draw()

    if twod_plot_file:
        fig = plt.gcf()
        fig.set_size_inches((5, 7))
        plt.savefig(plot_dir + twod_plot_file, bbox_inches='tight')

    # v_mean and sigma as function of r and r_p (old version of figure)
    plt.clf()
    fig, axes = plt.subplots(2, 1, sharex=True, num=1)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    ax = axes[0]
    ax.errorbar(sep, vlt, yerr=vlt_err, fmt=fmt[0], capthick=1,
                label=r'$\overline{v_\parallel}(r)$')
    ax.errorbar(sep, vlp, yerr=vlp_err, fmt=fmt[1], capthick=1,
                label=r'$\overline{v_\parallel}(r_\perp)$')
    ax.semilogx(basex=10, nonposy='clip')

    # Juszkiewicz et al. (1999, eq 6) model
#    xir = (r0/rtot)**gamma
#    xib = 3.0 / (3 - gamma) * (r0/rtot)**gamma
#    xibb = xib / (1 + xir)
#    alpha = 1.84 - 0.65*gamma
#    v_jsd = -2.0/3*H0*rtot*beta*xibb*(1 + alpha*xibb)
#    ax.plot(rtot, v_jsd, label='JSD')
    ax.set_ylabel(r'$\overline{v_{12}}\ [{\rm km/s}]$')
    ax.legend(loc=3)

    ax = axes[1]
    ax.errorbar(sep, slt_exp, yerr=slt_exp_err, fmt=fmt[0], capthick=1,
                label=r'$\sigma^e(r)$')
    ax.errorbar(sep, slp_exp, yerr=slp_exp_err, fmt=fmt[1], capthick=1,
                label=r'$\sigma^e(r_\perp)$')
    ax.errorbar(sep, slt, yerr=slt_err, fmt=fmt[2], capthick=1,
                label=r'$\sigma^G(r)$')
    ax.errorbar(sep, slp, yerr=slp_err, fmt=fmt[3], capthick=1,
                label=r'$\sigma^G(r_\perp)$')
    ax.semilogx(basex=10, nonposy='clip')
    ax.set_xlabel(r'$r\ / r_\perp\ [h^{-1} {\rm Mpc}]$')
    ax.set_ylabel(r'$\sigma_{12}\ [{\rm km/s}]$')
    ax.legend(loc=1)
    plt.draw()

    if plot_file_old:
        fig = plt.gcf()
        fig.set_size_inches(size)
        plt.savefig(plot_dir + plot_file_old, bbox_inches='tight')

    # v_mean and sigma as function of r_p for various pilim values (new ver)
    plt.clf()
    fig, axes = plt.subplots(2, 1, sharex=True, num=1)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Avearge over line-of-sight direction up to ipimax
    iplot = 0
    for ipimax in range(17, 20):
        pilim = pimin + ipimax*pistep
        if pimin < 0:
            pilim = 10**pilim
        label = r'$r_\parallel < {:5.0f}\ {{\rm Mpc}}/h$'.format(pilim)
        n_proj_arr = []
        vmean_proj_arr = []
        sigma_exp_proj_arr = []
        for ireal in range(nest):
            n_proj_arr.append(np.sum(n_red_arr[ireal][:ipimax, :], axis=0))
            vmean_proj_arr.append(np.ma.average(vmean_red_arr[ireal][:ipimax, :], axis=0,
                                                weights=n_red_arr[ireal][:ipimax, :]))
            sigma_exp_proj_arr.append(
                np.ma.average(sigma_exp_red_arr[ireal][:ipimax, :], axis=0,
                              weights=n_red_arr[ireal][:ipimax, :]))
        vlp = np.ma.average(vmean_proj_arr, axis=0, weights=n_proj_arr)
        vlp_err = np.sqrt(np.ma.average((vmean_proj_arr - vlp)**2, axis=0,
                                        weights=n_proj_arr))
        slp_exp = np.ma.average(sigma_exp_proj_arr, axis=0, weights=n_proj_arr)
        slp_exp_err = np.sqrt(np.ma.average((sigma_exp_proj_arr - slp_exp)**2,
                                            axis=0, weights=n_proj_arr))
        ax = axes[0]
        ax.errorbar(sep, vlp, yerr=vlp_err, fmt=line_list[iplot],
                    label=label, capthick=1)
        ax = axes[1]
        ax.errorbar(sep, slp_exp, yerr=slp_exp_err, fmt=line_list[iplot],
                    label=label, capthick=1)
        iplot += 1
    ax = axes[0]
    ax.semilogx(basex=10, nonposy='clip')
    ax.set_ylabel(r'$\overline{v_{12}}\ [{\rm km/s}]$')
#    ax.legend(loc=3)

    ax = axes[1]
    ax.semilogx(basex=10, nonposy='clip')
    ax.set_xlabel(r'$r_\perp\ [h^{-1} {\rm Mpc}]$')
    ax.set_ylabel(r'$\sigma_{12}\ [{\rm km/s}]$')
    ax.legend(loc=4)

    plt.draw()

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def pvd_mock_plot(infile):
    """
    Plot PVD from mocks.
    """
    data = np.loadtxt(infile)
    plt.clf()
    plt.plot(data[0], np.sqrt(data[2]))
    plt.xlabel('r [Mpc/h]')
    plt.ylabel('sigma [km/s]')
    plt.show()


def pvd_mock_cube(infile='Gonzalez_r21.txt', galfile='mock_cube.dat',
                  pvdfile='pvd_mock_cube.dat', rpmin=0.01, rpmax=100, nrp=20,
                  pimax=100, vmax=5000, nv=50, nlim=0, plot=False, qsub=False):
    """
    PVD from mock data cube with 3d v_pec info.
    Divide cube into 100 Mpc cells which also act as jackknfe regions.
    """
    lgrpmin = math.log10(rpmin)
    lgrpmax = math.log10(rpmax)

    cellsize = 100.0
    nc = 5
    ncell = nc**3
    data = np.loadtxt(infile, skiprows=1, delimiter=',')
    x, y, z, vx, vy, vz = data[:, 0], data[:, 1], data[:, 2], \
        data[:, 3], data[:, 4], data[:, 5]
    ngal = len(x)
    idx = np.floor(x/cellsize).astype(np.int32)
    idy = np.floor(y/cellsize).astype(np.int32)
    idz = np.floor(z/cellsize).astype(np.int32)
    idxcell = idz*nc*nc + idy*nc + idx
    cellHist, edges = np.histogram(idxcell, bins=np.arange(ncell**3+1)-0.5)

#    pdb.set_trace()
    fout = open(galfile, 'w')
    print(infile, file=fout)
    print(ngal, ncell, cellsize, lgrpmin, lgrpmax, nrp, pimax, vmax, nv,
          file=fout)
    icell = 0
    ncuse = 0
    ngsum = 0
    for iz in xrange(nc):
        for iy in xrange(nc):
            for ix in xrange(nc):
                ng = cellHist[icell]
                if ng > 0:
                    ncuse += 1
                    ngsum += ng
                    print(ix, iy, iz, icell, ng, file=fout)
                    idx = (idxcell == icell)
                    for i in xrange(ng):
                        print(x[idx][i], y[idx][i], z[idx][i],
                              vx[idx][i], vy[idx][i], vz[idx][i], file=fout)
                icell += 1
    fout.close()
    print('{} galaxies read, {} written out in {} out of {} cells'.format(
        ngal, ngsum, ncuse, ncell))
    if qsub:
        qsubcmd = 'qsub /research/astro/gama/loveday/Documents/Research/python/apollo_job.sh $BIN/pvd_mock_cube {} {}'
        cmd = qsubcmd.format(galfile, pvdfile)
    else:
        cmd = '$BIN/pvd_mock_cube {} {}'.format(galfile, pvdfile)
    istat = subprocess.call(cmd, shell=True)
    if istat:
        print('Error {} from pvd_mock_cube.c'.format(istat))


def pvd_mock_cube_plot(infile='pvd_mock_cube.dat', outfile='pvd_mock_cube.npz',
                       pilim=100, gamma=1.64, r0=6.16, beta=0.5,
                       vhist_rp_bins=(4, 6, 9, 13),
                       vhist_plot_file='pvd_mock_cube_hist.pdf',
                       plot_file='pvd_mock_cube.pdf', size=(5, 8)):
    """
    Plot PVD from mock data cube.
    """

    def chi2(p, fn, vbins, vhist, err):
        """reduced chi2 from exponential or Gaussian fit."""
        fit = fn(p, vbins)
        use = (fit > 5) * (err > 0)
        nuse = len(vbins[use])
        return np.sum(((vhist[use] - fit[use])/err[use])**2)/(nuse - len(p))

    def exp((mu, sigma, norm), vbins):
        """Exponential."""
        return norm * np.exp(-root2*np.abs(vbins - mu)/sigma)

    def gauss((mu, sigma, norm), vbins):
        """Gaussian."""
        return norm * np.exp(-(vbins - mu)**2/(2*sigma**2))

    def read_mom():
        """Read moments from input file and compute jacknife errors."""

        r = np.zeros((nrp, njack+1))
        vp = np.zeros((nrp, njack+1))
        sp = np.zeros((nrp, njack+1))
        vl = np.zeros((nrp, njack+1))
        sl = np.zeros((nrp, njack+1))
        vp_err = np.zeros(nrp)
        sp_err = np.zeros(nrp)
        vl_err = np.zeros(nrp)
        sl_err = np.zeros(nrp)
        for i in range(nrp):
            for ijack in range(njack+1):
                args = f.readline().split()
                r[i, ijack] = float(args[1])
                vp[i, ijack] = float(args[2])
                sp[i, ijack] = float(args[3])
                vl[i, ijack] = float(args[4])
                sl[i, ijack] = float(args[5])
            vp_err[i] = jack_err(vp[i, 1:])
            sp_err[i] = jack_err(sp[i, 1:])
            vl_err[i] = jack_err(vl[i, 1:])
            sl_err[i] = jack_err(sl[i, 1:])
        return r[:, 0], vp[:, 0], vp_err, sp[:, 0], sp_err, \
            vl[:, 0], vl_err, sl[:, 0], sl_err

    fmt = ['-', '--', '-.', ':']
    f = open(infile, 'r')
    f.readline()
    f.readline()
    args = f.readline().split()
    njack = int(args[1])
    nrp = int(args[2])
    lgrpmin = float(args[3])
    lgrpmax = float(args[4])
    lgrpstep = (lgrpmax - lgrpmin)/nrp
    pimax = float(args[5])
    nv = int(args[6])
    vmax = float(args[7])

    sep, vpt, vpt_err, spt, spt_err, vlt, vlt_err, slt, slt_err = read_mom()
    rproj, vpp, vpp_err, spp, spp_err, vlp, vlp_err, slp, slp_err = read_mom()

    vbins = np.zeros((nrp, nv, njack+1))
    vhist = np.zeros((nrp, nv, njack+1))
    vh_err = np.zeros((nrp, nv))
    for i in range(nrp):
        for iv in range(nv):
            args = f.readline().split()
            for ijack in range(njack+1):
                vbins[i, iv, ijack] = float(args[2*ijack])
                vhist[i, iv, ijack] = float(args[2*ijack + 1])
            vh_err[i, iv] = jack_err(vhist[i, iv, 1:])
    f.close()

    np.savez(outfile, sep=sep, vpt=vpt, vpt_err=vpt_err,
             spt=spt, spt_err=spt_err,
             vpp=vpp, vpp_err=vpp_err, spp=spp, spp_err=spp_err,
             vlt=vlt, vlt_err=vlt_err, slt=slt, slt_err=slt_err,
             vlp=vlp, vlp_err=vlp_err, slp=slp, slp_err=slp_err)

    # velocity histograms
    nrow = len(vhist_rp_bins)
    plt.clf()
    fig, axes = plt.subplots(nrow, 1, sharex=True, num=1)
    axes = axes.ravel()
    fig.subplots_adjust(left=0.18, bottom=0.05, hspace=0.0, wspace=0.0)
    fig.text(0.55, 0.0, r'$v_\parallel\ ({{\rm km/s}})$',
             ha='center', va='center')
    fig.text(0.06, 0.47, 'Frequency', ha='center', va='center',
             rotation='vertical')
    irow = 0
    for i in vhist_rp_bins:
        ax = axes[irow]
        vb = vbins[i, :, 0]
        vh = vhist[i, :, 0]
        use = vh > 0
        vb = vb[use]
        vh = vh[use]
        vhe = vh_err[i, use]
        ax.step(vb, vh, where='mid')
#        ax.errorbar(vb, vh, vhe, fmt=None, ecolor='b', capthick=1)

        p0 = (0.0, 500.0, np.amax(vh))
        out = scipy.optimize.fmin(chi2, p0, args=(exp, vb, vh, vhe),
                                  maxfun=10000, maxiter=10000, full_output=1,
                                  disp=0)
        pe = out[0]
        chi_e = out[1]
        efit = exp(pe, vb)
        ax.plot(vb, efit)
        out = scipy.optimize.fmin(chi2, p0, args=(gauss, vb, vh, vhe),
                                  maxfun=10000, maxiter=10000, full_output=1,
                                  disp=0)
        pg = out[0]
        chi_g = out[1]
        gfit = gauss(pg, vb)
        ax.plot(vb, gfit, '--')
        ax.set_ylim(1, 2*np.amax(vh))
        ax.semilogy(basey=10, nonposy='clip')
#        ax.set_ylabel('Frequency')
        r = 10**(lgrpmin + (i+0.5)*lgrpstep)
        label = r'$r_\bot = {:5.1f}\ h^{{-1}} {{\rm Mpc}}$'.format(r)
        ax.text(0.05, 0.85, label, transform=ax.transAxes)
        ax.text(0.05, 0.74, r'$\mu_e = {:5.0f}\ {{\rm km/s}}$'.format(pe[0]),
                transform=ax.transAxes)
        ax.text(0.05, 0.63, r'$\sigma_e = {:5.0f}\ {{\rm km/s}}$'.format(pe[1]),
                transform=ax.transAxes)
        ax.text(0.05, 0.52, r'$\chi^2_e = {:5.0f}$'.format(chi_e),
                transform=ax.transAxes)
        ax.text(0.95, 0.74, r'$\mu_G = {:5.0f}\ {{\rm km/s}}$'.format(pg[0]),
                ha='right', transform=ax.transAxes)
        ax.text(0.95, 0.63, r'$\sigma_G = {:5.0f}\ {{\rm km/s}}$'.format(pg[1]),
                ha='right', transform=ax.transAxes)
        ax.text(0.95, 0.52, r'$\chi^2_G = {:5.0f}$'.format(chi_g),
                ha='right', transform=ax.transAxes)
        irow += 1
#    ax.set_xlabel(r'$v_\parallel\ ({{\rm km/s}})$')
    plt.draw()

    if vhist_plot_file:
        fig = plt.gcf()
        fig.set_size_inches((4, 8))
        plt.savefig(plot_dir + vhist_plot_file, bbox_inches='tight')

#    plt.clf()
#    fig, axes = plt.subplots(2, 1, sharex=True, num=1)
#    fig.subplots_adjust(hspace=0.1, wspace=0.1)
#
#    ax = axes[0]
#    im = ax.imshow(np.flipud(vmean_pair_red), aspect='auto',
#                   extent=(10**lgrpmin, 10**lgrpmax, pimin, pimax))
#    ax.semilogx(basex=10, nonposy='clip')
#    plt.colorbar(im, ax=ax)
#    ax.set_xlabel(r'$r_\perp / {{\rm Mpc}}$')
#    ax.set_ylabel(r'$r_\parallel / {{\rm Mpc}}$')
#
#    ax = axes[1]
#    im = ax.imshow(np.flipud(sigma_pair_red), aspect='auto',
#                   extent=(10**lgrpmin, 10**lgrpmax, pimin, pimax))
#    ax.semilogx(basex=10, nonposy='clip')
#    plt.colorbar(im, ax=ax)
#    ax.set_xlabel(r'$r_\perp / {{\rm Mpc}}$')
#    ax.set_ylabel(r'$r_\parallel / {{\rm Mpc}}$')
#    plt.show()

    plt.clf()
    fig, axes = plt.subplots(2, 1, sharex=True, num=1)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    ax = axes[0]
    ax.errorbar(sep, vpt, yerr=vpt_err, fmt=fmt[0], capthick=1,
                label=r'$v_r(r)$')
    ax.errorbar(rproj, vpp, yerr=vpp_err, fmt=fmt[1], capthick=1,
                label=r'$v_r(r_\perp)$')
    ax.errorbar(sep, vlt, yerr=vlt_err, fmt=fmt[2], capthick=1,
                label=r'$v_\parallel(r)$')
    ax.errorbar(rproj, vlp, yerr=vlp_err, fmt=fmt[3], capthick=1,
                label=r'$v_\parallel(r_\perp)$')
    ax.semilogx(basex=10, nonposy='clip')

    # Juszkiewicz et al. (1999, eq 6) model
    xir = (r0/sep)**gamma
    xib = 3.0 / (3 - gamma) * (r0/sep)**gamma
    xibb = xib / (1 + xir)
    alpha = 1.84 - 0.65*gamma
    v_jsd = -2.0/3*H0*sep*beta*xibb*(1 + alpha*xibb)
    ax.plot(sep, v_jsd, label='JSD')
    ax.set_ylabel(r'$\overline{v_{12}}\ [{\rm km/s}]$')
    ax.legend(loc=3)

    ax = axes[1]
    ax.errorbar(sep, spt, yerr=spt_err, fmt=fmt[0], capthick=1)
    ax.errorbar(rproj, spp, yerr=spp_err, fmt=fmt[1], capthick=1)
    ax.errorbar(sep, slt, yerr=slt_err, fmt=fmt[2], capthick=1)
    ax.errorbar(rproj, slp, yerr=slp_err, fmt=fmt[3], capthick=1)
    ax.semilogx(basex=10, nonposy='clip')
    ax.set_xlabel(r'$r\ / r_\perp\ [h^{-1} {\rm Mpc}]$')
    ax.set_ylabel(r'$\sigma_{12}\ [{\rm km/s}]$')
    plt.draw()

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


# --------------------------------
# Averaging over mocks
# --------------------------------

def counts_av_mocks(root='vol'):
    """
    Sum mock pair counts over GAMA regions, then average over realisations.
    Individual realisations are stored as jackknife estimates in output file.
    """

    intemp = '{}_{}_{}_{:02d}.dat'
    rrtemp = 'rr_{}_{}.dat'
    outtemp = '{}_{}.dat'
    for pt in ('gg', 'gr'):
        pc_list = []
        for ireal in range(26):
            pc_sum_list = []
            for ifield in range(1, 4):
                pc = PairCounts(intemp.format(pt, root, ifield, ireal))
                pc_sum_list.append(pc)
            pcsum = PairCounts()
            pcsum.sum(pc_sum_list)
            pc_list.append(pcsum)
        pcav = PairCounts()
        pcav.average(pc_list)
        pcav.write(outtemp.format(pt, root))

    pc_sum_list = []
    for ifield in range(1, 4):
        pc = PairCounts(rrtemp.format('vol', ifield))
        pc_sum_list.append(pc)
    pcsum = PairCounts()
    pcsum.sum(pc_sum_list)
    pcsum.write(outtemp.format('rr', root))


def counts_av_samples(param='lum_c', intemp='{}_{}_{}_{}_{}_{:02d}.dat',
                      outtemp='{}_{}_{}_{}.dat'):
    """
    Sum mock subsample pair counts over GAMA regions,
    then average over realisations.
    """
    if 'lum' in param:
        Mlimits = def_mag_limits
    else:
        Mlimits = def_mass_limits
    for i in range(len(Mlimits)-1):
        Mlo = Mlimits[i]
        Mhi = Mlimits[i+1]
        for pt in ('gg', 'gr'):
            pc_list = []
            for ireal in range(26):
                pc_sum_list = []
                for ifield in range(1, 4):
                    pc = PairCounts(intemp.format(pt, param, Mlo, Mhi,
                                                  ifield, ireal))
                    pc_sum_list.append(pc)
                pcsum = PairCounts()
                pcsum.sum(pc_sum_list)
                pc_list.append(pcsum)
            pcav = PairCounts()
            pcav.average(pc_list)
            pcav.write(outtemp.format(pt, param, Mlo, Mhi))

        pc_sum_list = []
        for ifield in range(1, 4):
            pc = PairCounts('rr_{}_{}_{}_{}.dat'.format(param, Mlo, Mhi, ifield))
            pc_sum_list.append(pc)
        pcsum = PairCounts()
        pcsum.sum(pc_sum_list)
        pcsum.write('rr_{}_{}_{}.dat'.format(param, Mlo, Mhi))


def w_av(fileList, outfile):
    """Average over separate w(theta) determinations"""

    globList = []
    for file in fileList:
        globList += glob.glob(file)
    fileList = globList
    print(fileList)

    nfile = 0
    galfile = []
    ngal = 0
    nran = 0
    for infile in fileList:
        f = open(infile, 'r')
        galfile.append(f.readline())
        args = f.readline().split()
        ngal += int(args[0])
        nran += int(args[1])
        nlog = int(args[2])
        lgrmin = float(args[3])
        lgrmax = float(args[4])
        
        if nfile:
            if nlog != nlog0:
                print('Inconcistent binning, aborting')
                return
        else:
            nlog0 = nlog
            s = np.zeros(nlog)
            wsum = np.zeros(nlog)
            w2sum = np.zeros(nlog)
            gg = np.zeros(nlog)
            rr = np.zeros(nlog)
            gr = np.zeros(nlog)

        for i in range(nlog):
            data = f.readline().split()
            s[i] = float(data[0])
            wsum[i] += float(data[1])
            w2sum[i] += float(data[1])**2
            gg[i] += float(data[4])
            rr[i] += float(data[5])
            gr[i] += float(data[6])

        f.close()
        nfile += 1
        
    wmean = wsum/nfile
    werr = np.sqrt(w2sum/nfile - wmean*wmean)
    
    fout = open(outfile, 'w')
    print(galfile, file=fout)
    print(ngal, nran, nlog, lgrmin, lgrmax, file=fout)
    for i in range(nlog):
        print(s[i], wmean[i], wmean[i], werr[i], gg[i], rr[i], gr[i], file=fout)
    fout.close()


def counts_av(fileList, outfile):
    """Average over separate pair count files.
    Output file has individual estimates in place of jackknife estimates
    so that covariances can be calculated."""

    globList = []
    for file in fileList:
        globList += glob.glob(file)
    fileList = globList
    print(fileList)

    pc_list = []
    for infile in fileList:
        pc = PairCounts(infile)
        pc_list.append(pc)
    pcav = PairCounts()
    pcav.average(pc_list)
    pcav.write(outfile)


def xi_av(fileList, outfile):
    """Average over separate xi(s) and xi(sigma,pi) determinations.
    Output file has individual estimates in place of jackknife estimates
    so that covariances can be calculated.
    New version reads and writes files using class Xi.
    *** This routine is replaced by counts_av and is now obsolete*** """

    globList = []
    for file in fileList:
        globList += glob.glob(file)
    fileList = globList
    print(fileList)

    xi_list = []
    for infile in fileList:
        xi = Xi()
        xi.read(infile)
        xi_list.append(xi)
    xiav = Xi()
    xiav.average(xi_list)
    xiav.write(outfile)


def xi_av_old(fileList, outfile, s_rebin=1, pi_rebin=1, rp_rebin=1):
    """Average over separate xi(s) and xi(sigma,pi) determinations.
    Output file has individual estimates in place of jackknife estimates
    so that covariances can be calculated.
    NB: works only with old xi.c output format.
    Delete when rebinning moved to class PairCounts"""

    def rebin(npi, pi_rebin, nrp, rp_rebin, s1, s2, gg, gr, rr):
        """Rebin 2d correlation function to npi/pi_rebin * nrp/rp_rebin."""
        npibin = npi//pi_rebin
        nrpbin = nrp//rp_rebin
        s1mean = np.zeros((npibin, nrpbin))
        s2mean = np.zeros((npibin, nrpbin))
        ggsum = np.zeros((npibin, nrpbin))
        grsum = np.zeros((npibin, nrpbin))
        rrsum = np.zeros((npibin, nrpbin))
        xi = np.zeros((npibin, nrpbin))
#        pdb.set_trace()
        for i in range(0, nrp, rp_rebin):
            ib = i//rp_rebin
            for j in range(0, npi, pi_rebin):
                jb = j//pi_rebin
                for ii in range(i, i + rp_rebin):
                    for jj in range(j, j + pi_rebin):
                        s1mean[jb, ib] += gg[jj, ii] * s1[jj, ii]
                        s2mean[jb, ib] += gg[jj, ii] * s2[jj, ii]
                        ggsum[jb, ib] += gg[jj, ii]
                        grsum[jb, ib] += gr[jj, ii]
                        rrsum[jb, ib] += rr[jj, ii]
                if ggsum[jb, ib] > 0:
                    s1mean[jb, ib] /= ggsum[jb, ib]
                    s2mean[jb, ib] /= ggsum[jb, ib]
                if rrsum[jb, ib] > 0:
                    ggn = 2*ggsum[jb, ib]/ngal/(ngal - 1)
                    grn = grsum[jb, ib]/ngal/nran
                    rrn = 2*rrsum[jb, ib]/nran/(nran - 1)
                    xi[jb, ib] = (ggn - 2*grn + rrn)/rrn
        return s1mean, s2mean, ggsum, grsum, rrsum, xi

    globList = []
    for file in fileList:
        globList += glob.glob(file)
    fileList = globList
    print(fileList)

    nfile = 0
    ngal = 0
    nran = 0
    njack = len(fileList)
    for infile in fileList:
        f = open(infile, 'r')
        xiprog, ver, galFile = f.readline().split()
        info = eval(f.readline())
        args = f.readline().split()
        ngal += int(args[0])
        nran += int(args[1])
        if nfile == 0:
            ns = int(args[3])
            smin = float(args[4])
            smax = float(args[5])
            nrp = int(args[6])
            rpmin = float(args[7])
            rpmax = float(args[8])
            nrp_lin = int(args[9])
            rpmin_lin = float(args[10])
            rpmax_lin = float(args[11])
            npi = int(args[12])
            pimin = float(args[13])
            pimax = float(args[14])

            s = np.zeros(ns)
            xissum = np.zeros(ns)
            xis2sum = np.zeros(ns)
            xisjack = np.zeros((njack, ns))
            ggs = np.zeros(ns)
            rrs = np.zeros(ns)
            grs = np.zeros(ns)

            npibin = npi//pi_rebin
            nrpbin = nrp//rp_rebin
            nrpbin_lin = nrp_lin//rp_rebin
            s1 = np.zeros((npibin, nrpbin))
            s2 = np.zeros((npibin, nrpbin))
            xisum = np.zeros((npibin, nrpbin))
            xi2sum = np.zeros((npibin, nrpbin))
            xijack = np.zeros((njack, npibin, nrpbin))
            ggbsum = np.zeros((npibin, nrpbin))
            rrbsum = np.zeros((npibin, nrpbin))
            grbsum = np.zeros((npibin, nrpbin))
            xisum_lin = np.zeros((npibin, nrpbin_lin))
            xi2sum_lin = np.zeros((npibin, nrpbin_lin))
            xijack_lin = np.zeros((njack, npibin, nrpbin_lin))
            ggbsum_lin = np.zeros((npibin, nrpbin_lin))
            rrbsum_lin = np.zeros((npibin, nrpbin_lin))
            grbsum_lin = np.zeros((npibin, nrpbin_lin))

        for i in range(ns):
            data = f.readline().split()
            s[i] = float(data[0])
            ggs[i] += float(data[1])
            grs[i] += float(data[2])
            rrs[i] += float(data[3])
            xissum[i] += float(data[4])
            xis2sum[i] += float(data[4])**2
            xisjack[nfile, i] += float(data[4])

        s1 = np.zeros((npi, nrp))
        s2 = np.zeros((npi, nrp))
        gg = np.zeros((npi, nrp))
        gr = np.zeros((npi, nrp))
        rr = np.zeros((npi, nrp))
        for i in range(nrp):
            for j in range(npi):
                data = f.readline().split()
                s1[j, i] = float(data[0])
                s2[j, i] = float(data[1])
                gg[j, i] = float(data[2])
                gr[j, i] = float(data[3])
                rr[j, i] = float(data[4])
        s1mean, s2mean, ggsum, grsum, rrsum, xi = rebin(
            npi, pi_rebin, nrp, rp_rebin, s1, s2, gg, gr, rr)
        ggbsum += ggsum
        grbsum += grsum
        rrbsum += rrsum
        xisum += xi
        xi2sum += xi*xi
        xijack[nfile, :, :] = xi

        s1 = np.zeros((npi, nrp_lin))
        s2 = np.zeros((npi, nrp_lin))
        gg = np.zeros((npi, nrp_lin))
        gr = np.zeros((npi, nrp_lin))
        rr = np.zeros((npi, nrp_lin))
        for i in range(nrp_lin):
            for j in range(npi):
                data = f.readline().split()
                s1[j, i] = float(data[0])
                s2[j, i] = float(data[1])
                gg[j, i] = float(data[2])
                gr[j, i] = float(data[3])
                rr[j, i] = float(data[4])
        s1_linmean, s2_linmean, ggsum, grsum, rrsum, xi = rebin(
            npi, pi_rebin, nrp_lin, rp_rebin, s1, s2, gg, gr, rr)
        ggbsum_lin += ggsum
        grbsum_lin += grsum
        rrbsum_lin += rrsum
        xisum_lin += xi
        xi2sum_lin += xi*xi
        xijack_lin[nfile, :, :] = xi

        f.close()
        nfile += 1
#    pdb.set_trace()
    xismean = xissum/nfile
#    xierr = np.sqrt(xi2sum/nfile - ximean*ximean)
    ximean = xisum/nfile
#    xierr2 = np.sqrt(xi2sum2/nfile - ximean2*ximean2)
    ximean_lin = xi2sum_lin/nfile

    fout = open(outfile, 'w')
    info['mock'] = True
    info['err_type'] = 'mock'
    print(xiprog, ver, galFile, file=fout)
    print(info, file=fout)
    print(ngal, nran, njack, ns, smin, smax, nrpbin, rpmin, rpmax,
          nrpbin_lin, rpmin_lin, rpmax_lin, npibin, pimin, pimax, file=fout)
    for i in range(ns):
        xij = ' '.join(str(xi) for xi in xisjack[:, i].tolist())
        print(s[i], ggs[i], grs[i], rrs[i], xismean[i], xij, file=fout)
    for i in range(nrpbin):
        for j in range(npibin):
            xij = ' '.join(str(xi) for xi in xijack[:, j, i].tolist())
            print(s1mean[j, i], s2mean[j, i], ggbsum[j, i],
                  grbsum[j, i], rrbsum[j, i], ximean[j, i], xij, file=fout)
    for i in range(nrpbin_lin):
        for j in range(npibin):
            xij = ' '.join(str(xi) for xi in xijack_lin[:, j, i].tolist())
            print(s1_linmean[j, i], s2_linmean[j, i],
                  ggbsum_lin[j, i], grbsum_lin[j, i], rrbsum_lin[j, i],
                  ximean_lin[j, i], xij, file=fout)
    fout.close()


# -----------------
# Plotting routines
# -----------------

def zhist(galtemp='gal_{}_{}_{}.dat', rantemp='ran_{}_{}_{}.dat',
          param='lum_c', label=r'$M = [{:5.2f}, {:5.2f}]$', Mlimits=None,
          plot_file='zhist.pdf', landscape=False):
    """Plot redshift histograms for galaxy & random input files to xi.c."""

    def read_file(infile, nskip=3):
        # Read input file into array and return array of distances
        data = np.loadtxt(infile, skiprows=nskip)
        dist = np.sqrt(np.sum(data[:, 0:3]**2, axis=1))
        return dist

    if Mlimits is None:
        if 'lum' in param:
            Mlimits = def_mag_limits
        else:
            Mlimits = def_mass_limits

    if landscape:
        plot_size = (8, 5)
        sa_left = 0.12
        sa_bot = 0.1
    else:
        plot_size = (5, 8)
        sa_left = 0.18
        sa_bot = 0.08

    npanel = len(Mlimits) - 1
    plt.clf()
    nrow, ncol = util.two_factors(npanel, landscape)
    fig, axes = plt.subplots(nrow, ncol, sharex=False, sharey=False, num=1)
    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0.15, wspace=0.15)
    fig.text(0.55, 0.0, 'Distance (Mpc/h)', ha='center', va='center')
    fig.text(0.06, 0.5, 'Number', ha='center', va='center',
             rotation='vertical')
    for i in range(npanel):
        dg = read_file(galtemp.format(param, Mlimits[i], Mlimits[i+1]))
        rg = read_file(rantemp.format(param, Mlimits[i], Mlimits[i+1]))
        try:
            ax = axes.flat[i]
        except:
            ax = axes
        ax.hist((dg, rg), bins=20, normed=True, histtype='step')
        ax.text(0.1, 0.8, label.format(Mlimits[i], Mlimits[i+1]),
                transform=ax.transAxes)
    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def zhist_one(galfile='gal_0_1.dat', ranfile='ran_1.dat', nbin=50):
    """Plot redshift histograms for galaxy & random input files to xi.c."""

    def read_file(infile, nskip=3):
        # Read input file into array and return array of distances
        data = np.loadtxt(infile, skiprows=nskip)
        dist = np.sqrt(np.sum(data[:,0:3]**2, axis=1))
        return dist

    dg = read_file(galfile)
    rg = read_file(ranfile)
    plt.clf()
    plt.hist((dg, rg), bins=nbin, normed=True, histtype='step')
    plt.xlabel('Distance [Mpc/h]')
    plt.ylabel('Frequency')
    plt.show()


def zhist_mocks(galfile='gal_0_1.dat', ranfile='ran_1.dat'):
    """Plot redshift histograms for galaxy & random mocks."""

    def read_file(infile, nskip=3):
        # Read input file into array and return array of distances
        f = open(infile, 'r')
        info = eval(f.readline())
        rcut = info['rcut']
        f.close()
        data = np.loadtxt(infile, skiprows=nskip)
        dist = np.sqrt(np.sum(data[:,0:3]**2, axis=1))
        return rcut, dist

    dgh_array = np.zeros((20, 26))
    ngal = 0
    for ireal in xrange(26):
        dghsum = np.zeros(20)
        for ireg in xrange(1, 4):
            galfile = 'gal_vol_{:1d}_{:02d}.dat'.format(ireg, ireal)
            rcut, dg = read_file(galfile)
            dgh, edges = np.histogram(dg, bins=20, range=(0, rcut))
            dghsum += dgh
            ngal += len(dg)
        dgh_array[:, ireal] = dghsum

    dgmean = np.mean(dgh_array, axis=1)
    dgstd = np.std(dgh_array, axis=1)
    dgpc = np.percentile(dgh_array, (5, 95), axis=1)
    drhsum = np.zeros(20)
    nran = 0
    for ireg in xrange(1, 4):
        ranfile = 'ran_vol_{:1d}.dat'.format(ireg)
        rcut, dg = read_file(ranfile)
        drh, edges = np.histogram(dg, bins=20, range=(0, rcut))
        drhsum += drh
        nran += len(dg)

    norm = float(ngal)/nran/26
    r = (edges[:-1] + edges[1:]) / 2
    plt.clf()
#    plt.errorbar(r, dgmean, dgstd)
#    plt.errorbar(r, dgmean, dgpc)
#    plt.plot(r, drhsum*norm)
    plt.plot(r, dgh_array)
    plt.xlabel('Distance [Mpc/h]')
    plt.ylabel('Frequency')
    plt.show()


def sky_dist(galfile='gal_0_1.dat', ranfile=None):
    """Plot sky distribution for galaxy & random input files to xi.c."""

    def read_file(infile, nskip=3):
        # Read input file into array and return ra, dec arrays
        data = np.loadtxt(infile, skiprows=nskip)
        dist = np.sqrt(np.sum(data[:, 0:3]**2, axis=1))
        dec = np.rad2deg(np.arcsin(data[:, 2]/dist))
        ra = np.rad2deg(np.arctan2(data[:, 1], data[:, 0]))
        neg = ra < 0
        ra[neg] += 360
        return ra, dec

    rag, decg = read_file(galfile)
    plt.clf()
#    plt.subplot(211)
    if ranfile:
        rar, decr = read_file(ranfile)
        plt.scatter(rar, decr, 0.01, 'b', edgecolors='face')
    plt.scatter(rag, decg, 0.1, 'g', edgecolors='face')
#    plt.subplot(212)
    plt.show()


def plot_3d(galfile, ranfile, nskip=3,
            outgal='gal_coords.dat', outran='ran_coords.dat'):
    """
    Plot 3d distribution of galaxy & random input files to xi.c.
    Only works for single cell, skipping nskip lines at top.
    NB rotation is very slow, so also writes out simple file of
    Cartesian coords to load into topcat.
    """

    def read_file(infile, outfile, nskip):
        # Read input file into array and output file of cartesian coords
        data = np.loadtxt(infile, skiprows=nskip)
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]

        f = open(outfile, 'w')
        for i in xrange(len(x)):
            print(x[i], y[i], z[i], file=f)
        f.close()

        return x, y, z
    
    xg, yg, zg = read_file(galfile, outgal, nskip)
    xr, yr, zr = read_file(ranfile, outran, nskip)

#    mlab.points3d(xr, yr, zr)
#    mlab.points3d(xg, yg, zg, color='r')
    
##     plt.clf()
##     ax = plt.subplot(111, projection='3d')
##     ax.scatter(xr, yr, zr, s=0.01)
##     ax.scatter(xg, yg, zg, s=0.1, c='r')
##     plt.draw()
    

def w_plot(fileList, selList=None, fitRange=None):
    """Plot angular correlation function from wcorr.c"""

    globList = []
    for file in fileList:
        globList += glob.glob(file)
    fileList = globList
    print(fileList)
    globList = []
    for file in selList:
        globList += glob.glob(file)
    selList = globList
    print(selList)
    
    plt.clf()
    ifile = 0
    for infile in fileList:
        f = open(infile, 'r')
        galFile = f.readline()
        args = f.readline().split()
        nlog = int(args[2])

        s = np.zeros(nlog)
        xi = np.zeros(nlog)
        xierr = np.zeros(nlog)
        for i in range(nlog):
            data = f.readline().split()
            s[i] = float(data[0])
            xi[i] = float(data[1])
            xierr[i] = float(data[3])

        f.close()
        plt.plot(s, xi, 'o')
        plt.errorbar(s, xi, xierr, capthick=1)

        if fitRange:
##             idx = (fitRange[0] < s)*(s < fitRange[1] < s)
            idx = (fitRange[0] < s < fitRange[1])
            x = np.log10(s[idx])
            y = np.log10(xi[idx])
            w = 1.0/xierr[idx]**2
            p = np.polyfit(x, y, 1, w=w)
            gamma = 1 - p[0]
            A = 10**p[1]
            yfit = (10**(p[1] + p[0]*math.log10(fitRange[0])),
                    10**(p[1] + p[0]*math.log10(fitRange[1])))
            plt.plot(fitRange, yfit)

            if selList:
                r0 = limber(gamma, A, selList[ifile])
                print('gamma = ', gamma, ' A = ', A, ', ro = ', r0)
            else:
                print('gamma = ', gamma, ' A = ', A)
        ifile += 1

    plt.loglog(basex=10, basey=10, nonposy='clip')
    plt.xlabel(r'$\theta$/degrees')
    plt.ylabel(r'$w(\theta$)')
    plt.draw()


def xi_plot_cf(infile='xi_vol.dat', key='w_p', pi_max=40.0,
               binning=1, fit_range=(0.01, 5), logfit=0,
               xlimits=(0.01, 100), ylimits=(0.5, 5000), ic_rmax=0, neig=0,
               plot_jack=0, plot_file=None, size=(8, 4)):
    """Plot covariance matrix and power-law fit for single sample."""

    xi = xi_req(infile, key, binning=binning, pi_lim=pi_max)
    plt.clf()
    cov_ax = plt.subplot(221)
    covn_ax = plt.subplot(223)
    ax = plt.subplot(122)
    xi.plot(ax)
    if plot_jack:
        for ijack in range(xi.njack):
            ax.plot(xi.sep, xi.est[:, ijack+1])
    if fit_range:
        gamma, gamma_err, r0, r0_err, ic, _, _ = xi.fit(
            fit_range, logfit=logfit, ax=ax, cov_ax=cov_ax, covn_ax=covn_ax,
            ic_rmax=ic_rmax, neig=neig)
    ax.loglog(basex=10, basey=10, nonposy='clip')
#    ax.semilogx(basex=10, nonposy='clip')
    ax.set_xlabel(xlabel[key])
    ax.set_ylabel(ylabel[key])
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(size)
    plt.show()
#    print(xi.est, xi.cov.sig)


def xi_plot1(files=('xi_vol.dat',), key='w_p', pi_max=40.0,
             binning=1, fit_range=(0.1, 10), pl_div=None,
             xlimits=(0.01, 100), ylimits=(0.5, 5000), ic_rmax=0, neig=0,
             jack=0, outfile=None, plot_file=None, plot_size=(4, 4)):
    """Plot clustering results for given file list in single panel."""
    panels = []
    comps = []
    label = files
    panels.append({'files': files, 'comps': comps, 'label': label})
    xi_plot(key, panels, binning=binning, jack=jack, pi_max=pi_max,
            fit_range=fit_range, xlimits=xlimits, ylimits=ylimits,
            ic_rmax=ic_rmax, neig=neig, pl_div=pl_div, landscape=True,
            outfile=outfile, plot_file=plot_file, plot_size=plot_size)


def xi_plot_ev(clr='c', M=-21, filecomp='xi_Vt_{}_{}.dat',
               filetemp='xi_Vt_{}_{}_z*.dat', key='w_p',
               ic_rmax=0, neig=0, comp=None, plot_file='wp_ev_{}.pdf'):
    """Plot clustering results for redshift bins."""

    zfiles = glob.glob(filetemp.format(clr, M))
    panels = []
    for zfile in zfiles:
        files = (filecomp.format(clr, M), zfile)
        zlo = zfile[14:18]
        zhi = zfile[19:23]
        # label = zfile
        label = 'M < {}; {} < z < {}'.format(M, zlo, zhi)
        comps = []
        panels.append({'files': files, 'comps': comps, 'label': label})
    xi_plot(key, panels, xlimits=(0.1, 90), ic_rmax=ic_rmax, neig=neig,
            plot_file=plot_file.format(M))


def xi_plot_samples(intemp='xi_{}_{}_{}.dat', param='lum_c', key='w_p',
                    binning=1, pi_max=40, mock=True, comp='Li, Zehavi',
                    xlimits=(0.01, 90), ylimits=(0.1, 1e4), fit_range=None,
                    ic_rmax=0, neig=0, landscape=False):
    """Plot clsutering results for lum/mass bins.
    This replaces xi_plot_lum, xi_plot_mass etc."""

    if landscape:
        plot_size = (8, 5)
        sa_left = 0.12
        sa_bot = 0.1
    else:
        plot_size = (5, 8)
        sa_left = 0.18
        sa_bot = 0.08

    outfile = '{}_{}.dat'.format(key, param)
    plot_file = '{}_{}.pdf'.format(key, param)
    if 'lum' in param:
        Mlimits = def_mag_limits
        labeltemp = r'$M_r = [{}, {}]$'
    else:
        Mlimits = def_mass_limits
        labeltemp = r'$M_* = [{}, {}]$'
    panels = []
    for i in range(len(Mlimits)-1):
        files = []
        files.append(intemp.format(param, Mlimits[i], Mlimits[i+1]))
        if mock:
            files.append(gama_data + 'mocks/v1/' +
                         intemp.format(param, Mlimits[i], Mlimits[i+1]))
        comps = []
        if 'Farrow' in comp:
            if key == 'w_p':
                comps.append(Farrow_w_p(param, Mlimits[i], Mlimits[i+1]))
        if 'Li' in comp:
            if 'lum' in param and key == 'w_p':
                comps.append(Li_w_p('wrp-L{:5.1f}{:5.1f}.dat',
                                    Mlimits[i], Mlimits[i+1]))
            if 'mass' in param and key == 'w_p':
                comps.append(Li_w_p('wrp-M-{:04.1f}-{:04.1f}.dat',
                                    Mlimits[i], Mlimits[i+1]))
        if 'Zehavi' in comp:
            if 'lum' in param and key == 'w_p':
                comps.append(Zehavi_w_p(Mlimits[i], Mlimits[i+1]))
        label = labeltemp.format(Mlimits[i], Mlimits[i+1])
        panels.append({'files': files, 'comps': comps, 'label': label})
#        print(comps)
    xi_plot(key, panels, binning=binning, pi_max=pi_max,
            xlimits=xlimits, ylimits=ylimits,
            fit_range=fit_range, ic_rmax=ic_rmax, neig=neig, outfile=outfile,
            plot_file=plot_file, plot_size=plot_size)


def farrow_comp_plot(intemp='xi_f_M{}_{}_z{}_{}.dat', key='w_p',
                    binning=1, pi_max=40, pl_div=(5.33, 1.81),
                    xlimits=(0.01, 90), ylimits=(0.1, 8), fit_range=None,
                    ic_rmax=0, neig=0, landscape=False, plot_size=(4, 8)):
    """Plot correlation function for subsamples that match 
    Farrow+2015 selection (largest of first 5 samples, Table 2)."""

    if landscape:
        sa_left = 0.12
        sa_bot = 0.1
    else:
        sa_left = 0.18
        sa_bot = 0.08

    xlab = xlabel[key]
    ylab = ylabel[key]
    if pl_div:
        ylab = r'$w_p(r_p) / w_{\rm ref}(r_p)$'
    plot_file = 'farrow_comp.pdf'
    ftemp = (os.environ['HOME'] + '/Documents/Research/corrdata/Farrow2015/' +
             'farrow15-{:9.5f}-mag-{:9.5f}-{}-z-{}-wprp.dat')
    Mlims = (-22, -21, -20, -19, -18, -17)
    zlims = (0.5, 0.35, 0.24, 0.14, 0.02)
    nsamp = len(Mlims) - 1
    nrow, ncol = util.two_factors(nsamp)
    plt.clf()
    nrow, ncol = util.two_factors(nsamp, landscape)
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0, wspace=0)
    fig.text(0.5, 0.02, xlab, ha='center', va='center')
    fig.text(0.02, 0.5, ylab, ha='center', va='center', rotation='vertical')
    labeltemp = r'$M_r = [{}, {}], z = [{}, {}]$'
    if pl_div:
        gamma = pl_div[1]
        A = (pl_div[0]**gamma * scipy.special.gamma(0.5) *
             scipy.special.gamma(0.5*(gamma-1)) /
             scipy.special.gamma(0.5*gamma))
    for i in range(len(Mlims)-1):
        j = min(i, nsamp-2)  # Faintest (last) two samples share same z limits
        ax = axes.flat[i]
        infile = intemp.format(Mlims[i], Mlims[i+1], zlims[j+1], zlims[j])
        xi = xi_req(infile, key, binning=binning, pi_lim=pi_max)
        if pl_div:
            pl_fit = A * xi.sep**(1-gamma)
        else:
            pl_fit = 1
        ax.errorbar(xi.sep, (xi.est[:, 0] + xi.ic)/pl_fit,
                    xi.cov.sig/pl_fit, fmt='o')
        ffile = ftemp.format(Mlims[i], Mlims[i+1], zlims[j+1], zlims[j])
        data = np.loadtxt(ffile)
        r_p = data[:, 0]
        w_p = data[:, 1]
        w_p_err = data[:, 2]
        if pl_div:
            pl_fit = A * r_p**(1-gamma)
        else:
            pl_fit = 1
        ax.errorbar(r_p, w_p/pl_fit, w_p_err/pl_fit, fmt='s')
        ax.loglog(basex=10, basey=10, nonposy='clip')
        ax.set_ylim(ylimits)
        label = labeltemp.format(Mlims[i], Mlims[i+1], zlims[j+1], zlims[j])
        ax.text(0.2, 0.8, label, transform=ax.transAxes)
    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def xi_plot(key, panels, binning=1, jack=0, pi_max=40.0,
            fit_range=(0.01, 5), bias_par='M_mean', bias_scale=3.0,
            xlimits=(0.01, 100), ylimits=(0.5, 5e3), ic_rmax=0, neig=0,
            pl_range=(0.2, 9), pl_div=None, outfile=None, plot_file=None,
            plot_size=def_plot_size, landscape=False):
    """Plot clustering results according to key: xis, xi2, w_p, xir or bias.
    panels contains list of things to be plotted within each panel:
    a list of input files, comparison data and a panel label."""

    if landscape:
        sa_left = 0.12
        sa_bot = 0.1
    else:
        sa_left = 0.18
        sa_bot = 0.08

    xlab = xlabel[key]
    ylab = ylabel[key]

    npanel = len(panels)
    plt.clf()
    nrow, ncol = util.two_factors(npanel, landscape)
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0, wspace=0)
    fig.text(0.5, 0.02, xlab, ha='center', va='center')
    fig.text(0.02, 0.5, ylab, ha='center', va='center', rotation='vertical')
    irow, icol = 0, 0

    if outfile:
        fout = open(outfile, 'w')
    else:
        fout = None

    if pl_div and key=='w_p':
        # Convert power-law parameters from real to projected space
        gamma = pl_div[1]
        r0 = pl_div[0]
        A = (r0**gamma * scipy.special.gamma(0.5) *
             scipy.special.gamma(0.5*(gamma-1)) /
             scipy.special.gamma(0.5*gamma))
        pl_div[1] -= 1
        pl_div[0] = A**(1.0/(gamma-1))
        print(pl_div)

    for panel in panels:
        try:
            ax = axes[irow, icol]
        except:
            try:
                ax = axes[irow]
            except:
                ax = axes

        files = panel['files']
        i = 0
        clr = None
        for infile in files:
            if len(files) > 1:
                clr = clr_list[i]
            xi = xi_req(infile, key, binning=binning, pi_lim=pi_max)
            if key == 'xi2':
                xi.plot(ax, cbar=False)
            else:
                xi.plot(ax, jack=jack, color=clr_list[i], fout=fout,
                        pl_div=pl_div)
                if fit_range:
                    gamma, gamma_err, r0, r0_err, ic, _, _ = xi.fit(
                        fit_range, jack=jack, ax=ax, ic_rmax=ic_rmax,
                        neig=neig, color=clr)
            i += 1
        comps = panel['comps']
        for comp in comps:
            if comp:
                ax.errorbar(comp[0], comp[1], comp[2], color=clr_list[i],
                            fmt='s', capthick=1)
                i += 1

        if 'pl_pars' in panel:
            p = panel['pl_pars']
            if p:
                yfit = ((pl_range[0]/p[0])**-p[1], (pl_range[1]/p[0])**-p[1])
                ax.plot(pl_range, yfit, color=clr_list[i])

        if key != 'xi2':
            if (ylimits[0] <= 0):
                ax.semilogx(basex=10, nonposy='clip')
            else:
                ax.loglog(basex=10, basey=10, nonposy='clip')
            ax.axis(xlimits + ylimits)
        label = panel['label']
        if label:
            ax.text(0.2, 0.9, label, transform=ax.transAxes)

        icol += 1
        if icol >= ncol:
            icol = 0
            irow += 1
            if irow >= nrow:
                irow = 0
    if fout:
        fout.close()
    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def xir_mock_test(fit_range=(0.01, 5), ic_rmax=0, neig='full', verbose=0):
    """Test how well xi(r) can be recovered using mocks."""

    # First fit to xi(s) from cosmological redshift mocks
    xis = xi_req('xi_vol_cos.dat', 'xis')
    gamma, gamma_err, r0, r0_err, ic, gamma_jack, r0_jack = xis.fit(
            fit_range, ax=None, ic_rmax=ic_rmax, neig=neig,
            verbose=verbose)
    print('xis cosmo redshifts gamma = {:4.2f}+/-{:4.2f}, r0 = {:4.2f}+/-{:4.2f}'.format(
        np.mean(gamma_jack), np.std(gamma_jack),
        np.mean(r0_jack), np.std(r0_jack)))

    # Now fit to w_p(r_p) from observed redshift mocks for various pi_max
    for pi_lim in (5, 10, 15, 20, 30, 40, 50, 60, 100):
        w_p = xi_req('xi_vol.dat', 'w_p', pi_lim=pi_lim)
        xir = xi_req('xi_vol.dat', 'xir', pi_lim=pi_lim)
        gamma, gamma_err, r0, r0_err, ic, gamma_jack, r0_jack = w_p.fit(
                fit_range, ax=None, ic_rmax=ic_rmax, neig=neig,
                verbose=verbose)
        chi2 = xir.cov.chi2(xir.est[:, 0], xis.est[:-1, 0], neig)
        print('pi_lim = {}, gamma = {:4.2f}+/-{:4.2f}, r0 = {:4.2f}+/-{:4.2f}, chi2 {}'.format(
            pi_lim, np.mean(gamma_jack), np.std(gamma_jack),
            np.mean(r0_jack), np.std(r0_jack), chi2))


def xir_plot_mock(xi_obs_file='xi_vol.dat', xi_cos_file='xi_vol_cos.dat',
                  key='xir', binning=1, pi_max=40.0, pl_div=(4.84, 1.84),
                  xlimits=(0.01, 100), ylimits=(0.001, 1e5), ic_rmax=0, neig=0,
                  plot_file='xir_mock.pdf', plot_size=def_plot_size):
    """Compare xi(r) from inversion method with that from direct estimate
    using cosmological redshifts."""
    if pl_div:
        gamma = pl_div[1]
        r0 = pl_div[0]
    plt.clf()
    ax = plt.subplot(111)
    xi = xi_req(xi_obs_file, key='xir', binning=binning, pi_lim=pi_max)
    if pl_div:
        pl_fit = (xi.sep/r0)**(-gamma)
    else:
        pl_fit = 1
    ax.errorbar(xi.sep, (xi.est[:, 0] + xi.ic)/pl_fit,
                xi.cov.sig/pl_fit, fmt='o', color=clr_list[0])
#    xi.plot(ax, color=clr_list[0])
    xi = xi_req(xi_cos_file, key='xis', binning=binning, pi_lim=pi_max)
    if pl_div:
        pl_fit = (xi.sep/r0)**(-gamma)
    else:
        pl_fit = 1
    ax.errorbar(xi.sep, (xi.est[:, 0] + xi.ic)/pl_fit,
                xi.cov.sig/pl_fit, fmt='s', color=clr_list[1])
#    xi.plot(ax, color=clr_list[1])
    ax.loglog(basex=10, basey=10, nonposy='clip')
    plt.xlabel(xlabel[key])
    ylab = ylabel[key]
    if pl_div:
        ylab += r'$ / \xi_{\rm ref}(r)$'
    plt.ylabel(ylab)
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def plot_weights(file_list):
    """Plot weights as function of distance."""

    plt.clf()
    for infile in file_list:
        data = np.loadtxt(infile, skiprows=3)
        r = np.sqrt(np.sum(data[:,0:3]**2, axis=1))
        wt = data[:,3]
        plt.plot(r, wt, '.')
    plt.xlabel('r [Mpc/h]')
    plt.ylabel('Weight')
    plt.draw()


def xi2d_plot(infile, what='logxi', pilim=40.0, rplim=40.0, binning=0,
              pi_rebin=1, rp_rebin=1, mirror=True, cmap=None, xi_range=(-2, 1),
              sn_range=None, plot_file=None):
    """xi(sigma,pi) plot."""

    xi2 = xi_req(infile, 'xi2', binning=binning, pi_rebin=pi_rebin,
                 rp_rebin=rp_rebin, pi_lim=pilim, rp_lim=rplim)
    plt.clf()
    ax = plt.subplot(111)
    xi2.plot(ax, what, prange=xi_range, mirror=mirror, cmap=cmap)

    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def xi2d_plot_samples(intemp='xi_{}_{}_{}.dat', param='lum_c', what='logxi',
                      pilim=40.0, rplim=40.0, binning=0, mirror=True,
                      xi_range=(-2, 1), sn_range=None, landscape=False):
    """xi(sigma,pi) plot for multiple samples."""

    if landscape:
        plot_size = (8, 5)
        sa_left = 0.12
        sa_bot = 0.1
    else:
        plot_size = (5, 8)
        sa_left = 0.18
        sa_bot = 0.08

    if 'lum' in param:
        Mlimits = def_mag_limits
        labeltemp = r'$M_r = [{}, {}]$'
    else:
        Mlimits = def_mass_limits
        labeltemp = r'$M_* = [{}, {}]$'
    npanel = len(Mlimits) - 1
    xlab = xlabel['xi2']
    ylab = ylabel['xi2']
    plot_file = 'xi2d_{}.pdf'.format(param)

    plt.clf()
#    nrow, ncol = util.two_factors(npanel, landscape)
    nrow, ncol = 2, 4
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
    axes = axes.ravel()
    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0, wspace=0)
    fig.text(0.5, 0.02, xlab, ha='center', va='center')
    fig.text(0.02, 0.5, ylab, ha='center', va='center', rotation='vertical')

    for i in range(npanel):
        infile = intemp.format(param, Mlimits[i], Mlimits[i+1])
        label = labeltemp.format(Mlimits[i], Mlimits[i+1])
        ax = axes[i]
        xi = xi_req(infile, 'xi2', binning=binning, pi_lim=pilim, rp_lim=rplim)
        xi.plot(ax, cbar=False, aspect='equal')
        ax.text(0.2, 0.9, label, transform=ax.transAxes, color='w')

    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def xi2d_linear_model(infile, beta=0.45, r0=5, gamma=1.8,
                      pilim=40.0, rplim=40.0, binning=0, lgximin=-2, mirror=1):
    """Linear model for 2d correlation function."""

    xi2 = xi_req(infile, 'xi2', binning=binning, pi_lim=pilim, rp_lim=rplim)
    xi2.lin_model(beta, r0, gamma)
    plt.clf()
    ax = plt.subplot(111)
    xi2.plot(ax)
    plt.draw()


def xi2d_ft_plot(infile, pilim=40.0, rplim=40.0, binning=0, lgximin=-2):
    """2d FT plot."""

    xi2 = xi_req(infile, 'xi2', binning=binning, pi_lim=pilim, rp_lim=rplim)
    xi2.ft(lgximin)


def xi1d_cov_plot(infile, key=2, pi_max=40.0, plot_file=None):
    """Covariance plot."""

    xi = Xi(infile, pi_max)
    xi = xi.data[key]
    cov = Cov(xi.est[:, 1:])
    plt.clf()
    cov.plot()
    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def xi1d_eig_plot(infile, key=2, pi_max=40.0, plot_file=None):
    """Eigenvalue & vector plot."""

    xi = Xi(infile, pi_max)
    xi = xi.data[key]
    cov = Cov(xi.est[:, 1:])
    plt.clf()
    cov.plot_eig()
    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def xi2d_cov_plot(infile, pi_max=40.0, plot_file=None):
    """xi(sigma,pi) covariance plot."""

    xi = Xi(infile, pi_max)
    xi2 = xi.data['xi2']
    cov = jack_cov(xi2.est[:, :, 1:].reshape((xi2.nrp*xi2.npi, xi2.njack),
                   order='F'))
    plt.clf()
    ax = plt.subplot(111)
    cov.plot(ax=ax)
    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


# -----------------
# P(k, mu) fitting
# -----------------

def P_model(kvals, muvals, Pr, sigma, beta=0.45):
    """The model for P(k, mu) (Li+2006 eqn 3)."""
    Pmod = np.zeros((len(kvals), len(muvals)))
    for ik in xrange(len(kvals)):
        k = kvals[ik]
        sig = sigma[ik]/H0
        for imu in xrange(len(muvals)):
            mu = muvals[imu]
            Pmod[ik, imu] = Pr[ik] * (1 + beta*mu**2)**2 / (
                1 + 0.5*(k*mu*sig)**2)
    return Pmod


def P_k_mu_plot(infile, nsub=21, Prange=(0.1, 2e4), plot_file=None):
    """P(k, mu) plot."""

    xi2 = xi_req(infile, 'xi2')
    Pobs = P2d(xi2, nsub=nsub, err_type=xi2.err_type)

    plt.clf()
    ax = plt.subplot(111)
    Pobs.plot(ax, Prange)
    plt.draw()

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(6, 6)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def pvd_k_multi(intemp='xi_{}_{}_{}.dat', outtemp='pvd_k_{}_{}_{}.dat',
                param='lum_c', Mlimits=None, qsub=0):
    """PVD k fits for multiple datasets."""

    if Mlimits is None:
        if 'lum' in param:
            Mlimits = def_mag_limits
        else:
            Mlimits = def_mass_limits

    nsamp = len(Mlimits) - 1
    for i in range(nsamp):
        infile = intemp.format(param, Mlimits[i], Mlimits[i+1])
        outfile = outtemp.format(param, Mlimits[i], Mlimits[i+1])
        if qsub:
            python_commands = [
               "import corr",
               """corr.P_k_mu_fit('{}', '{}')""".format(infile, outfile)]
            print(python_commands)
            util.apollo_job(python_commands)
        else:
            P_k_mu_fit(infile, outfile)


def P_k_mu_fit(infile, outfile, nsub=21, smooth=20, neig=0, nchain=3,
               nstep=2000, Prange=(0.001, 2e4), plot_file=None):
    """Fit real space P(k) and PVD sigma(k) to P(k, mu)."""

    xi2 = xi_req(infile, 'xi2')
    Pobs = P2d(xi2, nsub=nsub, smooth=smooth, err_type=xi2.err_type)
    nk = len(Pobs.k)
    use = Pobs.P[:, :, 0] > 0
    cov = Cov(Pobs.P[use, 1:], xi2.err_type)
    cov.plot_eig()

    plt.clf()
    ax = plt.subplot(111)
    Pobs.plot(ax, Prange)
    plt.suptitle(infile)
    plt.show()

    # Starting guess at solution: params are lg P(k) and lg sigma(k)
    x0 = np.zeros(2*nk)
    idx = Pobs.P[:, 0, 0] > 0
    x0[idx] = np.log10(Pobs.P[idx, 0, 0])
#    pdb.set_trace()
    # x0[:nk] = np.log10(Pobs.P[:,0])
    x0[nk:] = np.log10(500)

    # Sample parameter space with emcee
    def lnprob(x):
        """Returns -0.5*chi^2 residual between model and observed P(k, mu).
        First half of array x contains lg Pr(k) values,
        second half lg sigma(k)."""

        # Avoid over/underflows
        if abs(np.max(x)) > 5:
            return -1e6
        Pr, sigma = 10**x[:nk], 10**x[nk:]
        Pmod = P_model(Pobs.k, Pobs.mu, Pr, sigma)
        chi2 = cov.chi2(Pobs.P[use, 0].flatten(), Pmod[use].flatten(), neig)
#        print x, chi2
#        pdb.set_trace()
        return -0.5*chi2

    ndim = 2*nk
    nwalkers = 10*ndim
    pos = np.tile(x0, (nwalkers, 1)) + 0.1*np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    for ichain in range(nchain):
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, nstep)
#        print sampler.chain[0, :, 0]
        print("Mean acceptance fraction: {0:.3f}"
              .format(np.mean(sampler.acceptance_fraction)))
        print("Autocorrelation time:", sampler.get_autocorr_time())
        plot_conv(sampler, nk, 5)
        plt.suptitle(infile + 'Chain {}'.format(ichain))
    res = 10**np.array(np.percentile(sampler.flatchain, [50, 16, 84], axis=0))

    Pr, sigma = res[0, :nk], res[0, nk:]
    Pr_range, sigma_range = res[1:, :nk], res[1:, nk:]
    print(Pr, sigma)
    Pmod = P_model(Pobs.k, Pobs.mu, Pr, sigma)

    outdict = {'infile': infile, 'Pobs': Pobs, 'Pmod': Pmod,
               'Pr': Pr, 'Pr_range': Pr_range,
               'sigma': sigma, 'sigma_range': sigma_range}
    pickle.dump(outdict, open(outfile, 'w'))

    if plot_file:
        plot_file.savefig()

    plt.clf()
    ax = plt.subplot(111)
    Pobs.plot(ax, Prange)
    ax.set_prop_cycle(None)
    for ik in xrange(nk):
        ax.plot(Pobs.mu, Pmod[ik, :])
    plt.suptitle(infile)
    plt.show()
    if plot_file:
        plot_file.savefig()

    # ans = raw_input('Return for next plot: ')
    plt.clf()
    fig, axes = plt.subplots(2, 1, sharex=True, num=1)
    fig.subplots_adjust(hspace=0, wspace=0.1)
    ax = axes[0]
    ax.errorbar(Pobs.k, Pr, capthick=1,
                yerr=((Pr-Pr_range[0, :], Pr_range[1, :]-Pr)), fmt='o')
    ax.loglog(basex=10, basey=10)
    ax.set_ylabel(r'$P_r(k)$')
    ax = axes[1]
    show = Pobs.k > 0.2
    ax.errorbar(Pobs.k[show], sigma[show], capthick=1,
                yerr=((sigma[show]-sigma_range[0, show],
                       sigma_range[1, show]-sigma[show])), fmt='o')
    ax.semilogx(basex=10)
    ax.set_ylabel(r'$\sigma_{12}(k)$')
    ax.set_xlabel(r'$k [h/{\rm Mpc}]$')
    plt.suptitle(infile)
    plt.show()
    if plot_file:
        plot_file.savefig()


def plot_conv(sampler, nk, nplot=0):
    """Plot MCMC convergance for nplot parameters."""

    plt.clf()
    nrow = nplot
    ncol = 2
    fig, axes = plt.subplots(nrow, ncol, num=1)
    fig.subplots_adjust(hspace=0, wspace=0.1)
    fig.text(0.5, 0.02, 'Step number', ha='center', va='center')
    fig.text(0.06, 0.5, r'$P(k)$', ha='center',
             va='center', rotation='vertical')
    fig.text(0.56, 0.5, r'$\sigma^2_{12}$', ha='center',
             va='center', rotation='vertical')
    nwalkers = sampler.chain.shape[0]
    nstep = 1
    if nplot > 0:
        nstep = nk//nplot
    ipar = 0
    for icol in range(ncol):
        for irow in range(nrow):
            ax = axes[irow, icol]
            for iw in xrange(nwalkers):
                ax.plot(sampler.chain[iw, :, ipar])
            ipar += nstep
    plt.show()


def pvd_k_plot(inroot='pvd_V_c_{}_{}.dat', labroot=r'${} < M_r < {}$',
               key=0, comp=True, plot_file=None):
    """Plot 0:P_s(k, mu), 1:P_r(k) or 2:sigma(k) with Li+2006 comparison."""

    Mlimits = def_mag_limits
    npanel = len(Mlimits) - 1
    if key == 0:
        xlabel = r'$\mu$'
    else:
        xlabel = r'$k\ [h\ {\rm Mpc}^{-1}]$'
        Liroot = ['pk-L{:4.1f}{:4.1f}.dat', 'pvd-L{:4.1f}{:4.1f}.dat'][key-1]
    ylabel = [r'$P_s(k, \mu)$', r'$P_r(k)$',
              r'$\sigma_{12}\ [{\rm km\ s}^{-1}]$'][key]
    xlab_pos = [0.1, 0.5, 0.1]
    ylab_pos = [0.1, 0.9, 0.9]
    nrow, ncol = util.two_factors(npanel)
    plt.clf()
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
    fig.subplots_adjust(left=0.125, hspace=0, wspace=0)
    util.fig_xlabel(xlabel)
    util.fig_ylabel(ylabel)
    irow, icol = 0, 0
    for i in range(npanel):
        Mlo, Mhi = Mlimits[i], Mlimits[i+1]
        dat = pickle.load(open(inroot.format(Mlo, Mhi), 'r'))
        label = labroot.format(Mlo, Mhi)
        ax = axes[irow, icol]
        Pobs = dat['Pobs']
        nk, nmu = Pobs.P.shape

        if key == 0:
            for ik in xrange(nk):
                ax.errorbar(Pobs.mu, Pobs.P[ik, :], yerr=Pobs.cov.sig[ik, :],
                            fmt='o', capthick=1)
            Pmod = dat['Pmod']
            ax.set_prop_cycle(None)
            for ik in xrange(nk):
                ax.plot(Pobs.mu, Pmod[ik, :])
            ax.set_xlim(-0.1, 1.0)
            ax.set_ylim(0.01, 5e4)
            ax.semilogy(basey=10)

        if key == 1:
            Pr = dat['Pr']
            Pr_range = dat['Pr_range']
            ax.errorbar(Pobs.k, Pr, capthick=1,
                        yerr=((Pr-Pr_range[0,:], Pr_range[1,:]-Pr)), fmt='o')
            if comp:
                try:
                    k, P, P_err = Li_w_p(Liroot, Mlo, Mhi)
                    ax.errorbar(k, P, P_err, fmt='s', capthick=1)
                except:
                    pass
            ax.set_xlim(0.08, 9)
            ax.set_ylim(5, 5e4)
            ax.loglog(basex=10, basey=10)

        if key == 2:
            sigma = dat['sigma']
            sigma_range = dat['sigma_range']
            show = Pobs.k > 0.2
            ax.errorbar(Pobs.k[show], sigma[show], capthick=1, 
                        yerr=((sigma[show]-sigma_range[0,show], 
                               sigma_range[1,show]-sigma[show])), fmt='o')
            if comp:
                try:
                    k, sig, sig_err = Li_w_p(Liroot, Mlo, Mhi)
                    plot = sig > 0
                    ax.errorbar(k[plot], sig[plot], sig_err[plot], fmt='s',
                                capthick=1)
                except:
                    pass
            ax.set_xlim(0.08, 9)
            ax.set_ylim(200, 950)
            ax.semilogx(basex=10)

        # ax.set_xlim(0.01, 50)
        # ax.set_ylim(0, 900)
        # ax.set_yticks(np.arange(0, 900, 200))
        ax.text(xlab_pos[key], ylab_pos[key], label, transform = ax.transAxes)
        icol += 1
        if icol >= ncol:
            icol = 0
            irow += 1
            if irow >= nrow:
                irow = 0
    plt.show()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def pvd_k_mock_plot(direct='pvd_av.npz', filenames=('mock_fourier.dat',),
                    k_range=(0.1, 100), P_range=(1e0, 1e4),
                    sig_range=(0, 1000), krfac=2*math.pi, alpha=0.1,
                    plot_file='pvd_k_mock.pdf', size=(5, 5)):
    """Plot P(k) and sigma(k) from mocks."""

    plt.clf()
    fig, axes = plt.subplots(2, 1, sharex=True, num=1)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    ax = axes[0]
    ax.loglog(basex=10, basey=10)
    ax.set_ylim(P_range)
    ax.set_ylabel(r'$P_r(k)$')
    ax = axes[1]
    # Plot direct estimate of sigma(r), using k = krfac / r
    data = np.load(direct)
    plt.plot(krfac/data['sep'], data['slt_exp'], 'k-', label='Direct')
    plt.fill_between(krfac/data['sep'], data['slt_exp'] - data['slt_exp_err'],
                     data['slt_exp'] + data['slt_exp_err'],
                     facecolor='k', alpha=alpha)
    ax.semilogx(basex=10)
    ax.set_xlim(k_range)
    ax.set_ylim(sig_range)
    ax.set_ylabel(r'$\sigma_{12}(k)\ [{\rm km\ s}^{-1}]$')
    ax.set_xlabel(r'$k\ [h\ {\rm Mpc}^{-1}]$')

    ifmt = 0
    for filename in filenames:
        ifmt += 1
        res = pickle.load(open(filename, 'r'))
        Pobs = res['Pobs']
        Pr = res['Pr']
        Pr_range = res['Pr_range']
        sigma = res['sigma']
        sigma_range = res['sigma_range']

        ax = axes[0]
        ax.errorbar(Pobs.k, Pr,
                    yerr=((Pr-Pr_range[0, :], Pr_range[1, :]-Pr)),
                    fmt=symb_list[ifmt], capthick=1)
        ax = axes[1]
        show = Pobs.k > 0.1
        ax.errorbar(Pobs.k[show], sigma[show],
                    yerr=((sigma[show]-sigma_range[0, show],
                           sigma_range[1, show]-sigma[show])),
                    fmt=symb_list[ifmt], capthick=1)

    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def Li_Pk_plot():
    """Plot Li+2006 P(kr), sigma_12 and P(k, mu) results."""

    li_dir = os.environ['HOME'] + '/Documents/Research/corrdata/Li2006/data/'
    k, Pr, Pr_err = Li_w_p(li_dir + 'pk-L-21.0-20.0.dat')
    Pr, Pr_err = Pr[3:-1], Pr_err[3:-1]
    k, sig, sig_err = Li_w_p(li_dir + 'pvd-L-21.0-20.0.dat')
    k, sig, sig_err = k[:-1], sig[:-1], sig_err[:-1]
    mu = np.linspace(0.0, 0.9, 10)
    Pmod = P_model(k, mu, Pr, sig**2)

    plt.clf()
    ax = plt.subplot(221)
    ax.errorbar(k, Pr, Pr_err, fmt='o', capthick=1)
    ax.loglog(basex=10, basey=10)
    ax.set_xlabel('k')
    ax.set_ylabel('P(k)')

    ax = plt.subplot(222)
    ax.errorbar(k, sig, sig_err, fmt='o', capthick=1)
    ax.semilogx(basey=10)
    ax.set_xlabel('k')
    ax.set_ylabel('sigma(k)')

    ax = plt.subplot(223)
    for ik in xrange(len(k)):
        ax.plot(mu, Pmod[ik, :], label='k = {}'.format(k[ik]))
    ax.semilogy(basey=10)
    ax.set_xlabel('mu')
    ax.set_ylabel('P(k)')
    ax.legend()
    plt.draw()

def bias_plot_lum(templist=('xi_lum_{}_{}.dat', 'xi_V_c_{}_{}.dat'), 
                  labellist=('Mag-limited', 'Vol-limited'), pi_max=40.0, 
                  bias_par='M_mean', bias_scale=3.0, bias_ref=2,
                  ic_rmax=0, fitcov=False, plot_file='bias_lum.pdf'):
    """Plot relative bias as function of luminosity."""

    def bplot(fileList, label):
        """Bias plot."""
        x = np.zeros(nb)
        y = np.zeros(nb)
        yerr = np.zeros(nb)
        ib = 0
        for infile in fileList:
            print(infile)
            xi = Xi(infile, pi_max)
            w_p = xi.data['w_p']
            if w_p.rmin < 0:
                bs = math.log10(bias_scale)
            else:
                bs = bias_scale
            jbias = int((bs - w_p.rmin)/w_p.rstep)
            x[ib] = xi.info[bias_par]
            y[ib] = w_p.est[jbias, 0]
            yerr[ib] = w_p.cov.sig[jbias]
            ib += 1
        yref = y[bias_ref]
        y /= yref
        yerr /= yref
        wt = 1.0/yerr**2
        ax.errorbar(x, y, yerr, fmt='o', label=label, capthick=1)

    compfile = os.environ['HOME'] + '/Documents/Research/corrdata/Zehavi2011/Table7.txt'
    data = np.loadtxt(compfile)
    xc = (-22.5, -21.5, -20.5, -19.5, -18.5, -17.5)
    yc = data[6, 1:12:2]/data[6, 5]
    ycerr = data[6, 2:13:2]/data[6, 5]
    print('Comparison scale', data[6, 0])

    plt.clf()
    Mlimits = mag_limits
    nb = len(Mlimits) - 1
    ax = plt.subplot(111)
    for temp, label in zip(templist, labellist):
        fileList =[]
        labelList =[]
        for i in range(nb):
            fileList.append(temp.format(Mlimits[i], Mlimits[i+1]))
        bplot(fileList, label)

    ax.errorbar(xc, yc, ycerr, fmt='rs', label='SDSS (Zehavi+ 2011)',
                capthick=1)
    # p = np.polyfit(x, y, 2, w=wt)
    # print 'poly coeffs', p
    # yfit = np.polyval(p, x)
    # ax.plot(x, yfit)
    if bias_par == 'M_mean': xlabel = r'log $L/L^*$'
    if bias_par == 'M_mean': xlabel = r'$M_r$'
    if bias_par == 'z_mean': xlabel = r'$z$'
    if bias_par == 'logmstar': xlabel = r'log $M_*/M_\odot$'
    ax.set_xlabel(xlabel)
    # ax.set_ylabel(r'$w_p({:4.2f} \ {{\rm Mpc}})$'.format(w_p.sep[jbias]))
    ax.set_ylabel(r'$b(M) / b(M^*)$')
    ax.set_xlim(-15, -23)
    ax.legend(loc=2)
    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(6, 6)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')

def bias_plot_mass(temp='xi_mass_c_{}_{}.dat', 
                   label='Mass-limited', pi_max=40.0, 
                   bias_par='logm_mean', bias_scale=3.0, bias_ref=4,
                   ic_rmax=0, fitcov=False, plot_file='bias_mass.pdf'):
    """Plot relative bias as function of mass."""

    def bplot(fileList, label):
        """Bias plot."""
        x = np.zeros(nb)
        y = np.zeros(nb)
        yerr = np.zeros(nb)
        ib = 0
        for infile in fileList:
            print(infile)
            xi = Xi(infile, pi_max)
            w_p = xi.data['w_p']
            if w_p.rmin < 0:
                bs = math.log10(bias_scale)
            else:
                bs = bias_scale
            jbias = int((bs - w_p.rmin)/w_p.rstep)
            x[ib] = xi.info[bias_par]
            y[ib] = w_p.est[jbias, 0]
            yerr[ib] = w_p.cov.sig[jbias]
            ib += 1
        yref = y[bias_ref]
        y /= yref
        yerr /= yref
        wt = 1.0/yerr**2
        ax.errorbar(x, y, yerr, fmt='o', label=label, capthick=1)

    plt.clf()
    Mlimits = mass_limits
    nb = len(Mlimits) - 1
    ax = plt.subplot(111)
    fileList = []
    xc = []
    yc = []
    ycerr = []
    for i in range(nb):
        fileList.append(temp.format(Mlimits[i], Mlimits[i+1]))
        wcomp = Li_w_p('wrp-M-{:04.1f}-{:04.1f}.dat', 
                       Mlimits[i], Mlimits[i+1])
        if wcomp:
            xc.append(0.5 * (Mlimits[i] + Mlimits[i+1]))
            yc.append(wcomp[1][11])
            ycerr.append(wcomp[2][11])
    bplot(fileList, label)
    yref = yc[2]
    yc /= yref
    ycerr /= yref
    ax.errorbar(xc, yc, ycerr, fmt='rs', label='SDSS (Li+ 2006)', capthick=1)
    # p = np.polyfit(x, y, 2, w=wt)
    # print 'poly coeffs', p
    # yfit = np.polyval(p, x)
    # ax.plot(x, yfit)
    if bias_par == 'M_mean': xlabel = r'log $L/L^*$'
    if bias_par == 'M_mean': xlabel = r'$M_r$'
    if bias_par == 'z_mean': xlabel = r'$z$'
    if bias_par == 'logm_mean': xlabel = r'log $M_*/M_\odot$'
    ax.set_xlabel(xlabel)
    # ax.set_ylabel(r'$w_p({:4.2f} \ {{\rm Mpc}})$'.format(w_p.sep[jbias]))
    ax.set_ylabel(r'$b(M) / b(M^*)$')
    ax.set_xlim(8, 11.5)
    ax.set_ylim(0, 2.5)
    ax.legend(loc=2)
    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(6, 6)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def vdist_test(nbin=64, rmin=0, rmax=64, r0=5.0, gamma=1.8, beta=0.5,
               sigma=600, hsmooth=0, plots=1):
    """Test recovery of velocity distribution from convolved beta model."""

    # Generate beta model convolved with fixed velocity dispersion
    xi_beta = Xi2d(nbin, rmin, rmax, nbin, rmin, rmax, 0, None)
    xi_beta.beta_model(beta, r0, gamma)
    xi_beta = xi_beta.reflect()
    plt.clf()
    ax = plt.subplot(111)
    xi_beta.plot(ax, prange=(-2, 3), mirror=0)
    plt.show()

    pvd = xi_beta.pistep*H0/(root2*sigma)*np.exp(-np.abs(
        root2*xi_beta.pic*H0/sigma))
    print('pvd sum = ', pvd.sum())

    conv = np.zeros((2*nbin, 2*nbin, 1))
    for irp in xrange(2*nbin):
        conv[:, irp, 0] = np.convolve(xi_beta.est[:, irp, 0], pvd, 'same')
    xi_model = copy.deepcopy(xi_beta)
    xi_model.est = conv

    plt.clf()
    ax = plt.subplot(111)
    xi_model.plot(ax, prange=(-2, 3), mirror=0)
    plt.show()

    xi_model.vdist(hsmooth=hsmooth, plots=plots)


def vdist(infile='xi_vol.dat', pi_max=64, pi_max_fit=40, rp_max=64,
          fitRange=(0.1, 10), beta_fix=None, meansep=0, ic_rmax=0,
          binning=0, neig=0, hsmooth=0, plots=1, plot_size=def_plot_size):
    """Velocity distribution function via Fourier transform of xi2d."""

    def vdist_model(p):
        """Velocitiy distribution from beta model.
        Parameters p = (beta, lg(sigma)."""

        if len(p) == 2:
            beta = p[0]
            sigma = 10**p[1]
        else:
            beta = beta_fix
            sigma = 10**p[0]
#        xi_beta = Xi2d(xi2.nrp//2, 0, xi2.rpmax,
#                       xi2.npi//2, 0, xi2.pimax, 0, None)
        xi_beta = copy.deepcopy(xi2)
        xi_beta.njack = 0
        xi_beta.beta_model(beta, r0=r0, gamma=gamma, meansep=meansep)
#        xi_beta = xi_beta.reflect()
        if fn == 'exp':
            pvd = xi_beta.pistep*H0/(root2*sigma)*np.exp(-np.abs(
                root2*xi_beta.pic*H0/sigma))
        if fn == 'gauss':
            pvd = xi_beta.pistep*H0/(root2pi*sigma)*np.exp(
                -(xi_beta.pic*H0)**2/(2*sigma**2))
        conv = np.zeros((xi2.nrp, xi2.npi, 1))
        for irp in xrange(xi2.nrp):
            conv[:, irp, 0] = np.convolve(xi_beta.est[:, irp, 0], pvd, 'same')
        xi_model = copy.deepcopy(xi_beta)
        xi_model.est = conv

#        plt.clf()
#        ax = plt.subplot(111)
#        xi_model.plot(ax, prange=(-2, 3), mirror=0)
#        plt.show()
#        pdb.set_trace()
        return xi_model.vdist(hsmooth=hsmooth, plots=0)

    def chi2(p):
        """chi2 from exponential or Gaussian distribution-convolved beta model.
        Note that First data point is always unity, so is excluded."""

        _, ratio_mod, _, _, _, _ = vdist_model(p)
        chi2 = ratio_cov.chi2(ratio[1:], ratio_mod[1:], neig=neig)
#        print p, chi2
        return chi2

    # Power-law fit to projected correlation function
    w_p = xi_req(infile, 'w_p', binning=1, pi_lim=pi_max_fit, rp_lim=rp_max)
    plt.clf()
    ax = plt.subplot(111)
    w_p.plot(ax)
    plt.loglog(basex=10, basey=10, nonposy='clip')
    plt.xlabel(r'$r_\bot\ [h^{-1} {\rm Mpc}]$')
    plt.ylabel(r'$w_p(r_\bot)$')
    gamma, gamma_err, r0, r0_err, ic, gamma_jack, r0_jack = w_p.fit(
        fitRange, ax=ax, ic_rmax=ic_rmax, neig=0)
    plt.show()

    # Velocity distribution from 2d correlation function
    xi2 = xi_req(infile, 'xi2', binning=binning, pi_lim=pi_max, rp_lim=rp_max)
    xi2 = xi2.reflect()
    plt.clf()
    ax = plt.subplot(111)
    xi2.plot(ax, mirror=0)
    plt.show()
    freq, ratio, ratio_cov, v, fv, fv_cov = xi2.vdist(
        hsmooth=hsmooth, plots=plots)

    ratio_cov.plot(norm=1, label='Ratio')
    plt.show()
    fv_cov.plot(norm=1, label='f(v)')
    plt.show()

    ratio_mod = {}
    fv_mod = {}
    for fn in ('exp', 'gauss'):
        if beta_fix:
            p0 = (2.78,)
        else:
            p0 = (0.5, 2.78)

        ndof = len(v)-1  # First data point is always unity
        if neig > 0 and neig < ndof:
            ndof = neig
        ndof -= len(p0)
        out = scipy.optimize.fmin(chi2, p0, maxfun=10000, maxiter=10000,
                                  full_output=1, disp=0)
        p = out[0]
        if len(p) == 2:
            print('{} fit parameters: beta = {}, sigma = {}'.format(
                fn, p[0], 10**p[1]))
        else:
            print('{} fit parameters: beta = {}, sigma = {}'.format(
                fn, beta_fix, 10**p[0]))
        print('chi2/nu = {}/{}'.format(out[1], ndof))
        _, ratio_mod[fn], _, _, fv_mod[fn], _ = vdist_model(p)

    plt.clf()
    plt.errorbar(freq, ratio, np.append(0, ratio_cov.sig), fmt='o', capthick=1)
    plt.plot(freq, ratio_mod['exp'], '-', freq, ratio_mod['gauss'], '--')
    plt.xlabel(r'$k\ [100\ \mathrm{km\ s}^{-1}]^{-1}$')
    plt.ylabel(r'$F[f(v)]$')
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    if xi2.err_type == 'mock':
        plotfile = 'mock_Ffv.pdf'
    else:
        plotfile = 'Ffv.pdf'
    plt.savefig(plot_dir + plotfile, bbox_inches='tight')

    plt.clf()
    plt.errorbar(v, fv, fv_cov.sig, fmt='o', capthick=1)
    plt.plot(v, fv_mod['exp'], '-', v, fv_mod['gauss'], '--')
    plt.xlabel(r'$v\ [100\ \mathrm{km\ s}^{-1}]$')
    plt.ylabel(r'$f(v)$')
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(plot_size)
    if xi2.err_type == 'mock':
        plotfile = 'mock_fv.pdf'
    else:
        plotfile = 'fv.pdf'
    plt.savefig(plot_dir + plotfile, bbox_inches='tight')


def plot_stream(lgr_range=(-1, 2), r0=5.0, gamma=1.8, beta=0.5):
    """Plot streaming models."""

    lgr = np.linspace(lgr_range[0], lgr_range[1], 50)
    r = 10**lgr
    xir = (r0/r)**gamma
    xib = 3.0 / (3 - gamma) * (r0/r)**gamma
    xibb = xib / (1 + xir)
    alpha = 1.84 - 0.65*gamma
    v_bean = -H0*beta*r*xir/(1 + xir)
    v_dp = -H0*r/(1 + (r/r0)**2)
    v_jsd = -2.0/3*H0*r*beta*xibb*(1 + alpha*xibb)
    plt.clf()
    plt.plot(r, v_bean, label='Bean')
    plt.plot(r, v_dp, label='DP')
    plt.plot(r, v_jsd, label='JSD')
    plt.semilogx(basex=10)
    plt.xlabel('r [Mpc/h]')
    plt.ylabel('v(r) [km/s]')
    plt.legend()
    plt.draw()


def pvd_stream_mocks(qsub=False):
    """Run pvd_stream on mocks."""

#    for pred_err in ('None', 'jack'):
#    flowmod = 'jsd'
#    flowmod = 'pvd_av.npz'
    pred_err = 'None'
    err_est = 'jack'
    logfit = 1
    interplog = 1
    neig = 0
    infile = 'xi_vol.dat'
    for xirmod in ('w_p_fit', 'xir_meas'):
        for flowmod in ('jsd', 'pvd_vol.npz'):
            outfile = 'pvd_stream_vol_{}_{}.dat'.format(flowmod, xirmod)
            plotfile = 'pvd_stream_vol_{}_{}.pdf'.format(flowmod, xirmod)
            if qsub:
                python_commands = [
                   "import corr",
                   """corr.pvd_stream('{}', '{}',
                                      flowmod='{}',
                                      xirmod='{}',
                                      pi_max_fit=50, neig={},
                                      err_est='{}', pred_err='{}',
                                      logfit={}, interplog={},
                                      plot_file='{}')""".format(
                                      infile, outfile, flowmod, xirmod,
                                      neig, err_est, pred_err, logfit,
                                      interplog, plotfile)]
                print(python_commands)
                util.apollo_job(python_commands)
            else:
                pvd_stream(infile, outfile,
                           flowmod=flowmod, xirmod=xirmod,
                           pi_max_fit=50, neig=neig,
                           err_est=err_est, pred_err=pred_err, logfit=logfit,
                           interplog=interplog, plot_file=plotfile)


def pvd_stream_samples(intemp='xi_{}_{}_{}.dat',
                       outtemp='pvd_stream_{}_{}_{}.dat',
                       param='lum_c', Mlimits=None):
    """Calc PVD via streaming model for several subsamples."""

    if Mlimits is None:
        if 'lum' in param:
            Mlimits = def_mag_limits
        else:
            Mlimits = def_mass_limits

    nsamp = len(Mlimits) - 1
    for i in range(nsamp):
        print(intemp.format(param, Mlimits[i], Mlimits[i+1]))
        pvd_stream(intemp.format(param, Mlimits[i], Mlimits[i+1]),
                   outtemp.format(param, Mlimits[i], Mlimits[i+1]))


def pvd_stream(infile, outfile, pi_max=40, pi_max_fit=40, rp_max=100,
               fitRange=(0.01, 5), ic_rmax=0, binning=2, neig=0, flowmod='jsd',
               xirmod='xir_meas', vmod='exp', err_est='jack', pred_err='None',
               lg_xi_range=(-2, 2), ylim=200, logfit=0, interplog=0,
               plot_int=0, plot_file=None):
    """Calculate PVD using streaming model (Zehavi+2002 eqn 18)."""

    def integrand(y, sigma, xir, rp, pi, what='xi'):
        """The integrand (1 + xi(r)) * f(v) if what='xi',
            or xi_err(r)^2 * f(v) if what='err',
            or f(v) if what='f'."""

        r = (rp**2 + y**2)**0.5
        xi, xi_err = xir.interp(r, log=interplog)

        # These velocities replace r by abs(y) to convert pairwise to los vel
        if flowmod == 'bean':
            vmean = -0.5*H0*abs(y)*xi/(1 + xi)
        if flowmod == 'dp':
            vmean = -H0*abs(y)/(1 + (r/r0)**2)
        if flowmod == 'jsd':
            xibar = (3 * r0**gamma * r**-gamma) / (3 - gamma)
            xibbar = xibar / (1 + xi)
            vmean = -2.0/3.0*H0*abs(y)*fgrow*xibbar * (1 + alpha*xibbar)
        if flowmod == 'mock':
            # NB mock vmean(r) is already a los velocity
            vmean = np.interp(math.log10(r), flow_lgr, flow_vmean)

        v = H0 * (pi - y) - vmean
        if vmod == 'exp':
            f = math.exp(-root2*abs(v)/sigma) / (root2*sigma)
        if vmod == 'gauss':
            f = math.exp(-v**2/(2*sigma**2)) / (root2pi*sigma)

        if what == 'f':
            return f

        if what == 'err':
            return xi_err**2*f
        if 'jack' in what:
            ijack = int(what[4:])
            xi, err = xir.interp(r, jack=ijack, log=interplog)
        return (1 + xi) * f

    def xi_pred(sigma, xir, rp, pivals, ijack=-1, sig_min=10,
                epsabs=1e-5, epsrel=1e-5, check_int=False):
        """Convolution of xi(r) with f(v) (Zehavi+2002 eqn 14)."""

        npi = len(pivals)
        xip = np.zeros(npi)
        xip_jack = np.zeros((npi, xir.njack))

        for ipi in xrange(npi):
            pi = pivals[ipi]

            # Set integration limits to +/- 5 sigma from peak in f(v)
            ymin = -6*sigma/H0
            ymax = pi + 6*sigma/H0
            points = (0, pi)
            if check_int:
                # Check that f(v) integrates to unity
                res = scipy.integrate.quad(
                    integrand, ymin, ymax,
                    args=(sigma, xir, rp, pi, 'f'),
                    epsabs=epsabs, epsrel=epsrel, points=points)
                res = H0*res[0]
                if (res < 0.95 or res > 1.05):
                    print(sigma, rp, pi, res)
                    pdb.set_trace()
            if ijack > -1:
                what = 'jack{}'.format(ijack)
            else:
                what = 'xi'
            res = scipy.integrate.quad(
                integrand, ymin, ymax,
                args=(sigma, xir, rp, pi, what),
                epsabs=epsabs, epsrel=epsrel, points=points)
            xip[ipi] = H0*res[0] - 1

            if pred_err == 'jack':
                for ijack in range(xir.njack):
                    res = scipy.integrate.quad(
                        integrand, ymin, ymax,
                        args=(sigma, xir, rp, pi, 'jack{}'.format(ijack)),
                        epsabs=epsabs, epsrel=epsrel, points=points)
                    xip_jack[ipi, ijack] = H0*res[0] - 1
        return xip, xip_jack

    def xi_chi2(lgsigma, xi2, cov, xir, pibins, irp, rp, ijack=0):
        """Chi2 from model fit (Zehavi+2002 eqn 18)."""
        xip, xip_jack = xi_pred(
            10**lgsigma, xir, rp, xi2.pi[pibins, irp], ijack)
#        delta = xi2.jack[pibins, irp, :] - xis_jack
#        cov = Cov(delta, xir.err_type)
#        pdb.set_trace()
        if logfit:
            chi2 = cov.chi2(
                np.log1p(np.fmax(-0.99, xi2.est[pibins, irp, ijack])),
                np.log1p(np.fmax(-0.99, xip)), neig)
        else:
            chi2 = cov.chi2(xi2.est[pibins, irp, ijack], xip, neig)
        return chi2

    # Main routine starts here
    if plot_file:
        pdf = PdfPages(plot_file)
    else:
        pdf = None

    print(infile)
    xi2 = xi_req(infile, 'xi2', binning=binning, pi_lim=pi_max, rp_lim=rp_max)
    xi2.cov = Cov(xi2.est[:, :, 1:], xi2.err_type)
    w_p = xi2.w_p(pi_lim=pi_max, rp_lim=rp_max)
    if xirmod == 'w_p_fit' or flowmod == 'jsd':
        gamma, gamma_err, r0, r0_err, ic, gamma_jack, r0_jack = w_p.fit(
            fitRange, ax=None, ic_rmax=ic_rmax, neig=0)
        fgrow = 0.5
        alpha = 1.2 - 0.65*gamma
    if xirmod == 'w_p_fit':
        xir = w_p
#        xir.est = (r0/xir.sep)**gamma
#        xir.jack = (r0_jack/np.tile(xir.sep, (xir.njack, 1)).T)**gamma_jack
        xir.est = (([r0] + r0_jack)/np.tile(
            xir.sep, (xir.njack+1, 1)).T)**([gamma] + gamma_jack)
        xir.cov = Cov(xir.est[:, 1:], xir.err_type)
    if xirmod == 'xir_meas':
        xir = w_p.xir()
    if xirmod not in ('w_p_fit', 'xir_meas'):
        # Read xi(r) from mock catalogue with cosmological redshifts
        xir = xi_req(xirmod, 'xis', pi_lim=pi_max)

    # Plot xi(r) estimates
    plt.clf()
    ax = plt.subplot(111)
    xir.plot(ax)
    for ijack in range(xir.njack+1):
        xir.plot(ax, ijack)
    plt.loglog(basex=10, basey=10, nonposy='clip')
    plt.xlabel(r'$r\ [h^{-1} {\rm Mpc}]$')
    plt.ylabel(r'$\xi(r)$')
    util.pdfsave(pdf)

    # Read mean los peculiar velocity vmean(r) for mocks
    if os.path.isfile(flowmod):
        data = np.load(flowmod)
        flow_lgr = np.log10(data['sep'])
        flow_vmean = data['vlt']
        flowmod = 'mock'
        plt.clf()
        plt.plot(flow_lgr, flow_vmean)
        plt.xlabel(r'$\log\ r\ [h^{-1} {\rm Mpc}]$')
        plt.ylabel(r'$\bar{v}_\parallel(r)\ [{\rm km s}^{-1}]$')
        util.pdfsave(pdf)

    nrp = xi2.nrp
    sigma = np.zeros(nrp)
    sig_lo = np.zeros(nrp)
    sig_hi = np.zeros(nrp)
    vfit = np.zeros(nrp)
    vfit_lo = np.zeros(nrp)
    vfit_hi = np.zeros(nrp)
    npi = min(xi2.npi, int((pi_max_fit - xi2.pimin)/xi2.pistep))
    print(nrp, npi, 'rp, pi bins')
    xi_mod = np.zeros((npi, nrp))
    chi_red = np.ones(nrp)
    sigbins = np.arange(50, 1000, 50)

    # Fit to each rp bin
    rp = np.zeros(nrp)
    for irp in xrange(nrp):
        pibins = np.arange(npi)
        use = xi2.galpairs[:npi, irp, 0] > 0
        pibins = pibins[use]
        if neig in (0, 'all'):
            nfit = len(pibins)
        else:
            nfit = neig
        if len(pibins) > 0:
            rp[irp] = np.ma.average(xi2.rp[pibins, irp],
                                    weights=xi2.galpairs[pibins, irp, 0])
            r = rp[irp]

            # Covariance matrix and eigenvalues for this slice of xi2
            if logfit:
                cov = Cov(np.log1p(np.fmax(-0.99, xi2.est[pibins, irp, 1:])),
                          xi2.err_type)
                label = 'log(1 + xi)'
            else:
                cov = Cov(xi2.est[pibins, irp, 1:], xi2.err_type)
                label = 'xi'
            cov.plot(norm=True, label=label)
            util.pdfsave(pdf)

            # Grid search to find approx sigma
            chi2 = [xi_chi2(np.log10(sig), xi2, cov, xir, pibins, irp, r)
                    for sig in sigbins]
            sigmin = sigbins[np.argmin(chi2)]

            out = scipy.optimize.fmin(
                xi_chi2, np.log10(sigmin),
                args=(xi2, cov, xir, pibins, irp, r),
                xtol=0.001, ftol=0.001,
                maxfun=10000, maxiter=10000, full_output=1, disp=0)
            lgsig = out[0][0]
            sigma[irp] = 10**lgsig
            if nfit > 1:
                chi_red[irp] = out[1]/(nfit-1)

            if err_est == 'jack':
                sig_jack = np.zeros(njack)
                for ijack in xrange(njack):
                    out = scipy.optimize.fmin(
                        xi_chi2, lgsig,
                        args=(xi2, cov, xir, pibins, irp, r, ijack),
                        xtol=0.001, ftol=0.001,
                        maxfun=10000, maxiter=10000, full_output=1, disp=0)
                    sig_jack[ijack] = 10**out[0][0]
                sig_err = jack_err(sig_jack, xi2.err_type)
                sig_lo[irp] = sig_err
                sig_hi[irp] = sig_err
            else:
                # Find upper and lower 1-sigma errors on sigma
                # nsig=2 since xi_chi2 returns chi2 not -ln L
                lo, hi = util.like_err(
                    xi_chi2, lgsig, limits=(0, 3),
                    args=(xi2, cov, xir, pibins, irp, r), nsig=2.0)
                sig_lo[irp] = sigma[irp] - 10**(lgsig-lo)
                sig_hi[irp] = 10**(lgsig+hi) - sigma[irp]

    #        print xi2.pi[pibins, irp]
    #        print xi2.est[pibins, irp]
    #        print xi2.cov.sig[pibins, irp]

            # Plot covariance matrices
            xi_mod[pibins, irp], xip_jack = xi_pred(
                sigma[irp], xir, rp[irp], xi2.pi[pibins, irp])
#            delta = xi2.jack[pibins, irp, :] - xip_jack
#            plt.clf()
#            ax = plt.subplot(131)
#            cov = Cov(xi2.jack[pibins, irp, :], xir.err_type)
#            cov.plot(ax=ax, label='xi2')
#            ax = plt.subplot(132)
#            cov = Cov(xip_jack, xir.err_type)
#            cov.plot(ax=ax, label='xi model')
#            ax = plt.subplot(133)
#            cov = Cov(delta, xir.err_type)
#            cov.plot(ax=ax, label='xi2 - model')
#            util.pdfsave(pdf)
#            cov.plot_eig()
#            util.pdfsave(pdf)

            if plot_int:
                # Plot integrand (1 + xi(r) * f(v) at given pi
                for ipi in pibins:
                    pi = xi2.pi[ipi, irp]
                    ymin = -6*sigma[irp]/H0
                    ymax = pi + 6*sigma[irp]/H0
                    plt.clf()
                    util.fnplot1(integrand, ymin, ymax,
                                 args=(sigma[irp], xir, rp[irp], pi))
                    util.fnplot1(integrand, ymin, ymax,
                                 args=(sigma[irp], xir, rp[irp], pi, 'f'))
                    plt.xlabel('y')
                    plt.ylabel('(1 + xi(r) * f(v)')
                    plt.title(r'$r_\perp = {:5.2f}, r_\parallel = {:5.2f}$'.format(
                        rp[irp], pi))
                    util.pdfsave(pdf)

#            try:
            plt.clf()
            plt.subplot(121)
            plt.plot(sigbins, chi2)
            ymax = np.amax(chi2)
            if ymax > 0:
                plt.semilogy(basey=10, nonposy='clip')
            plt.xlabel(r'$\sigma$ [km/s]')
            plt.ylabel(r'$\chi^2$')
            plt.title(r'$r_\perp = {:5.2f},\ \chi^2_\nu = {:5.2f}$'.format(
                rp[irp], chi_red[irp]))

            xi_mod_err = jack_err(xip_jack, err_type=xir.err_type)
            plt.subplot(122)
            plt.errorbar(xi2.pi[pibins, irp], xi2.est[pibins, irp, 0],
                         xi2.cov.sig[pibins, irp], fmt='o', capthick=1)
            plt.errorbar(xi2.pi[pibins, irp], xi_mod[pibins, irp],
                         xi_mod_err, fmt='s', capthick=1)
            _, ymax = plt.ylim()
            if binning < 2:
                plt.semilogy(basey=10, nonposy='clip')
            else:
                plt.loglog(basex=10, basey=10, nonposy='clip')
            plt.xlabel(r'$r_\parallel$ [Mpc/h]')
            plt.ylabel(r'$\xi$')
            plt.title(r'$\sigma = {:5.2f} -{:5.2f} +{:5.2f}$'.format(
                sigma[irp], sig_lo[irp], sig_hi[irp]))
            util.pdfsave(pdf)
#            if sigma[irp] < 1:
#                pdb.set_trace()
#            except:
#                print('Error plotting chi2 or xi',
#                      chi2, xi2.est[pibins, irp, 0])

    if flowmod == 'fit':
        outdict = {'rp': rp, 'chi_red': chi_red,
                   'sigma': sigma, 'sig_lo': sig_lo, 'sig_hi': sig_hi,
                   'vfit': vfit, 'vfit_lo': vfit_lo, 'vfit_hi': vfit_hi}
    else:
        outdict = {'rp': rp, 'sigma': sigma, 'sig_lo': sig_lo,
                   'sig_hi': sig_hi, 'chi_red': chi_red}
    pickle.dump(outdict, open(outfile, 'w'))
#    print 'sigma:', sigma
#    print 'sig_lo:', sig_lo
#    print 'sig_hi:', sig_hi

    if flowmod == 'fit':
        plt.clf()
        plt.subplot(311)
        plt.errorbar(rp, sigma, (sig_lo, sig_hi), fmt='o', capthick=1)
        plt.semilogx(basex=10, nonposy='clip')
        plt.ylabel(r'$\sigma_{12}(r_\bot)\ [{\rm km s}^{-1}]$')
        plt.subplot(312)
        plt.errorbar(rp, vfit, (vfit_lo, vfit_hi), fmt='o', capthick=1)
        plt.semilogx(basex=10, nonposy='clip')
        plt.ylabel(r'$\overline{v}(r_\bot)\ ([\rm km s}^{-1}]$')
        plt.subplot(313)
        plt.plot(rp, chi_red, 'o')
        plt.loglog(basex=10, basey=10, nonposy='clip')
        plt.ylabel(r'$\chi^2/\nu$')
        plt.xlabel(r'$r_\bot\ [{\rm h}^{-1} {\rm Mpc}]$')
        util.pdfsave(pdf)
    else:
        plt.clf()
        plt.subplot(211)
        plt.errorbar(rp, sigma, (sig_lo, sig_hi), fmt='o', capthick=1)
        plt.ylim(0, 1500)
        try:
            plt.semilogx(basex=10, nonposy='clip')
        except:
            pass
        plt.ylabel(r'$\sigma_{12}(r_\bot)\ [{\rm km s}^{-1}]$')
        plt.subplot(212)
        plt.plot(rp, chi_red, 'o')
        try:
            plt.loglog(basex=10, basey=10, nonposy='clip')
        except:
            pass
        plt.ylabel(r'$\chi^2/\nu$')
        plt.xlabel(r'$r_\bot\ [{\rm h}^{-1} {\rm Mpc}]$')
        util.pdfsave(pdf)

    pibins = np.arange(npi)
    plt.clf()
    plt.subplot(131)
    dat = xi2.est[pibins, :, 0]
    logdat = np.zeros((npi, nrp)) + lg_xi_range[0]
    pos = dat > 0
    logdat[pos] = np.log10(dat[pos])
    plt.imshow(logdat, aspect=1, vmin=lg_xi_range[0], vmax=lg_xi_range[1])
    plt.xlabel(r'$r_\perp$ bin')
    plt.ylabel(r'$r_\parallel$ bin')
    plt.title('Data')

    plt.subplot(132)
    dat = xi_mod
    logdat = np.zeros((npi, nrp)) + lg_xi_range[0]
    pos = dat > 0
    logdat[pos] = np.log10(dat[pos])
    plt.imshow(logdat, aspect=1, vmin=lg_xi_range[0], vmax=lg_xi_range[1])
    plt.xlabel(r'$r_\perp$ bin')
    plt.ylabel(r'$r_\parallel$ bin')
    plt.title('Model')

    plt.subplot(133)
    dat = (xi2.est[pibins, :, 0] - xi_mod) / xi2.cov.sig[pibins, :]
    plt.imshow(dat, aspect=1, vmin=-3, vmax=3)
    plt.xlabel(r'$r_\perp$ bin')
    plt.ylabel(r'$r_\parallel$ bin')
    plt.title('Residual (+- 3 sigma)')
    util.pdfsave(pdf)
    if pdf:
        pdf.close()


def pvd_stream_test(infile='xi_z_obs_av.dat', pi_max=40, rp_max=100,
                    fitRange=(0.1, 10), ic_rmax=0, neig=0, flowmod='mock',
                    xirmod='mock'):
    """Test varying parameters for streaming model PVD from mocks."""
    for pi_max_fit in (10, 20, 30, 40):
        outfile = 'mock_stream_pi_max_fit_{}.dat'.format(pi_max_fit)
        pvd_stream(infile, outfile, pi_max=pi_max, pi_max_fit=pi_max_fit,
                   flowmod=flowmod, xirmod=xirmod)


def pvd_stream_plot(direct='pvd_vol.npz',
                    infiles=('pvd_stream_vol_measflow.dat',
                             'pvd_stream_vol.dat', 'pvd_beta_neig0.dat'),
                    rplim=(0.01, 10), siglim=(0, 1000), alpha=0.1,
                    plot_file=None, plot_size=def_plot_size):
    """Plot pvd_stream results (sigma only)."""

    plt.clf()
    data = np.load(direct)
    plt.plot(data['sep'], data['slp_exp'], 'k-', label='Direct')
    plt.fill_between(data['sep'], data['slp_exp'] - data['slp_exp_err'],
                     data['slp_exp'] + data['slp_exp_err'],
                     facecolor='k', alpha=alpha)
    try:
        filelist = glob.glob(infiles)
    except:
        filelist = infiles

    print(filelist)
    iplot = 0
    for infile in filelist:
        dict = pickle.load(open(infile, 'r'))
        if 'beta' in dict:
            print('beta, gamma, r0: {:4.2f} {:4.2f} {:4.1f}'.format(
                dict['beta'], dict['gamma'], dict['r0']))
#        print 'sigma:', dict['sigma']
#        print 'sig_lo:', dict['sig_lo']
#        print 'sig_hi:', dict['sig_hi']
        plt.errorbar(dict['rp'], dict['sigma'],
                     (dict['sig_lo'], dict['sig_hi']),
                     fmt=symb_list[iplot], label=infile, capthick=1)
        iplot += 1
    plt.xlim(rplim)
    plt.ylim(siglim)
    plt.semilogx(basex=10, nonposy='clip')
    plt.xlabel(r'$r_\bot\ [h^{-1} {\rm Mpc}]$')
    plt.ylabel(r'$\sigma_{12}\ ([\rm km/s}]$')
#    plt.legend()
    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def pvd_stream_plot3(direct='pvd_av.npz',
                    infiles=('mock_stream_flow_mock.dat',
                             'mock_stream_flow_jsd.dat'),
                    siglim=(0, 1000), vlim=None, chilim=(0.1, 500),
                    plot_file=None, plot_size=(5, 8)):
    """Plot pvd_stream results, one panel each for sigma, vmean, and chi2."""

    plt.clf()
    fig, axes = plt.subplots(3, 1, sharex=True, num=1)
    fig.subplots_adjust(hspace=0, wspace=0)
    data = np.load(direct)
    axes[0].errorbar(data['sep'], data['slp_exp'], data['slp_exp_err'],
                     fmt=symb_list[0], label='Direct', capthick=1)
    axes[1].errorbar(data['sep'], data['vlp'], data['vlp_err'],
                     fmt=symb_list[0], label='Direct', capthick=1)
    try:
        filelist = glob.glob(infiles)
    except:
        filelist = infiles

    print(filelist)
    iplot = 0
    for infile in filelist:
        iplot += 1
        dict = pickle.load(open(infile, 'r'))
#        print 'sigma:', dict['sigma']
#        print 'sig_lo:', dict['sig_lo']
#        print 'sig_hi:', dict['sig_hi']
        axes[0].errorbar(dict['rp'], dict['sigma'],
                         (dict['sig_lo'], dict['sig_hi']),
                         fmt=symb_list[iplot], label=infile, capthick=1)
        try:
            axes[1].errorbar(dict['rp'], dict['vfit'],
                             (dict['vfit_lo'], dict['vfit_hi']),
                             fmt=symb_list[iplot], label=infile, capthick=1)
        except:
            pass
        axes[2].plot(dict['rp'], dict['chi_red'],
                     symb_list[iplot], label=infile)
    ax = axes[0]
    ax.semilogx(basex=10, nonposy='clip')
    ax.set_ylabel(r'$\sigma_{12}\ [{\rm km/s}]$')
    ax.set_ylim(siglim)
    ax = axes[1]
    ax.semilogx(basex=10, nonposy='clip')
    ax.set_ylabel(r'$\overline{v}_{12}\ [{\rm km/s}]$')
    ax.set_ylim(vlim)
    ax = axes[2]
    ax.loglog(basex=10, basey=10, nonposy='clip')
    ax.set_ylim(chilim)
    ax.set_xlabel(r'$r_\bot\ [h^{-1} {\rm Mpc}]$')
    ax.set_ylabel(r'$\chi^2/\nu$')
#    plt.legend()
    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def pvd_beta_samples(intemp='xi_{}_{}_{}.dat',
                     outtemp='pvd_beta_{}_{}_{}.dat',
                     param='lum_c', Mlimits=None, neig=0, err_est='jack'):
    """Calc PVD via beta model for several subsamples."""

    if Mlimits is None:
        if 'lum' in param:
            Mlimits = def_mag_limits
        else:
            Mlimits = def_mass_limits

    nsamp = len(Mlimits) - 1
    for i in range(nsamp):
        print(intemp.format(param, Mlimits[i], Mlimits[i+1]))
        pvd_beta_fit(intemp.format(param, Mlimits[i], Mlimits[i+1]),
                     outtemp.format(param, Mlimits[i], Mlimits[i+1]),
                     neig=neig, err_est=err_est)


def pvd_beta_mock_samples(intemp='xi_{}_{}_{}.dat',
                          outtemp='pvd_beta_ind_{}_{}_{}.dat',
                          param='lum_c', Mlimits=None, neig=0, err_est=None):
    """Calc PVD via beta model for each mock xi(rp,pi), then average,
    for several subsamples."""

    if Mlimits is None:
        if 'lum' in param:
            Mlimits = def_mag_limits
        else:
            Mlimits = def_mass_limits

    nsamp = len(Mlimits) - 1
    for i in range(nsamp):
        infile = intemp.format(param, Mlimits[i], Mlimits[i+1])
        print(infile)
        beta_list = []
        sigma_list = []
        for ijack in range(1, 27):
            fitpars = pvd_beta_fit(infile, None, ijack=ijack, neig=neig,
                                   err_est=err_est)
            beta_list.append(fitpars['beta'])
            sigma_list.append(fitpars['sigma'])
            pdb.set_trace()
        fitpars['beta'] = np.mean(beta_list)
        fitpars['beta_err'] = np.std(beta_list)
        fitpars['sigma'] = np.mean(sigma_list)
        fitpars['beta_lo'] = np.std(sigma_list)
        fitpars['beta_hi'] = np.std(sigma_list)
        outfile = outtemp.format(param, Mlimits[i], Mlimits[i+1])
        pickle.dump(fitpars, open(outfile, 'w'))


def pvd_beta_vl_samples(intemp='xi_{}_{}_{}.dat',
                        outtemp='pvd_beta_{}_{}_{}.dat',
                        param='lum-18_c', err_est='jack',
                        Mlimits=(-23, -22, -21, -20, -19, -18), neig=0):
    """Calc PVD via beta model for several subsamples."""

    if Mlimits is None:
        if 'lum' in param:
            Mlimits = def_mag_limits
        else:
            Mlimits = def_mass_limits

    nsamp = len(Mlimits) - 1
    for i in range(nsamp):
        print(intemp.format(param, Mlimits[i], Mlimits[i+1]))
        pvd_beta_fit(intemp.format(param, Mlimits[i], Mlimits[i+1]),
                     outtemp.format(param, Mlimits[i], Mlimits[i+1]),
                     neig=neig, err_est=err_est)


def pvd_beta_mock_vol(qsub=False):
    """Run pvd_beta on mock volume-limited sample."""

    neig = 0
#    xirmod = 'xir_meas'
    err_est = 'jack'
    logfit = 0
    binning = 2
    infile = 'xi_vol.dat'
    plotfile = None
#    for neig in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'full'):
#    for neig in (8,):
    for beta in (0.45,):
        for xirmod in ('xir_meas', 'w_p_fit'):
            outfile = 'pvd_beta_{}_{}.dat'.format(beta, xirmod)
            if qsub:
                python_commands = [
                   "import corr",
                   """corr.pvd_beta_fit('{}', '{}', beta={},
                                    pi_max=40, binning={}, neig={},
                                    err_est='{}', xirmod='{}'
                                    logfit={}, plot_file={})""".format(
                                    infile, outfile, beta, binning, neig,
                                    err_est, xirmod, logfit, plotfile)]
                print(python_commands)
                util.apollo_job(python_commands)
            else:
                pvd_beta_fit(infile, outfile, beta,
                             pi_max=40, binning=binning, neig=neig,
                             err_est=err_est, xirmod=xirmod, logfit=logfit,
                             plot_file=plotfile)


def pvd_beta_fit(infile, outfile, beta='fit', ijack=0, pi_max=40, rp_max=100,
                 fitRange=(0.01, 5), ic_rmax=0, binning=2, neig=0,
                 err_est='jack', lg_xi_range=(-2, 2), ylim=200, logfit=1,
                 xirmod='xir_meas', epsabs=1e-5, epsrel=1e-5, plot_file=None):
    """Find best-fit beta-parameter in convolved beta model."""

    def chi2(beta, r0, gamma, xi2, ijack=0, binning=2, neig=0, logfit=0):
        """Return total chi2 from convolved beta model fit."""
        fitpars = pvd_beta(beta, xir, r0, gamma, xi2, ijack=ijack,
                           neig=neig, logfit=logfit, err_est=None,
                           binning=binning, plots=False)
        print(beta, fitpars['chi2tot'])
#        pdb.set_trace()
        return fitpars['chi2tot']

    # Read and plot observed correlation functions
    xi2 = xi_req(infile, 'xi2', binning=binning, pi_lim=pi_max, rp_lim=rp_max)
    w_p = xi2.w_p(pi_lim=pi_max, rp_lim=rp_max)
    xir = w_p.xir()
    if xirmod == 'xir_meas':
        # xibar and xibbar integrals (Hawkins+ App A)
        # Exclude bins where xi is negative or s/n < 1
        use = xir.est[:, ijack] > xir.cov.sig
        r = xir.sep[use]
        nbin = len(r)
        if nbin == 0:
            print('No reliable xi(r) measurements, using power-law fit')
            xirmod = 'w_p_fit'
        else:
            xib = np.zeros(nbin)
            xibb = np.zeros(nbin)
            #  treat xi(r) as a piecewise power law
            x = np.log10(r)
            y = np.log10(xir.est[use, ijack])
            for ir in range(nbin-1):
                m = (y[ir+1] - y[ir]) / (x[ir+1] - x[ir])
                c = y[ir] - m*x[ir]
                gamma = -m
                A = 10**c
                if ir == 0:
                    #  extrapolate first two bins to zero separation
                    xib_int = r[ir]**(3-gamma) * A/(3-gamma)
                    xibb_int = r[ir]**(5-gamma) * A/(5-gamma)
                    xib[ir] = xib_int * 3/r[ir]**3
                    xibb[ir] = xibb_int * 5/r[ir]**5
                xib_int += (r[ir+1]**(3-gamma) - r[ir]**(3-gamma)) * A/(3-gamma)
                xibb_int += (r[ir+1]**(5-gamma) - r[ir]**(5-gamma)) * A/(5-gamma)
                xib[ir+1] = xib_int * 3/r[ir+1]**3
                xibb[ir+1] = xibb_int * 5/r[ir+1]**5
            xir.r = r
            xir.xi0 = xir.est[use, ijack]
            xir.xi2 = xir.xi0 - xib
            xir.xi4 = xir.xi0 + 2.5*xib - 3.5*xibb

    plt.clf()
    ax = plt.subplot(111)
    w_p.plot(ax, ijack)
    print(w_p.est[:, ijack])
    if np.max(w_p.est[:, ijack]) > 0:
        plt.loglog(basex=10, basey=10, nonposy='clip')
    plt.xlabel(r'$r_\bot\ [h^{-1} {\rm Mpc}]$')
    plt.ylabel(r'$w_p(r_\bot)$')
    gamma, gamma_err, r0, r0_err, ic, gamma_jack, r0_jack = w_p.fit(
        fitRange, jack=ijack, ax=ax, ic_rmax=ic_rmax, neig=neig, logfit=logfit)
    plt.show()
    if r0 == 0.0:
        print('Aborting due to ill-determined xi_r(r)')
        return

    plt.clf()
    ax = plt.subplot(111)
    xir.plot(ax)
    if xirmod == 'xir_meas':
        plt.plot(r, 3*r0**gamma/r**3/(3-gamma)*r**(3-gamma), label='xib_pl')
        plt.plot(r, xib, label='xib')
        plt.plot(r, xibb, label='xibb')
        plt.plot(r, xir.xi2, label='xi2')
        plt.plot(r, xir.xi4, label='xi4')
    plt.loglog(basex=10, basey=10, nonposy='clip')
    plt.xlabel(r'$r [h^{-1} {\rm Mpc}]$')
    plt.ylabel(r'$\xi(r)$')
    plt.legend()
    plt.show()

    plt.clf()
    ax = plt.subplot(111)
    xir.plot(ax)
    if xirmod == 'xir_meas':
        plt.plot(r, xib, label='xib')
        plt.plot(r, xibb, label='xibb')
        plt.plot(r, xir.xi2, label='xi2')
        plt.plot(r, xir.xi4, label='xi4')
    plt.semilogx(basex=10)
    plt.xlabel(r'$r [h^{-1} {\rm Mpc}]$')
    plt.ylabel(r'$\xi(r)$')
    plt.legend()
    plt.show()

    if xirmod != 'xir_meas':
        xir = None
    if beta == 'fit':
        out = scipy.optimize.fmin(
            chi2, 0.5, args=(r0, gamma, xi2, ijack, binning, neig, logfit),
            xtol=0.01, ftol=1, maxfun=10000,
            maxiter=10000, full_output=1, disp=0)
        beta = out[0][0]
    fitpars = pvd_beta(beta, xir, r0, gamma, xi2, ijack=ijack, neig=neig,
                       logfit=logfit, err_est=err_est, binning=binning,
                       plots=True)
    if outfile:
        pickle.dump(fitpars, open(outfile, 'w'))
    rp = fitpars['rp']
    sigma = fitpars['sigma']
    sig_lo = fitpars['sig_lo']
    sig_hi = fitpars['sig_hi']
    chi_red = fitpars['chi_red']
    plt.clf()
    plt.subplot(211)
    plt.errorbar(rp, sigma, (sig_lo, sig_hi), fmt='o', capthick=1)
    plt.ylim(0, 1500)
    try:
        plt.semilogx(basex=10, nonposy='clip')
    except:
        pass
    plt.ylabel(r'$\sigma_{12}(r_\bot)\ [{\rm km s}^{-1}]$')
    plt.subplot(212)
    plt.plot(rp, chi_red, 'o')
    try:
        plt.loglog(basex=10, basey=10, nonposy='clip')
    except:
        pass
    plt.ylabel(r'$\chi^2/\nu$')
    plt.xlabel(r'$r_\bot\ [{\rm h}^{-1} {\rm Mpc}]$')
    plt.show()

    npi = min(xi2.npi, int((pi_max - xi2.pimin)/xi2.pistep))
    pibins = np.arange(npi)
    plt.clf()
    plt.subplot(131)
    dat = xi2.est[pibins, :, ijack]
    logdat = np.zeros((npi, xi2.nrp)) + lg_xi_range[0]
    pos = dat > 0
    logdat[pos] = np.log10(dat[pos])
    plt.imshow(logdat, aspect=1, vmin=lg_xi_range[0], vmax=lg_xi_range[1])
    plt.xlabel(r'$r_\perp$ bin')
    plt.ylabel(r'$r_\parallel$ bin')
    plt.title('Data')

    plt.subplot(132)
    dat = fitpars['xi_mod']
    logdat = np.zeros((npi, xi2.nrp)) + lg_xi_range[0]
    pos = dat > 0
    logdat[pos] = np.log10(dat[pos])
    plt.imshow(logdat, aspect=1, vmin=lg_xi_range[0], vmax=lg_xi_range[1])
    plt.xlabel(r'$r_\perp$ bin')
    plt.ylabel(r'$r_\parallel$ bin')
    plt.title('Model')

    plt.subplot(133)
    xi2.cov = Cov(xi2.est[pibins, :, 1:], xi2.err_type)
    dat = (xi2.est[pibins, :, 0] - fitpars['xi_mod']) / xi2.cov.sig
    plt.imshow(dat, aspect=1, vmin=-3, vmax=3)
    plt.xlabel(r'$r_\perp$ bin')
    plt.ylabel(r'$r_\parallel$ bin')
    plt.title('Residual (+- 3 sigma)')
    plt.show()
    return fitpars


def pvd_beta(beta, xir, r0, gamma, xi2, ijack=0, pi_max=40, rp_max=100,
             neig=0, err_est='jack', lg_xi_range=(-2, 2), ylim=200, logfit=0,
             binning=2, plots=False):
    """Calculate PVD from beta model (Hawkins+2003 Sec 4.1)."""

    def Lorentz(sigma, k):
        """Lorentzian corresponding to exponential PVD sigma'.
           k should be in (km/s)^-1."""
        return 1.0/(1 + 2*(math.pi*sigma*k)**2)

    def xi_pred(sigma, irp):
        """Convolution of beta model xi(r_par) with f(v)."""
#        if binning < 2:
#            pvd_ft = Lorentz(sigma, k_pi/H0)
#        else:
        pvd = xi_beta.pistep*H0/(root2*sigma)*np.exp(-np.abs(
            root2*xi_betar.pic*H0/sigma))
#            pvd = np.exp(-np.abs(root2*pivals*H0/sigma))/(root2*sigma)
#            pvd_ft = np.fft.fftshift(np.fft.fft(pvd))
#        xip = np.abs(np.fft.ifft(xi_beta_ft[:, irp]*pvd_ft))
        xip = np.convolve(xi_betar.est[:, irp, ijack], pvd, 'same')
        xip = xip[xi_betar.est.shape[0]/2:]
        if binning == 2:
            # Rebin xi_model into same r_par bins as xi2
            xip = np.interp(xi2.pic, xi_beta.pic, xip)
        return xip

    def xi_chi2(lgsigma, xi2, cov, pibins, irp, rp, ijack=0):
        """Chi2 from model fit (Hawkins+2003 eqn 20)."""
        sigma = 10**lgsigma
        xip = xi_pred(sigma, irp)[pibins]
        if logfit:
            chi2 = cov.chi2(
                np.log1p(np.fmax(-0.99, xi2.est[pibins, irp, ijack])),
                np.log1p(np.fmax(-0.99, xip)), neig)
        else:
            chi2 = cov.chi2(xi2.est[pibins, irp, ijack], xip, neig)
        pdb.set_trace()
        return chi2

    # Generate linear infall model
    if binning < 2:
        # With linear r_par binning, xi_beta has same shape as xi2
        xi_beta = copy.deepcopy(xi2)
        xi_beta.est = np.zeros((xi2.npi, xi2.nrp, 1))
    else:
        # With log r_par binning, make xi_beta linearly spaced in r_par
        # with bin size pimin
        beta_npi = int(10**xi2.pimax / 10**xi2.pimin)
        xi_beta = Xi2d(xi2.nrp, xi2.rpmin, xi2.rpmax, beta_npi,
                       0.0, 10**xi2.pimax, 0, None)
    xi_beta.beta_model(beta, xir, r0, gamma)
    nrp = xi2.nrp
    xi_betar = xi_beta.reflect(0)

    if plots:
        plt.clf()
        ax = plt.subplot(111)
        xi2.plot(ax, jack=ijack, prange=(-2, 3), mirror=0)
        plt.show()
        plt.clf()
        ax = plt.subplot(111)
        xi_beta.plot(ax, prange=(-2, 3), mirror=0)
        plt.show()

        # Plot beta model convolved with fixed velocity dispersion
        sigma = 600.0
        pvd = xi_beta.pistep*H0/(root2*sigma)*np.exp(-np.abs(
            root2*xi_betar.pic*H0/sigma))
        print('pvd sum = ', pvd.sum())
        plt.clf()
        plt.plot(pvd)
        plt.xlabel(r'$r_\parallel$ bin')
        plt.ylabel('pvd')
        plt.show()

        conv = np.zeros((xi_betar.npi, nrp))
        for irp in xrange(nrp):
            conv[:, irp] = np.convolve(xi_betar.est[:, irp, 0], pvd, 'same')
        xi_model = copy.deepcopy(xi_beta)
        xi_model.est[:, :, 0] = conv[conv.shape[0]/2:, :]

        plt.clf()
        ax = plt.subplot(111)
        xi_model.plot(ax, prange=(-2, 3), mirror=0)
        plt.show()

        if binning == 2:
            # Rebin xi_model into same r_par bins as xi2
            xim = xi_model.est
            xi_model = copy.deepcopy(xi2)
            for irp in xrange(xi2.nrp):
                xi_model.est[:, irp, 0] = np.interp(
                    xi2.pic, xi_beta.pic, xim[:, irp, 0])
            plt.clf()
            ax = plt.subplot(111)
            xi_model.plot(ax, prange=(-2, 3), mirror=0)
            plt.show()

        # Check that xi2, xi_beta and xi_model give consistent w_p
        plt.clf()
        ax = plt.subplot(111)
        wp = xi2.w_p(rp_max, pi_max)
        wp.plot(ax, jack=ijack)
        wp = xi_beta.w_p(rp_max, pi_max)
        wp.plot(ax)
        wp = xi_model.w_p(rp_max, pi_max)
        wp.plot(ax)
        ax.loglog(basex=10, basey=10, nonposy='clip')
        plt.show()

    # Fit PVD to each rp bin
    chi2tot = 0
    sigma = np.zeros(nrp)
    sig_lo = np.zeros(nrp)
    sig_hi = np.zeros(nrp)
    npi = min(xi2.npi, int((pi_max - xi2.pimin)/xi2.pistep))
#    print nrp, npi, 'rp, pi bins'
    xi_mod = np.zeros((npi, nrp))
    chi_red = np.ones(nrp)
    sigbins = np.arange(50, 1000, 50)
    rp = np.zeros(nrp)
    for irp in xrange(nrp):
        pibins = np.arange(npi)
        use = ((xi2.galpairs[:npi, irp, ijack] > 0) *
               (xi2.est.mask[:npi, irp, ijack] is False))
        pibins = pibins[use]
        if neig in (0, 'all', 'full'):
            nfit = len(pibins)
        else:
            nfit = neig
        pdb.set_trace()
        if len(pibins) > 0:
            rp[irp] = np.ma.average(xi2.rp[pibins, irp],
                                    weights=xi2.galpairs[pibins, irp, ijack])
            r = rp[irp]

            # Covariance matrix and eigenvalues for this slice of xi2
            if logfit:
                cov = Cov(np.log1p(np.fmax(-0.99, xi2.est[pibins, irp, 1:])),
                          xi2.err_type)
                label = 'log(1 + xi)'
            else:
                cov = Cov(xi2.est[pibins, irp, 1:], xi2.err_type)
                label = 'xi'
            if plots:
                cov.plot(norm=True, label=label)
                plt.show()
                cov.plot_eig()

            # Grid search to find approx sigma
            chi2 = [xi_chi2(np.log10(sig), xi2, cov, pibins, irp, r, ijack)
                    for sig in sigbins]
            sigmin = sigbins[np.argmin(chi2)]

            out = scipy.optimize.fmin(
                xi_chi2, np.log10(sigmin),
                args=(xi2, cov, pibins, irp, r, ijack),
                xtol=0.001, ftol=0.001,
                maxfun=10000, maxiter=10000, full_output=1, disp=0)
            lgsig = out[0][0]
            chi2tot += out[1]
            if math.isnan(chi2tot):
                pdb.set_trace()
            sigma[irp] = 10**lgsig
            if nfit > 1:
                chi_red[irp] = out[1]/(nfit-1)

            if err_est == 'jack':
                sig_jack = np.zeros(njack)
                for ijack in xrange(njack):
                    out = scipy.optimize.fmin(
                        xi_chi2, lgsig,
                        args=(xi2, cov, pibins, irp, r, ijack),
                        xtol=0.001, ftol=0.001,
                        maxfun=10000, maxiter=10000, full_output=1, disp=0)
                    sig_jack[ijack] = 10**out[0][0]
                sig_err = jack_err(sig_jack, xi2.err_type)
                sig_lo[irp] = sig_err
                sig_hi[irp] = sig_err
            if err_est == 'dchi':
                # Find upper and lower 1-sigma errors on sigma
                # nsig=2 since xi_chi2 returns chi2 not -ln L
                lo, hi = util.like_err(
                    xi_chi2, lgsig, limits=(0, 3),
                    args=(xi2, cov, pibins, irp, r, ijack), nsig=2.0)
                sig_lo[irp] = sigma[irp] - 10**(lgsig-lo)
                sig_hi[irp] = 10**(lgsig+hi) - sigma[irp]

    #        print xi2.pi[pibins, irp]
    #        print xi2.est[pibins, irp]
    #        print xi2.cov.sig[pibins, irp]

            sig = sigma[irp]
            xi_mod[:, irp] = xi_pred(sig, irp)
#            if binning < 2:
#                pvd_ft = Lorentz(sig, k_pi/H0)
#            else:
#                pvd = np.exp(-np.abs(root2*pivals*H0/sig))/(root2*sig)
#                pvd_ft = np.fft.fftshift(np.fft.fft(pvd))
            if plots:
                plt.clf()
    #            plt.subplot(121)
    #            plt.plot(k_pi, pvd_ft)
    #            plt.xlabel(r'$k_\parallel$ (h/Mpc)')
    #            plt.ylabel(r'FT[$P(v)$]')
    #            plt.title(r'$\sigma = {:5.2f} -{:5.2f} +{:5.2f}$'.format(
    #                sig, sig_lo[irp], sig_hi[irp]))
    #            plt.subplot(122)
                plt.errorbar(xi2.pi[pibins, irp], xi2.est[pibins, irp, ijack],
                             cov.sig, fmt='o', capthick=1)
                plt.plot(xi_beta.pic, xi_beta.est[:, irp, 0])
                plt.plot(xi2.pi[pibins, irp], xi_mod[pibins, irp])
    #            _, ymax = plt.ylim()
    #            if ymax > 0:
    #                plt.semilogy(basey=10, nonposy='clip')
                if binning == 2:
                    plt.loglog(basex=10, basey=10, nonposy='clip')
                else:
                    plt.semilogy(basey=10, nonposy='clip')
                plt.xlabel(r'$r_\parallel$ [Mpc/h]')
                plt.ylabel(r'$\xi$')
                plt.title(r'$r_\perp = {:5.2f}, \sigma = {:5.2f} -{:5.2f} +{:5.2f}$'.format(
                    r, sig, sig_lo[irp], sig_hi[irp]))
    #            if sig < 1:
    #                pdb.set_trace()
                plt.show()

    fitpars = {'beta': beta, 'r0': r0, 'gamma': gamma, 'neig': neig,
               'logfit': logfit, 'chi2tot': chi2tot, 'rp': rp, 'sigma': sigma,
               'sig_lo': sig_lo, 'sig_hi': sig_hi, 'chi_red': chi_red,
               'xi_mod': xi_mod}
#    pdb.set_trace()
    return fitpars


def pvd_scale_dep(est='beta', param='lum_c', ylims=(0, 1300), landscape=False,
                  alpha=0.1, plot_file_temp='pvd_{}_{}.pdf'):
    """Plot sigma(r) in mass/lum bins with mock and Li+2006 comparison.
    est is one of 'beta', 'stream' or 'k'."""

    if est in ('beta', 'stream'):
        xlims = (0.01, 5)
        xlabel = r'$r_\bot\ [h^{-1}\ {\rm Mpc}]$'
    else:
        xlims = (0.1, 20)
        xlabel = r'$k\ [h\ {\rm Mpc}^{-1}]$'
    if landscape:
        plot_size = (8, 5)
        sa_left = 0.12
        sa_bot = 0.1
    else:
        plot_size = (5, 8)
        sa_left = 0.18
        sa_bot = 0.08

    intemp = 'pvd_{}_{}_{}_{}.dat'
    mocktemp = gama_data + 'mocks/v1/pvd_{}_ind_{}_{}_{}.dat'
    dmocktemp = gama_data + 'mocks/v1/pvd_{}_{}.npz'
    plot_file = plot_file_temp.format(est, param)
    if 'lum' in param:
        Mlimits = def_mag_limits
        Liroot = 'pvd-L{:4.1f}{:4.1f}.dat'
        labroot = r'${} < M_r < {}$'
    else:
        Mlimits = def_mass_limits
        Liroot = 'pvd-M-{:04.1f}-{:04.1f}.dat'
        labroot = r'${} < \lg(M/M_\odot) < {}$'

    npanel = len(Mlimits) - 1  # drop last two mass bins
    nrow, ncol = util.two_factors(npanel, landscape)
    plt.clf()
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
    axes = axes.ravel()
    fig.subplots_adjust(left=sa_left, bottom=sa_bot, hspace=0, wspace=0)
    fig.text(0.55, 0.02, xlabel, ha='center', va='center')
    fig.text(0.06, 0.5, r'$\sigma_{12}\ ({\rm km\ s}^{-1})$',
             ha='center', va='center', rotation='vertical')
    for i in range(npanel):
        Mlo, Mhi = Mlimits[i], Mlimits[i+1]
        label = labroot.format(Mlo, Mhi)
        ax = axes[i]

        # GAMA estimate
        dat = pickle.load(open(intemp.format(est, param, Mlo, Mhi), 'r'))
        if est == 'k':
            sep = dat['Pobs'].k
        else:
            sep = dat['rp']
        if 'sig_lo' in dat:
            sig_lo = dat['sig_lo']
            sig_hi = dat['sig_hi']
        else:
            sig_lo = dat['sigma'] - dat['sigma_range'][0]
            sig_hi = dat['sigma_range'][1] - dat['sigma']
        ax.errorbar(sep, dat['sigma'], (sig_lo, sig_hi), fmt=symb_list[0],
                    capthick=1)

        # Direct mock estimate
        data = np.load(dmocktemp.format(Mlo, Mhi))
        sep = data['sep']
        if est == 'k':
            sep = 2*math.pi/sep
        ax.plot(sep, data['slp_exp'], 'b-', label='Direct (r_perp)')
        ax.fill_between(sep, data['slp_exp'] - data['slp_exp_err'],
                        data['slp_exp'] + data['slp_exp_err'],
                        facecolor='b', alpha=alpha)
        # Indirect mock estimate
        dat = pickle.load(open(mocktemp.format(est, param, Mlo, Mhi), 'r'))
        if est == 'k':
            sep = dat['Pobs'].k
        else:
            sep = dat['rp']
        if 'sig_lo' in dat:
            sig_lo = dat['sig_lo']
            sig_hi = dat['sig_hi']
        else:
            sig_lo = dat['sigma'] - dat['sigma_range'][0]
            sig_hi = dat['sigma_range'][1] - dat['sigma']
        ax.errorbar(sep, dat['sigma'], (sig_lo, sig_hi), fmt=symb_list[1],
                    capthick=1)
        if est == 'k':
            try:
                k, sig, sig_err = Li_w_p(Liroot, Mlo, Mhi)
                plot = sig > 0
                ax.errorbar(k[plot], sig[plot], sig_err[plot],
                            fmt=symb_list[2], capthick=1)
            except:
                pass
        ax.semilogx(basex=10, nonposy='clip')
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
#        ax.set_yticks(np.arange(0, 900, 200))
        if est == 'beta':
            label += r'; $\beta = {:4.2f}$'.format(dat['beta'])
        ax.text(0.05, 0.9, label, transform=ax.transAxes)
#    plt.tight_layout(h_pad=0, w_pad=0)
    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def pvd_type_dep(est='beta', param='lum_c',
                 rbins=((0, 3), (3, 6), (6, 9), (9, 12)),
                 intemp=('pvd_{}_{}_{}_{}.dat',
                         gama_data + 'mocks/v1/pvd_{}_{}_{}_{}.dat',
                         'pvd_{}_{}_{}_{}.dat'),
                 mag_limits=def_mag_limits, mass_limits=def_mass_limits,
                 ylims=(0, 1300), plot_file='pvd_type_{}.pdf', landscape=False):
    """Type dependence of PVD."""

    plot_file = plot_file.format(param)
    if 'lum' in param:
        Mlimits = mag_limits
        xlims = (-23, -15.1)
        xlabel = r'$M_r - 5 \lg h$'
        Liroot = 'pvd-L{:4.1f}{:4.1f}.dat'
    else:
        Mlimits = mass_limits
        xlims = (mass_limits[0], mass_limits[-2])
        xlabel = r'$\lg(M/M_\odot h)$'
        Liroot = 'pvd-M-{:04.1f}-{:04.1f}.dat'

    npanel = len(rbins)
    nest = len(intemp)
    nm = len(Mlimits) - 1
    mags = np.zeros(nm)
    sigma = np.zeros((npanel, nest, nm))
    sigerr = np.zeros((npanel, nest, nm))
    label = [' ', ' ', ' ', ' ']
    labroot = r'${:4.2f} < r_\bot / h^{{-1}} {{\rm Mpc}} < {:4.2f}$'
    for im in range(nm):
        Mlo, Mhi = Mlimits[im], Mlimits[im+1]
        mags[im] = 0.5 * (Mlo + Mhi)
        for iest in range(nest):
            if (iest == 2):
                infile = intemp[iest].format(est, 'lum-18_c', Mlo, Mhi)
            else:
                infile = intemp[iest].format(est, param, Mlo, Mhi)
            try:
                dat = pickle.load(open(infile, 'r'))
                rp = dat['rp']
                sig = dat['sigma']
                sig_lo = dat['sig_lo']
                sig_hi = dat['sig_hi']
                sige = sig_lo + sig_hi
                for ip in range(npanel):
                    ilo = rbins[ip][0]
                    ihi = rbins[ip][1]
                    var = sige[ilo:ihi]**2
                    sigma[ip, iest, im] = np.average(sig[ilo:ihi],
                                                     weights=1.0/var)
                    sigerr[ip, iest, im] = (np.sum(1.0/var))**-0.5
                    if iest == 0 and im == 0:
                        label[ip] = labroot.format(rp[ilo], rp[ihi])
            except:
                pass
    if landscape:
        ncol = npanel
        nrow = 1
        plot_size = (8, 4)
    else:
        nrow = npanel
        ncol = 1
        plot_size = (4, 8)

    plt.clf()
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, num=1)
    fig.subplots_adjust(left=0.2, bottom=0.05, hspace=0, wspace=0)
    fig.text(0.55, 0.0, xlabel, ha='center', va='center')
    fig.text(0.06, 0.5, r'$\sigma_{12}\ ({\rm km\ s}^{-1})$',
             ha='center', va='center', rotation='vertical')
    for ip in range(npanel):
        ax = axes[ip]
        for iest in range(nest):
            idx = sigma[ip, iest, :] > 0
            ax.errorbar(mags[idx], sigma[ip, iest, idx], sigerr[ip, iest, idx],
                        fmt=sl_list[iest], capthick=1)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
#        ax.set_yticks(np.arange(0, 900, 200))
        ax.text(0.05, 0.85, label[ip], transform=ax.transAxes)
    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


# -----------------
# Utility routines
# -----------------

# Correlation function estimators
def ls(counts, i2d):
    """Landay-Szalay autocorrelation function."""
    if i2d < 0:
        ggn = counts['gg'].pcn
        grn = counts['gr'].pcn
        rrn = counts['rr'].pcn
    else:
        ggn = counts['gg'].pc2_list[i2d]['pcn']
        grn = counts['gr'].pc2_list[i2d]['pcn']
        rrn = counts['rr'].pc2_list[i2d]['pcn']
    est = np.ma.masked_invalid((ggn - 2*grn) / rrn + 1)
    return est


def lsx(counts, i2d):
    """Landay-Szalay cross-correlation function."""
    if i2d < 0:
        Ggn = counts['Gg'].pcn
        Grn = counts['Gr'].pcn
        grn = counts['gr'].pcn
        rrn = counts['rr'].pcn
    else:
        Ggn = counts['Gg'].pc2_list[i2d]['pcn']
        Grn = counts['Gr'].pc2_list[i2d]['pcn']
        grn = counts['gr'].pc2_list[i2d]['pcn']
        rrn = counts['rr'].pc2_list[i2d]['pcn']
    est = np.ma.masked_invalid((Ggn - Grn - grn) / rrn + 1)
    return est


def dpx(counts, i2d):
    """Davis-Peebles cross-correlation function."""
    if i2d < 0:
        Ggn = counts['Gg'].pcn
        Grn = counts['Gr'].pcn
    else:
        Ggn = counts['Gg'].pc2_list[i2d]['pcn']
        Grn = counts['Gr'].pc2_list[i2d]['pcn']
    est = np.ma.masked_invalid(Ggn/Grn - 1)
    return est


def xi_req(infile, key, binning=1, pi_rebin=1, rp_rebin=1,
           pi_lim=100, rp_lim=100):
    """Return requested xi with its covariance, according to key and binning,
    from pair-count files.
    Binning = 0 for lin-lin, 1 for log-lin, 2 for log-log."""
    ggfile = infile.replace('xi', 'gg')
    grfile = infile.replace('xi', 'gr')
    rrfile = infile.replace('xi', 'rr')
    gg = PairCounts(ggfile, pi_rebin=pi_rebin, rp_rebin=rp_rebin)
    gr = PairCounts(grfile, pi_rebin=pi_rebin, rp_rebin=rp_rebin)
    rr = PairCounts(rrfile, pi_rebin=pi_rebin, rp_rebin=rp_rebin)
    counts = {'gg': gg, 'gr': gr, 'rr': rr}
    xi = Xi()
    return xi.est(counts, ls, key=key, binning=binning, pi_lim=pi_lim,
                  rp_lim=rp_lim)


def jack_err(ests, err_type='jack'):
    """
    Jackknife or mock error from array of estimates.
    """
    err = np.std(ests)
    nest = len(ests)
    if err_type == 'jack':
        err *= math.sqrt(nest-1)

    # Set large error if variance is zero (something wrong)
    # if err <= 0.0: err = 99.0
    return err


def jack_cov(ests):
    """Covariance matrix from jackknife estimates."""

    ncorr = len(ests[0])-1
    cov = ncorr*np.cov(ests, bias=1)
    return cov

# Routines for reading comparison data


def Farrow_w_p(param, lo, hi):
    """Farrow et al 2015 w_p."""
    files = {'vlum_c-23-22': 'farrow15--23.0-mag0.1--22.0-0.014-z-0.5-wprp.dat',
             'vlum_c-22-21': 'farrow15--22.0-mag0.1--21.0-0.01-z-0.38-wprp.dat',
             'vlum_c-21-20': 'farrow15--21.0-mag0.1--20.0-0.01-z-0.26-wprp.dat',
             'vlum_c-20-19': 'farrow15--20.0-mag0.1--19.0-0.01-z-0.176-wprp.dat',
             'vlum_c-19-18': 'farrow15--19.0-mag0.1--18.0-0.01-z-0.116-wprp.dat'}
    key = '{}{}{}'.format(param, lo, hi)
#    print(key)
    if key in files:
        fname = (os.environ['HOME'] +
                 '/Documents/Research/corrdata/Farrow2015/' + files[key])
        print(fname)
        data = np.loadtxt(fname)
        r_p = data[:, 0]
        w_p = data[:, 1]
        w_p_err = data[:, 2]
        return r_p, w_p, w_p_err
    else:
        return None


def Li_w_p(template, lo, hi):
    """Li et al 2006 w_p."""
    try:
        fname = (os.environ['HOME'] +
                 '/Documents/Research/corrdata/Li2006/data/' +
                 template.format(lo, hi))
        print(fname)
        data = np.loadtxt(fname)
        r_p = data[:, 0]
        w_p = data[:, 1]
        w_p_err = data[:, 2]
        return r_p, w_p, w_p_err
    except:
        return None

def Zehavi_w_p(lo, hi):
    """Zehavi et al 2011 w_p."""
    if lo < -23 or lo > -18:
        return None

    fname = (os.environ['HOME'] + 
             '/Documents/Research/corrdata/Zehavi2011/Table7.txt')
    wcol = 47 + 2*lo
    ecol = wcol + 1
    data = np.loadtxt(fname)
    r_p = data[:,0]
    w_p = data[:,wcol]
    w_p_err = data[:,ecol]
    return r_p, w_p, w_p_err


def P_k_cube(infile='Gonzalez_r21.txt', nbins=512, nk=16, rmax=500.0):
    """Power spectrum for mock data cube."""

    assert (nbins <= 512)  # memory limit
    assert (rmax <= 500)  # sim box size
    bins = np.linspace(0, rmax, nbins+1)
    data = np.loadtxt(infile, delimiter=',', skiprows=1)
    counts, edges = np.histogramdd(data[:, :3], (bins, bins, bins))
    ps3d = np.abs(np.fft.fftn(counts))**2
#    print counts.shape, ps3d.shape
#    util.plot_3d_array(counts)
#    util.plot_3d_array(ps3d)
    rbin = rmax/nbins
    ksamp = np.fft.fftfreq(nbins, rbin)
#    print ksamp
    ksq = ksamp**2
    k_3d = np.sqrt(ksq[:, None, None] + ksq[:, None] + ksq)
#    print k_3d
#    pdb.set_trace()
    kmin = ksamp[1]
    kmax = abs(ksamp[nbins//2])
    klims = np.linspace(kmin, kmax, nk+1)
    kmean = np.zeros(nk)
    Pmean = np.zeros(nk)
    Perr = np.zeros(nk)
    for ik in range(nk):
        sel = (klims[ik] <= k_3d) * (k_3d < klims[ik+1])
        kmean[ik] = np.mean(k_3d[sel])
        Pmean[ik] = np.mean(ps3d[sel])
        Perr[ik] = scipy.stats.sem(ps3d[sel])

#    print kmin, kmax, klims
#    print kmean, Pmean, Perr
    plt.clf()
    plt.errorbar(kmean, Pmean, yerr=Perr, capthick=1)
#    plt.loglog(basex=10, basey=10, nonposy='clip')
    plt.semilogy(basey=10, nonposy='clip')
    plt.xlabel(r'$k\ [h\ {\rm Mpc}^{-1}]$')
    plt.ylabel(r'$P(k)$')
    plt.show()


def read_stats(infile):
    """Return ngal and zcut from specified file."""
    f = open(infile, 'r')
    info = eval(f.readline())
    args = f.readline().split()
    f.close()
    ngal = int(args[0])
    try:
        zcut = info['zcut']
    except:
        zcut = 0
    return ngal, zcut


def gama_stats():
    """Return mean ngal and zcut for various samples."""
    for samp in ('-16_-15', '-17_-16', '-18_-17', '-19_-18', '-20_-19',
                 '-21_-20', '-22_-21', '-23_-22'):
        infile = 'gal_lum_c_{}.dat'.format(samp)
        ngal, zcut = read_stats(infile)
        print(samp, 'ngal, zcut = {} {}'.format(ngal, zcut))


def mock_stats(filetemp='gal_lum_c_{}_{}_{:02d}.dat',
               samples=('-16_-15', '-17_-16', '-18_-17', '-19_-18', '-20_-19',
                        '-21_-20', '-22_-21', '-23_-22')):
    """Return mock mean ngal for various samples."""
    for samp in samples:
        nlist = []
        for ireal in xrange(26):
            ngal = 0
            for reg in xrange(1, 4):
                infile = filetemp.format(samp, reg, ireal)
                ng, zcut = read_stats(infile)
                ngal += ng
            nlist.append(ngal)
        ngal = np.mean(nlist)
        nerr = np.std(nlist)
        print(samp, 'ngal = {} +- {}'.format(round(ngal), round(nerr)))
