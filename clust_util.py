# Classes and utilities for galaxy clustering

import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import scipy.stats
import subprocess

from astropy.table import Table

import gal_sample as gs
import util

# Global parameters
gama_data = os.environ['GAMA_DATA']
job_script = os.environ['PYTHONPATH'] + '/apollo_job.sh'
qsub_xi_cmd = 'qsub ' + job_script + ' $BIN/xi '

def_binning = (-2, 2, 20, 0, 100, 100)
def_theta_max = 12

# J3 pars are (gamma, r0, rmax), respectively.
# If rmax < 0.1, then the first parameter is assumed to be J3 itself.
# This is necessary to correctly normalise cross-correlations.
#def_J3_pars = (1.84, 5.59, 30)
def_J3_pars = (2000, 0, 0)

def_plot_size = (5, 3.5)
xlabel = {'xis': r'$s\ [h^{-1}{\rm Mpc}]$',
          'xi2': '',
          'w_p': r'$r_p\ [h^{-1}{\rm Mpc}]$',
          'xir': r'$r\ [h^{-1}{\rm Mpc}]$', 'bias': r'$M_r$'}
ylabel = {'xis': r'$\xi(s)$', 'xi2': '',
          'w_p': r'$w_p(r_p)$', 'xir': r'$\xi(r)$',
          'bias': r'$b(M) / b(M^*)$'}

# Directory to save plots
plot_dir = '.'


def test(Mlim=-20, key='w_p', binning=1, nfac=1,
         pi_max=40, xlimits=(0.01, 100), calc=1):
    """Test basic functionality of sample selection, correlation function
    calculation and plotting on a small data sample."""

    galfile = 'gal_test.dat'
    ranfile = 'ran_test.dat'
    xifile = 'xi_test.dat'
    if calc:
        samp = gs.GalSample()
        samp.read_gama()
        samp.add_vmax()
        samp.vol_limit(Mlim)
        xi_sel(samp, galfile, ranfile, xifile, nfac, set_vmax=False,
               mask=gama_data+'/mask/zcomp.ply', run=1)

    # Plot the results
    plt.clf()
    ax = plt.gca()
    xi = xi_req(xifile, key, binning=binning, pi_lim=pi_max)
    if key == 'xi2':
        xi.plot(ax, cbar=False)
    else:
        xi.plot(ax)
    plt.loglog(basex=10, basey=10)
    plt.xlabel('r')
    plt.ylabel('w_p(r)')
    plt.show()


def xtest(Mlim=(-21, -20.5, -20.45), key='w_p', binning=1, nfac=1,
          pi_max=40, xlimits=(0.01, 100), pi_lim=100, rp_lim=100):
    """Test cross-correlation using two luminosity-selected samples."""

    xi_cmd = '$BIN/xi '
    selcol = 'ABSMAG_R'
    samp = gs.GalSample()
    samp.read_gama()
    samp.vis_calc()
    samp.add_vmax()
    sel_dict = {selcol: (Mlim[0], Mlim[1])}
    xi_sel(samp, 'gal1.dat', 'ran1.dat', '', nfac, sel_dict=sel_dict,
           set_vmax=False, mask=gama_data+'/mask/zcomp.ply', run=0, J3wt=True)
    sel_dict = {selcol: (Mlim[1], Mlim[2])}
#    samp.vol_limit(Mlim[2])  # 2nd sample is now volume-limited
    xi_sel(samp, 'gal2.dat', 'ran2.dat', '', nfac, sel_dict=sel_dict,
           set_vmax=False, mask=gama_data+'/mask/zcomp.ply', run=0, J3wt=True)

    cmd = xi_cmd + 'gal1.dat gg_1_1.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'gal1.dat ran1.dat gr_1_1.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'ran1.dat rr_1_1.dat'
    subprocess.call(cmd, shell=True)

    cmd = xi_cmd + 'gal2.dat gg_2_2.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'gal2.dat ran2.dat gr_2_2.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'ran2.dat rr_2_2.dat'
    subprocess.call(cmd, shell=True)

    cmd = xi_cmd + 'gal1.dat gal2.dat gg_1_2.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'gal1.dat ran2.dat gr_1_2.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'gal2.dat ran1.dat gr_2_1.dat'
    subprocess.call(cmd, shell=True)
    cmd = xi_cmd + 'ran1.dat ran2.dat rr_1_2.dat'
    subprocess.call(cmd, shell=True)

    # Plot the results

    gg = PairCounts('gg_1_1.dat')
    Gg = PairCounts('gg_1_2.dat')
    GG = PairCounts('gg_2_2.dat')

    gr = PairCounts('gr_1_1.dat')
    gR = PairCounts('gr_1_2.dat')
    Gr = PairCounts('gr_2_1.dat')
    GR = PairCounts('gr_2_2.dat')

    rr = PairCounts('rr_1_1.dat')
    Rr = PairCounts('rr_1_2.dat')
    RR = PairCounts('rr_2_2.dat')

    counts = {'gg': gg, 'Gg': Gg, 'GG': GG,
              'gr': gr, 'gR': gR, 'Gr': Gr, 'GR': GR,
              'rr': rr, 'Rr': Rr, 'RR': RR}
    xi = Xi()
    xi0 = xi.est(counts, dpx, key=key, binning=binning,
                 pi_lim=pi_lim, rp_lim=rp_lim)
    xi1 = xi.est(counts, lsx, key=key, binning=binning,
                 pi_lim=pi_lim, rp_lim=rp_lim)
    xi2 = xi.est(counts, lsx2r, key=key, binning=binning,
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


def xi_sel(samp, galfile, ranfile, xifile, nfac, sel_dict=None, set_vmax=False,
           mask=gama_data+'/mask/zcomp.ply', run=0, binning=def_binning,
           theta_max=def_theta_max, maxran=1000000, J3wt=True):
    """Output selected galaxy and random samples for xi.c."""

    samp.t['den'] = np.zeros(len(samp.t))
    samp.t['Vmax_out'] = np.ones(len(samp.t))
    samp.t['weight'] = np.ones(len(samp.t))

    samp.select(sel_dict)
    ts = samp.tsel()
    use = samp.t['use']
    zgal = ts['z']
    ngal = len(zgal)
    if nfac*ngal > maxran:
        nfac = round(maxran/ngal)
        print('nfac changed to ', nfac)
    rancat = gs.GalSample(zlimits=samp.zlimits)
    rancat.cosmo = samp.cosmo
    rancat.t = Table()
    if samp.vol_limited:
        J3_pars = (0, 0, 0)
        nran = nfac*ngal
        zran = util.ran_fun(
                samp.vol_ev, samp.zlimits[0], samp.zlimits[1], nran)
        rancat.t['den'] = np.zeros(nran)
    else:
        if J3wt:
            J3_pars = def_J3_pars
        else:
            J3_pars = (0, 0, 0)
        ndupe = np.array(np.round(
                nfac*ts['Vmax_raw'] / ts['Vmax_dec']).astype(np.int32))
#        print(ndupe)
        nran = np.sum(ndupe)
        zran = np.zeros(nran)
        j = 0
        for i in range(ngal):
            ndup = ndupe[i]
            zran[j:j+ndup] = util.ran_fun(
                    samp.vol_ev, np.array(ts['zlo'])[i],
                    np.array(ts['zhi'])[i], ndup)
            j += ndup

        # Density for minimum variance weighting
        zran_hist, bin_edges = np.histogram(zran, bins=50,
                                            range=samp.zlimits)
        zran_hist = zran_hist*ngal/nran  # Note cannot use *= due int->float
        zstep = bin_edges[1] - bin_edges[0]
        zcen = bin_edges[:-1] + 0.5*zstep
        V_int = samp.area/3.0 * samp.cosmo.dm(bin_edges)**3
        Vbin = np.diff(V_int)
        denbin = zran_hist/Vbin

        samp.t['den'][use] = np.interp(ts['z'], zcen, denbin)
        rancat.t['den'] = np.interp(zran, zcen, denbin)
#    rancat.t['den'] = np.zeros(nran)

    rancat.t['use'] = np.ones(nran, dtype=np.bool)
    rancat.t['z'] = zran
    # Vmax weighting
    if set_vmax:
        samp.t['Vmax_out'][use] = ts['Vmax_dec']
        rancat.t['Vmax_out'] = ts['Vmax_dec']
    else:
        samp.t['Vmax_out'][use] = np.ones(ngal)
        rancat.t['Vmax_out'] = np.ones(nran)
    samp.t['weight'][use] = np.ones(ngal)
    rancat.t['weight'] = np.ones(nran)

    samp.info = {'file': galfile, 'njack': gs.njack,
                 'set_vmax': set_vmax, 'err_type': 'jack'}
    if sel_dict:
        samp.info.update(sel_dict)
    samp.xi_output(galfile, binning, theta_max, J3_pars)
    print(ngal, ' galaxies written to', galfile)

    # Generate random coords using ransack
    ranc_file = 'ransack.dat'
    cmd = "$MANGLE_DIR/bin/ransack -r{} {} {}".format(nran, mask, ranc_file)
    subprocess.call(cmd, shell=True)
    data = np.loadtxt(ranc_file, skiprows=1)
    rancat.t['RA'] = data[:, 0]
    rancat.t['DEC'] = data[:, 1]
    rancat.assign_jackknife()
    rancat.info = {'file': ranfile, 'njack': gs.njack,
                   'set_vmax': set_vmax, 'err_type': 'jack'}
    rancat.xi_output(ranfile, binning, theta_max, J3_pars)
    print(nran, ' randoms written to', ranfile)

    if run == 1:
        # Run the clustering code executable in $BIN/xi, compiled from xi.c
        cmd = '$BIN/xi {} {}'.format(galfile, xifile.replace('xi', 'gg', 1))
        subprocess.call(cmd, shell=True)
        cmd = '$BIN/xi {} {} {}'.format(
                galfile, ranfile, xifile.replace('xi', 'gr', 1))
        subprocess.call(cmd, shell=True)
        cmd = '$BIN/xi {} {}'.format(ranfile, xifile.replace('xi', 'rr', 1))
        subprocess.call(cmd, shell=True)
    if run == 2:
        cmd = qsub_xi_cmd + galfile + ' ' + xifile.replace('xi', 'gg', 1)
        subprocess.call(cmd, shell=True)
        cmd = qsub_xi_cmd + ranfile + ' ' + xifile.replace('xi', 'rr', 1)
        subprocess.call(cmd, shell=True)
        cmd = qsub_xi_cmd + galfile + ' ' + ranfile + ' ' + xifile.replace(
                'xi', 'gr', 1)
        subprocess.call(cmd, shell=True)


def ran_z_gen(samp, nfac):
    """Generate random redshifts nfac times larger than gal catalogue."""

    ndupe = np.round(
            nfac * samp.t['V_max'] / samp.t['Vdc_max']).astype(np.int32)

    ngal = len(samp.t)
    nran = np.sum(ndupe)
    zran = np.zeros(nran)
    j = 0
    for i in range(ngal):
        ndup = ndupe[i]
        zran[j:j+ndup] = util.ran_fun(
                samp.vol_ev, samp.t['zlo'][i], samp.t['zhi'][i], ndup)
        j += ndup
    return zran


# -----------------
# Plotting routines
# -----------------

def cat_stats(infile, nskip=3):
    """Simple catalogue statistics."""

    data = np.loadtxt(infile, skiprows=nskip)
    r = (data[:, 0]**2 + data[:, 1]**2 + data[:, 2]**2)**0.5
    print('mean distance:', np.mean(r))
    print('mean weight:', np.mean(data[:, 3]))
    print('mean density:', np.mean(data[:, 4]))
    print('mean Vmax:', np.mean(data[:, 5]))


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


def zhist_one(galfile='gal_0_1.dat', ranfile='ran_1.dat', nbin=50, normed=True):
    """Plot redshift histograms for galaxy & random input files to xi.c."""

    def read_file(infile, nskip=3):
        # Read input file into array and return array of distances
        data = np.loadtxt(infile, skiprows=nskip)
        dist = np.sqrt(np.sum(data[:, 0:3]**2, axis=1))
        return dist

    dg = read_file(galfile)
    rg = read_file(ranfile)
    plt.clf()
    plt.hist((dg, rg), bins=nbin, normed=normed, histtype='step')
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
#    irow, icol = 0, 0

    if outfile:
        fout = open(outfile, 'w')
    else:
        fout = None

    if pl_div and key == 'w_p':
        # Convert power-law parameters from real to projected space
        gamma = pl_div[1]
        r0 = pl_div[0]
        A = (r0**gamma * scipy.special.gamma(0.5) *
             scipy.special.gamma(0.5*(gamma-1)) /
             scipy.special.gamma(0.5*gamma))
        pl_div[1] -= 1
        pl_div[0] = A**(1.0/(gamma-1))
        print(pl_div)

    for panel, ax in zip(panels, axes):
#        try:
#            ax = axes[irow, icol]
#        except:
#            try:
#                ax = axes[irow]
#            except:
#                ax = axes

        files = panel['files']
        i = 0
        for infile in files:
            c = next(ax._get_lines.prop_cycler)['color']
            try:
                xi = xi_req(infile, key, binning=binning, pi_lim=pi_max)
                if key == 'xi2':
                    xi.plot(ax, cbar=False)
                else:
                    xi.plot(ax, jack=jack, color=c, fout=fout, pl_div=pl_div)
                    if fit_range:
                        fit = xi.fit(
                            fit_range, jack=jack, ax=ax, ic_rmax=ic_rmax,
                            neig=neig, color=c)
                i += 1
            except IOError:
                print('Unable to read ', infile)
        comps = panel['comps']
        for comp in comps:
            if comp:
                ax.errorbar(comp[0], comp[1], comp[2], color=c,
                            fmt='s', capthick=1)
                i += 1

        if 'pl_pars' in panel:
            p = panel['pl_pars']
            if p:
                yfit = ((pl_range[0]/p[0])**-p[1], (pl_range[1]/p[0])**-p[1])
                ax.plot(pl_range, yfit, color=c)

        if key != 'xi2':
            if (ylimits[0] <= 0):
                ax.semilogx(basex=10, nonposy='clip')
            else:
                ax.loglog(basex=10, basey=10, nonposy='clip')
            ax.axis(xlimits + ylimits)
        label = panel['label']
        if label:
            ax.text(0.2, 0.9, label, transform=ax.transAxes)

#        icol += 1
#        if icol >= ncol:
#            icol = 0
#            irow += 1
#            if irow >= nrow:
#                irow = 0
    if fout:
        fout.close()
    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(plot_size)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


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
    """Landay-Szalay cross-correlation function with single random catalogue."""
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


def lsx2r(counts, i2d):
    """Landay-Szalay cross-correlation function with two random catalogues."""
    if i2d < 0:
        Ggn = counts['Gg'].pcn
        Grn = counts['Gr'].pcn
        gRn = counts['gR'].pcn
        Rrn = counts['Rr'].pcn
    else:
        Ggn = counts['Gg'].pc2_list[i2d]['pcn']
        Grn = counts['Gr'].pc2_list[i2d]['pcn']
        gRn = counts['gR'].pc2_list[i2d]['pcn']
        Rrn = counts['Rr'].pc2_list[i2d]['pcn']
    est = np.ma.masked_invalid((Ggn - Grn - gRn) / Rrn + 1)
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


class Cat(object):
    """Galaxy or random catalogue."""

    def __init__(self, ra, dec, r, weight=None, den=None, Vmax=None, info=None):
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
        for i in range(self.nobj):
            if njack > 1:
                for jack in range(njack):
                    if (gs.ra_jack[jack] <= self.ra[i] <= gs.ra_jack[jack] + 4):
                        ijack = jack
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
        self.wa = float(args[1])
        self.nb = float(args[2])
        self.wb = float(args[3])
        self.njack = int(args[4])
        self.n2d = int(args[5])

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
#            self.pc[i, :] = map(float, data[1:])
            self.pc[i, :] = [float(data[j]) for j in range(1, len(data))]
        if self.wb > 0:
            self.pcn = self.pc/self.wa/self.wb
        else:
            self.pcn = 2*self.pc/self.wa/(self.wa - 1)

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
#                    pc[j, i, :] = map(float, data[2:])
                    pc[j, i, :] = [float(data[k]) for k in range(2, len(data))]

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

            if self.wb > 0:
                pcn = pc/self.wa/self.wb
            else:
                pcn = 2*pc/self.wa/(self.wa - 1)
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
        self.wa = np.sum([pcs[i].wa for i in ests])
        self.wb = np.sum([pcs[i].wb for i in ests])
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
        self.wa = np.mean([pcs[i].wa for i in ests])
        self.wb = np.mean([pcs[i].wb for i in ests])
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
        print(self.na, self.wa, self.nb, self.wb, self.njack, self.n2d, file=f)

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
#        print(self.sep, self.est[:, jack])
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

        fit_dict = {'gamma': 0, 'gamma_err': 0, 'gamma_jack': 0,
                    'r0': 0, 'r0_err': 0, 'r0_jack': 0,
                    'chisq': 0, 'nu': 0, 'ic': 0}
        idx = ((fit_range[0] < self.sep) * (self.sep < fit_range[1]) *
               (self.galpairs > 0) * np.all(self.est > 0, axis=1))
        if len(self.sep[idx]) < 2:
            print('Insufficient valid bins for fit')
#            pdb.set_trace()
            return fit_dict
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
        fit_dict = {'gamma': gamma, 'gamma_err': gamma_err, 
                    'gamma_jack': gamma_jack,
                    'r0': r0, 'r0_err': r0_err, 'r0_jack': r0_jack,
                    'chisq': chisq, 'nu': nu, 'ic': self.ic}
        return fit_dict
#        return gamma, gamma_err, r0, r0_err, self.ic, gamma_jack, r0_jack

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

    def vdist(self, ijack=0, lgximin=-2, hsmooth=0, neig=0, plots=1):
        """Velocity distribution function via Fourier transform of
        2d correlation function."""

        def vdist_samp(ximap, plots):
            """Velocity distribution for single xi estimate."""

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
        freq, ratio, v, fv = vdist_samp(hann*self.est[:, :, ijack], plots)

        if self.njack > 0:
            ratio_jack = np.zeros((len(fv), self.njack))
            fv_jack = np.zeros((len(fv), self.njack))
            for jjack in xrange(self.njack):
                freq, ratio_jack[:, jjack], v, fv_jack[:, jjack] = vdist_samp(
                    hann*self.est[:, :, jjack+1], 0)
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

        lg_k_min, lg_k_max, nk = -1.0, 1.0, 10
#        lg_k_min, lg_k_max, nk = -1.0, 2, 15
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
                if self.cov > 0:
                    return (obs - model)**2 / self.cov
                else:
                    return 0
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


