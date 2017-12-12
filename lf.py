# LF routines utilising new gal_sample utilities

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pdb
import scipy.special
import gal_sample as gs

# Constants
ln10 = math.log(10)


def lfr(outfile='lfr.dat',
        colname='ABSMAG_R', Mmin=-25, Mmax=-12, nbin=26, zmin=0.002, zmax=0.65,
        clean_photom=1, use_wt=1):
    """r-band LF using density-corrected Vmax."""

    samp = gs.GalSample()
    samp.read_gama()
    samp.add_vmax()
    lf = LF(samp.tsel(), colname)
    lf.plot(finish=True)


def smf(outfile='smf.dat',
        colname='logmstar', Mmin=6, Mmax=12, nbin=24, zmin=0.002, zmax=0.65,
        clean_photom=1, use_wt=1):
    """Stellar mass function using density-corrected Vmax."""

    samp = gs.GalSample()
    samp.read_gama()
    samp.stellar_mass()
    samp.add_vmax()
    lf = LF(samp.tsel(), colname, Mmin=Mmin, Mmax=Mmax, nbin=nbin)
    lf.plot(finish=True)


def blf_test(outfile='blf.dat',
             cols=('ABSMAG_R', 'logmstar'), arange=((-25, -12), (6, 12)),
             bins=(13, 12), zmin=0.002, zmax=0.65, clean_photom=1, use_wt=1):
    """Mr-stellar mass bivariate function using density-corrected Vmax."""

    samp = gs.GalSample()
    samp.read_gama()
    samp.stellar_mass()
    samp.add_vmax()
    lf = LF2(samp.tsel(), cols, bins, arange)
    lf.plot(finish=True)


def bbd_petro(outfile='bbd_petro.dat',
              cols=('ABSMAG_R', 'R_SB_ABS'), arange=((-25, -12), (16, 26)),
              bins=(26, 20), zmin=0.002, zmax=0.65, clean_photom=1, use_wt=1):
    """Petrosian BBD using density-corrected Vmax."""

    samp = gs.GalSample()
    samp.read_gama()
    samp.add_vmax()
    lf = LF2(samp.tsel(), cols, bins, arange)
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
    lf = LF2(samp.tsel(), cols, bins, arange)
    lf.plot(chol_fit=True, finish=True)


def group_lf(mbins=(12.00, 12.34, 12.68, 13.03, 13.37, 13.71, 14.05),
             mdbins=(12, 14, 14.5, 15, 16.5),
             ndbins=(1, 1.8, 2.2, 2.5, 5), nmin=5, nmax=500, edge_min=0.9,
             Mmin=-25, Mmax=-14, nbin=22, colname='ABSMAG_R'):
    """Galaxy LF by group mass."""
#    mbins=(12, 13, 13.5, 14, 16)
    samp = gs.GalSample()
    samp.read_gama()
    samp.add_vmax()
    samp.group_props()
    t = samp.t
    t['log_massden'] = t['log_mass'] - np.log10(math.pi*t['Rad50']**2)
    t['log_numden'] = np.log10(t['Nfof'] / (math.pi*t['Rad50']**2))

    plt.clf()
    sel = np.logical_not(t['log_mass'].mask)
    plt.hist((t['log_mass'][sel],
              t['log_mass'][sel * np.array(t['Nfof'] >= nmin)]),
             bins=12, range=(10, 16))
    plt.xlabel('log (M/M_sun)')
    plt.ylabel('Frequency')
    plt.show()

    plt.clf()
    sel = np.logical_not(t['log_mass'].mask) * np.array(t['Nfof'] >= nmin)
    plt.hist(t['log_massden'][sel])
    plt.xlabel(r'log Mass density [M_sun Mpc$^{-2}$]')
    plt.ylabel('Frequency')
    plt.show()

#    plt.clf()
#    plt.hist(np.log10(t['LumBfunc'][np.logical_not(t['log_mass'].mask)]))
#    plt.xlabel('log Lum')
#    plt.ylabel('Frequency')
#    plt.show()
#
    plt.clf()
    plt.scatter(t['Nfof'] + np.random.random(len(t['Nfof'])) - 0.5,
                t['log_mass'], s=0.1, c=np.log10(t['Nfof']))
    plt.xlabel('Nfof')
    plt.ylabel('log (M/M_sun)')
    plt.semilogx(basex=10)
    plt.show()
#
#    plt.clf()
#    plt.scatter(t['LumBfunc'], t['log_mass'], s=0.1, c=t['Nfof'])
#    plt.xlabel('Lum')
#    plt.ylabel('log mass')
#    plt.semilogx(basex=10)
#    plt.show()

    print(len(t), 'galaxies before reliable group selection')
    sel = (np.array(t['GroupEdge'] > edge_min) *
           np.logical_not(t['log_mass'].mask) *
           np.array(t['Nfof'] >= nmin))
    samp.t = t[sel]
    print(len(samp.t), 'galaxies after reliable group selection')

    plot_samples(samp, 'log_mass', mbins, '{} < log M < {}',
                 outfile='group_lf_mass.txt')
    plot_samples(samp, 'log_massden', mdbins, '{} < log Mden < {}')

    samp.vol_limit(-17.5)
    samp.group_limit(nmin)
    print(len(samp.t), 'galaxies after volume limiting to ', samp.zlim)

    plt.clf()
    plt.hist(samp.t['log_numden'])
    plt.xlabel(r'log Number density [Mpc$^{-2}$]')
    plt.ylabel('Frequency')
    plt.show()

    plt.clf()
    plt.scatter(samp.t['z'], samp.t['log_numden'], s=0.1)
    plt.xlabel(r'Redshift')
    plt.ylabel(r'log Number density [Mpc$^{-2}$]')
    plt.show()

    plot_samples(samp, 'log_numden', ndbins, '{} < log nden < {}')


def plot_samples(samp, selcol, bins, label_template, lfcol='ABSMAG_R',
                 Mmin=-25, Mmax=-14, nbin=22, outfile=None):
    """Plot LF for sub-samples selected by column selcol in given bins."""

    plt.clf()
    ax = plt.gca()
    plt.semilogy(basey=10, nonposy='clip')
    plt.xlabel('M')
    plt.ylabel(r'$\phi(M)$')
    lf_list = []
    if outfile:
        f = open(outfile, 'w')
    for i in range(len(bins)-1):
        sel_dict = {selcol: (bins[i], bins[i+1])}
        label = label_template.format(bins[i], bins[i+1])
        samp.select(sel_dict)
        norm = len(samp.t)/len(samp.tsel())
        lf = LF(samp.tsel(), lfcol, Mmin=Mmin, Mmax=Mmax, nbin=nbin, norm=norm)
        if outfile:
            lf.write(f, label)
        lf.schec_fit()
#        print(lf.alpha, lf.alpha_err, lf.Mstar, lf.Mstar_err,
#              lf.lpstar, lf.lpstar_err)
        lf.plot(ax=ax, label=label)
        lf_list.append(lf)
    if outfile:
        f.close()
    plt.ylim(1e-7, 1)
    plt.legend()
    plt.show()

    plt.clf()
    ax = plt.gca()
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$M^*$')
    for i in range(len(bins)-1):
        lf_list[i].like_cont(ax=ax)
    plt.show()


class LF():
    """LF data and methods."""

    def __init__(self, t, colname, Mmin=-25, Mmax=-12, nbin=26, norm=1,
                 Vmax='Vmax_dec'):
        """Initialise new LF instance from specified table and column."""

        self.Mmin, self.Mmax, self.nbin = Mmin, Mmax, nbin
        bins = np.linspace(Mmin, Mmax, nbin+1)
        self.Mbin = bins[:-1] + 0.5*np.diff(bins)
        absval = t[colname]
        wt = t['cweight']/t[Vmax]
        self.phi, edges = np.histogram(absval, bins, weights=wt)
        self.phi *= norm/np.diff(bins)

        # Jackknife errors
        njack = gs.njack
        self.njack = njack
        self.phi_jack = np.zeros((njack, len(self.phi)))
        for jack in range(njack):
            idx = t['jack'] != jack
            self.phi_jack[jack, :], edges = np.histogram(
                absval[idx], bins, weights=wt[idx])
            self.phi_jack[jack, :] *= float(njack)/(njack-1)/np.diff(bins)
        self.phi_err = norm*np.sqrt((njack-1) * np.var(self.phi_jack, axis=0))

    def write(self, f, label):
        """Output to specified file."""
        print('# ', label, file=f)
        for i in range(len(self.Mbin)):
            print(self.Mbin[i], self.phi[i], self.phi_err[i], file=f)

    def schec_fit(self, schec_guess=(-1, -20, -5)):
        """Schechter function fit."""

        res = scipy.optimize.fmin(
                schec_resid, schec_guess, (self.Mbin, self.phi, self.phi_err),
                xtol=0.001, ftol=0.001, full_output=1, disp=0)
        self.alpha, self.Mstar, self.lpstar = res[0]
        alpha_jack, Mstar_jack, lpstar_jack = [], [], []
        for jack in range(self.njack):
            res = scipy.optimize.fmin(
                    schec_resid, res[0],
                    (self.Mbin, self.phi_jack[jack, :], self.phi_err),
                    xtol=0.001, ftol=0.001, full_output=1, disp=0)
            alpha_jack.append(res[0][0])
            Mstar_jack.append(res[0][1])
            lpstar_jack.append(res[0][2])
        self.alpha_err = np.sqrt((self.njack-1) * np.var(alpha_jack, axis=0))
        self.Mstar_err = np.sqrt((self.njack-1) * np.var(Mstar_jack, axis=0))
        self.lpstar_err = np.sqrt((self.njack-1) * np.var(lpstar_jack, axis=0))

    def like_cont(self, lc_step=32, lc_sigma=4, ax=None):
        """alpha-Mstar likelihood contours."""

        self.chi2map = np.zeros([lc_step, lc_step])

        xmin = self.alpha - lc_sigma*self.alpha_err
        xmax = self.alpha + lc_sigma*self.alpha_err
        dx = (xmax - xmin)/lc_step
        ymin = self.Mstar - lc_sigma*self.Mstar_err
        ymax = self.Mstar + lc_sigma*self.Mstar_err
        dy = (ymax - ymin)/lc_step
        self.lc_limits = [xmin, xmax, ymin, ymax]

        # chi2 minimum
        chi2min = schec_resid((self.alpha, self.Mstar, self.lpstar),
                              self.Mbin, self.phi, self.phi_err)
        self.v = [chi2min + 4, ]
        for ix in range(lc_step):
            al = xmin + (ix+0.5)*dx
            for iy in range(lc_step):
                ms = ymin + (iy+0.5)*dy
                res = scipy.optimize.fmin(
                        lambda lpstar: schec_resid(
                                (al, ms, lpstar),
                                self.Mbin, self.phi, self.phi_err), -2,
                        xtol=0.001, ftol=0.001, full_output=1, disp=0)
                self.chi2map[iy, ix] = res[1]

        if ax:
            c = next(ax._get_lines.prop_cycler)['color']
            ax.contour(self.chi2map, self.v, aspect='auto', origin='lower',
                       extent=self.lc_limits, linestyles='solid', colors=c)

    def plot(self, ax=None, label=None, xlim=None, ylim=None,
             ls='-', finish=False):
        """Plot LF and optionally the Schechter fn fit."""

        if ax is None:
            plt.clf()
            ax = plt.subplot(111)
        c = next(ax._get_lines.prop_cycler)['color']
        ax.errorbar(self.Mbin, self.phi, self.phi_err, fmt='o', color=c,
                    label=label)
        if hasattr(self, 'alpha'):
            x = np.linspace(self.Mmin, self.Mmax, 100)
            y = Schechter(x, self.alpha, self.Mstar, 10**self.lpstar)
            show = y > 1e-10
            ax.plot(x[show], y[show], ls, color=c)
        if xlim:
            ax.xlim(xlim)
        if xlim:
            ax.xlim(xlim)
        if finish:
            ax.semilogy(basey=10, nonposy='clip')
            ax.set_xlabel(r'$M_r$')
            ax.set_ylabel(r'$\phi$')
            plt.show()


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

    def write(self, f, label):
        """Output to specified file."""
        print('# ', label, file=f)
        for i in range(len(self.Mbin)):
            print(self.Mbin[i], self.phi[i], self.phi_err[i], file=f)

    def plot(self, ax=None, label=None, ngmin=5, vmin=-6, vmax=-1.5,
             chol_fit=0, finish=1):
        """Plot bivariate LF."""

        if ax is None:
            plt.clf()
            ax = plt.subplot(111)
        extent = self.arange[0] + self.arange[1]
        log_phi = np.log10(self.phi.T)
        log_phi = np.ma.array(log_phi, mask=np.isnan(log_phi))
        plt.imshow(log_phi, aspect='auto', origin='lower',
                   extent=extent, interpolation='nearest',
                   vmin=vmin, vmax=vmax)
        cb = plt.colorbar()
        cb.set_label(r'$\log_{10} \phi$')

        """Least-squares Choloniewski fn fit to phi(M, mu)."""
        if chol_fit:
            chol_par_name = ('alpha', '   M*', ' phi*', ' beta', '  mu*',
                             'log sigma')
            M = np.tile(self.Mbin1, (len(self.Mbin2), 1))
            mu = np.tile(self.Mbin2, (len(self.Mbin1), 1)).transpose()

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
            phi = self.phi.T
            phi_err = self.phi_err.T
            exclude = self.ngal.T < ngmin
            phi_err[exclude] = 1e6
            use = self.ngal.T >= ngmin
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


def schec_resid(schec_pars, M, phi, phi_err, sigma=0):
    """Return chi^2 residual between binned phi estimate and Schechter fit."""

    alpha, Mstar, lpstar = schec_pars
    fit = Schechter(M, alpha, Mstar, 10**lpstar)
    if sigma > 0:
        scale = sigma/np.mean(np.diff(M))
        ng = int(math.ceil(3*scale))
        gauss = scipy.stats.norm.pdf(np.arange(-ng, ng+1), scale=scale)
        fit = np.convolve(fit, gauss, 'same')

    idx = phi_err > 0
    fc = np.sum(((phi[idx]-fit[idx]) / phi_err[idx])**2)
    return fc


def Schechter(M, alpha, Mstar, phistar):
    L = 10**(0.4*(Mstar-M))
    schec = 0.4*ln10*phistar*L**(alpha+1)*np.exp(-L)
    return schec
