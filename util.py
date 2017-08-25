# Useful utilities

from __future__ import division

import math
import numpy as np
import os
import pdb
import pylab as plt
import scipy.optimize
import scipy.stats
import subprocess

#from astLib import astCalc
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits

# Constants
ln10 = math.log(10)
gama_data = os.environ['GAMA_DATA']


class CosmoLookupOld(object):
    """Distance and volume-element lookup tables.
    NB volume element is per unit solid angle."""

    def __init__(self, H0, omega_l, zRange, nbin=1000):
        c = 3e5
        astCalc.H0 = H0
        astCalc.OMEGA_L = omega_l
        astCalc.OMEGA_M0 = 1 - omega_l
        self._zrange = zRange
        self._z = np.linspace(zRange[0], zRange[1], nbin)
        nz = self._z.size
        self._dm = np.zeros(nz)
        self._dV = np.zeros(nz)
        for i in xrange(nz):
            self._dm[i] = astCalc.dm(self._z[i])
            self._dV[i] = c/H0*self._dm[i]*self._dm[i]/math.sqrt(astCalc.Ez2(self._z[i]))
        self._dist_mod = 5*np.log10((1+self._z) * self._dm) + 25
    def dm(self, z):
        """Transverse comoving distance."""
        return np.interp(z, self._z, self._dm)

    def dl(self, z):
        """Luminosity distance."""
        return (1+z)*np.interp(z, self._z, self._dm)

    def da(self, z):
        """Angular diameter distance."""
        return np.interp(z, self._z, self._dm)/(1+z)

    def dV(self, z):
        """Volume element per unit solid angle."""
        return np.interp(z, self._z, self._dV)

    def dist_mod(self, z):
        """Distance modulus."""
        return np.interp(z, self._z, self._dist_mod)


class CosmoLookup(object):
    """Distance and volume-element lookup tables.
    NB volume element is differential per unit solid angle."""

    def __init__(self, H0, omega_l, zRange, nz=1000):
        cosmo = FlatLambdaCDM(H0=H0, Om0=1-omega_l)
        self._zrange = zRange
        self._z = np.linspace(zRange[0], zRange[1], nz)
        self._dm = cosmo.comoving_distance(self._z)
#        self._dV = cosmo.comoving_volume(self._z)
        self._dV = cosmo.differential_comoving_volume(self._z)
        self._dist_mod = cosmo.distmod(self._z)

    def dm(self, z):
        """Comoving distance."""
        return np.interp(z, self._z, self._dm)

    def dl(self, z):
        """Luminosity distance."""
        return (1+z)*np.interp(z, self._z, self._dm)

    def da(self, z):
        """Angular diameter distance."""
        return np.interp(z, self._z, self._dm)/(1+z)

    def dV(self, z):
        """Volume element per unit solid angle."""
        return np.interp(z, self._z, self._dV)

    def dist_mod(self, z):
        """Distance modulus."""
        return np.interp(z, self._z, self._dist_mod)


def ran_dist(x, p, nran):
    """Generate nran random points according to distribution p(x)"""

    if np.amin(p) < 0:
        print('ran_dist warning: pdf contains negative values!')
    cp = np.cumsum(p)
    y = (cp - cp[0]) / (cp[-1] - cp[0])
    r = np.random.random(nran)
    return np.interp(r, y, x)


def ran_fun(f, xmin, xmax, nran, args=None, nbin=1000):
    """Generate nran random points according to pdf f(x)"""

    x = np.linspace(xmin, xmax, nbin)
    if args:
        p = f(x, *args)
    else:
        p = f(x)
    return ran_dist(x, p, nran)


def ran_fun2(f, xmin, xmax, ymin, ymax, nran, args=(), nbin=1000, pplot=False):
    """Generate nran random points according to 2d pdf f(x,y)"""

    dx = float(xmax - xmin)/nbin
    dy = float(ymax - ymin)/nbin
    x = np.linspace(xmin + 0.5*dx, xmax - 0.5*dx, nbin)
    y = np.linspace(ymin + 0.5*dy, ymax - 0.5*dy, nbin)
    xv, yv = np.meshgrid(x, y)
    p = f(xv, yv, *args)
    if pplot:
        plt.clf()
        plt.imshow(p, aspect='auto', origin='lower',
                   extent=(xmin, xmax, ymin, ymax), interpolation='nearest')
        plt.colorbar()
        plt.draw()
    binno = xrange(nbin * nbin)
    bins = (ran_dist(binno, p, nran)).astype(int)
    j = bins / nbin
    i = bins % nbin
    xran = x[i]
    yran = y[j]

    # Add uniform random offsets to avoid quantization
    xoff = dx * (np.random.random(nran) - 0.5)
    yoff = dy * (np.random.random(nran) - 0.5)

    return xran + xoff, yran + yoff


def ran_fun_test():
    """Test ran_fun"""

    def fun(x, y):
        return np.cos(x)**2 * np.sin(y)**2

    xr, yr = ran_fun2(fun, -5, 5, -5, 5, 10000, pplot=True)
    plt.scatter(xr, yr, 0.1, c='w')
    plt.draw()


def smooth(x, window_len=10, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]

def factor(n):
    """
    Generator for getting factors for a number
    From http://blog.dhananjaynene.com/2009/01/2009-is-not-a-prime-number-a-python-program-to-compute-factors/
    """
    yield 1
    i = 2
    limit = n**0.5
    while i <= limit:
        if n % i == 0:
            yield i
            n = n / i
            limit = n**0.5
        else:
            i += 1
    if n > 1:
        yield n

def prime_factors(n):
    """
    Return the prime factors of the given number.
    From http://wj32.wordpress.com/2007/12/08/prime-factorization-sieve-of-eratosthenes/
    """
    factors = []
    lastresult = n
    
    # 1 is a special case
    if n == 1:
        return [1]
    
    while 1:
        if lastresult == 1:
            break
        
        c = 2
        
        while 1:
            if lastresult % c == 0:
                break
            
            c += 1
        
        factors.append(c)
        lastresult /= c
    
    return factors

def two_factors(n, landscape=False):
    """
    Return smallest pair of integers whose product is n or greater.
    Useful for choosing numbers of rows and columns in multi-panel plots.
    Set landscape=True for ncol >= nrow, default (portrait) is ncol <= nrow.
    """

    # First check if n is a square
    root = math.sqrt(n)
    if root == int(root):
        return int(root), int(root)
    
    pfac = prime_factors(n)
    nfac = len(pfac)
    if nfac == 2:
        if landscape:
            return pfac[0], pfac[1]
        else:
            return pfac[1], pfac[0]

    if nfac > 2:
        fac = reduce(lambda prod, j: prod*pfac[j], xrange(nfac-1), 1), pfac[-1]
        n1, n2 = np.sort(fac)[0], np.sort(fac)[1]
        if landscape:
            return n1, n2
        else:
            return n2, n1

    # Not factorizable
    if n > 10:
        n2 = int(math.ceil(math.sqrt(n)))
        n1 = int(math.ceil(float(n)/n2))
    else:
        n2 = n
        n1 = 1
    if landscape:
        return n1, n2
    else:
        return n2, n1


def poisson_probability(actual, mean):
    # Better to use scipy.stats.poisson.pmf
    
    # naive:   math.exp(-mean) * mean**actual / factorial(actual)

    # iterative, to keep the components from getting too large or small:
    p = math.exp(-mean)
    for i in range(actual):
        p *= mean
        p /= i+1
    return p

def fnplot(func, xmin, xmax, nbin=50, args=()):
    """Plot specified function over given limits."""
    xp = np.linspace(xmin, xmax, nbin)
    yp = func(xp, *args)
    plt.plot(xp, yp)

def fnplot1(func, xmin, xmax, nbin=50, args=()):
    """Plot specified function over given limits for function that can
    only be called with a scalar argument."""
    xp = np.linspace(xmin, xmax, nbin)
    yp = np.zeros(nbin)
    for i in xrange(nbin):
        yp[i] = func(xp[i], *args)
    plt.plot(xp, yp)

def kcheck(file='kcorr.fits', zrange=(0.0, 0.5)):
    """Check polynomial fit to k-corrections for consistency."""
    hdulist = pyfits.open(file)
    header = hdulist[1].header
    z0 = header['Z0']

    tbdata = hdulist[1].data
    sel = ((tbdata.field('z_tonry') >= zrange[0]) * 
           (tbdata.field('z_tonry') < zrange[1]) *
           (tbdata.field('nQ') > 2))
    tbdata = tbdata[sel]
    ngal = len(tbdata)
    z = tbdata.field('z_tonry')
    kc = tbdata.field('kcorr_r')
    pc = tbdata.field('pcoeff_r')
    k_rec = [np.polyval(pc[i, :], z[i] - z0) for i in xrange(ngal)]
    plt.clf()
    plt.scatter(z, kc - k_rec, 0.1)
    plt.xlabel('z')
    plt.ylabel('Kc - Kp')
    plt.draw()

def rebin_mean(a, nx, ny):
    """Rebin 2d array by factors nx and ny by averaging."""
    shape = (a.shape[0]/nx, a.shape[1]/ny)
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def rebin_sum(a, nx, ny):
    """Rebin 2d array by factors nx and ny by summing."""
    shape = (a.shape[0]/nx, a.shape[1]/ny)
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).sum(-1).sum(1)

def rebin_quad(a, nx, ny):
    """Rebin 2d array by factors nx and ny by adding in quadrature."""
    shape = (a.shape[0]/nx, a.shape[1]/ny)
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return np.sqrt((a*a).reshape(sh).sum(-1).sum(1))

def mock_ke_corr(z):
    """k+e corrections for gama mocks: Robotham et al 2011 eqn (8)."""
    z_p = 0.2
    Q = 1.75
    a = (0.2085, 1.0226, 0.5237, 3.5902, 2.3843)
    corr = Q*z
    for i in range(5):
        corr += a[i]*(z - z_p)**i
    return corr


def vol_limits(infile=gama_data+'kcorr_auto_z01.fits',  mlim=19.8, Q=0.81,
               Mlims=(-23, -22, -21, -20, -19, -18, -17, -16, -15),
               plot_fac=0.1, kplot=0):
    """Determine redshift limits corresponding to given absolute magnitude
    limits to yield volume-limited samples"""

    def Mvol(zlim):
        """Returns abs mag corresponding to given redshift for
        volume-limited sample."""
        # Take K-corr as 95-percentile of objects nearby in redshift
        # (larger K-corr --> brighter Mag)
        dz = 0.01
        idx = (z > zlim - dz)*(z < zlim + dz)
        k = scipy.stats.scoreatpercentile(kc[idx], 95)
        return mlim - cosmo.dist_mod(zlim) - k + Q*(zlim-z0)

    hdulist = fits.open(infile)
    header = hdulist[1].header
    H0 = 100.0
    z0 = header['z0']
    zrange = (0.002, 0.65)
    Mrange = (-15, -23)
    cosmo = CosmoLookup(H0, header['omega_l'], zrange)

    tbdata = hdulist[1].data
    idx = ((tbdata.field('survey_class') > 3) *
           (tbdata.field('r_petro') < mlim) *
           (tbdata.field('z') > 0) * (tbdata.field('nq') > 2))
    tbdata = tbdata[idx]
    mag = tbdata.field('r_petro')
    z = tbdata.field('z_tonry')
    kc = tbdata.field('kcorr_r')
    hdulist.close()

    print('mag range', np.min(mag), np.max(mag))
    Mabs = mag - cosmo.dist_mod(z) - kc + Q*(z-z0)
    plt.clf()
    if kplot:
        ax = plt.subplot(211)
        ax.scatter(z, kc, 0.1, 'k', edgecolors='face')
        ax.set_xlim(zrange)
        ax.set_ylim(-0.9, 1.5)
        ax.set_ylabel(r'$K(z)$')
        ax = plt.subplot(212)
    else:
        ax = plt.subplot(111)

    show = np.random.random(len(z)) < plot_fac
    ax.scatter(z[show], Mabs[show], 1, 'k', edgecolors='none')
    ax.set_xlim(zrange)
    ax.set_ylim(Mrange)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$^{0.1}M_r$')

    z_list = []
    for Mlim in Mlims:
        if Mvol(zrange[1]) - Mlim > 0:
            zlim = zrange[1]
        else:
            zlim = scipy.optimize.brentq(
                lambda z: Mvol(z) - Mlim, zrange[0], zrange[1],
                xtol=1e-5, rtol=1e-5)
        z_list.append(zlim)
        print(Mlim, zlim)
        ax.plot((zrange[0], zlim, zlim), (Mlim, Mlim, Mrange[1]))
    plt.show()
    return z_list


def schec_fit(M, phi, phi_err, schec_par, sigma=0,
              afix=False, likeCont=False, loud=False):
    """Least-squares Schechter fn fit to binned estimate.
    If sigma > 0, fit Schechter fn convolved with Gaussian."""

    alpha, Mstar, lpstar = schec_par
    prob = 0.32
    nbin = len(phi_err > 0)
    if afix:
        nu = nbin - 2
    else:
        nu = nbin - 3
    dchisq = scipy.special.chdtri(nu, prob)
    if loud:
        print(nu, dchisq)

    if afix:
        x0 = [Mstar, lpstar]
        res = scipy.optimize.fmin(
            lambda Mstar, lpstar, alpha, M, phi, phi_err, sigma:
            schec_resid((alpha, Mstar, lpstar), M, phi, phi_err, sigma),
            x0, (alpha, M, phi, phi_err, sigma),
            xtol=0.001, ftol=0.001, full_output=1, disp=0)
        xopt = res[0]
        chi2 = res[1]
        Mstar = xopt[0]
        lpstar = xopt[1]
        alpha_err = [0, 0]
    else:
        x0 = [alpha, Mstar, lpstar]
        res = scipy.optimize.fmin(schec_resid, x0, (M, phi, phi_err, sigma),
                                  xtol=0.001, ftol=0.001, full_output=1, disp=0)
        xopt = res[0]
        chi2 = res[1]
        alpha = xopt[0]
        Mstar = xopt[1]
        lpstar = xopt[2]
        alpha_err = like_err(lambda Mstar, lpstar, alpha, M, phi, phi_err, sigma:
                           schec_resid((alpha, Mstar, lpstar), M, phi, phi_err, sigma),
                           alpha, limits=(alpha-5, alpha+5),
                           marg=(Mstar, lpstar), args=(M, phi, phi_err, sigma),
                           nsig=2*dchisq)
        if loud:
            print('  alpha %6.2f - %6.2f + %6.2f' % (alpha, alpha_err[0], alpha_err[1]))
        
    Mstar_err = like_err(lambda lpstar, Mstar, alpha, M, phi, phi_err, sigma:
                       schec_resid((alpha, Mstar, lpstar), M, phi, phi_err, sigma),
                       Mstar, limits=(Mstar-5, Mstar+5),
                       marg=(lpstar), args=(alpha, M, phi, phi_err, sigma),
                       nsig=2*dchisq)
    lpstar_err = like_err(lambda Mstar, lpstar, alpha, M, phi, phi_err, sigma:
                        schec_resid((alpha, Mstar, lpstar), M, phi, phi_err, sigma),
                        lpstar, limits=(lpstar-5, lpstar+5),
                        marg=(Mstar), args=(alpha, M, phi, phi_err, sigma),
                        nsig=2*dchisq)
    
    if loud:
        print('  Mstar %6.2f - %6.2f + %6.2f' % (Mstar, Mstar_err[0], Mstar_err[1]))
        print('lpstar %6.4f - %6.4f + %6.4f' % (lpstar, lpstar_err[0], lpstar_err[1]))
    res = {'alpha': alpha, 'alpha_err': alpha_err, 
           'Mstar': Mstar, 'Mstar_err': Mstar_err, 
           'lpstar': lpstar, 'lpstar_err': lpstar_err, 
           'chi2': chi2, 'nu': nu}

    if likeCont:
        if loud:
            print("M*, phi* 2-sigma contours ...")
        prob = 0.05
        nu = len(phi_err > 0) # no free parameters
        dchisq = scipy.special.chdtri(nu, prob)
        print(nu, dchisq)
        nstep = 32
        chi2map = np.zeros([nstep, nstep])

        xmin = Mstar - 3*Mstar_err[0]
        xmax = Mstar + 3*Mstar_err[1]
        dx = (xmax - xmin)/nstep
        ymin = lpstar - 3*lpstar_err[0]
        ymax = lpstar + 3*lpstar_err[1]
        dy = (ymax - ymin)/nstep

        # chi2 minimum
        chi2min = schec_resid((alpha, Mstar, lpstar), M, phi, phi_err, sigma)
        v = [chi2min + dchisq,]
        for ix in range(nstep):
            ms = xmin + (ix+0.5)*dx
            for iy in range(nstep):
                ps = ymin + (iy+0.5)*dy
                chi2map[iy, ix] = schec_resid((alpha, ms, ps), M, phi, phi_err, sigma)

        res.append({'chi2map': chi2map, 'v': v, 
                    'limits': [xmin, xmax, ymin, ymax]})
    return res
        
def schec_resid(dchec_par, M, phi, phi_err, sigma=0):
    """Return chi^2 residual between binned phi estimate and Schechter fit."""

    alpha, Mstar, lpstar = schec_par
    fit = Schechter(M, alpha, Mstar, 10**lpstar)
    if sigma > 0:
        scale = sigma/np.mean(np.diff(M))
        ng = int(math.ceil(3*scale))
        gauss = scipy.stats.norm.pdf(np.arange(-ng, ng+1), scale=scale)
        fit = np.convolve(fit, gauss, 'same')

    idx = phi_err > 0
    fc = np.sum(((phi[idx]-fit[idx]) / phi_err[idx])**2)
    return fc

def schec_plot(alpha, Mstar, phistar, Mmin, Mmax, lineStyle=':', axes=None):
    nstep = 100
    x = np.linspace(Mmin, Mmax, nstep)
    y = Schechter(x, alpha, Mstar, phistar)
    if axes:
        axes.plot(x, y, lineStyle)
    else:
        plt.plot(x, y, lineStyle)

def Schechter(M, alpha, Mstar, phistar):
    L = 10**(0.4*(Mstar-M))
    schec = 0.4*ln10*phistar*L**(alpha+1)*np.exp(-L)
    return schec

def saund_fit(M, phi, phiErr, alpha, Mstar, sigma, lpstar):
    """Least-squares Saunders fn fit to binned estimate."""

    x0 = [alpha, Mstar, sigma, lpstar]
    res = scipy.optimize.fmin(saund_resid, x0, (M, phi, phiErr),
                              xtol=0.001, ftol=0.001, full_output=1, disp=0)
    xopt = res[0]
    chi2 = res[1]
    alpha = xopt[0]
    Mstar = xopt[1]
    sigma = xopt[2]
    lpstar = xopt[3]
    saund_par = {'alpha': alpha, 'Mstar': Mstar, 'sigma': sigma, 
                 'lpstar': lpstar, 'chi2': chi2, 'nu': len(M)-4}
    return saund_par
        
def saund_resid(alpha, Mstar, sigma, lpstar, M, phi, phiErr):
    """Return chi^2 residual between binned phi estimate and Saunders fit."""

    fc = 0
    phistar = 10**lpstar
    for ibin in range(len(M)):
        if phiErr[ibin] > 0:
            diff = phi[ibin] - phistar*saunders_lf(10**(0.4*(Mstar - M[ibin])),
                                                   alpha, sigma)
            fc += (diff/phiErr[ibin])**2
    return fc

def saunders_lf(L, alpha, sigma):
    """Saunders et al. (1990) LF fit"""
    return L**(alpha)*np.exp(-(np.log10(1 + L))**2/(2.0*sigma**2))

def saund_plot(alpha, Mstar, sigma, phistar, Mmin, Mmax,
              lineStyle=':', axes=None):
    nstep = 100
    M = np.linspace(Mmin, Mmax, nstep)
    L = 10**(0.4*(Mstar - M))
    phi = phistar*saunders_lf(L, alpha, sigma)
    if axes:
        axes.plot(M, phi, lineStyle)
    else:
        plt.plot(M, phi, lineStyle)

def like_err(fn, xmin, cons=None, limits=None, marg=None, args=(), nsig=1.0):
    """Return one parameter, nsig-sigma lower and upper errors about
    minimum xmin of -log likelihood function fn, marginalising over
    parameters marg.  Since it is called by scipy.optimize.fmin,
    supplied function fn must take parameters in order: marg, x, args.
    If fn returns chi^2 = -2 ln L, then nsig should be doubled.
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

def chisq_quad_fit(x, chi2, nfit=2, dchi2=1):
    """Fits quadratic to (x, chi2) values +/- nfit bins either side of minimum
    and returns x values corresponding to minimum chi2 and 1-sigma 
    lower and upper limits."""

    assert len(x) == len(chi2)
    imin = np.argmin(chi2)
    ilo = max(0, imin-nfit)
    ihi = min(len(x), imin+nfit+1)
    (a, b, c) = np.polyfit(x[ilo:ihi], chi2[ilo:ihi], 2)
    xmin = -b/(2*a)
    chi2min = a*xmin**2 + b*xmin + c
    xlo = (-b - math.sqrt(4*a*dchi2))/(2*a)
    xhi = (-b + math.sqrt(4*a*dchi2))/(2*a)
    return {'xmin': xmin, 'chi2min': chi2min, 'xlo': xlo, 'xhi': xhi}


def apollo_job(python_commands, job_script='python_job_script.sh'):
    """Create and run apollo job script to execute supplied python commands."""

    f = open(job_script, 'w')
    print >> f, """
#!/bin/bash

# import required environment variables such as PYTHONPATH
#$ -v PYTHONPATH=/research/astro/gama/loveday/Documents/Research/python
#$ -o /lustre/scratch/astro/loveday
#$ -j y
#$ -m e
#$ -M J.Loveday@sussex.ac.uk
#
# specify the queue with optional architecture spec following @@, e.g. mps.q@@mps_amd
#$ -q mps.q
# estimated runtime
##$ -l d_rt=08:00:00
# catch kill and suspend signals
#$ -notify
cd {}
python <<EOF
""".format(os.getcwd())
    for line in python_commands:
        print >> f, line
    print >> f, "EOF"
    f.close()

    cmd = 'chmod a+x ' + job_script
    subprocess.call(cmd, shell=True)

    cmd = 'qsub ' + job_script
    subprocess.call(cmd, shell=True)

def fig_xlabel(label, yc=0.01):
    """Add single label centred on x-axis when using multiple panels."""
    fig = plt.gcf()
    xc = 0.5 * (fig.subplotpars.left + fig.subplotpars.right)
    fig.text(xc, yc, label, ha='center', va='center')

def fig_ylabel(label, xc=0.05):
    """Add single label centred on y-axis when using multiple panels."""
    fig = plt.gcf()
    yc = 0.5 * (fig.subplotpars.bottom + fig.subplotpars.top)
    fig.text(xc, yc, label, ha='center', va='center', rotation='vertical')


def plot_3d_array(arr):
    """Plot 3d array by slice."""
    n = int(math.sqrt(arr.shape[-1]))
    plt.clf()
    fig, axes = plt.subplots(n, n, sharex=True, sharey=True, num=1)
    fig.subplots_adjust(hspace=0, wspace=0)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.imshow(arr[:, :, i*n + j])
    plt.show()


def pdfsave(pdf):
    """Draw figure, save to pdf file if specified, and show."""
    plt.draw()
    if pdf:
        pdf.savefig()
    plt.show()
