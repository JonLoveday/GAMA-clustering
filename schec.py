#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:12:31 2018

Implements Schechter function models in Sherpa

@author: loveday
"""

import math
import numpy as np

from sherpa.models import model

__all__ = ('SchecMag', 'SchecMass', 'SaundersMag', 'SaundersMass')
ln10 = math.log(10)


def _lognormal(pars, M):
    """Evaluate a normal function in magnitude or log-mass space.

    Parameters
    ----------
    pars: sequence of 3 numbers
        The order is M_c, sigma_c, lgps (log_10 phi*).
        These numbers are assumed to be valid.
    M: sequence of magnitudes or log masses
        The grid on which to evaluate the model. It is expected
        to be a floating-point type.

    Returns
    -------
    phi: sequence of numbers
        The model evaluated on the input grid.

    Notes
    -----
    """

    (M_c, sigma_c, lgps) = pars
    phi = 10**lgps * np.exp(-(M - M_c)**2 / (2*sigma_c**2))
    return phi


class LogNormal(model.ArithmeticModel):
    """A normal function in magnitude or log-mass space.

    The model parameters are:

    M_c
        The central magnitude.
    sigma_c
        The standard deviation.
    lgps
        log10(phi*).

    """

    def __init__(self, name='lognormal'):
        self.M_c = model.Parameter(name, 'M_c', -21.5, min=-23, max=-19)
        self.sigma_c = model.Parameter(name, 'sigma_c', 0.5, min=0.1, max=1.0)
        self.lgps = model.Parameter(name, 'lgps', -4, min=-7, max=-2)

        model.ArithmeticModel.__init__(self, name,
                                       (self.M_c, self.sigma_c, self.lgps))

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""

        return _lognormal(pars, x)


def _schecmag(pars, M):
    """Evaluate a Schechter function in magnitude space.

    Parameters
    ----------
    pars: sequence of 3 numbers
        The order is Mstar, alpha, lgps (log_10 phi*).
        These numbers are assumed to be valid.
    M: sequence of magnitudes
        The grid on which to evaluate the model. It is expected
        to be a floating-point type.

    Returns
    -------
    phi: sequence of numbers
        The model evaluated on the input grid.

    Notes
    -----
    """

    (Mstar, alpha, lgps) = pars
    L = 10**(0.4*(Mstar - M))
    phi = 0.4*ln10 * 10**lgps * L**(1+alpha) * np.exp(-L)
    return phi


class SchecMag(model.ArithmeticModel):
    """A Schechter function in magnitude space.

    The model parameters are:

    Mstar
        The characteristic magnitude.
    alpha
        The faint-end slope.
    lgps
        log10(phi*).

    """

    def __init__(self, name='schecmag'):
        self.Mstar = model.Parameter(name, 'Mstar', -20.2, min=-22, max=-19)
        self.alpha = model.Parameter(name, 'alpha', -1.2, min=-2, max=1)
        self.lgps = model.Parameter(name, 'lgps', -2, min=-8, max=0)

        model.ArithmeticModel.__init__(self, name,
                                       (self.Mstar, self.alpha, self.lgps))

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""

        return _schecmag(pars, x)


def _schecmagsq(pars, M):
    """Evaluate a modified Schechter function in magnitude space.
    See Yang+2008 eqn 5.

    Parameters
    ----------
    pars: sequence of 3 numbers
        The order is Mstar, alpha, lgps (log_10 phi*).
        These numbers are assumed to be valid.
    M: sequence of magnitudes
        The grid on which to evaluate the model. It is expected
        to be a floating-point type.

    Returns
    -------
    phi: sequence of numbers
        The model evaluated on the input grid.

    Notes
    -----
    """

    (Mstar, alpha, lgps) = pars
    L = 10**(0.4*(Mstar - M))
    phi = 0.4*ln10 * 10**lgps * L**(1+alpha) * np.exp(-L**2)
    return phi


class SchecMagSq(model.ArithmeticModel):
    """A modified Schechter function in magnitude space.

    The model parameters are:

    Mstar
        The characteristic magnitude.
    alpha
        The faint-end slope.
    lgps
        log10(phi*).

    """

    def __init__(self, name='schecmagsq'):
        self.Mstar = model.Parameter(name, 'Mstar', -20.2, min=-22, max=-19)
        self.alpha = model.Parameter(name, 'alpha', -1.2, min=-2, max=1)
        self.lgps = model.Parameter(name, 'lgps', -2, min=-8, max=0)

        model.ArithmeticModel.__init__(self, name,
                                       (self.Mstar, self.alpha, self.lgps))

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""

        return _schecmagsq(pars, x)


def _schecmaggen(pars, M):
    """Evaluate a generalised Schechter function in magnitude space.
    Within the exponent, L/L* is raised to the power beta.

    Parameters
    ----------
    pars: sequence of 4 numbers
        The order is Mstar, alpha, beta, lgps (log_10 phi*).
        These numbers are assumed to be valid.
    M: sequence of magnitudes
        The grid on which to evaluate the model. It is expected
        to be a floating-point type.

    Returns
    -------
    phi: sequence of numbers
        The model evaluated on the input grid.

    Notes
    -----
    """

    (Mstar, alpha, beta, lgps) = pars
    L = 10**(0.4*(Mstar - M))
    phi = 0.4*ln10 * 10**lgps * L**(1+alpha) * np.exp(-L**beta)
    return phi


class SchecMagGen(model.ArithmeticModel):
    """A generalised Schechter function in magnitude space.

    The model parameters are:

    Mstar
        The characteristic magnitude.
    alpha
        The faint-end slope.
    beta
        The power to which L/L* is raised within the exponent
    lgps
        log10(phi*).

    """

    def __init__(self, name='schecmaggen'):
        self.Mstar = model.Parameter(name, 'Mstar', -20.2, min=-22, max=-19)
        self.alpha = model.Parameter(name, 'alpha', -1.2, min=-2, max=1)
        self.beta = model.Parameter(name, 'beta', 1, min=0, max=3)
        self.lgps = model.Parameter(name, 'lgps', -2, min=-8, max=0)

        model.ArithmeticModel.__init__(
            self, name, (self.Mstar, self.alpha, self.beta, self.lgps))

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""

        return _schecmaggen(pars, x)


def _schecmassgen(pars, M):
    """Evaluate a generalised Schechter function in log-mass space.
    Within the exponent, M/M** is raised to the power beta.

    Parameters
    ----------
    pars: sequence of 4 numbers
        The order is Mstar, alpha, beta, lgps (log_10 phi*).
        These numbers are assumed to be valid.
    M: sequence of magnitudes
        The grid on which to evaluate the model. It is expected
        to be a floating-point type.

    Returns
    -------
    phi: sequence of numbers
        The model evaluated on the input grid.

    Notes
    -----
    """

    (Mstar, alpha, beta, lgps) = pars
    L = 10**(M - Mstar)
    phi = ln10 * 10**lgps * L**(1+alpha) * np.exp(-L**beta)
    return phi


class SchecMassGen(model.ArithmeticModel):
    """A generalised Schechter function in log-mass space.

    The model parameters are:

    Mstar
        The characteristic mass.
    alpha
        The faint-end slope.
    beta
        The power to which L/L* is raised within the exponent
    lgps
        log10(phi*).

    """

    def __init__(self, name='schecmassgen'):
        self.Mstar = model.Parameter(name, 'Mstar', 10.5, min=9, max=11)
        self.alpha = model.Parameter(name, 'alpha', -1.2, min=-2, max=1)
        self.beta = model.Parameter(name, 'beta', 1, min=0, max=2.5)
        self.lgps = model.Parameter(name, 'lgps', -2, min=-8, max=0)

        model.ArithmeticModel.__init__(
            self, name, (self.Mstar, self.alpha, self.beta, self.lgps))

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""

        return _schecmassgen(pars, x)


def _schecmass(pars, M):
    """Evaluate a Schechter function in log-mass space.

    Parameters
    ----------
    pars: sequence of 3 numbers
        The order is log Mstar, alpha, lgps (log_10 phi*).
        These numbers are assumed to be valid.
    M: sequence of log masses
        The grid on which to evaluate the model. It is expected
        to be a floating-point type.

    Returns
    -------
    phi: sequence of numbers
        The model evaluated on the input grid.

    Notes
    -----
    """

    (Mstar, alpha, lgps) = pars
    L = 10**(M - Mstar)
    phi = ln10 * 10**lgps * L**(1+alpha) * np.exp(-L)
    return phi


class SchecMass(model.ArithmeticModel):
    """A Schechter function in log mass space.

    The model parameters are:

    Mstar
        The characteristic log mass.
    alpha
        The low-mass slope.
    lgps
        log10(phi*).

    """

    def __init__(self, name='schecmass'):
        self.Mstar = model.Parameter(name, 'Mstar', 10.5, min=9, max=12)
        self.alpha = model.Parameter(name, 'alpha', -1.2, min=-2, max=1)
        self.lgps = model.Parameter(name, 'lgps', -2, min=-8, max=0)

#        model.RegriddableModel1D.__init__(self, name,
        model.ArithmeticModel.__init__(self, name,
                                       (self.Mstar, self.alpha, self.lgps))

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""

        return _schecmass(pars, x)


def _schecmasssq(pars, M):
    """Evaluate a modified Schechter function in log-mass space.
    See Yang+2009 eqn 16.

    Parameters
    ----------
    pars: sequence of 3 numbers
        The order is log Mstar, alpha, lgps (log_10 phi*).
        These numbers are assumed to be valid.
    M: sequence of log masses
        The grid on which to evaluate the model. It is expected
        to be a floating-point type.

    Returns
    -------
    phi: sequence of numbers
        The model evaluated on the input grid.

    Notes
    -----
    """

    (Mstar, alpha, lgps) = pars
    L = 10**(M - Mstar)
    phi = ln10 * 10**lgps * L**(1+alpha) * np.exp(-L**2)
    return phi


class SchecMassSq(model.ArithmeticModel):
    """A modified Schechter function in log mass space.

    The model parameters are:

    Mstar
        The characteristic log mass.
    alpha
        The low-mass slope.
    lgps
        log10(phi*).

    """

    def __init__(self, name='schecmasssq'):
        self.Mstar = model.Parameter(name, 'Mstar', 10.5, min=9, max=12)
        self.alpha = model.Parameter(name, 'alpha', -1.2, min=-2, max=1)
        self.lgps = model.Parameter(name, 'lgps', -2, min=-8, max=0)

        model.ArithmeticModel.__init__(self, name,
                                       (self.Mstar, self.alpha, self.lgps))

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""

        return _schecmasssq(pars, x)


def _saundersmass(pars, M):
    """Evaluate Saunders SMF in log-mass space.
    See Saunders+1990 eqn 6.1.

    Parameters
    ----------
    pars: sequence of 4 numbers
        The order is log Mstar, alpha, sigma, lgps (log_10 phi*).
        These numbers are assumed to be valid.
    M: sequence of log masses
        The grid on which to evaluate the model. It is expected
        to be a floating-point type.

    Returns
    -------
    phi: sequence of numbers
        The model evaluated on the input grid.

    Notes
    -----
    """

    (Mstar, alpha, sigma, lgps) = pars
    L = 10**(M - Mstar)
    phi = (ln10 * 10**lgps * L**(1+alpha) *
           np.exp(-np.log10(1+L)**2/(2.0*sigma**2)))
    return phi


class SaundersMass(model.ArithmeticModel):
    """Saunders SMF in log mass space.

    The model parameters are:

    Mstar
        The characteristic log mass.
    alpha
        The low-mass slope.
    sigma
        Width of the Gaussian.
    lgps
        log10(phi*).

    """

    def __init__(self, name='saundersmass'):
        self.Mstar = model.Parameter(name, 'Mstar', 10.5, min=9, max=12)
        self.alpha = model.Parameter(name, 'alpha', -1.2, min=-2, max=1)
        self.sigma = model.Parameter(name, 'sigma', 1, min=0.01, max=10)
        self.lgps = model.Parameter(name, 'lgps', -2, min=-8, max=0)

        model.ArithmeticModel.__init__(
            self, name, (self.Mstar, self.alpha, self.sigma, self.lgps))

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""

        return _saundersmass(pars, x)


def _saundersmag(pars, M):
    """Evaluate Saunders LF in magnitude space.
    See Saunders+1990 eqn 6.1.

    Parameters
    ----------
    pars: sequence of 4 numbers
        The order is log Mstar, alpha, sigma, lgps (log_10 phi*).
        These numbers are assumed to be valid.
    M: sequence of magnitudes
        The grid on which to evaluate the model. It is expected
        to be a floating-point type.

    Returns
    -------
    phi: sequence of numbers
        The model evaluated on the input grid.

    Notes
    -----
    """

    (Mstar, alpha, sigma, lgps) = pars
    L = 10**(0.4*(Mstar - M))
    phi = (0.4*ln10 * 10**lgps * L**(1+alpha) *
           np.exp(-np.log10(1+L)**2/(2.0*sigma**2)))
    return phi


class SaundersMag(model.ArithmeticModel):
    """Saunders LF in log magnitude space.

    The model parameters are:

    Mstar
        The characteristic log mass.
    alpha
        The low-mass slope.
    sigma
        Width of the Gaussian.
    lgps
        log10(phi*).

    """

    def __init__(self, name='saundersmag'):
        self.Mstar = model.Parameter(name, 'Mstar', -20.2, min=-22, max=-19)
        self.alpha = model.Parameter(name, 'alpha', -1.2, min=-2, max=1)
        self.sigma = model.Parameter(name, 'sigma', 1, min=0.001, max=10)
        self.lgps = model.Parameter(name, 'lgps', -2, min=-8, max=0)

        model.ArithmeticModel.__init__(
            self, name, (self.Mstar, self.alpha, self.sigma, self.lgps))

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""

        return _saundersmag(pars, x)
