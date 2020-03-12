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

__all__ = ('SchecMag', 'SchecMagDbl', 'SchecMass', )
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
        self.M_c = model.Parameter(name, 'M_c', 1)
        self.sigma_c = model.Parameter(name, 'sigma_c', 1)
        self.lgps = model.Parameter(name, 'lgps', 1)

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
        self.Mstar = model.Parameter(name, 'Mstar', 1)
        self.alpha = model.Parameter(name, 'alpha', 1)
        self.lgps = model.Parameter(name, 'lgps', 1)

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
        self.Mstar = model.Parameter(name, 'Mstar', 1)
        self.alpha = model.Parameter(name, 'alpha', 1)
        self.lgps = model.Parameter(name, 'lgps', 1)

        model.ArithmeticModel.__init__(self, name,
                                       (self.Mstar, self.alpha, self.lgps))

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""

        return _schecmagsq(pars, x)


def _schecmagdbl(pars, M):
    """Evaluate a double Schechter function in magnitude space.

    Parameters
    ----------
    pars: sequence of 5 numbers
        The order is alpha1, alpha2, Mstar, lgps1, lgps2 (log_10 phi*).
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

    (alpha1, alpha2, Mstar, lgps1, lgps2) = pars
    L = 10**(0.4*(Mstar - M))
    phi = 0.4*ln10 * np.exp(-L) * (10**lgps1 * L**(1+alpha1) + 10**lgps2 * L**(1+alpha2)) 
    return phi


#class SchecMag(model.RegriddableModel1D):
class SchecMagDbl(model.ArithmeticModel):
    """A double Schechter function in magnitude space.

    The model parameters are:

    alpha1, alpha2
        The faint-end slopes.
    Mstar
        The characteristic magnitude.
    lgps1, lgps2
        log10(phi*) coresponding to the two components.

    """

    def __init__(self, name='schecmagdbl'):
        self.alpha1 = model.Parameter(name, 'alpha1', 1)
        self.alpha2 = model.Parameter(name, 'alpha2', 1)
        self.Mstar = model.Parameter(name, 'Mstar', 1)
        self.lgps1 = model.Parameter(name, 'lgps1', 1)
        self.lgps2 = model.Parameter(name, 'lgps2', 1)

#        model.RegriddableModel1D.__init__(self, name,
        model.ArithmeticModel.__init__(self, name,
                                       (self.alpha1, self.alpha2, self.Mstar,
                                        self.lgps1, self.lgps2))

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""

        return _schecmagdbl(pars, x)


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
        self.Mstar = model.Parameter(name, 'Mstar', 1)
        self.alpha = model.Parameter(name, 'alpha', 1)
        self.lgps = model.Parameter(name, 'lgps', 1)

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
        self.Mstar = model.Parameter(name, 'Mstar', 1)
        self.alpha = model.Parameter(name, 'alpha', 1)
        self.lgps = model.Parameter(name, 'lgps', 1)

        model.ArithmeticModel.__init__(self, name,
                                       (self.Mstar, self.alpha, self.lgps))

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""

        return _schecmasssq(pars, x)
