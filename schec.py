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

__all__ = ('SchecMag', 'SchecMass', )
ln10 = math.log(10)


def _schecmag(pars, M):
    """Evaluate a Schechter function in magnitude space.

    Parameters
    ----------
    pars: sequence of 3 numbers
        The order is alpha, Mstar, lgps (log_10 phi*).
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

    (alpha, Mstar, lgps) = pars
    L = 10**(0.4*(Mstar - M))
    phi = 0.4*ln10 * 10**lgps * L**(1+alpha) * np.exp(-L)
    return phi


#class SchecMag(model.RegriddableModel1D):
class SchecMag(model.ArithmeticModel):
    """A Schechter function in magnitude space.

    The model parameters are:

    alpha
        The faint-end slope.
    Mstar
        The characteristic magnitude.
    lgps
        log10(phi*).

    """

    def __init__(self, name='schecmag'):
        self.alpha = model.Parameter(name, 'alpha', 1)
        self.Mstar = model.Parameter(name, 'Mstar', 1)
        self.lgps = model.Parameter(name, 'lgps', 1)

#        model.RegriddableModel1D.__init__(self, name,
        model.ArithmeticModel.__init__(self, name,
                                       (self.alpha, self.Mstar, self.lgps))

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""

        return _schecmag(pars, x)


def _schecmass(pars, M):
    """Evaluate a Schechter function in log-mass space.

    Parameters
    ----------
    pars: sequence of 3 numbers
        The order is alpha, log Mstar, lgps (log_10 phi*).
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

    (alpha, Mstar, lgps) = pars
    L = 10**(M - Mstar)
    phi = ln10 * 10**lgps * L**(1+alpha) * np.exp(-L)
    return phi


#class SchecMag(model.RegriddableModel1D):
class SchecMass(model.ArithmeticModel):
    """A Schechter function in log mass space.

    The model parameters are:

    alpha
        The low-mass slope.
    Mstar
        The characteristic log mass.
    lgps
        log10(phi*).

    """

    def __init__(self, name='schecmass'):
        self.alpha = model.Parameter(name, 'alpha', 1)
        self.Mstar = model.Parameter(name, 'Mstar', 1)
        self.lgps = model.Parameter(name, 'lgps', 1)

#        model.RegriddableModel1D.__init__(self, name,
        model.ArithmeticModel.__init__(self, name,
                                       (self.alpha, self.Mstar, self.lgps))

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""

        return _schecmass(pars, x)
