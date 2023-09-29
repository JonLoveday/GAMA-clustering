# 4MOST simulated catalogues

import math
import numpy as np
from numpy.random import default_rng
rng = default_rng()
from astropy.table import Table, join
from astropy import units as u
#import KCorrect as KC
import pdb
import matplotlib as mpl
import pylab as plt
import mpmath
#import pyqt_fit.kde
import scipy.integrate
import scipy.optimize
import scipy.signal
import scipy.stats
# import statistics
# import statsmodels.nonparametric.kde
# import jswml
import time
import util

# Ticks point inwards on all axes
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['mathtext.fontset'] = 'dejavusans'

# Extinction coeffs multiplying E(B-V) from Yuan+2013, Table 2, col 2.
# Y was interpolated between z and J.
ext_coef = {'FUV': 4.89, 'NUV': 7.24, 'u':4.39, 'g': 3.30, 'r': 2.31,
            'i': 1.71, 'z': 1.29, 'Y': 1.1, 'J': 0.72, 'H': 0.46,
            'Ks': 0.306, 'W1': 0.18, 'W2': 0.16}

def sample(input='Dan/BG_4MOST.fits', output='BG_4MOST_samp.fits', nout=10000):
    """Random sample a catalogue"""

    tin = Table.read(input)
    nin = len(tin)
    sel = rng.choice(nin, nout, replace=False)
    tout = tin[sel]
    tout.write(output)
    
def ext_corr(input='BG_4MOST_samp.fits'):
    """Extinction corrected magnitudes"""

    tin = Table.read(input)
    r_corr = tin['rtot'] - tin['ebv']*ext_coef['r']
    i_corr = r_corr - tin['ri'] - tin['ebv']*ext_coef['i']
    z_corr = i_corr - tin['iz'] - tin['ebv']*ext_coef['z']
    Y_corr = z_corr - tin['zy'] - tin['ebv']*ext_coef['Y']
    J_corr = Y_corr - tin['yj'] - tin['ebv']*ext_coef['J']
    J_obs = tin['rtot'] - tin['ri'] - tin['iz'] - tin['zy'] - tin['yj']
    plt.clf()
    plt.hist((J_obs, J_corr), bins=20, range=(15, 20))
    plt.xlabel('J-corr')
    plt.show()


def ebv_map(input='BG_4MOST_samp.fits'):
    """E(B-V) map"""

    tin = Table.read(input)
    plt.clf()
    plt.scatter(tin['RA'], tin['DEC'], c=np.log10(tin['ebv']))
    plt.colorbar()
    plt.show()

    