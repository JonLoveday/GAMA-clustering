#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-corrections from SEDs
Created on Wed Feb 28 14:38:42 2018

@author: loveday
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb

from astLib import astSED
from astropy.table import Table

lf_data = os.environ['LF_DATA']


def read_m09(infile='lowzDr12.fits',
             sedfile=lf_data+'Maraston2009/M09_models/M09_composite_bestfitLRG.sed',
             zbinning=(0.0, 0.6, 30)):
    """Read Maraston+09 SEDs; reproduce their Fig 1 (right panels)."""
    nz = zbinning[-1]
    data = np.loadtxt(sedfile)
    ages, idxs = np.unique(data[:, 0], return_index=True)
    print(ages)
    plt.clf()
    m09_dir = {}
    for i in range(len(idxs)-1):
        ilo = idxs[i]
        ihi = idxs[i+1]
        spec = astSED.SED(data[ilo:ihi, 1], data[ilo:ihi, 2])
        m09_dir[ages[i]] = spec
        plt.plot(data[ilo:ihi, 1], data[ilo:ihi, 2])
    plt.xlabel(r'$\lambda [A]$')
    plt.ylabel(r'$F_\lambda$')
    plt.show()

    pbfile = lf_data + 'Doi2010/ugriz_atmos.txt'
    doi_u = astSED.Passband(pbfile, normalise=0, transmissionColumn=1)
    doi_g = astSED.Passband(pbfile, normalise=0, transmissionColumn=2)
    doi_r = astSED.Passband(pbfile, normalise=0, transmissionColumn=3)
    doi_i = astSED.Passband(pbfile, normalise=0, transmissionColumn=4)
    doi_z = astSED.Passband(pbfile, normalise=0, transmissionColumn=5)
    gunn_u = astSED.Passband(lf_data+'Gunn2001/filter_u.txt',
                             normalise=0, transmissionColumn=1)
    gunn_g = astSED.Passband(lf_data+'Gunn2001/filter_g.txt',
                             normalise=0, transmissionColumn=1)
    gunn_r = astSED.Passband(lf_data+'Gunn2001/filter_r.txt',
                             normalise=0, transmissionColumn=1)
    gunn_i = astSED.Passband(lf_data+'Gunn2001/filter_i.txt',
                             normalise=0, transmissionColumn=1)
    gunn_z = astSED.Passband(lf_data+'Gunn2001/filter_z.txt',
                             normalise=0, transmissionColumn=1)
#    logl = np.logspace(math.log10(spec.dispersion[0].value),
#                       math.log10(spec.dispersion[-1].value),
#                       num=len(spec.dispersion), endpoint=True)
#    pdb.set_trace()
    specz = spec.copy()
    specz.redshift(0.5)
    plt.clf()
    plt.plot(spec.wavelength, spec.flux/np.max(spec.flux))
    plt.plot(specz.wavelength, specz.flux/np.max(spec.flux))
#    plt.plot(pass_u.wavelength, pass_u.transmission)
#    plt.plot(pass_g.wavelength, pass_g.transmission)
#    plt.plot(pass_r.wavelength, pass_r.transmission)
#    plt.plot(pass_i.wavelength, pass_i.transmission)
#    plt.plot(pass_z.wavelength, pass_z.transmission)
    gunn_u.plot()
    gunn_g.plot()
    gunn_r.plot()
    gunn_i.plot()
    gunn_z.plot()
    doi_u.plot()
    doi_g.plot()
    doi_r.plot()
    doi_i.plot()
    doi_z.plot()
#    plt.plot(spec.dispersion, spec.flux)
#    plt.plot(spec_log.dispersion, spec_log.flux)
#    plt.xrange(2000, 8000)
    plt.xlabel(r'$\lambda [A]$')
    plt.ylabel(r'$F_\lambda$')
    plt.show()

    z = np.linspace(*zbinning)
    gr_doi = np.zeros(nz)
    ri_doi = np.zeros(nz)
    gr_gunn = np.zeros(nz)
    ri_gunn = np.zeros(nz)

    table = Table.read(infile)

    plt.clf()
    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.set_size_inches(6, 12)
    ax[0].scatter(table['z'], table['modelMag_g'] - table['modelMag_r'],
                  s=0.01, c='k')
    ax[0].set_ylim(0.5, 2.5)
    ax[0].set_ylabel(r'$(g-r)$')
    ax[1].scatter(table['z'], table['modelMag_r'] - table['modelMag_i'],
                  s=0.01, c='k')
    ax[1].set_xlim(0, 0.6)
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel(r'$(r-i)$')
    ax[1].set_xlabel('Redshift')

    zmean = np.zeros(nz-1)
    gr_mean = np.zeros(nz-1)
    ri_mean = np.zeros(nz-1)
    for iz in range(nz-1):
        sel = (z[iz] <= table['z']) * (table['z'] < z[iz+1])
        zmean[iz] = np.mean(table['z'][sel])
        gr_mean[iz] = np.mean(table['modelMag_g'][sel] - table['modelMag_r'][sel])
        ri_mean[iz] = np.mean(table['modelMag_r'][sel] - table['modelMag_i'][sel])
    ax[0].plot(zmean, gr_mean)
    ax[1].plot(zmean, ri_mean)

    for age in [5., 10., 12.]:
        spec = m09_dir[age]
        specz = spec.copy()
        for iz in range(nz):
            specz.redshift(z[iz])
            gr_doi[iz] = specz.calcColour(doi_g, doi_r, 'AB')
            ri_doi[iz] = specz.calcColour(doi_r, doi_i, 'AB')
            gr_gunn[iz] = specz.calcColour(gunn_g, gunn_r, 'AB')
            ri_gunn[iz] = specz.calcColour(gunn_r, gunn_i, 'AB')

        ax[0].plot(z, gr_doi)
        ax[1].plot(z, ri_doi, label='age {} Gyr'.format(age))
    ax[1].legend()
    plt.show()


def doi_reformat(infile=lf_data + 'Doi2010/ugriz.txt'):
    """Multiply Doi et al CCD response function by atmospheric transmission."""

    data = np.loadtxt(infile)
    data[:, 1] *= data[:, 6]
    data[:, 2] *= data[:, 6]
    data[:, 3] *= data[:, 6]
    data[:, 4] *= data[:, 6]
    data[:, 5] *= data[:, 6]
    np.savetxt(lf_data + 'Doi2010/ugriz_atmos.txt', data, fmt='%.4f',
               header='# lambda  u_atmos  g_atmos  r_atmos  i_atmos  z_atmos')


def gunn_reformat(infile=lf_data + 'Gunn2001/filter_curves.fits',
                  outfile=lf_data + 'Gunn2001/filter_{}.txt'):
    """Multiply Gunn CCD response function by atmospheric transmission."""

    bands = 'ugriz'
    for iband in range(5):
        table = Table.read(infile, hdu=iband+1)
        data = np.zeros((len(table), 2))
        data[:, 0] = table['wavelength']
        data[:, 1] = table['resbig'] * table['xatm']
        np.savetxt(outfile.format(bands[iband]), data, fmt='%.4f')
