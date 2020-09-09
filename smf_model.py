#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:18:55 2019

Empirical SMF model based on Toczak+2017

@author: loveday
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
import scipy

from astropy.cosmology import Planck15 as cosmo, z_at_value
import astropy.units as u
import util


class SMP():
    """Stellar mass particle."""

    def __init__(self, mass, age=0):
        """Initialise new SMP instance from specified smf."""
        self.mass = mass
        self.minit = mass
        self.age = age

    def mass_loss(self, dt):
        """Mass loss according to eqn (16) of Moster, Naab & White (2013).
        dt is time change in Gyr"""
        self.age += dt
        self.mass = (1 - 0.05*math.log((self.age + 3e-4)/3e-4))*self.minit


class Galaxy():
    """Galaxy data and methods."""

    def __init__(self, mass, sfr):
        """Initialise new Galaxy instance from specified smf."""
        self.mtype = 0  # 0=unmerged, 1=minor, 2=major merger
        self.sfr = sfr
        self.smps = [SMP(mass, age=np.random.random())]

    def mass(self):
        """Return log sum of mass of stellar mass particles."""
        return np.sum(np.array([smp.mass for smp in self.smps]))

    def lg_mass(self):
        """Return log sum of mass of stellar mass particles."""
        return math.log10(self.mass())

    def form_stars(self, dt):
        """Form new stars by adding additional SMP"""
        if self.sfr > 0:
            self.smps.append(SMP(self.sfr*dt*1e9))

    def set_sfr(self, s0, M0, gamma):
        """Set new SFR according to Tomczak+2016 eqn (2)."""
        if self.sfr > 0:
            self.sfr = 10**(s0 - math.log10(1 + (self.mass()/M0)**-gamma))

    def smp_plot(self):
        """Plot mass-age distribution for a galaxy's SMPs."""
        plt.clf()
        for smp in self.smps:
            plt.plot(math.log10(smp.mass), smp.age, '.')
        plt.xlabel(r'log $M^*/M_\odot$')
        plt.ylabel('Age [Gyr]')
        plt.show()


class GalPop():
    """Galaxy population data and methods."""

    def __init__(self, N=10000, zstart=5, zend=0.1, dt=0.1, fmerge_end=0.5,
                 merger_mass_loss=0.3, bins=np.linspace(6, 12, 25),
                 rf_file='red_frac.pkl'):
        """Initialise new GalPop instance from smf_init."""

        self.N0 = N
        self.ngal = N
        self.bins = bins
        self.dm = bins[1] - bins[0]
        self.lgM = bins[:-1] + 0.5*self.dm
        self.zstart, self.zend, self.z = zstart, zend, zstart
        self.t = cosmo.age(self.z).value
        self.dt = dt
        tend = cosmo.age(zend).value
        self.nstep = int((tend - self.t)/dt) + 1
        self.nrem = self.nstep
        sfr_pars = sfr_z_pars(self.z)
        s0, M0, gamma = sfr_pars
        M = 10**(util.ran_fun(smf_init, bins[0], bins[-1], N))
        sfr = 10**(s0 - np.log10(1 + (M/M0)**-gamma))

        self.galaxies = [Galaxy(M[i], sfr[i]) for i in range(N)]
        self.phi_init = smf_init(self.lgM)*self.ngal/np.sum(smf_init(self.lgM))
        self.nmerge_minor = 0
        self.nmerge_major = 0
        self.merger_mass_loss = merger_mass_loss
        self.ic_mass = 0
        # Normalize merger rate to obtain end merged fraction
        res = scipy.integrate.quad(merger_rate, self.t, tend, args=(1))
        mint = res[0]
#        self.m0 = fmerge_end*N/mint/(1 + fmerge_end)
        self.m0 = fmerge_end*N/mint
    #    qf_step = qf_end/nstep
    #    print(nstep, 'time steps')

        # Target quiescent fraction
        (Mfit, rf) = pickle.load(open(rf_file, 'rb'))
        self.qf_end = np.interp(self.lgM, Mfit, rf)
        self.smf_plot(logy=1)

    def lg_masses(self):
        """Return rray of current galaxy log-masses."""
        return np.array([galaxy.lg_mass() for galaxy in self.galaxies])

    def smf_plot(self, split='quiescent', logy=0, outfile=None):
        """Plot current SMF."""
        lgm = self.lg_masses()
        if outfile:
            hist, edges = np.histogram(lgm, self.bins)
            pickle.dump((self.lgM, hist), open(outfile, 'wb'))
        plt.clf()
        if split == 'mtype':
            mtype = np.array([galaxy.mtype for galaxy in self.galaxies])
            plt.hist((lgm[mtype == 0], lgm[mtype == 1], lgm[mtype == 2]),
                     self.bins, histtype='barstacked')
        else:
            sfr = np.array([galaxy.sfr for galaxy in self.galaxies])
            plt.hist((lgm[sfr == 0], lgm[sfr > 0]),
                     self.bins, histtype='barstacked', color=('r', 'b'))
        plt.plot(self.lgM, self.phi_init, 'g')
        if logy:
            plt.semilogy(nonposy='clip')
            ylim = plt.ylim()
            plt.ylim(1, ylim[1])
        plt.xlabel(r'log $M^*/M_\odot$')
        plt.ylabel('N')
        plt.show()

    def sfr_plot(self):
        """Plot SFR-M* relation at current redshift."""
        sfr_pars = sfr_z_pars(self.z)
        s0, M0, gamma = sfr_pars
        lgsfr = s0 - np.log10(1 + (10**self.lgM/M0)**-gamma)
        plt.clf()
        plt.plot(self.lgM, lgsfr)
        plt.xlabel(r'log $M^*/M_\odot$')
        plt.ylabel(r'log SFR [$M_\odot$ / yr]')
        plt.show()

    def qf_plot(self):
        """Plot current and target quiescent fraction."""
        lgm = self.lg_masses()
        sfr = np.array([galaxy.sfr for galaxy in self.galaxies])
        hq, edges = np.histogram(lgm[sfr == 0], self.bins)
        h, edges = np.histogram(lgm, self.bins)
        qf = hq/h
        plt.clf()
        plt.plot(self.lgM, qf, label='Current')
        plt.plot(self.lgM, self.qf_end, label='Target (GAMA groups)')
        plt.legend()
        plt.xlabel(r'log $M^*/M_\odot$')
        plt.ylabel('Red fraction')
        plt.show()

    def quench(self):
        """Quench galaxies with mass-dependent probability."""
        lgm = self.lg_masses()
        sfr = np.array([galaxy.sfr for galaxy in self.galaxies])
        hq, edges = np.histogram(lgm[sfr == 0], self.bins)
        h, edges = np.histogram(lgm, self.bins)
        qf = hq/h
        fq = (self.qf_end - qf)/(1-qf)/self.nrem
#        print(self.qf_end, qf, fq, self.nrem)
        for galaxy in self.galaxies:
            if galaxy.sfr > 0:
                im = int((galaxy.lg_mass() - self.bins[0])/self.dm)
                if im < 0:
                    im = 0
                if im >= len(self.bins):
                    im = len(self.bins) - 1
                if np.random.random() < fq[im]:
                    galaxy.sfr = 0

    def merge(self):
        """Merge randomly-selected galaxies."""
        merge_rate = self.dt * self.m0 * (1 + self.z)**2.7
        nminor = np.random.poisson(0.75*merge_rate)
        nmajor = np.random.poisson(0.25*merge_rate)
        print('forming', nminor, nmajor, 'minor, major mergers')
        iminor, imajor = 0, 0
        while (iminor < nminor) or (imajor < nmajor):
            merge = False
            igal = np.random.choice(self.ngal, size=2, replace=False)
            gal = [self.galaxies[i] for i in igal]
            mass = [g.mass() for g in gal]
            if mass[0] < mass[1]:
                ilo, glo, ghi = igal[0], gal[0], gal[1]
            else:
                ilo, glo, ghi = igal[1], gal[1], gal[0]
            mratio = glo.mass() / ghi.mass()
            if (mratio < 0.25) and (iminor < nminor):
                iminor += 1
                self.nmerge_minor += 1
                ghi.mtype = 1
                merge = True
            if (mratio >= 0.25) and (imajor < nmajor):
                imajor += 1
                self.nmerge_major += 1
                ghi.mtype = 2
                merge = True

            if merge:
                # Remove merger_mass_loss fraction from lower-mass gal;
                # tranfer rest to higher mass gal
                for smp in glo.smps:
                    self.ic_mass += smp.mass * self.merger_mass_loss
                    smp.mass *= (1 - self.merger_mass_loss)
                    ghi.smps.append(smp)
                self.ngal -= 1
                del self.galaxies[ilo]
                del glo

    def update(self):
        """Update galaxy masses at each time step."""

        self.quench()
        self.merge()
        sfr_pars = sfr_z_pars(self.z)
        self.nrem -= 1
        for galaxy in self.galaxies:
            for smp in galaxy.smps:
                smp.mass_loss(self.dt)
            galaxy.form_stars(self.dt)
            galaxy.set_sfr(*sfr_pars)

    def quiescent_fraction(self):
        """Fraction of quiescent galaxies."""
        nq = 0
        for galaxy in self.galaxies:
            if galaxy.sfr == 0:
                nq += 1
        return nq/len(self.galaxies)

    def evolve(self):
        """Main routine for evolving the SMF."""

        while(self.z > self.zend):
            self.t += self.dt
            self.z = z_at_value(cosmo.age, self.t*u.Gyr)
            self.update()
            print(f'z={self.z:5.3f}, t={self.t:5.3f}Gyr, '
                  f'qf={self.quiescent_fraction():5.3f}, '
                  f'nminor={self.nmerge_minor}, nmajor={self.nmerge_major}')
            self.smf_plot(logy=1)


def smf_init_old(lgM, alpha=-1.5, lgMstar=10.7):
    """Initial SMF."""
    L = 10**(lgM - lgMstar)
    return L**(1+alpha) * np.exp(-L)


def smf_init(lgM, a1=-0.14, ps1=0.21, a2=-1.52, ps2=0.14, lgMstar=10.77):
    """Initial SMF from Tmczak+2017 Table 4, lowest density bin."""
    L = 10**(lgM - lgMstar)
    return (ps1*L**(1+a1) + ps2*L**(1+a2)) * np.exp(-L)


def sfr_z_pars(z):
    """Redshift-dependent SFR parameters from Tomczak+2016 eqn (4)."""
    s0 = 0.448 + 1.220*z - 0.174*z**2
    M0 = 10**(9.458 + 0.865*z - 0.132*z**2)
    gamma = 1.091
    return (s0, M0, gamma)


def red_frac(infile='smf_incomp_vmax.pkl'):
    """Red fraction from GAMA."""
    lf_dict = pickle.load(open(infile, 'rb'))
    plt.clf()
    for i in range(3, 6):
        phir = lf_dict[f'M{i}allr']
        phib = lf_dict[f'M{i}allb']
        sel = phir.comp * phib.comp
        Mbin = phir.Mbin[sel]
        rf = phir.phi[sel]/(phir.phi[sel] + phib.phi[sel])
        rf_err = (rf**2 * ((phir.phi_err[sel]/phir.phi[sel])**2 +
                           (phib.phi_err[sel]/phib.phi[sel])**2))**0.5
        plt.errorbar(Mbin, rf, rf_err, label=f'M{i}')
    plt.legend()
    plt.xlabel(r'log $M^*/M_\odot$')
    plt.ylabel('Red fraction')
    plt.show()
    return Mbin, rf


def merger_rate(t, m0):
    z = z_at_value(cosmo.age, t*u.Gyr)
    return m0*(1 + z)**2.7


def smf_evolve(N=10000, zend=0.1, fmerge_end=0.8, bins=np.linspace(6, 12, 24),
               outfile=None):
    """Main routine for evolving the SMF."""

    gal_pop = GalPop(N=N, zend=zend, fmerge_end=fmerge_end, bins=bins)
    gal_pop.evolve()
    gal_pop.qf_plot()
    gal_pop.smf_plot(split='mtype', logy=1, outfile=outfile)
    print(f'{gal_pop.ngal} galaxies at end')
    print(f'{gal_pop.nmerge_minor/gal_pop.N0:5.3f} minor, '
          f'{gal_pop.nmerge_major/gal_pop.N0:5.3f} major merger fraction')
    print(gal_pop.nrem)
    ic_mf = gal_pop.ic_mass / (gal_pop.ic_mass + np.sum(10**gal_pop.lg_masses()))
    print(f'IC mass fraction {ic_mf:5.3f}')


def plot(inlist):
    """Plot SMFs from inlist."""
    plt.clf()
    for infile in inlist:
        lgM, h = pickle.load(open(infile, 'rb'))
        plt.plot(lgM, h, label=infile)
    plt.semilogy()
    plt.legend()
    plt.xlabel(r'log $M^*/M_\odot$')
    plt.ylabel('phi')
    plt.show()
    