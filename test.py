#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:31:19 2019

@author: loveday
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def cbar():
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=12.8, vmax=14.2)
    scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.clf()
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, num=1)
    fig.subplots_adjust(top=0.93)
    cbar_ax = fig.add_axes([0.13, 0.97, 0.75, 0.02])
    cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                                   orientation='horizontal')
    plt.axis((-2, 1, -20, -15))
    plt.show()


def polar():
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # Compute areas and colors
    N = 150
    r = 2 * np.random.rand(N)
    theta = 2 * np.pi * np.random.rand(N)
    area = 200 * r**2
    colors = theta

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

    ax.set_thetamin(120)
    ax.set_thetamax(135)
    ax.set_xlabel('RA')
    ax.set_ylabel('z')
    plt.show()


def scatter(nsmall=10000, nlarge=100):
    plt.clf()
    x, y = np.random.random(nsmall), np.random.random(nsmall)
    plt.scatter(x, y, s=0.1, c='k', alpha=0.5)
    x, y, c = np.random.random(nlarge), np.random.random(nlarge), np.random.random(nlarge)
    plt.scatter(x, y, s=100, c=c, alpha=1)
    plt.savefig('test_scatter.png', bbox_inches='tight')
    plt.show()


def argtest():
    def fun_list(arg=[1, 2]):
        print(arg)
        arg[0] = 9

    def fun(arg=1):
        print(arg)
        arg = 9

    fun()
    fun()
    fun_list()
    fun_list()

        