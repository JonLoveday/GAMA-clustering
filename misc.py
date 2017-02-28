# Miscellaneous unrelated utilities

from __future__ import division

import csv
import glob
import h5py
import math
import numpy as np
import os
import pickle
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astLib import astCalc
#import KCorrect as KC
import pdb
import matplotlib
import pylab as plt
from mpl_toolkits.axes_grid import AxesGrid
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
import mpmath
import pyqt_fit.kde
import scipy.integrate
import scipy.optimize
import scipy.signal
import scipy.stats
# import statistics
# import statsmodels.nonparametric.kde
# import jswml
import time
import util

# Directoty to save plots
plot_dir = os.environ['HOME'] + '/Documents/tex/papers/gama/jswml/'

def counts_old(infile='TilingCatv10.fits'):
    """Number counts"""

    mmin = 14
    mmax = 20
    slope = 0.52
    nbins = int(10*(mmax - mmin))
    fmin = 1
    fmax = 10**(slope*(mmax - mmin))
    
    hdulist = fits.open(infile)
    tbdata = hdulist[1].data
    mag = tbdata.field('R_PETRO')
    bins = mmin + np.log10(np.linspace(fmin, fmax, nbins))/slope
    mp = 10**(slope*(mag - mmin))
##     n, bins, patches = plt.hist(mag, 60, [14, 20])
    plt.clf()
    plt.subplot(211)
    print plt.hist(mp, 60, [fmin, fmax])
    plt.subplot(212)
    print plt.hist(mag, bins, [mmin, mmax])
    plt.draw()
    hdulist.close()

def counts(colname, mrange=(14, 20), nbins=60, infile='kcorrz.fits'):
    """Number counts"""

    hdulist = fits.open(infile)
    tbdata = hdulist[1].data
    mag = tbdata.field(colname)
    nq = tbdata.field('nq')
    sel = nq > 2
    hdulist.close()

    plt.clf()
    plt.semilogy(basey=10, nonposy='clip')
    plt.hist(mag, nbins, mrange)
    plt.hist(mag[sel], nbins, mrange)
    plt.xlabel(colname)
    plt.draw()

def imcomp(infile='coaddGalaxyXstripe82.fit'):
    """Stripe 82 imaging completeness"""

    mmin = 14
    mmax = 20
    mstep = 0.1
    nm = int((mmax - mmin)/mstep)
    smin = 18.0
    smax = 26.0
    sstep = 0.1
    ns = int((smax - smin)/sstep)
    
    hdulist = fits.open(infile)
    tbdata = hdulist[1].data
    match = tbdata.field('objid_2') > 0
    mag1 = (tbdata.field('petroMagCor_r_1'))[match]
    mag2 = (tbdata.field('petroMagCor_r_2'))[match]
    sb1 = mag1 + 2.5*np.log10(2*math.pi*((tbdata.field('petror50_r_1'))[match])**2)
    hdulist.close()

    tck = scipy.interpolate.bisplrep(mag1, sb1, mag1 - mag2)
    mags = np.linspace(mmin, mmax, nm)
    sbs = np.linspace(smin, smax, ns)
    fit = scipy.interpolate.bisplev(mags, sbs, tck)

    plt.figure()
    plt.pcolor(mags, sbs, fit)
    plt.colorbar()
    plt.show()

def tcomp(infile='TilingCatv43.fits', magname='r_petro', 
          plotname=r'$r_{\rm Petro}$',
          mrange=(14, 19.8), mstep=0.1, yrange=(0.0, 1.1),
          plot_file='tcomp.pdf'):
    """GAMA target completeness."""

    nm = int((mrange[1] - mrange[0])/mstep)
    mags = np.linspace(mrange[0], mrange[1], nm)
    
    hdulist = fits.open(infile)
    tbdata = hdulist[1].data

    # Observed objects excluding F1 and F3 fillers and bad vis_class 2-5
    sel = ((tbdata.field('survey_class') >= 3) * 
           (tbdata.field('r_petro') < 19.8) *
           ((tbdata.field('vis_class') < 2) + (tbdata.field('vis_class') > 5)))
    tbdata = tbdata[sel]
    observed = tbdata.field('survey_code') > 0
    hdulist.close()

    tarhist, edges = np.histogram(tbdata.field(magname), nm, mrange)
    obshist, edges = np.histogram(tbdata[observed].field(magname), nm, mrange)
    use = tarhist > 0
    mags = mags[use]
    tarhist = tarhist[use]
    obshist = obshist[use]
    fobs = obshist.astype(float)/tarhist
    fsig = fobs/np.sqrt(tarhist)

    plt.clf()
    fig, axes = plt.subplots(2, sharex=True, num=1)
    ax = axes[0]
    ax.step(mags, tarhist)
    ax.step(mags, obshist)
    ax.set_ylabel('Frequency')
    
    ax = axes[1]
    ax.step(mags, obshist.astype(float)/tarhist)
    ax.set_xlabel(plotname)
    ax.set_ylabel('Target completeness')
    ax.set_ylim(yrange)
    plt.draw()

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
        print 'plot saved to ', plot_dir + plot_file


def zcomp(infile='kcorrz01.fits', magname='fibermag_r',
          plotname=r'$r_{\rm fibre}\ ({\rm mag})$',
          mrange=(17, 22.5), mstep=0.1, p0=(22, 2, 0.5), yrange=(0.0, 1.1),
          plot_file='zcomp.pdf'):
    """GAMA redshift success rate."""

    def sigmoid(p, x):
        if (len(p) == 2):
            return 1.0/(1 + np.exp(p[1]*(x - p[0])))
        return (1.0/(1 + np.exp(p[1]*(x-p[0]))))**p[2]

    def sigmoid_resid(p, x, y, yerr):
        use = np.isfinite(y) * np.isfinite(yerr)
        return (y[use] - sigmoid(p, x[use]))/yerr[use]

    nm = int((mrange[1] - mrange[0])/mstep)
    mags = np.linspace(mrange[0], mrange[1], nm)

    hdulist = fits.open(infile)
    tbdata = hdulist[1].data

    # Observed objects excluding F1 and F3 fillers and bad vis_class 2-5
    sel = ((tbdata.field('survey_class') >= 3) *
           (tbdata.field('survey_code') > 0) *
           (tbdata.field('r_petro') < 19.8) *
           ((tbdata.field('vis_class') < 2) + (tbdata.field('vis_class') > 5)))
    tbdata = tbdata[sel]
    goodz = (tbdata.field('nq') > 2)
    hdulist.close()

    obshist, edges = np.histogram(tbdata.field(magname), nm, mrange)
    goodhist, edges = np.histogram(tbdata[goodz].field(magname), nm, mrange)
    use = obshist > 0
    mags = mags[use]
    obshist = obshist[use]
    goodhist = goodhist[use]
    fgood = goodhist.astype(float)/obshist
    fsig = fgood/np.sqrt(obshist)
    res = scipy.optimize.leastsq(sigmoid_resid, p0, (mags, fgood, fsig),
                                 xtol=0.001, ftol=0.001, full_output=1)
    popt, cov, info, mesg, ier = res
    chi2 = (info['fvec']**2).sum()
    nu = len(mags) - len(popt)
    print 'sigmoid parameters', popt, 'chi2, nu', chi2, nu

    plt.clf()
    fig, axes = plt.subplots(2, sharex=True, num=1)
    ax = axes[0]
    ax.step(mags, obshist)
    ax.step(mags, goodhist)
    ax.set_ylabel('Frequency')

    ax = axes[1]
    ax.step(mags, goodhist.astype(float)/obshist)
    ax.plot(mags, sigmoid(popt, mags))
    ax.set_xlabel(plotname)
    ax.set_ylabel('Redshift success')
    ax.set_ylim(yrange)
    plt.draw()

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
        print 'plot saved to ', plot_dir + plot_file

def zcomp_g10(infile='G10CosmosCatv01.fits', magname='I_MAG_AUTO_06',
              mrange=(18.8, 25), mstep=0.1, p0=(22, 2, 0.5), 
              yrange=(0.0, 1.1), plot_file='zcomp_g10.pdf'):
    """G10 redshift success rate."""

    plotname = magname
    nm = int((mrange[1] - mrange[0])/mstep)
    mags = np.linspace(mrange[0], mrange[1], nm)
    
    hdulist = fits.open(infile)
    tbdata = hdulist[1].data
    sel = ((149.55 < tbdata.field('RA_06')) * (tbdata.field('RA_06') < 150.65) *
           (1.80 < tbdata.field('DEC_06')) * (tbdata.field('DEC_06') < 2.73) *
           (tbdata.field('STAR_06') == 0))
    tbdata = tbdata[sel]
    goodz = (tbdata.field('Z_USE') < 3)
    hdulist.close()

    obshist, edges = np.histogram(tbdata.field(magname), nm, mrange)
    goodhist, edges = np.histogram(tbdata[goodz].field(magname), nm, mrange)
    use = obshist > 0
    mags = mags[use]
    obshist = obshist[use]
    goodhist = goodhist[use]
    fgood = goodhist.astype(float)/obshist
    fsig = fgood/np.sqrt(obshist)
    # res = scipy.optimize.leastsq(sigmoid_resid, p0, (mags, fgood, fsig),
    #                              xtol=0.001, ftol=0.001, full_output=1)
    # popt, cov, info, mesg, ier = res
    # chi2 = (info['fvec']**2).sum()
    # nu = len(mags) - len(popt)
    # print 'sigmoid parameters', popt, 'chi2, nu', chi2, nu

    plt.clf()
    fig, axes = plt.subplots(2, sharex=True, num=1)
    ax = axes[0]
    ax.step(mags, obshist)
    ax.step(mags, goodhist)
    ax.set_ylabel('Frequency')
    
    ax = axes[1]
    ax.step(mags, goodhist.astype(float)/obshist)
    # ax.plot(mags, sigmoid(popt, mags))
    ax.set_xlabel(plotname)
    ax.set_ylabel('Redshift success')
    ax.set_ylim(yrange)
    plt.draw()

    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
        print 'plot saved to ', plot_dir + plot_file

def magErrHist(infile):
    """Histogram SDSS mag errors"""

    band = 'ugriz'
    hdulist = fits.open(infile)
    tbdata = hdulist[1].data
    magerr = (tbdata.field('magerr'))
    hdulist.close()

    plt.figure()
    for iband in range(5):
        plt.subplot(5, 1, iband+1)
        plt.hist(magerr[:,iband], 20, [0.0, 0.5])
        plt.xlabel(r'$M_%c$' % band[iband])
        plt.ylabel(r'$N$')
    plt.show()

def coaddTest(infile='lsb_cand.fits'):
    """Check results of eyeballing coadd galaxy candidates"""

    nm = 30
    dm = 0.2
    mags = np.linspace(14, 20, nm+1)
    ns = 50
    ds = 0.2
    sbs = np.linspace(18, 28, ns+1)
    
    hdulist = fits.open(infile)
    tbdata = hdulist[1].data
    mag = tbdata.field('petroMagCor_r')
    r50 = tbdata.field('petror50_r')
    vc = tbdata.field('vis_class')
    good = vc == 1
    bad = vc > 1
    sb = mag + 2.5*np.log10(2*math.pi*r50**2)
    hdulist.close()

    plt.clf()
    plt.plot(mag[good], sb[good], 'b,', mag[bad], sb[bad], 'r,')
    tothist, xedges, yedges = np.histogram2d(sb, mag, [sbs, mags])
    goodhist, xedges, yedges = np.histogram2d(sb[good], mag[good], [sbs, mags])
    goodfrac = goodhist/tothist
    zero = tothist == 0
    goodfrac[zero] = 1
    badhist, xedges, yedges = np.histogram2d(sb[bad], mag[bad], [sbs, mags])
    badfrac = badhist/tothist
    badfrac[zero] = 1
##     plt.contour(mags[0:12] + 0.25, sbs[0:16] + 0.25, tothist,
##                 levels=[1, 2, 4, 8, 16, 32, 64])
##     plt.contour(mags[0:nm] + dm/2, sbs[0:ns] + ds/2, badfrac,
##                 levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    plt.axis([16, 20, 18, 28])
    plt.xlabel('r mag'); plt.ylabel('r sb')
    plt.draw()


def vis_class_copy(vcfile='v5/tiling-cat-v5-dr6-ch.fits',
                   infile='v10/post_class.txt.bak',
                   outfile='v10/post_class.txt'):
    """Copy and reassign Cullan's classifications to post_class.txt."""

    # Mapping from old to new classes.  Anything else will have class 0
    new_class = {4:4, 5:3, 6:1}

    # Read vis_classes
    hdulist = fits.open(vcfile)
    tbdata = hdulist[1].data
    cata_index = tbdata.field('cata_index')
    vis_class = tbdata.field('vis_class')
    hdulist.close()

    # vis-class[cataid] dictionary
    vc = {}
    for i in xrange(len(cata_index)):
        vc[cata_index[i]] = vis_class[i]
        
    # Add classifications to post_class.txt
    fin = open(infile, 'r')
    fout = open(outfile, 'w')
    first = 1
##    pdb.set_trace()
    for line in fin:
        line = line.rstrip('\n')
        if first:
            print >> fout, line, 'post_class'
            first = 0
        else:
            data = line.split()
            cataid = int(data[0])
            try:
                nc = new_class[vc[cataid]]
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                nc = 0
            print >> fout, line, nc
                
    fin.close()
    fout.close()


def jpg_disp_wx(jpgfile='ex.jpg'):
    """Read and siaplay a jpg image using wx.
    See http://www.velocityreviews.com/forums/t353096-display-of-jpeg-images-from-python.html"""

    import wx
    a = wx.PySimpleApp()
    wximg = wx.Image(jpgfile) #, wx.BITMAP_TYPE_JPEG)
    wxbmp=wximg.ConvertToBitmap()
    f = wx.Frame(None, -1, "Show JPEG demo")
    f.SetSize( wxbmp.GetSize() )
    wx.StaticBitmap(f,-1,wxbmp,(0,0))
    f.Show(True)
    def callback(evt,a=a,f=f):
        # Closes the window upon any keypress
        f.Close()
        a.ExitMainLoop()
        wx.EVT_CHAR(f, callback)
        a.MainLoop()

def jpg_disp_tk(jpgfile='ex.jpg'):
    """Read and display a jpg image using tkinter"""

    import Image
    import ImageTk
    import Tkinter
    
    def button_click_exit_mainloop (event):
        event.widget.quit() # this will cause mainloop to unblock.

    root = Tkinter.Tk()
    image1 = Image.open(jpgfile)
    root.geometry('%dx%d' % (image1.size[0],image1.size[1]))
    tkpi = ImageTk.PhotoImage(image1)
    label_image = Tkinter.Label(root, image=tkpi)
    label_image.place(x=0,y=0,width=image1.size[0],height=image1.size[1])
    root.title(jpgfile)
    
    prompt = '      Press any key      '
    label1 = Tkinter.Label(root, text=prompt, width=len(prompt))
    label1.pack()

    def key(event):
        if event.char == event.keysym:
            msg = 'Normal Key %r' % event.char
        elif len(event.char) == 1:
            msg = 'Punctuation Key %r (%r)' % (event.keysym, event.char)
        else:
            msg = 'Special Key %r' % event.keysym
        label1.config(text=msg)

    root.bind_all('<Key>', key)

    root.mainloop()
    
def jpg_disp(jpgfile='ex.jpg'):
    """Read and display a jpg image"""

    import curses, Image
    image1 = Image.open(jpgfile)

    plt.clf()
    plt.imshow(image1)
    plt.draw()

def spiral(rmax):
    """Generate a square spiral"""

    c0 = np.array([0,0])
    c = np.array([0,0])
    i = 1
    idict = {}
    plt.clf()

    for r in range(1,rmax):
        c = c0 - (r, r)
        for dir in (np.array((1,0)), np.array((0,1)),
                    np.array((-1,0)), np.array((0,-1))):
            c += dir
            while max(abs(c - c0)) <= r:
                plt.text(c[0], c[1], '%d' % i)
                idict[str(c)] = i
##                 print c, i
                i += 1
                c += dir
            c -= dir
    plt.axis([-rmax, rmax, -rmax, rmax])
    plt.draw()
    print idict

def coordDist():
    """Coord distance vs z plot for Omega_m = 1 Universe"""
    def Dc(z):
        return 2*3e5/100*(1 - (1 + z)**-0.5)
    
    z = np.arange(0,1,0.01)
    plt.clf()
    plt.plot(z, Dc(z))
    plt.draw()

def fileTest(file='/scratch/junk'):
    """Find max file size"""

    f = open(file, 'w')
    try:
        while 1:
            print >> f, 'Testing'
    except:
        f.close()

def button():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.random.rand(10))

    def onclick(event):
        print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
            event.button, event.x, event.y, event.xdata, event.ydata)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

def fitsTest(infile='test.fits', outfile='test2.fits'):
    """Test fits table creation & modification."""
    
    counts = np.array([312, 334, 308, 317])
    names = np.array(['NGC1', 'NGC2', 'NGC3', 'NGC4'])
    c1 = fits.Column(name='target', format='10A', array=names)
    c2 = fits.Column(name='counts', format='J', unit='DN', array=counts)
    tbhdu = fits.new_table([c1, c2])
    tbhdu.writeto('test.fits', clobber=True)

    hdulist = fits.open(infile)
    header = hdulist[1].header
    tbdata = hdulist[1].data
    cols = hdulist[1].columns
    cols.add_col(fits.Column(name='bpt_type', format='A'))
    tbhdu = fits.new_table(fits.ColDefs(cols))
    tbhdu.writeto(outfile, clobber=True)
    hdulist.close()

def csvTest():
    group_dict = {}
    data = np.loadtxt('G3CRef194v04.dat', dtype=np.int32, skiprows=1)
    for pair in data:
        group_dict[pair[0]] = pair[1]
    return group_dict


def cone_plot(infile='kcorrz01.fits', z_limits=(0, 0.5), alpha=0.5,
              sdss=False, plot_file=None):
    """GAMA cone plots"""

    def setup_axes(fig, rect, extremes):
        """Axes for a GAMA region"""

        # rotate a bit for better orientation
        tr_rotate = Affine2D().translate(-0.5*(extremes[0] + extremes[1]), 0)

        # scale degree to radians
        tr_scale = Affine2D().scale(np.pi/180., 1.)

        tr = tr_rotate + tr_scale + PolarAxes.PolarTransform()

        grid_locator1 = MaxNLocator(5)
##         tick_formatter1 = angle_helper.FormatterHMS()

        grid_locator2 = MaxNLocator(5)

##         ra0, ra1 = 8.*15, 14.*15
##         cz0, cz1 = 0, 14000
        grid_helper = floating_axes.GridHelperCurveLinear(
            tr, extremes=extremes,
            grid_locator1=grid_locator1,
            grid_locator2=grid_locator2)
##                                         tick_formatter1=tick_formatter1,
##                                         tick_formatter2=None,
##                                         )

        ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
        fig.add_subplot(ax1)

        # adjust axis
        ax1.axis["left"].set_axis_direction("bottom")
        ax1.axis["right"].set_axis_direction("top")

        ax1.axis["bottom"].set_visible(False)
        ax1.axis["top"].set_axis_direction("bottom")
        ax1.axis["top"].toggle(ticklabels=True, label=True)
        ax1.axis["top"].major_ticklabels.set_axis_direction("top")
        ax1.axis["top"].label.set_axis_direction("top")

        ax1.axis["left"].label.set_text(r'z')
        ax1.axis["top"].label.set_text('RA')


        # create a parasite axes whose transData in RA, z
        aux_ax = ax1.get_aux_axes(tr)

        aux_ax.patch = ax1.patch # for aux_ax to have a clip path as in ax
        ax1.patch.zorder=0.9     # but this has a side effect that the patch is
                                 # drawn twice, and possibly over some other
                                 # artists. So, we decrease the zorder a bit to
                                 # prevent this.

        return ax1, aux_ax


    hdulist = fits.open(infile)
    tbdata = hdulist[1].data
    if sdss:
        sel = np.isfinite(tbdata.field('zconf'))
        tbdata = tbdata[sel]
        sel = ((tbdata.field('zconf') > 0.8) * 
               (tbdata.field('dec') > -2) * (tbdata.field('dec') < 3) * 
               (tbdata.field('petromagcor_r') < 17.6))
    else:
        sel = ((tbdata.field('survey_class') >= 3) * (tbdata.field('nQ') >= 3) *
               (tbdata.field('r_petro') < 19.8) *
               ((tbdata.field('vis_class') < 2) + (tbdata.field('vis_class') > 5)))
    tbdata = tbdata[sel]
    z = tbdata.field('z')
    ra = tbdata.field('RA')
    dec = tbdata.field('DEC')
    print len(z), 'objects selected'
    
    ra_limits = ((129, 141), (174, 186), (211.5, 223.5))
    dec_limits = ((-2, 3), (-3, 2), (-2, 3))
    rect = (311, 312, 313)
    label = ('G09', 'G12', 'G15')
    
    fig = plt.figure(1, figsize=(8, 4))
    fig.clf()
    fig.subplots_adjust(hspace=0.2, left=0.05, right=0.95)

    for ireg in range(3):
        ax, aux_ax = setup_axes(fig, rect[ireg], ra_limits[ireg] + z_limits)
        sel = ((ra >= ra_limits[ireg][0]) * (ra <= ra_limits[ireg][1]) * 
               (dec >= dec_limits[ireg][0]) * (dec <= dec_limits[ireg][1]))
        aux_ax.scatter(ra[sel], z[sel], s=0.01, alpha=alpha)
        ax.text(0.05, 0.7, label[ireg], transform = ax.transAxes)

    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')
    
def deblend_fix_flux(infile='TilingCatv16_dr6.fits',
                     fixfile='deblend-fix-fluxes.fits',
                     outfile='TilingCatv16_dr6_fix.fits'):
    """Correct magnitudes of incorrectly deblended sources"""

    # Read fixed magnitudes
    hdulist = fits.open(fixfile)
    tbdata = hdulist[1].data
    cataid = tbdata.field('cataid')
    extinction_r = tbdata.field('extinction_r')
    petro_u_fix = {}
    model_u_fix = {}
    petro_g_fix = {}
    model_g_fix = {}
    petro_r_fix = {}
    model_r_fix = {}
    petro_i_fix = {}
    model_i_fix = {}
    petro_z_fix = {}
    model_z_fix = {}
    for i in range(len(cataid)):
        id = cataid[i]
        petro_u_fix[id] = tbdata.field('PETROMAG_U_ADJ')[i] - 1.87*extinction_r[i]
        model_u_fix[id] = tbdata.field('MODELMAG_U_ADJ')[i] - 1.87*extinction_r[i]
        petro_g_fix[id] = tbdata.field('PETROMAG_G_ADJ')[i] - 1.38*extinction_r[i]
        model_g_fix[id] = tbdata.field('MODELMAG_G_ADJ')[i] - 1.38*extinction_r[i]
        petro_r_fix[id] = tbdata.field('PETROMAG_R_ADJ')[i] - extinction_r[i]
        model_r_fix[id] = tbdata.field('MODELMAG_R_ADJ')[i] - extinction_r[i]
        petro_i_fix[id] = tbdata.field('PETROMAG_I_ADJ')[i] - 0.76*extinction_r[i]
        model_i_fix[id] = tbdata.field('MODELMAG_I_ADJ')[i] - 0.76*extinction_r[i]
        petro_z_fix[id] = tbdata.field('PETROMAG_Z_ADJ')[i] - 0.54*extinction_r[i]
        model_z_fix[id] = tbdata.field('MODELMAG_Z_ADJ')[i] - 0.54*extinction_r[i]
    hdulist.close()
    
    # Read input file
    hdulist = fits.open(infile)
    tbdata = hdulist[1].data
    cataid = tbdata.field('cataid')
    for i in range(len(cataid)):
        id = cataid[i]
        if petro_u_fix.has_key(id):
            tbdata.field('petroMagCor_u')[i] = petro_u_fix[id]
            tbdata.field('modelMagCor_u')[i] = model_u_fix[id]
            tbdata.field('petroMagCor_g')[i] = petro_g_fix[id]
            tbdata.field('modelMagCor_g')[i] = model_g_fix[id]
            tbdata.field('petroMagCor_r')[i] = petro_r_fix[id]
            tbdata.field('modelMagCor_r')[i] = model_r_fix[id]
            tbdata.field('petroMagCor_i')[i] = petro_i_fix[id]
            tbdata.field('modelMagCor_i')[i] = model_i_fix[id]
            tbdata.field('petroMagCor_z')[i] = petro_z_fix[id]
            tbdata.field('modelMagCor_z')[i] = model_z_fix[id]
            
    hdu = fits.BinTableHDU(tbdata, header=hdulist[1].header)
    hdu.writeto(outfile, clobber=True)


def bpt_sdss(infile, outfile):
    """Classify galaxy according to BPT: unknown, quiescent, starforming,
    composite or agn.  See Kewley et al 2006, MNRAS, 372, 961."""
    
    # Read line data
    hdulist = fits.open('/export/scratch/loveday/sdss/data/DR7/gal_line_dr7_v5_2.fit.gz', memmap=True)
    header = hdulist[1].header
    tbdata = hdulist[1].data

    # Default classification is unknown
    bpt_type = np.array(['u']*len(tbdata))

    # Require s/n >= 3 in all four lines,otherwise classify as quiescent
    # See http://www.mpa-garching.mpg.de/SDSS/DR7/raw_data.html for
    # error correction factors
    idx = ((tbdata.field('NII_6584_FLUX')/
            tbdata.field('NII_6584_FLUX_ERR')/2.039 >= 3) *
           (tbdata.field('H_ALPHA_FLUX')/
           tbdata.field('H_ALPHA_FLUX_ERR')/2.473 >= 3) *
           (tbdata.field('OIII_5007_FLUX')/
            tbdata.field('OIII_5007_FLUX_ERR')/1.566 >= 3) *
           (tbdata.field('H_BETA_FLUX')/
           tbdata.field('H_BETA_FLUX_ERR')/1.882 >= 3))
    bpt_type[~idx] = 'q'
    print len(np.flatnonzero(~idx)), ' quiescent galaxies'

    # For remaining galaxies, classify according to flux ratios
    NII_Halpha = np.log10(tbdata.field('NII_6584_FLUX')/
                          tbdata.field('H_ALPHA_FLUX'))
    OIII_Hbeta = np.log10(tbdata.field('OIII_5007_FLUX')/
                          tbdata.field('H_BETA_FLUX'))

    plt.clf()
    
    # Starforming
    idx = (bpt_type != 'q')*(OIII_Hbeta < 0.61/(NII_Halpha - 0.05) + 1.3)
    bpt_type[idx] = 's'
    plt.scatter(NII_Halpha[idx], OIII_Hbeta[idx], s=0.01, c='k',
                edgecolors='face')
    print len(np.flatnonzero(idx)), ' starforming galaxies'
    
    # Composite
    idx = ((bpt_type != 'q')*(OIII_Hbeta > 0.61/(NII_Halpha - 0.05) + 1.3)*
           (OIII_Hbeta < 0.61/(NII_Halpha - 0.47) + 1.19))
    bpt_type[idx] = 'c'
    plt.scatter(NII_Halpha[idx], OIII_Hbeta[idx], s=0.01, c='b',
                edgecolors='face')
    print len(np.flatnonzero(idx)), ' composite galaxies'
    
    # AGN
    idx = (bpt_type != 'q')*(OIII_Hbeta > 0.61/(NII_Halpha - 0.47) + 1.19)
    bpt_type[idx] = 'a'
    plt.scatter(NII_Halpha[idx], OIII_Hbeta[idx], s=0.01, c='r',
                edgecolors='face')
    print len(np.flatnonzero(idx)), ' AGN galaxies'
    plt.axis([-2, 1, -1.2, 1.5])
    plt.draw()
    
    # BPT dictionary
    key = [str(tbdata[i].field('PLATEID')) + '_' + str(tbdata[i].field('FIBERID')) for i in xrange(len(tbdata))]
    bpt_dict = {key[i]:bpt_type[i] for i in xrange(len(tbdata))}
    hdulist.close()
    
    # Read galaxies from input file and add bpt_type column
    hdulist = fits.open(infile)
    header = hdulist[1].header
    tbdata = hdulist[1].data
    cols = hdulist[1].columns

    # BPT lookup
    key = [str(tbdata[i].field('PLATE')) + '_' + str(tbdata[i].field('FIBERID')) for i in xrange(len(tbdata))]
    bpt_type = np.array(['u']*len(tbdata))
    for i in xrange(len(tbdata)):    
        try:
            bpt_type[i] = bpt_dict[key[i]]
        except:
            bpt_type[i] = 'u'

    
    cols.add_col(fits.Column(name='bpt_type', format='A', unit='None',
                               null='u', array=bpt_type))
    tbhdu = fits.new_table(fits.ColDefs(cols))
    tbhdu.writeto(outfile, clobber=True)
    hdulist.close()


#def bpt_gama(infile, outfile):
def bpt_gama():
    """Classify galaxy according to BPT: unknown, quiescent, starforming,
    composite or agn.  See Kewley et al 2006, MNRAS, 372, 961."""

    # Read line data
    hdulist = fits.open('EmLinesPhysGAMAII.fits')
    header = hdulist[1].header
    tbdata = hdulist[1].data

    # Default classification is unknown
    bpt_type = np.array(['u']*len(tbdata))

    # Require s/n >= 3 in all four lines,otherwise classify as quiescent
    idx = ((tbdata.field('NIIRFLUX')/tbdata.field('NIIRFLUX_ERR') >= 3) *
           (tbdata.field('HAFLUX')/tbdata.field('HAFLUX_ERR') >= 3) *
           (tbdata.field('OIIIRFLUX')/tbdata.field('OIIIRFLUX_ERR') >= 3) *
           (tbdata.field('HBFLUX')/tbdata.field('HBFLUX_ERR') >= 3))
    bpt_type[~idx] = 'q'
    print len(np.flatnonzero(~idx)), ' quiescent galaxies'

    # For remaining galaxies, classify according to flux ratios
    NII_Halpha = np.log10(tbdata.field('NIIRFLUX')/tbdata.field('HAFLUX'))
    OIII_Hbeta = np.log10(tbdata.field('OIIIRFLUX')/tbdata.field('HBFLUX'))

    plt.clf()

    # Starforming
    idx = (bpt_type != 'q')*(OIII_Hbeta < 0.61/(NII_Halpha - 0.05) + 1.3)
    bpt_type[idx] = 's'
    plt.scatter(NII_Halpha[idx], OIII_Hbeta[idx], s=0.01, c='k',
                edgecolors='face')
    print len(np.flatnonzero(idx)), ' starforming galaxies'

    # Composite
    idx = ((bpt_type != 'q') * (OIII_Hbeta > 0.61/(NII_Halpha - 0.05) + 1.3) *
           (OIII_Hbeta < 0.61/(NII_Halpha - 0.47) + 1.19))
    bpt_type[idx] = 'c'
    plt.scatter(NII_Halpha[idx], OIII_Hbeta[idx], s=0.01, c='b',
                edgecolors='face')
    print len(np.flatnonzero(idx)), ' composite galaxies'

    # AGN
    idx = (bpt_type != 'q')*(OIII_Hbeta > 0.61/(NII_Halpha - 0.47) + 1.19)
    bpt_type[idx] = 'a'
    plt.scatter(NII_Halpha[idx], OIII_Hbeta[idx], s=0.01, c='r',
                edgecolors='face')
    print len(np.flatnonzero(idx)), ' AGN galaxies'

    idx = (bpt_type == 'u')
    print len(np.flatnonzero(idx)), ' unclassified'

    plt.axis([-2, 1, -1.2, 1.5])
    plt.draw()

#    # BPT dictionary
#    key = [str(tbdata[i].field('PLATEID')) + '_' +
#           str(tbdata[i].field('FIBERID')) for i in xrange(len(tbdata))]
#    bpt_dict = {key[i]: bpt_type[i] for i in xrange(len(tbdata))}
#    hdulist.close()
#
#    # Read galaxies from input file and add bpt_type column
#    hdulist = fits.open(infile)
#    header = hdulist[1].header
#    tbdata = hdulist[1].data
#    cols = hdulist[1].columns
#
#    # BPT lookup
#    key = [str(tbdata[i].field('PLATE')) + '_' +
#           str(tbdata[i].field('FIBERID')) for i in xrange(len(tbdata))]
#    bpt_type = np.array(['u']*len(tbdata))
#    for i in xrange(len(tbdata)):
#        try:
#            bpt_type[i] = bpt_dict[key[i]]
#        except:
#            bpt_type[i] = 'u'
#
#    cols.add_col(fits.Column(name='bpt_type', format='A', unit='None',
#                             null='u', array=bpt_type))
#    tbhdu = fits.new_table(fits.ColDefs(cols))
#    tbhdu.writeto(outfile, clobber=True)
#    hdulist.close()


def groupProps(infile, outfile):
    """Add group properties to GAMA catalogue."""

    hdulist = fits.open(infile, memmap=True)
    header = hdulist[1].header
    tbdata = hdulist[1].data
    cols = hdulist[1].columns
    cataid = tbdata.field('cataid')

    # Initialize all group_ids to zero to allow for ungrouped galaxies
    group_id = {id: 0 for id in cataid}

    # Read group ids for grouped galaxies
    file = os.environ['GAMA_DATA'] + '/groups/G3Cv04/G3CRef194v04.dat'
    data = np.loadtxt(file, dtype=np.int32)
    for item in data:
        group_id[item[0]] = item[1]
        
    # Defaults for ungrouped galaxies
    group_mass = {0: -1}
    vel_disp = {0: -1}
    n_fof = {0: -1}
    rad50 = {0: -1}
    rel_den = {0: -1}
        
    # Read group data (see Robotham et al 2011 sec 4.3 re 'A' factor)
    file = os.environ['GAMA_DATA'] + '/groups/G3Cv04/G3CFoFGroup194v04.dat'
    data = np.loadtxt(file, skiprows=1)
    for item in data:
        grpid = int(item[0])
        n_fof[grpid] = int(item[1])
        rad50[grpid] = item[7]
        rel_den[grpid] = item[10]
        vel_disp[grpid] = item[13]
        A = -1.2 + 20.7/math.sqrt(item[1]) + 2.3/math.sqrt(item[6])
        group_mass[grpid] = A*item[18]
        
    # Assign group properties for each galaxy
    nf = np.array([n_fof[group_id[id]] for id in cataid])
    radii = np.array([rad50[group_id[id]] for id in cataid])
    densities = np.array([rel_den[group_id[id]] for id in cataid])
    masses = np.array([group_mass[group_id[id]] for id in cataid])
    sigmas = np.array([vel_disp[group_id[id]] for id in cataid])

    # add_col gives spurious(?) AttributeErrors; ignore these
    try:
        cols.add_col(fits.Column(name='group_nfof', format='I', unit='None',
                                   array=nf))
    except:
        pass
    try:
        cols.add_col(fits.Column(name='group_rad50', format='E', unit='Mpc/h',
                                   array=radii))
    except:
        pass
    try:
        cols.add_col(fits.Column(name='group_relden', format='E', unit='h/Mpc',
                                   array=densities))
    except:
        pass
    try:
        cols.add_col(fits.Column(name='group_sigma', format='E', unit='km/s',
                                   array=sigmas))
    except:
        pass
    try:
        cols.add_col(fits.Column(name='group_mass', format='E', unit='M_sun',
                                   array=masses))
    except:
        pass

    tbhdu = fits.new_table(fits.ColDefs(cols), header=header)
    tbhdu.writeto(outfile, clobber=True)
    hdulist.close()
    
def add_colour(infile, outfile, limits=(-23, -15.5, 0.1, 1.2), 
               zrange=(0.002, 0.65)):
    """Add blue/red colour classification according to Loveday+ 2012 eqn 3."""

    def colourCut(mag):
        """Return g-r colour cut corresponding to r-band absolute mag"""
        return 0.15 - 0.03*mag

    hdulist = fits.open(infile)
    header = hdulist[1].header
    H0 = 100.0
    omega_l = header['OMEGA_L']
    cosmo = util.CosmoLookup(H0, omega_l, zrange)

    tbdata = hdulist[1].data
    cols = hdulist[1].columns
    z = tbdata.field('z')
    M_r = tbdata.field('r_petro') - cosmo.dist_mod(z) - tbdata.field('kcorr_r')
    gr = ((tbdata.field('g_model') - tbdata.field('kcorr_g')) - 
          (tbdata.field('r_model') - tbdata.field('kcorr_r')))

    colours = np.array(len(z) * ['u'])
    grcut = colourCut(M_r)
    sel = (gr < grcut)
    colours[sel] = 'b'
    sel = (gr >= grcut)
    colours[sel] = 'r'

    cols.add_col(fits.Column(name='colour', format='A', unit='None',
                               array=colours))
    tbhdu = fits.new_table(fits.ColDefs(cols), header=header)
    tbhdu.writeto(outfile, clobber=True)
    hdulist.close()
    
    mags = np.linspace(limits[0], limits[1], 50)
    cutline = colourCut(mags)
    cut = colourCut(M_r)
    nb = (gr < cut).sum()
    nr = (gr > cut).sum()
    print 'nblue, nred = ', nb, nr

    zlims = [[0.002, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.65]]
    fig = plt.figure(1)
    plt.clf()
    grid = AxesGrid(fig, 111, # similar to subplot(111)
                    nrows_ncols = (2,2), # creates nr*nc grid of axes
                    axes_pad=0.0, # pad between axes in inch.
                    aspect=False)

    for iz in range(4):
        ax = grid[iz]
        idx = (zlims[iz][0] <= z)*(z < zlims[iz][1])
        hist, xedges, yedges = np.histogram2d(gr[idx], M_r[idx], 50, 
                                              [limits[2:4], limits[0:2]])
        ax.contour(hist, 10, extent=limits)
        ax.plot(mags, cutline, '-')
        ax.axis(limits)
        ax.set_ylabel(r'$(g-r)$'); ax.set_xlabel(r'$M_r')
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
        title = r'${:2.1f} < z < {:2.1f}$'.format(zlims[iz][0], zlims[iz][1])
        ax.text(0.6, 0.9, title, transform = ax.transAxes)
        nb = (gr[idx] < cut[idx]).sum()
        nr = (gr[idx] >= cut[idx]).sum()
        print zlims[iz], ' nblue, nred = ', nb, nr
    plt.draw()

def sbPlot(infile):
    """Plot SB histogram and completeness."""

    # Photometric completeness from Blanton et al 2005, ApJ, 631, 208, Table 1
    # Modified to remove decline at bright end and to prevent negative 
    # completeness values at faint end
    sb_tab = (18, 19, 19.46, 19.79, 20.11, 20.44, 20.76, 21.09, 21.41, 21.74,
              22.06, 22.39, 22.71, 23.04, 23.36, 23.69, 24.01, 24.34, 26.00)
    im_tab = (1.0, 1.0, 0.99, 0.97, 0.98, 0.98, 0.98, 0.97, 0.96, 0.96, 0.97,
              0.94, 0.86, 0.84, 0.76, 0.63, 0.44, 0.33, 0.01)

    hdulist = fits.open(infile)
    header = hdulist[1].header
    tbdata = hdulist[1].data

    rpet = tbdata.field('petroMagCor_r')
    r50 = tbdata.field('petror50_r')
    sb = rpet + 2.5*np.log10(2*math.pi*r50*r50)
    hdulist.close()
    
    plt.clf()
    plt.plot(sb_tab, im_tab)
    plt.hist(sb, bins=80, range=(18, 26), normed=True, histtype='step')
    plt.axis([18, 26, 0.0, 1.1])
    plt.xlabel(r'$\mu_{50, r}\ /$ mag arcsec$^{-2}$')
    plt.ylabel('Imaging completeness')
    plt.draw()
    
def ellipse(aa, bb, theta):
    """Test ellipse plotting."""    

    plt.clf()
    el = Ellipse((0,0), aa, bb, theta, fc='none', linestyle='dotted')
    ax = plt.subplot(111, aspect='equal')
    ax.add_artist(el)
    ax.axis((-1,1,-1,1))
    plt.draw()
    
def error_ellipse(sigx, sigy, corr):
    """Test error ellipse plotting."""    

    sigxy = sigx*sigy*corr
    cov = ((sigx**2, sigxy), (sigxy, sigy**2))
    w, v = np.linalg.eig(cov)
    aa = math.sqrt(w[0])
    bb = math.sqrt(w[1])
    try:
        theta = (180.0/math.pi)*0.5*math.atan(2*sigxy/(sigx**2 - sigy**2))
    except:
        theta = 0.0
    ellipse(aa, bb, theta)

def projTest():
    """Distribution of apparent axis ratio of circular disk
    viewed at random angles."""

    ntest = 1000
    q = []
    for i in xrange(ntest):
        x, y = 0, 1
        while x < y:
            theta = 0.5*math.pi*np.random.rand(1)
            x = math.sin(theta)
            y = np.random.rand(1)
        q += (math.cos(theta),)
    
    plt.clf()
    plt.hist(q, bins=20)
    plt.draw()
    
def read_mask(filename='/export/scratch/loveday/gama/masks/mask_year3_v4/mask_0/tilmask.np_08000.sm_0.20.ng_15_50.dat'):
    """Read Peder's GAMA mask"""

    data = np.fromfile(filename)
    mask = np.reshape(data, (8000,8000), 'F')
    plt.clf()
    plt.imshow(mask)
    plt.draw()

def tycho_mask():
    """Plot MASK_IC as function of d/Rs"""

    x = np.arange(1.0, 5.0, 0.1)
    y = 1.0/x
    plt.clf()
    plt.plot(x,y)
    plt.draw()
    
def kcorr_check():
    """Check kcorrect transformation between different zref."""

    def read_kz(file, ngal=20):
        hdulist = fits.open(file)
        tbdata = hdulist[1].data
        kcoeff = tbdata.field('kcoeffr')[0:ngal,:]
        return kcoeff
    
    zref = 0.1
    ngal = 20
    kc0 = read_kz('/export/scratch/loveday/gama/v16/dr6/tonryz0/kcorrz.fits')
    kc01 = read_kz('/export/scratch/loveday/gama/v16/dr6/tonryz01/kcorrz.fits')
    zbins = np.arange(0.05, 0.5, 0.05)
    plt.clf()
    for igal in xrange(ngal):
        kz1_obs = np.polyval(kc01[igal], zbins-zref)
        kz1_pred = k = np.polyval(kc0[igal], zbins) - np.polyval(kc0[igal], zref) - 2.5*math.log10(1 + zref)
        dkz = kz1_pred - kz1_obs
    
        plt.plot(kz1_obs, dkz)
    plt.xlabel('k(z, zref)')
    plt.ylabel('Predicted - observed')
    plt.draw()

def mag_select(infile, outfile):
    """Select galaxies on apparent mag to give flat mag distribution between
    17.5 and 19.8."""
    
    # Read galaxies from input file
    hdulist = fits.open(infile)
    header = hdulist[1].header
    tbdata = hdulist[1].data
    mag = tbdata.field('petroMagCor_r')
    ngal = len(mag)
    print ngal, ' galaxies read'
    
    # Histogram the data by magnitude
    mmin = 17.5
    mmax = 19.8
    mstep = 0.1    
    nbin = int((mmax - mmin)/mstep) + 1
    mbins = np.linspace(mmin, mmax, nbin)
    hist, edges = np.histogram(mag, mbins)
    ibin = np.floor((mag - mmin)/mstep).astype(int)
    
    # Select galaxies with p inversely proportional to counts in bin
    # (zero for galaxies outside mag range)
    p = np.zeros(ngal)
    idx = (ibin >=0)*(ibin < nbin)
    p[idx] = float(min(hist))/hist[ibin[idx]]
    randoms = np.random.random(ngal)
    idx = p > randoms
    
    newtbdata = tbdata[idx]
    nout = len(newtbdata)
    hdu = fits.BinTableHDU(newtbdata)
    hdu.writeto(outfile, clobber=True)
    hdulist.close()
    print nout, ' galaxies selected'

def k_comp(infile1='kcorr_z00v01.fits', infile2='../mlum/kcorr.fits'):
    """Compare k-corrections between ugriz and FNugrizYJHK."""    

    # Read galaxies from input file
    hdulist = fits.open(infile1)
    header = hdulist[1].header
    tbdata = hdulist[1].data
    kcorr1 = tbdata.field('kcorr')
    pcu1 = tbdata.field('kcoeffu')
    hdulist.close()
    print kcorr1[0], pcu1[0]
    
    hdulist = fits.open(infile2)
    header = hdulist[1].header
    tbdata = hdulist[1].data
    kcorr2 = tbdata.field('kcorr')
    pc2 = tbdata.field('pcoeff')
    hdulist.close()
    print kcorr2[0][2:7], np.reshape(pc2[0], (5, 11)).transpose()[2]
    pdb.set_trace()

def abs_vals(infile='kcorr.fits', outfile='absmag.fits'):
    """Output file of ra, dec, abs mag"""
             
    hdulist = fits.open(infile)
    header = hdulist[1].header
    H0 = 100.0
    omega_l = header['OMEGA_L']
    z0 = header['Z0']
    area = header['AREA']
    area *= (math.pi/180.0)*(math.pi/180.0)
    zmin = 0.002
    zmax = 0.5
    cosmo = util.CosmoLookup(H0, omega_l, (zmin, zmax))

    print 'H0, omega_l, z0, area/Sr = ', H0, omega_l, z0, area

    tbdata = hdulist[1].data
    nq = tbdata.field('nq')
    idx = ((tbdata.field('nq') > 2) * (tbdata.field('z_tonry') > zmin) *
           (tbdata.field('z_tonry') < zmax))
    tbdata = tbdata[idx]
    ra = tbdata.field('ra')         
    dec = tbdata.field('dec')
    z = tbdata.field('z_tonry')
    kc = tbdata.field('kcorr_r')
    M_r_pet = tbdata.field('petromagcor_r') - 5*np.log10(cosmo.dl(z)) - 25 - kc
    M_r_ser = tbdata.field('gal_mag_10re_r') - 5*np.log10(cosmo.dl(z)) - 25 - kc
    cols = fits.ColDefs([
        fits.Column(name='ra', format='E', array=ra),
        fits.Column(name='dec', format='E', array=dec),
        fits.Column(name='z', format='E', array=z),
        fits.Column(name='M_r_pet', format='E', array=M_r_pet),
        fits.Column(name='M_r_ser', format='E', array=M_r_ser)
        ])
    tbhdu = fits.new_table(cols)
    tbhdu.writeto(outfile, clobber=True)
    
def ss_plot():
    """Plot Solar System data"""
    
    group_dict = {}
    data = np.loadtxt('SolarSystem_data.txt',
                      dtype=([('name', np.str, 16),
                              ('radius_km', np.float),
                              ('radius_earth', np.float),
                              ('volume_km3', np.float),
                              ('volume_earth', np.float),
                              ('mass_kg', np.float),
                              ('mass_earth', np.float),
                              ('density', np.float),
                              ('gravity_mss', np.float),
                              ('gravity_earth', np.float),
                              ('type',np.str, 16)]),
                      delimiter=',', skiprows=2)
    print data['density'], data['radius_km']
    
    plt.clf()
    plt.semilogy(basey=10, nonposy='clip')
    plt.plot(data['density'], data['radius_km'], '.')
    for i in xrange(len(data['name'])):
        plt.text(data['density'][i], data['radius_km'][i], data['name'][i],
                 fontsize=8)
    plt.xlabel('Density (g/cm$^3$)')
    plt.ylabel('Radius (km)')
    plt.draw()

def planet_lt(r, d, a):
    """Luminosity and effective temperature for planets"""

    L_sun = 3.84e26
    sigma = 5.67e-8
    lum = L_sun*r**2*(1-a)/(4*d**2)
    Te = (L_sun * (1-a) /(16 * math.pi * d**2 * sigma))**0.25
    print lum, Te
 
def mgc_phot():
    """Compare B_MGC and SDSS r magnitudes."""
    dat = np.loadtxt('mgc_zcat.txt', usecols=(14, 18, 35))
    sel = (dat[:,1] == 1) * (dat[:,2] < 18) * (dat[:,2] > 10)
    B = dat[sel,0]
    r = dat[sel,2]
    print len(B), 'galaxies'
    print 'B - r = ', np.mean(B-r), np.std(B-r)
    plt.clf()
    plt.scatter(r, B-r, s=0.1)
    plt.xlabel('r mag')
    plt.ylabel('B - r')
    plt.draw()
    
def J3(r0, gamma, rmax):
    """4 pi J3 given xi(r) parameters."""
    print 4*math.pi * r0**gamma / (3 - gamma) * rmax**(3-gamma)

def mock_M_z(infile='/research/astro/gama/loveday/gama/g3cv6/G3CMockGalv06.fits',
             mlim=19.8, zrange=(0.002, 0.5), vol=1):
    """
    Plot M-z relation for mocks.
    """

    def ke_corr(z):
        """k+e corrections for mocks: Robotham et al 2011 eqn (8)."""

        z_p = 0.2
        Q = 1.75
        a = (0.2085, 1.0226, 0.5237, 3.5902, 2.3843)
        corr = Q*z
        for i in range(5):
            corr += a[i]*(z - z_p)**i
        return corr
    
    # Read input file into structure
    hdulist = fits.open(infile)
    header = hdulist[1].header
    tbdata = hdulist[1].data
    hdulist.close()
    sel = (tbdata.field('Volume') == vol)
    tbdata = tbdata[sel]
    H0 = 100.0
    omega_l = 0.75
    cosmo = util.CosmoLookup(H0, omega_l, zrange)

    z = tbdata.field('Z')
    appMag = tbdata.field('Rpetro')
    dm = tbdata.field('DM_100_25_75')
    absMag = appMag - dm - ke_corr(z)

    plt.clf()
    plt.subplot(221)
    zz = np.linspace(0.0, 0.5)
    kec = ke_corr(zz)
    kc = kec - 1.75*zz
    plt.plot(zz, kec)
    plt.xlabel('Redshift')
    plt.ylabel(r'$k+e$ corr')
    plt.subplot(222)
    plt.plot(zz, kc)
    plt.xlabel('Redshift')
    plt.ylabel(r'k-corr')
    plt.subplot(223)
    plt.scatter(z, absMag, s=0.01)
    plt.xlabel('Redshift')
    plt.ylabel(r'$M_r$')
    plt.draw()

def taylor_zmax_comp(infile='StellarMassesv15kc00.fits'):
    """Compare Taylor zmax_19p8 with linear k-correction prediction."""

    global cosmo

    mlim = 19.8
    hdulist = fits.open(infile)
    H0 = 70.0
    omega_l = 0.7
    z0 = 0.0
    zmin = 0.002
    zmax = 0.65
    cosmo = util.CosmoLookup(H0, omega_l, (zmin, zmax))
    tbdata = hdulist[1].data
    idx = ((tbdata.field('survey_class') >= 3) * (tbdata.field('nq') > 2) * 
           (tbdata.field('z_tonry') > zmin) * (tbdata.field('z_tonry') < zmax) *
           (tbdata.field('zmax_19p8') > 0))
    tbdata = tbdata[idx]
    z = tbdata.field('z_tonry')
    zmax_ned = tbdata.field('zmax_19p8')
    appmag = tbdata.field('fitphot_r')
    absmag = tbdata.field('absmag_r')
    kc = appmag - absmag - cosmo.dist_mod(z)
    ngal = len(z)
    kcoeff = np.zeros((ngal, 2))
    kcoeff[:, 0] = kc
    hdulist.close()

    zmax_est = map(lambda i: 
                   jswml.zdm(mlim - absmag[i], kcoeff[i,:], (zmin, zmax), 0.0), 
                   xrange(ngal))

    plt.clf()
    plt.scatter(z, zmax_est - zmax_ned, 0.01)
    plt.ylim(-0.1, 0.5)
    plt.xlabel('z')
    plt.ylabel('Delta zmax (kcorrect est. - Taylor)')
    plt.draw()

def taylor_kcorr(infile='StellarMassesv15kc00.fits'):
    """Taylor k-corrections."""

    mlim = 19.8
    hdulist = fits.open(infile)
    H0 = 70.0
    omega_l = 0.7
    z0 = 0.0
    zmin = 0.002
    zmax = 0.65
    cosmo = util.CosmoLookup(H0, omega_l, (zmin, zmax))
    tbdata = hdulist[1].data
    idx = ((tbdata.field('survey_class') >= 3) * (tbdata.field('nq') > 2) * 
           (tbdata.field('z_tonry') > zmin) * (tbdata.field('z_tonry') < zmax))
    tbdata = tbdata[idx]
    z = tbdata.field('z_tonry')
    appmag = tbdata.field('fitphot_r')
    absmag = tbdata.field('absmag_r')
    kct = appmag - absmag - cosmo.dist_mod(z)
    kcb = tbdata.field('kcorr_r')
    urt = tbdata.field('uminusr')
    urb = ((tbdata.field('u_model') - tbdata.field('kcorr_u')) - 
           (tbdata.field('r_model') - tbdata.field('kcorr_r')))
    hdulist.close()

    plt.clf()
    plt.subplot(311)
    plt.scatter(z, kct, 0.01, c=urt, vmin=0, vmax=3, edgecolors='face')
    plt.ylim(-0.5, 1.5)
    plt.xlabel('z')
    plt.ylabel('kcorr (Taylor)')
    cb = plt.colorbar()
    cb.set_label(r'$(u - r)_{\rm SED}$')
    plt.subplot(312)
    plt.scatter(z, kcb, 0.01, c=urb, vmin=0, vmax=3, edgecolors='face')
    plt.ylim(-0.5, 1.5)
    plt.xlabel('z')
    plt.ylabel('kcorr (Blanton)')
    cb = plt.colorbar()
    cb.set_label(r'$(u - r)_{\rm model}$')
    plt.subplot(313)
    plt.scatter(z, kcb - kct, 0.01, c=urt, vmin=0, vmax=3, edgecolors='face')
    plt.ylim(-0.3, 0.3)
    plt.xlabel('z')
    plt.ylabel('delta kcorr (Blanton - Taylor)')
    cb = plt.colorbar()
    cb.set_label(r'$(u - r)_{\rm SED}$')
    plt.draw()

def taylor_fit_kcorr(infile='StellarMassesv15.fits'):
    """Fit quadratic to Taylor k-corrections."""

    mlim = 19.8
    hdulist = fits.open(infile)
    H0 = 70.0
    omega_l = 0.7
    z0 = 0.0
    zmin = 0.002
    zmax = 0.65
    cosmo = util.CosmoLookup(H0, omega_l, (zmin, zmax))
    tbdata = hdulist[1].data
    idx = ((tbdata.field('survey_class') >= 3) * (tbdata.field('nq') > 2) * 
           (tbdata.field('z_tonry') > zmin) * (tbdata.field('z_tonry') < zmax) *
           (tbdata.field('zmax_19p8') > 0))
    tbdata = tbdata[idx]
    ngal = len(tbdata)
    za = np.zeros((ngal, 3))
    ka = np.zeros((ngal, 3))
    z = tbdata.field('z_tonry')
    zmax_19p8 = tbdata.field('zmax_19p8')
    appmag = tbdata.field('fitphot_r')
    absmag = tbdata.field('absmag_r')
    k_zobs = appmag - absmag - cosmo.dist_mod(z)
    k_zmax = mlim - absmag - cosmo.dist_mod(zmax_19p8)
    za[:,1] = z
    za[:,2] = zmax_19p8
    ka[:,1] = k_zobs
    ka[:,2] = k_zmax
    hdulist.close()

    kcoeff = [np.polyfit(za[i,:], ka[i,:], 2) for i in xrange(ngal)]
    zz = np.linspace(zmin, zmax)
    plt.clf()
    for i in xrange(50):
        plt.plot(za[i,:], ka[i,:], 'o')
        kf = np.polyval(kcoeff[i], zz)
        plt.plot(zz, kf)
    plt.ylim(-1, 1)
    plt.xlabel('z')
    plt.ylabel('kcorr')
    plt.draw()


def taylor_clr_plot(infile='StellarMassesv15.fits', zmin=0.002, zmax=0.65):
    """Taylor colour-mass diagram."""

    hdulist = fits.open(infile)
    tbdata = hdulist[1].data
    idx = ((tbdata.field('survey_class') >= 3) * (tbdata.field('nq') > 2) *
           (tbdata.field('z_tonry') > zmin) * (tbdata.field('z_tonry') < zmax) *
           (tbdata.field('zmax_19p8') > 0) * (tbdata.field('logmstar') > 0))
    tbdata = tbdata[idx]
    hdulist.close()

    plt.clf()
    plt.scatter(tbdata.field('logmstar'), tbdata.field('gminusi_stars'), 0.1)
    plt.xlabel(r'$\log(M/M_\odot)$')
    plt.ylabel(r'$(g-i)^*$')
    plt.draw()

def absMag(appMag, distance):
    """Abs mag given ap[parent and distance in pc."""
    return appMag - 5*math.log10(distance) + 5

def kcorr_plot(infile='kcorr_auto_z00_vec04.fits'):
    """Plot k-correction by colour."""

    mlim = 19.8
    zmin, zmax = 0.002, 0.65
    hdulist = fits.open(infile)
    tbdata = hdulist[1].data
    idx = ((tbdata.field('r_petro') < mlim) * (tbdata.field('nq') > 2) * 
           (tbdata.field('z_tonry') > zmin) * (tbdata.field('z_tonry') < zmax))
    tbdata = tbdata[idx]
    z = tbdata.field('z_tonry')
    kc = tbdata.field('kcorr_r')
    try:
        clr = ((tbdata.field('mag_auto_u') - tbdata.field('A_u') - 
                tbdata.field('kcorr_u')) - 
               (tbdata.field('mag_auto_r') - tbdata.field('A_r') - 
                tbdata.field('kcorr_r')))
        ylabel = 'kcorr (auto)'
    except:
        clr = ((tbdata.field('u_model') - tbdata.field('kcorr_u')) - 
               (tbdata.field('r_model') - tbdata.field('kcorr_r')))
        ylabel = 'kcorr (model)'

    hdulist.close()

    plt.clf()
    plt.subplot(111)
    plt.scatter(z, kc, 0.1, c=clr, vmin=0, vmax=3, edgecolors='face')
    plt.axis((zmin, zmax, -0.3, 1.5))
    plt.xlabel('z')
    plt.ylabel(ylabel)
    cb = plt.colorbar()
    cb.set_label(r'$(u - r)$')
    plt.draw()


def mass_comp(infile='kcorrz01.fits', lgmmin=6, lgmmax=13, lgmstep=0.2,
              fluxcorr=False, pord=3, plot_file='mass_comp.png', size=(5,8)):
    """Stellar mass completeness as function of redshift.
    See Pozzetti+2010, Sec 5.2 and Moustakas+ 2013, Sec 4.3."""

    mlim = 19.8
    H0 = 100.0
    zmin, zmax = 0.002, 0.65
    zlimits = np.linspace(0, 0.6, 61)
    nz = len(zlimits) - 1

    hdulist = fits.open(infile)
    tbdata = hdulist[1].data
    hdulist.close()
    idx = ((tbdata.field('r_petro') < mlim) * (tbdata.field('nq') > 2) *
           (tbdata.field('z_tonry') > zmin) * (tbdata.field('z_tonry') < zmax) *
           (tbdata.field('survey_class') > 3) * (tbdata.field('logmstar') > 0))
    tbdata = tbdata[idx]
    z = tbdata.field('z_tonry')
    r = tbdata.field('r_petro')
    logM = tbdata.field('logmstar') - 2*math.log10(H0/70.0)
    if fluxcorr:
        logM += np.log10(tbdata.field('fluxscale'))

    plt.clf()
    fig, axes = plt.subplots(3, sharex=True, num=1)
    plt.subplots_adjust(hspace=0)
    for ic in xrange(3):
        clr = 'cbr'[ic]
        selc = tbdata.field('colour') > 'a'
        if ic > 0:
            selc *= (tbdata.field('colour') == clr)
        ax = axes[ic]
        ax.scatter(z, logM, 0.1, c='k', edgecolors='face', alpha=0.1)
        ax.set_xlim(0.0, zmax)
        ax.set_ylim(6, 13)
        ax.set_ylabel(r'$\log(M/M_\odot)$')

#        zlimits = np.percentile(z[selc], zpc)
        zmed = np.zeros(nz)
        logMmin = np.zeros(nz)
        for iz in xrange(nz):
            sel = selc * (z >= zlimits[iz]) * (z < zlimits[iz+1])
            mfnt = np.percentile(r[sel], 80)
            fnt = sel * (r > mfnt)
            logMlim = logM[fnt] + 0.4*(r[fnt] - mlim)
            ax.scatter(z[fnt], logMlim, 0.1, c='b', edgecolors='face')
            zmed[iz] = np.percentile(z[sel], 50)
            logMmin[iz] = np.percentile(logMlim, 95)
        ax.plot(zmed, logMmin, 'go')
        pfit = np.polyfit(zmed, logMmin, pord)
        ax.plot(zmed, np.polyval(pfit, zmed), 'g-')
        ax.text(0.9, 0.1, clr, transform=ax.transAxes)
        print clr, pfit
    ax.set_xlabel(r'Redshift $z$')
    plt.draw()
    fig = plt.gcf()
    fig.set_size_inches(size)
    plt.savefig(plot_file, bbox_inches='tight')


def mass_plot(infile='kcorrz01.fits', lgmmin=6, lgmmax=13, lgmstep=0.2,
              fluxcorr=False):
    """Plot mass-related relations."""

    Mr_sun = 4.76
    mlim = 19.8
    H0 = 100.0
    omega_l = 0.7
    zmin, zmax = 0.002, 0.65
    cosmo = util.CosmoLookup(H0, omega_l, (zmin, zmax))

    hdulist = fits.open(infile)
    tbdata = hdulist[1].data
    idx = ((tbdata.field('r_petro') < mlim) * (tbdata.field('nq') > 2) * 
           (tbdata.field('z_tonry') > zmin) * (tbdata.field('z_tonry') < zmax) *
           (tbdata.field('survey_class') > 3) * (tbdata.field('logmstar') > 0))
    tbdata = tbdata[idx]
    z = tbdata.field('z_tonry')
    r = tbdata.field('r_petro')
    kc = tbdata.field('kcorr_r')
    Mr = r - cosmo.dist_mod(z) - kc
    logM = tbdata.field('logmstar') - 2*math.log10(H0/70.0)
    if fluxcorr:
        logM += np.log10(tbdata.field('fluxscale'))
    hdulist.close()

    nbin = int((lgmmax - lgmmin) / lgmstep)
    Mass_bin = np.linspace(lgmmin + 0.5*lgmstep, lgmmax - 0.5*lgmstep, nbin)
    Mr_lim = np.zeros(nbin)
    for i in xrange(nbin):
        lgmlo = lgmmin + i*lgmstep
        lgmhi = lgmlo + lgmstep
        sel = (logM >= lgmlo) * (logM < lgmhi)
        Mr_lim[i] = np.percentile(Mr[sel], 95)

    plt.clf()
    fig, axes = plt.subplots(3, sharex=True, num=1)
    ax = axes[0]
    # ax.plot(Mass_bin, Mr_lim)
    # ax.scatter(logM, Mr, 0.1, c=z, vmin=zmin, vmax=zmax, edgecolors='face')
    # ax.set_xlim(lgmmin, lgmmax)
    # ax.set_ylim(-25, -12)
    # ax.set_ylabel(r'$M_r$')
    ax.scatter(z, Mr, 0.1, c=z, vmin=zmin, vmax=zmax, edgecolors='face')
    ax.set_ylim(-12, -25)
    ax.set_ylabel(r'$M_r$')

    ax = axes[1]
    ax.scatter(z, logM, 0.1, c=z, vmin=zmin, vmax=zmax, edgecolors='face')
    ax.set_ylim(lgmmin, lgmmax)
    ax.set_ylabel(r'$\log(M/M_\odot)$')

    ax = axes[2]
    ax.scatter(z, logM + 0.4*(Mr - Mr_sun), 0.1, c=z, vmin=zmin, vmax=zmax, 
               edgecolors='face')
    ax.set_ylim(-2, 2)
    ax.set_ylabel(r'$\log(M/L)$')

    ax.set_xlabel(r'$z$')
    plt.show()

    plt.clf()
    fig, axes = plt.subplots(ncols=2, sharey=True, num=1)
    ax = axes[0]
    ml = logM + 0.4*(Mr - Mr_sun)
    ylab = r'$\log(M/L)$'
    # ylab = r'$\log(M/M_\odot) + M_r$'
    # ml = logM + Mr
    ax.scatter(logM, ml, 0.1, c=z, vmin=zmin, vmax=zmax, 
               edgecolors='face')
    ax.set_xlim(lgmmin, lgmmax)
    ax.set_xlabel(r'$\log(M/M_\odot)$')
    ax.set_ylabel(ylab)
    ax = axes[1]
    ax.scatter(Mr, ml, 0.1, c=z, vmin=zmin, vmax=zmax, 
               edgecolors='face')
    ax.set_xlim(-12, -25)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(r'$M_r$')

    plt.show()

    plt.clf()
    ax = plt.subplot(111)
    ax.scatter(Mr, logM, 0.1, c=z, vmin=zmin, vmax=zmax, 
               edgecolors='face')
    ax.set_xlim(-12, -25)
    ax.set_ylim(lgmmin, lgmmax)
    ax.set_xlabel(r'$M_r$')
    ax.set_ylabel(r'$\log(M/M_\odot)$')
    plt.show()


def zhist(infile='kcorrz01.fits'):
    """GAMA redshift histogram."""

    Mr_sun = 4.76
    mlim = 19.8
    H0 = 100.0
    omega_l = 0.7
    zmin, zmax = 0.002, 0.65
    cosmo = util.CosmoLookup(H0, omega_l, (zmin, zmax))

    hdulist = fits.open(infile)
    tbdata = hdulist[1].data
    idx = ((tbdata.field('r_petro') < mlim) * (tbdata.field('nq') > 2) *
           (tbdata.field('z_tonry') > zmin) * (tbdata.field('z_tonry') < zmax) *
           (tbdata.field('survey_class') > 3) * (tbdata.field('logmstar') > 0))
    tbdata = tbdata[idx]
    z = tbdata.field('z_tonry')
    hdulist.close()

    plt.clf()
    plt.hist(z, 60, (0.0, 0.6))
    plt.xlabel('Redshift')
    plt.ylabel('Frequency')
    plt.show()


def kde_test(area=0.0001, nsamp=10, alpha=-1.1, Mstar=-20.5, 
             phistar=0.01, Mlims=(-24, -15), nM=36, zlims=(0.002, 0.5), 
             mlims=(14, 19.8)):
    """Test pyqt_fit kernel density estimator."""

    def schec(M):
        """Schechter function."""
        L = 10**(0.4*(Mstar - M))
        ans = 0.4 * math.log(10) * phistar * L**(alpha+1) * np.exp(-L)
        return ans

    def zdm(dmod, zlims):
        """Calculate redshift z corresponding to distance modulus dmod, solves
        dmod = m - M = DM(z),
        where z is constrained to lie in range zlims."""

        if cosmo.dist_mod(zlims[0]) - dmod > 0:
            return zlims[0]
        if cosmo.dist_mod(zlims[1]) - dmod < 0:
            return zlims[1]
        z = scipy.optimize.brentq(lambda z: cosmo.dist_mod(z) - dmod,
                                  zlims[0], zlims[1], xtol=1e-5, rtol=1e-5)
        return z

    H0 = 100
    omega_l = 0.7
    cosmo = util.CosmoLookup(H0, omega_l, (zlims[0], zlims[1]))

    L1 = 10**(0.4*(Mstar - Mlims[1]))
    L2 = 10**(0.4*(Mstar - Mlims[0]))
    Vs = area * (cosmo.dm(zlims[1])**3 - cosmo.dm(zlims[0])**3)
    den = phistar * mpmath.gammainc(alpha+1, L1, L2)
    ndat = int(Vs * den)
    print 'Vol, density, ndat:', Vs, den, ndat

    phi_bin = np.zeros((nsamp, nM))
    phi_kde = np.zeros((nsamp, nM))
    bin_edges = np.linspace(Mlims[0], Mlims[1], nM + 1)
    dM = (Mlims[1] - Mlims[0])/float(nM)
    Mbins = bin_edges[:-1] + 0.5*dM

    for isamp in xrange(nsamp):
        z = util.ran_fun(cosmo.dV, zlims[0], zlims[1], ndat)
        Mabs = util.ran_fun(schec, Mlims[0], Mlims[1], ndat)
        mapp = Mabs + cosmo.dist_mod(z)
        sel = (mapp >= mlims[0]) * (mapp < mlims[1])
        z, Mabs, mapp = z[sel], Mabs[sel], mapp[sel]
        nsel = len(z)

        dmod = mlims[1] - Mabs
        zmax = map(lambda i: zdm(dmod[i], zlims), xrange(nsel))
        V = area * cosmo.dm(z)**3
        Vmax = area * cosmo.dm(zmax)**3
        wt = 1.0/Vmax

        phi_bin[isamp, :], edges = np.histogram(Mabs, bin_edges, weights=wt)
        phi_bin[isamp, :] /= dM

        kde = pyqt_fit.kde.KDE1D(Mabs, lower=Mlims[0], upper=Mlims[1], 
                                 weights=wt)
        # scale = kde.bandwidth/dM
        # ng = int(math.ceil(3*scale))
        # gauss = scipy.stats.norm.pdf(np.arange(-ng, ng+1), scale=scale)
        # phi_kde[isamp, :] = scipy.signal.deconvolve(kde(Mbins), gauss)[0] * wt.sum()
        phi_kde[isamp, :] = kde(Mbins) * wt.sum()

        print 'Sample {}: {} visible galaxies, wtsum ={}, kde bw = {}'.format(
            isamp, nsel, wt.sum(), kde.bandwidth)

    phi_bin_mean = np.mean(phi_bin, axis=0)
    phi_bin_err = np.std(phi_bin, axis=0)/math.sqrt(nsamp-1)
    schec_par = util.schec_fit(Mbins, phi_bin_mean, phi_bin_err, 
                               (alpha, Mstar, math.log10(phistar)))
    print 'Binned phi: alpha = {:5.2f} +- {:5.2f}, M* = {:5.2f} +- {:5.2f}, log phi* = {:5.2f} +- {:5.2f}'.format(
        schec_par['alpha'], schec_par['alpha_err'][0],
        schec_par['Mstar'], schec_par['Mstar_err'][0],
        schec_par['lpstar'], schec_par['lpstar_err'][0])

    phi_kde_mean = np.mean(phi_kde, axis=0)
    phi_kde_err = np.std(phi_kde, axis=0)/math.sqrt(nsamp-1)
    schec_par = util.schec_fit(Mbins, phi_kde_mean, phi_kde_err, 
                               (alpha, Mstar, math.log10(phistar)), 
                               sigma=kde.bandwidth)
    print 'KDE phi: alpha = {:5.2f} +- {:5.2f}, M* = {:5.2f} +- {:5.2f}, log phi* = {:5.2f} +- {:5.2f}'.format(
        schec_par['alpha'], schec_par['alpha_err'][0],
        schec_par['Mstar'], schec_par['Mstar_err'][0],
        schec_par['lpstar'], schec_par['lpstar_err'][0])

    plt.clf()
    plt.errorbar(Mbins, phi_bin_mean, yerr=phi_bin_err, fmt='o')
    plt.errorbar(Mbins, phi_kde_mean, yerr=phi_kde_err, fmt='o', label='KDE')
    plt.xlabel(r'$M$')
    plt.semilogy(basey=10, nonposy='clip')
    plt.ylim(1e-6, 1)
    plt.ylabel(r'$\phi(M)$')
    plt.legend(loc=2)
    plt.draw()

def gsum(x, mu, sigma, norm=1):
    """Return sum of Gaussians at points x."""
    mu = np.array(mu)[:, np.newaxis]
    try:
        sigma = np.array(sigma)[:, np.newaxis]
    except:
        sigma = sigma*np.ones(len(mu))
        sigma = np.array(sigma)[:, np.newaxis]
    try:
        norm = np.array(norm)[:, np.newaxis]
    except:
        norm = norm*np.ones(len(mu))
        norm = np.array(norm)[:, np.newaxis]

    rtp = math.sqrt(2*math.pi)
    return np.sum(norm*np.exp(-(x-mu)**2/(2*sigma**2))/(rtp*sigma), axis=0)

def norm_test(mu=(0, 0.5), sigma=(1, 0.5)):
    """Test scipy.stats.norm."""
    plt.clf()
    xvals = np.linspace(-2, 2, 50)
    plt.plot(xvals, gsum(xvals, mu, sigma))
    plt.draw()
             
def nifty_numpy():
    """Some nifty things with numpy to avoid looping."""

    # Create 2d array with elements a[i]*a[j] from 1d array a
    a = np.array((0,1,2,3))
    a = a[:, np.newaxis]          # Insert new axis, making a Nx1
    b = a*a.transpose()

    # given 1d arrys of lower and upper limits, ilo, ihi, and 1d array a,
    # calculate array [a[ilo[0]:ihi[0]].sum(), a[ilo[1]:ihi[1]].sum(), ...]
    a = np.array((0,1,2,3))
    ilo = np.array((0,0,1,1))
    ihi = np.array((1,2,3,4))
    np.array((a[ilo[0]:ihi[0]].sum(), a[ilo[1]:ihi[1]].sum()))

def J3(gamma, r0, rmax):
    return r0**gamma / (3-gamma) * rmax**(3-gamma)

def mag_plot(infile='kcorrz01.fits', dmrange=(-6,6), dmlim=2, 
             outliers='outliers.txt', plot_file='dmhist.pdf'):
    """Plot Petro-Sersic mag histograms."""

    hdulist = fits.open(infile)
    tbdata = hdulist[1].data
    hdulist.close()
    idx = (tbdata.field('survey_class') > 3)
    tbdata = tbdata[idx]
    ntarg = len(tbdata)
    idx = (tbdata.field('bn_objid') < 0)
    # tbdata = tbdata[idx]
    nnbn = len(tbdata[idx])
    r_pet = tbdata.field('r_petro')
    r_ser = tbdata.field('r_sersic')
    red = (tbdata.field('bn_objid') < 0) * (tbdata.field('colour') == 'r')
    good = np.fabs(r_pet[idx] - r_ser[idx]) < dmlim
    ngood = len(tbdata[good])
    print '{} out of {} targets ({})with no bright neighbours'.format(
        nnbn, ntarg, float(nnbn)/ntarg)
    print '{} ({})with dm < {}'.format(
        ngood, float(ngood)/nnbn, dmlim)
    bad = np.fabs(r_pet[idx] - r_ser[idx]) > dmlim
    nbad = len(tbdata[idx][bad])

    f = open(outliers, 'w')
    print >> f, 'r_petro-r_sersic ra dec'
    for i in xrange(nbad):
        print >> f, r_pet[idx][bad][i] - r_ser[idx][bad][i], tbdata.field('ra')[idx][bad][i], tbdata.field('dec')[idx][bad][i]
    f.close()

    plt.clf()
    plt.hist(r_pet - r_ser, 48, dmrange, histtype='step')
#     plt.hist(r_pet[idx] - r_ser[idx], 48, dmrange, histtype='bar')
    plt.hist(r_pet[idx] - r_ser[idx], 48, dmrange, histtype='step', color='k',
             ls='dotted', lw=2)
    plt.hist(r_pet[red] - r_ser[red], 48, dmrange, histtype='step', color='r',
             ls='dashed')
    plt.plot((-2, -2), (1, 1e6), 'g:')
    plt.plot((2, 2), (1, 1e6), 'g:')
    plt.semilogy(basey=10, nonposy='clip')
    plt.axis((-6, 6, 1, 2e5))
    plt.xlabel(r'$r_{\rm Petro} - r_{\rm Sersic}\ ({\rm mag})$')
    plt.ylabel('Frequency')
    plt.draw()
    if plot_file:
        fig = plt.gcf()
        fig.set_size_inches(5, 4)
        plt.savefig(plot_dir + plot_file, bbox_inches='tight')


def stream_plot(r0=5.0, z=0.0):
    """Plot different streaming models."""
    gamma = 1.46
    omega_m = 0.3
    fgrow = omega_m**0.6
    alpha = 1.2 - 0.65*gamma
    h = 0.7
    H0 = 100*h
    rp = 10**np.linspace(-2, 2)
    r = rp/h
    a = 1.0/(1 + z)
    x = r/a
    xi = (r0/x)**gamma
    xibar = (3 * r0**gamma * x**-gamma) / (3 - gamma)
    xibbar = xibar / (1 + xi)
    plt.clf()
    v = -2.0/3.0*H0*r*fgrow*xibbar * (1 + alpha*xibbar)
    plt.plot(rp, v, label='JSD')
    v = -H0*r/(1 + (r/r0)**2)
    plt.plot(rp, v, label='DP83')
    v = -2.0/3.0*H0*r*fgrow*xibbar
    plt.plot(rp, v, label='8c')
    plt.semilogx()
    plt.xlabel('r (Mpc/h)')
    plt.ylabel('v_12 (km/s)')
    plt.legend(loc=3)
    plt.draw()


def mock_plot(infile='kcorrz00.fits'):
    """Compare calculated and given absolute mags in mocks."""

    H0 = 100
    omega_l = 0.7
    zrange = (0.002, 0.65)
    cosmo = util.CosmoLookup(H0, omega_l, zrange)
    hdulist = fits.open(infile)
    tbdata = hdulist[1].data
    hdulist.close()
    z = tbdata.field('redshift_obs')
    kc = tbdata.field('kcorr_r')
    abs_mag = tbdata.field('sdss_r_obs_app') - cosmo.dist_mod(z) - kc

    plt.clf()
    plt.scatter(tbdata.field('sdss_r_rest_abs'),
                tbdata.field('sdss_r_rest_abs') - abs_mag, 0.1,
                c=tbdata.field('chi2'), edgecolors='none')
    plt.axis((-25, -12, -0.5, 0.5))
    plt.colorbar()
    plt.xlabel('sdss_r_rest_abs')
    plt.ylabel('delta mag')
    plt.show()


def sb_limits():
    """LSST SB limits as fn of time."""
    years = (1, 2, 3, 5, 10)
    sb = (25.2, 25.7, 26.1, 26.5, 27)
    plt.clf()
    plt.plot(years, sb)
    plt.plot((years[0], years[-1]), (26, 26), ':', label='~1:10 mass ratio')
    plt.xlabel('Time (years)')
    plt.ylabel(r'SB (r mag arcsec$^{-2}$)')
    plt.legend(loc=4)
    plt.show()


def vpec_plot(infile='GAMA-0/Gonzalez13.DB.MillGas.field1.core.0.hdf5',
              zlo=0.002, zhi=0.5, mlim=19.8, scale_fac=1e-3):
    """Peculiar velocity map for mock catalogue."""

    c = 3e5
    with h5py.File(infile, 'r') as f:
        mag = f['Data']['apprSo_tot_ext'][:]
        z_obs = f['Data']['z_obs'][:]
        sel = (mag < mlim) * (zlo <= z_obs) * (z_obs < zhi)
        z_obs = z_obs[sel]
        ra = f['Data']['ra'][sel]
        z_cos = f['Data']['z_cos'][sel]
    vpec = c * ((1+z_obs)/(1+z_cos) - 1)

    plt.clf()
    plt.hist(vpec)
    plt.xlabel('vpec (km/s)')
    plt.ylabel('Frequency')
    plt.show()

    plt.scatter(ra, z_cos, s=0.01, c=vpec, edgecolors='face')
    plt.colorbar()
    plt.xlabel('RA')
    plt.ylabel('Redshift')
    plt.show()


def zspace_sep(coords):
    """Redshift-space separations of two coords (ra, dec, r)."""

    x, y, z = np.zeros(2), np.zeros(2), np.zeros(2)
    for i in range(2):
        coord = coords[i]
        ra = coord[0]
        dec = coord[1]
        r = coord[2]
        x[i] = r*np.cos(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
        y[i] = r*np.sin(np.deg2rad(ra))*np.cos(np.deg2rad(dec))
        z[i] = r*np.sin(np.deg2rad(dec))

    dx = x[0] - x[1]
    dy = y[0] - y[1]
    dz = z[0] - z[1]
    dsq = dx*dx + dy*dy + dz*dz
    rsq = x**2 + y**2 + z**2
    r = np.sqrt(rsq)
    costheta = (rsq[0] + rsq[1] - dsq)/(2.0*r[0]*r[1])  # cosine rule
    if abs(costheta) < 1:
        t = math.sqrt((1-costheta)/(1+costheta))  # t = tan(0.5*theta)
    else:
        t = 0

    rp = (r[0] + r[1])*t
    pi = abs(r[0] - r[1])
    print 'DP83 rp, pi = ', rp, pi

    v0 = np.array((x[0], y[0], z[0]))
    v1 = np.array((x[1], y[1], z[1]))
    s = v0 - v1
    l = 0.5*(v0+v1)
    pi = abs(np.dot(s, l)/math.sqrt(np.dot(l, l)))
    rp = math.sqrt(np.dot(s, s) - pi**2)
#    print v0, v1, s, l
    print 'Fisher rp, pi = ', rp, pi


def cov_test(ndim=10, ngen=100, m=1, c=0):
    """Test fitting y = mx + c with correlated errors."""

    def chi2((m, c)):
        """
        Chi^2 residual between obs and model, using first neig eigenvectors
        (Norberg+2009, eqn 12).  By default (neig=0), use diagonal elements
        only.  Set neig='full' for full covariance matrix,
        'all' for all e-vectors."""

        fit = m*x + c
        if neig == 0:
            if len(obs) > 1:
                diag = np.diag(cov)
                nonz = diag > 0
                return np.sum((obs[nonz] - fit[nonz])**2 / diag[nonz])
            else:
                return (obs - fit)**2 / cov
        if neig == 'full':
            return (obs-fit).T.dot(icov).dot(obs-fit)
        yobs = eig_vec.T.dot(siginv).dot(obs)
        yfit = eig_vec.T.dot(siginv).dot(fit)
        if neig == 'all':
            return np.sum((yobs - yfit)**2 / eig_val)
        else:
            return np.sum((yobs[:neig] - yfit[:neig])**2 / eig_val[:neig])

#    cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
#    cov = np.triu(cov)
#    cov += cov.T - np.diag(cov.diagonal())
#    cov = np.dot(cov, cov)

    cov = np.ones((ndim, ndim))
    for j in range(ndim):
        for i in range(j):
            cov[i, j] = 1.0/(1 + j - i)
            cov[j, i] = 1.0/(1 + j - i)
    plt.clf()
    extent = (0, ndim, 0, ndim)
    plt.imshow(cov, aspect=1, interpolation='none',
               extent=extent, origin='lower')
    plt.title('Cov')
    plt.colorbar()
    plt.show()

    icov = np.linalg.inv(cov)
    siginv = np.diag(1.0/np.sqrt(np.diag(cov)))
    cnorm = np.nan_to_num(siginv.dot(cov).dot(siginv))
    plt.imshow(cnorm, aspect=1, interpolation='none',
               extent=extent, origin='lower')
    plt.title('Norm Cov')
    plt.colorbar()
    plt.show()

    eig_val, eig_vec = np.linalg.eigh(cnorm)
    print eig_val
    print eig_vec
    idx = eig_val.argsort()[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]
    print eig_val
#    print eig_vec

    x = np.arange(ndim)
    mean = m*x + c
    pf = np.zeros((3, ngen))
    pe = np.zeros((3, ngen, ndim+1))
#    print 'diag full eig(1-10)'
    plt.clf()
    for i in range(ngen):
        obs = np.random.multivariate_normal(mean, cov)
        plt.plot(x, obs)
        neig = 'all'
        out = scipy.optimize.fmin(chi2, (m, c), xtol=0.001, ftol=0.001,
                                  maxfun=10000, maxiter=10000, full_output=1,
                                  disp=0)
        pf[0, i] = out[0][0]
        pf[1, i] = out[0][1]
        pf[2, i] = out[1]
        for neig in range(ndim+1):
            out = scipy.optimize.fmin(chi2, (m, c), xtol=0.001, ftol=0.001,
                                      maxfun=10000, maxiter=10000,
                                      full_output=1, disp=0)
            pe[0, i, neig] = out[0][0]
            pe[1, i, neig] = out[0][1]
            pe[2, i, neig] = out[1]
    plt.show()
    print('Full cov: m = {:5.2f} +- {:5.2f}, c = {:5.2f} +- {:5.2f}, chi2 = {:5.2f} +- {:5.2f}'.format(
            pf[0, :].mean(), pf[0, :].std(), pf[1, :].mean(), pf[1, :].std(),
            pf[2, :].mean(), pf[2, :].std()))
    for neig in range(ndim+1):
        print('neig = {}: m = {:5.2f} +- {:5.2f}, c = {:5.2f} +- {:5.2f}, chi2 = {:5.2f} +- {:5.2f}'.format(
                neig, pe[0, :, neig].mean(), pe[0, :, neig].std(),
                pe[1, :, neig].mean(), pe[1, :, neig].std(),
                pe[2, :, neig].mean(), pe[2, :, neig].std()))


def ran_test(n):
    """Variance of n standard-normally dsitributed random numbers."""
    print(np.var(np.random.rand(n)))


def fft_test(tmin=0, tmax=10, nbin=8):
    """Test FFT by convolving signal with a smoothing kernel."""
    root2 = 2**0.5
    t = np.linspace(tmin, tmax, nbin, endpoint=False)
    tmid = 0.5*(tmin+tmax) #- (tmax-tmin)/nbin
#    tmid = 0
#    signal = np.zeros(nbin)
#    signal[nbin/2] = 1
#    signal[nbin/2 + 1] = 1
#    signal[20] = 1
    signal = np.exp(-5*(t-tmid)**2)
#    print signal
#    print np.fft.fft(signal)
#    print np.fft.hfft(signal)
    smooth = 0.5*np.exp(-np.abs(t-tmid))
    print scipy.integrate.quad(lambda t: np.exp(-5*(t-tmid)**2),
                               tmin, tmax)
    print scipy.integrate.quad(lambda t: 0.5*np.exp(-np.abs(t-tmid)),
                               tmin, tmax)
    plt.clf()
    plt.plot(t, signal, t, smooth)
    plt.show()

    signal_ft = np.fft.fft(signal)
    smooth_ft = np.fft.fft(smooth)
    signal_rft = np.fft.rfft(signal)
    smooth_rft = np.fft.rfft(smooth)
    f = np.fft.fftfreq(nbin, tmax/nbin)
#    print f
    lorentz = 2.0/(1 + (2*math.pi*f)**2)
#    print f
#    print lorentz
#    print smooth_ft
#    print smooth_rft
    plt.clf()
    plt.plot(f, np.real(signal_ft), 'b-', f, np.imag(signal_ft), 'b:',
             f, np.real(smooth_ft), 'g-', f, np.imag(smooth_ft), 'g:')
    plt.plot(f, lorentz)
    plt.show()
    plt.clf()
    plt.plot(signal_rft)
    plt.plot(smooth_rft)
    plt.show()

    conv = np.fft.fftshift(np.fft.ifft(signal_ft*smooth_ft))
    rconv = np.fft.fftshift(np.fft.irfft(signal_rft*smooth_rft))
    tim = time.clock()
    lconv = np.fft.ifft(signal_ft*lorentz)
    tim_ifft = time.clock()
    np_conv = np.convolve(signal, smooth, 'same')
    tim_np = time.clock()
    sp_conv = scipy.signal.fftconvolve(signal, smooth, 'same')
    tim_sp = time.clock()
    plt.clf()
#    plt.plot(t, conv, t, np.fft.fftshift(conv))
    plt.plot(t, conv, t, rconv-0.01, t, lconv, t, np_conv, t, sp_conv-0.01)
#    print lconv
    plt.show()
    print('Timings: ifft:', tim_ifft - tim, 'numpy:', tim_np - tim_ifft,
          'scipy:', tim_sp - tim_np)


def exp_test():
    x = np.linspace(0.001, 10)
    logx = np.log(x)
    expx = np.exp(x)
    y = np.exp(-x)
    plt.clf()
    plt.plot(logx, y)
    plt.show()
    print logx, y


def cosmo_comp(H0=100, omega_l=0.7, zrange=(0, 1), nz=1000):
    """Compare cosmologigal distance calaculations using astlib.astCalc
    and astropy.cosmology."""

    astCalc.H0 = H0
    astCalc.OMEGA_L = omega_l
    astCalc.OMEGA_M0 = 1 - omega_l
    cosmo = FlatLambdaCDM(H0=H0, Om0=1-omega_l)

    z = np.linspace(zrange[0], zrange[1], nz)
    dc_astropy = cosmo.comoving_distance(z)
    dct_astropy = cosmo.comoving_transverse_distance(z)

    plt.clf()
    plt.plot(z, dc_astropy.value - dct_astropy.value)
    plt.show()


def error_bars(capthick=None):
    """Demonstrate missing errorbar caps in pdf output."""
    print(matplotlib.__version__)
    plt.clf()
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    yerr = np.ones(len(y))
    plt.errorbar(x, y, yerr, capthick=capthick)
    plt.draw()
    plt.savefig('error_bars.pdf')


def subplot_test():
    """Test fancy subplots."""
    plt.clf()
    ax1 = plt.subplot2grid((19,1), (0,0), rowspan=1)
    ax2 = plt.subplot2grid((19,1), (2,0), rowspan=5)
    ax3 = plt.subplot2grid((19,1), (8,0), rowspan=5)
    ax4 = plt.subplot2grid((19,1), (14,0), rowspan=5)
    ax1.plot((0, 0), (1, 1))
    ax2.plot((0, 0), (1, 1))
    ax3.plot((0, 0), (1, 1))
    ax4.plot((0, 0), (1, 1))
#    plt.subplots_adjust(hspace=0.2)
    plt.xlabel('RA')
    ax3.set_ylabel('Dec')
    plt.draw()
