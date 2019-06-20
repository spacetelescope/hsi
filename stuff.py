from __future__ import print_function, division

import glob
import os
import os.path
import time
import numpy as np
import ustab.interfere.combine as combine
import ustab.interfere.convertraw as hic
from ustab.interfere.convertraw import convert_all
from ustab.interfere.spatial import unwrap_raw_frame, unwrap
from ustab.safio import write as saf_write
import ustab.fdio as fdio 
import ustab.datio as hdio
import ustab.safio as hsio
import zernike
import numpy.fft as nf
#from matplotlib import pyplot as plt
from scipy.signal import gaussian
import astropy.io.fits as fits
from scipy import signal
from scipy.ndimage import convolve

path_prefix = '/drives/server42/vol2/AAA_USTAB_DATA'

    
def ndcalc(baseline, measurement):
    """calculate numerator and denominator values for phase calculation
    initial version only handles sequences of 4-phase
    ufunc-like"""
    
    b0, b90, b180, b270 = baseline
    m0, m90, m180, m270 = measurement
    Nm = m270 - m90
    Dm = m0   - m180
    Nb = b270 - b90
    Db = b0   - b180
    numerator = Nm*Db - Dm*Nb
    denominator = Dm*Db + Nm*Nb
    return numerator, denominator

def smooth(arr, windowsize):
    """requires 2-d array"""
    if windowsize not in np.arange(10**2) +1:
        print('windowsize provided is: ', windowsize)
        raise ValueError("windowsize must be odd positive integer")
    kernel = np.ones((windowsize, windowsize))
    return convolve(arr, kernel)
    
def calcPhaseDiff(baseline, measurement, windowsize):
    """Ufunc-like"""
    ws = windowsize
    numerator, denominator = ndcalc(baseline, measurement)
    return np.arctan2(smooth(numerator, ws), smooth(denominator, ws))

def exported_to_raw(inpath, outpath):
    convert_all(inpath, outpath)

def raw_to_wrapped(inpath, outpath):
    flist = glob.glob(inpath+'*.dat')
    print(flist)
    for rfile in flist:
        print("computing wrapped phase for ", rfile)
        wrapped = dat_to_wrapped_phase(rfile)
        outnum = rfile.split('_')[-1][:-4]
        outroot = rfile.split('_')[0] # bug
        ufile = outpath + 'wrapped_' + outnum + '.saf'
        saf_write(ufile, wrapped)

def extract_time_series(path, x, y):
    fl = glob.glob(path+'*.saf')
    ts = np.zeros((len(fl),))
    for i,fn in enumerate(fl):
        im = saf_read(fn)[1]
        ts[i] = im[y,x]
    return ts


class MaskSet:
    def __init__(self, mask, center, radius):
        self.mask = mask
        self.center = center
        self.radius = radius
        self.indices = np.where(mask)


def exported_to_raw(inpath, outpath):
    convert_all(inpath, outpath)


def raw_to_unwrapped(inpath, outpath):
    flist = glob.glob(inpath+'*.dat')
    for rfile in flist:
        print("computing wrapped phase for ", rfile)
        wrapped = 1  # dat_to_wrapped_phase(rfile)
        outnum = rfile.split('_')[-1][:-4]
        outroot = rfile.split('_')[0]  # bug
        ufile = outroot + outnum
        saf_write(ufile, wrapped)

def calc_frac_modulation(refcube, imcube, mask):
    n, d = ndcalc(refcube, imcube)
    modulation = np.sqrt(n**2 + d**2)/2
    intensity = ((refcube+imcube).sum(axis=0)*mask)
    max_intensity = intensity.max()
    modulation = (modulation*mask)
    intensity_threshold = max_intensity/10
    intensity_mask = intensity > intensity_threshold
    nmask = intensity_mask.sum()
    frac_modulation = modulation[intensity_mask]/intensity[intensity_mask]
    print('max_intensity: ', max_intensity)
    print('nmask: ', nmask)
    return frac_modulation.sum()/nmask
    

def temporal_correction(oldphase, newphase):
    '''
    compute n*twopi phase adjustment that must be subtracted to newphase to
    make the resultant value fall within +/- pi of the old value
    '''
    diff = newphase - oldphase + np.pi
    diffmod2pi = (diff % (2 * np.pi))
    corr = diff - diffmod2pi
    return corr


def temporal_unwrap(timeseries):
    ut = timeseries.copy()
    prevval = ut[0]
    for i, val in enumerate(ut[1:]):
        corr = temporal_correction(prevval, val)
        #print("i: %d, prevval %f,  val %f, corr %f" % (i, prevval, val, corr))
        ut[i+1] -= corr
        prevval = ut[i+1]
    return ut


def doit_old(inpath, maskset, prevrefphase=None):
    '''
    take a single raw image, compute wrapped phase, spatially unwrap,
    temporally unwrap, fit zernikes, save unwrapped image, zernike results
    and reference point value to disk.
    '''
    print("processing ", inpath)
    center = maskset.center
    rhdr, rawframe = hdio.read(inpath)
    unwrapped = unwrap_raw_frame(rawframe, mask=maskset.mask)
    refphase = unwrapped[center[1], center[0]]
    phasecorr = temporal_correction(prevrefphase, refphase)
    unwrapped -= phasecorr
    y, x = maskset.indices
    radius = maskset.radius
    zcoeff = zernike.fit_gzernike(x, y, unwrapped[maskset.indices],
                                  10, center, radius)
    return unwrapped, zcoeff, unwrapped[center[1], center[0]]


def doall_old(inpath, maskfile):
    '''
    Here inpath is the basic directory that contains all the associated data.
    /raw is appended to it to get to the raw data. Results are stored in the
    results subdirectory.
    '''
    globpat = os.path.join(inpath, 'raw/frame_*.dat')
    filelist = glob.glob(globpat)
    # filelist = glob.glob(os.path.join(inpath, 'raw/frame_*.dat'))
    filelist.sort()
    print(filelist)
    mask = hsio.read(maskfile)[1]
    outdir = os.path.join(inpath, 'results')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    print(outdir)
    maskset = MaskSet(mask, center=(120, 120), radius=110)
    prevrefphase = 0
    zmatrix = np.zeros((len(filelist), 10))
    refphase = np.zeros((len(filelist),))
    zresfile = open(outdir + '/zernikes.txt', 'w')
    presfile = open(outdir + '/phasetimeseries.txt', 'w')
    for i, rfile in enumerate(filelist):
        print(i, rfile)
        uim, zcoeff, rphase = doit_old(rfile, maskset, prevrefphase)
        zmatrix[i] = zcoeff
        refphase[i] = rphase
        zresfile.write(10*'%e,' % tuple(zcoeff) + '\n')
        presfile.write(str(rphase)+'\n')
    zresfile.close()
    presfile.close()

def doit_new(inpath, maskset, prevrefphase=None):
    '''
    take a single raw image, compute wrapped phase, spatially unwrap,
    temporally unwrap, fit zernikes, save unwrapped image, zernike results
    and reference point value to disk.
    '''
    print("processing ", inpath)
    center = maskset.center
    rawframe = fdio.read(inpath)
    unwrapped = unwrap_raw_frame(rawframe, mask=maskset.mask)
    refphase = unwrapped[center[1], center[0]]
    phasecorr = temporal_correction(prevrefphase, refphase)
    unwrapped -= phasecorr
    y, x = maskset.indices
    radius = maskset.radius
    zcoeff = zernike.fit_gzernike(x, y, unwrapped[maskset.indices],
                                  10, center, radius)
    return unwrapped, zcoeff, unwrapped[center[1], center[0]]


def doall_new(inpath, maskfile):
    '''
    Here inpath is the basic directory that contains all the associated data.
    /raw is appended to it to get to the raw data. Results are stored in the
    results subdirectory.
    '''
    globpat = os.path.join(inpath, '*.rawframe')
    filelist = glob.glob(globpat)
    # filelist = glob.glob(os.path.join(inpath, 'raw/frame_*.dat'))
    # Given the variable number length, sorting is more complex
    filenum = [int(os.path.split(item)[1].split('.')[0]) for item in filelist]
    print(filenum)
    zlist = list(zip(filenum, filelist))
    zlist.sort()
    filelist = [item[1] for item in zlist]
    print(filelist)
    mask = hsio.read(maskfile)[1]
    outdir = os.path.join(inpath, '000_results')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    print(outdir)
    maskset = MaskSet(mask, center=(120, 120), radius=110)
    prevrefphase = 0
    zmatrix = np.zeros((len(filelist), 10))
    refphase = np.zeros((len(filelist),))
    zresfile = open(outdir + '/zernikes.txt', 'w')
    presfile = open(outdir + '/phasetimeseries.txt', 'w')
    for i, rfile in enumerate(filelist):
        print(i, rfile)
        uim, zcoeff, rphase = doit_new(rfile, maskset, prevrefphase)
        zmatrix[i] = zcoeff
        refphase[i] = rphase
        zresfile.write(10*'%e,' % tuple(zcoeff) + '\n')
        presfile.write(str(rphase)+'\n')
    zresfile.close()
    presfile.close()

bozo = 22


def doit_speckle(inpath, maskset, refcube, prevrefphase, windowsize=5, output=True):
    '''
    take a single raw image, compute wrapped phase, spatially unwrap,
    temporally unwrap, fit zernikes, save unwrapped image, zernike results
    and reference point value to disk.
    '''
    if output:
        print("processing ", inpath, windowsize)
    center = maskset.center
    rawframe = fdio.read(inpath)
    cube = combine.separate_pixelated(rawframe)
    phase = calcPhaseDiff(refcube, cube, windowsize)
    unwrapped = unwrap(phase) * maskset.mask
    refphase = (unwrapped*maskset.mask).sum()/maskset.mask.astype(np.int64).sum()
    phasecorr = temporal_correction(prevrefphase, refphase)
    unwrapped -= phasecorr
    y, x = maskset.indices
    radius = maskset.radius
    zcoeff = zernike.fit_gzernike(x, y, unwrapped[maskset.indices],
                                  10, center, radius)
    return unwrapped, zcoeff, unwrapped[center[1], center[0]]



def doall_speckle(inpath, maskset, range=None, suffix=''):
    '''
    Here inpath is the basic directory that contains all the associated data.
    /raw is appended to it to get to the raw data. Results are stored in the
    results subdirectory.
    '''
    globpat = os.path.join(inpath, '*.rawframe')
    filelist = glob.glob(globpat)
    # filelist = glob.glob(os.path.join(inpath, 'raw/frame_*.dat'))
    # Given the variable number length, sorting is more complex
    filenum = [int(os.path.split(item)[1].split('.')[0]) for item in filelist]
    print(filenum)
    zlist = list(zip(filenum, filelist))
    zlist.sort()
    filelist = [item[1] for item in zlist]
    if range is not None:
        filelist = filelist[range[0]:range[1]]
        foffset = range[0]
    else:
        foffset = 0
        reffile = filelist[0]
    print(filelist)
    mask = maskset.mask
    #mask = hsio.read(maskfile)[1]
    rawref = fdio.read(os.path.join(inpath,'0.rawframe'))
    refcube = combine.separate_pixelated(rawref)
    outdir = os.path.join(inpath, '000_results')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    print(outdir)
    # maskset = MaskSet(mask, center=(240, 240), radius=180)
    prevrefphase = 0
    zmatrix = np.zeros((len(filelist), 10))
    refphase = np.zeros((len(filelist),))
    if range is None:
        zresfile = open(outdir + '/zernikes{}.txt'.format(suffix), 'w')
        presfile = open(outdir + '/phasetimeseries{}.txt'.format(suffix), 'w')
    else:
        zresfile = open(outdir + '/p_zernikes%s%03d.txt' % (suffix,foffset), 'w')
        presfile = open(outdir + '/p_phasetimeseries%s%03d.txt' % (suffix,foffset), 'w')        
    for i, rfile in enumerate(filelist):
        print(i+foffset, rfile)
        uim, zcoeff, rphase = doit_speckle(rfile, maskset, refcube, prevrefphase)
        zmatrix[i] = zcoeff
        refphase[i] = rphase
        zresfile.write(10*'%e,' % tuple(zcoeff) + '\n')
        presfile.write(str(rphase)+'\n')
    zresfile.close()
    presfile.close()

def doall_speckle_rebaseline(inpath, maskfile, rebaseline=1000):
    '''
    Here inpath is the basic directory that contains all the associated data.
    /raw is appended to it to get to the raw data. Results are stored in the
    results subdirectory.
    '''
    globpat = os.path.join(inpath, '*.rawframe')
    filelist = glob.glob(globpat)
    # filelist = glob.glob(os.path.join(inpath, 'raw/frame_*.dat'))
    # Given the variable number length, sorting is more complex
    filenum = [int(os.path.split(item)[1].split('.')[0]) for item in filelist]
    print(filenum)
    zlist = list(zip(filenum, filelist))
    zlist.sort()
    filelist = [item[1] for item in zlist]
    print(filelist)
    mask = hsio.read(maskfile)[1]
    maskset = MaskSet(mask, center=(240, 240), radius=180)
    outdir = os.path.join(inpath, '000_results')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        print(outdir)
    zlist = []
    niter = len(filelist)//rebaseline
    print('niter:', niter)
    tfilelist = filelist.copy()
    tfilelist.append('bozo') # just to allow slicing to work on the last iter
    for i in range(niter):
        for j in range(5):
            print(50*'*')
        rfilelist = tfilelist[i*rebaseline:(i+1)*rebaseline+1]
        rawref = fdio.read(rfilelist[0])
        refcube = combine.separate_pixelated(rawref)
        prevrefphase = 0
        zmatrix = np.zeros((rebaseline+1, 10))
        zresfile = open(outdir + '/zernikes.txt', 'w')
        for i, rfile in enumerate(rfilelist):
             if rfile=='bozo':
                 break
             print(i, rfile)
             uim, zcoeff, rphase = doit_speckle(rfile, maskset, refcube, prevrefphase)
             zmatrix[i] = zcoeff
        zlist.append(zmatrix)
    # now splice segments together
    szmatrix = np.zeros((len(filelist)+1, 10))
    szmatrix[0:rebaseline+1,:] = zlist[0]
    for i in range(niter-1):
        szmatrix[(i+1)*rebaseline:(i+2)*rebaseline+1,:] = szmatrix[(i+1)*rebaseline,:] + zlist[i+1]
    szresfile = open(outdir +'/szernikes.txt','w')
    for i in range(len(filelist)):
        szresfile.write(10*'%e,' % tuple(szmatrix[i,:]) + '\n')
    szresfile.close()

def segment_processing(dpat, nsegments):
    # first delete all partial result files that exist
    ##deletelist = glob.glob(os.path.join(dpat,'000_results/p_*.txt')
    ##return deletelists
    toffsets = np.arange(nsegments, dtype=np.int32)
    dlist = glob.glob(dpat)
    seglist = []
    for ddir in dlist:
        files = glob.glob(os.path.join(ddir,'*.rawframe'))
        nframes = len(files)
        offsets = toffsets * (nframes // nsegments)
        end = offsets.copy()
        end[:-1] = offsets[1:]
        end[-1] = nframes
        for i in range(nsegments):
            seglist.append((ddir, (offsets[i], end[i])))
    return seglist

def merge_partials(dpat):
    dlist = glob.glob(dpat)
    for ddir in dlist:
        fout = open(os.path.join(ddir, '000_results/pm_zernikes.txt'), 'w')
        pzfiles = glob.glob(os.path.join(ddir, '000_results/p_zernikes*.txt'))
        pzfiles.sort()
        for pzf in pzfiles:
            fin = open(pzf)
            fout.write(fin.read())
            fin.close()
    fout.close   

def delete_partials(dpat):
    delete_list = glob.glob(os.path.join(dpat, '000_results/p_*.txt'))
    for f in delete_list:
        os.remove(f)
        
def joinall_old(dpath, root):
    '''
    Given a list of integers, plot on the same plot with vertical
    offsets, the corresponding piston spectra.
    '''

    all_zp = np.zeros((1300, 2500))
    for i in range(1,1301):
        fn = 'szernikes.txt'
        z = loadz(os.path.join(dpath, root % i, 'results', fn))[1]
        all_zp[i-1, :] = temporal_unwrap(z[:, 0])
    hdu = fits.PrimaryHDU(all_zp)
    hdu.writeto('allpiston.fits')
    return all_zp

def loadts(filename):
    lines = open(filename).readlines()
    flines = [float(item.strip()) for item in lines]
    return np.array(flines)


def loadz(filename):
    lines = open(filename).readlines()
    flines = [item.strip() for item in lines]
    fflines = [item.split(',')[:10] for item in flines]
    ffflines = [[float(item2) for item2 in item] for item in fflines]
    if len(ffflines) != 2500:
        valid = False
    else:
        valid = True
    return valid, np.array(ffflines)


def plot_all_spec(dpath, root, nlist):
    '''
    Given a list of integers, plot on the same plot with vertical
    offsets, the corresponding piston spectra.
    '''
    plt.ion()
    plt.clf()
    freqarr = np.arange(1250)/10.
    avg_zp = np.zeros(2500)
    dlist = glob.glob(os.path.join(dpath,'2018*'))
    for i, ddir in enumerate(dlist[:nlist]):
        fn = 'zernikes.txt'
        z = loadz(os.path.join(ddir,'000_results', fn))
        zp = temporal_unwrap(z[1][:, 0])
        avg_zp += zp
    avg_zp = avg_zp / nlist
    window = np.kaiser(2500, 8)
    fzp = np.abs(nf.fft(avg_zp*window))[:len(z[1])//2]
    plt.plot(freqarr[50:], fzp[50:])
    return fzp


def plot_spectrum(freq, spec, annotate=True):
    plt.ion()
    plt.clf()
    plt.plot(freq, spec)
    #plt.title('Averaged Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (picometers)')
    if annotate:
        plt.annotate('Stimulus', size=20, xy=(46, 0.32 * RAD2PM),
                     xytext=(55, .4 * RAD2PM),
                     arrowprops=dict(facecolor='black', shrink=0.1))

def get_completed(dpath, fpat):
    flist = glob.glob(os.path.join(dpath, fpat, 'results/zernikes.txt'))
    nlist = [int(item[-25:-21]) for item in flist]
    return nlistput

def monitor_results(dpath, fpat, fn):
        nlist = get_completed(dpath, fpat)
        freq, fzp = avg_all_spec(dpath, fpat[:-1]+'%04d', nlist)
        #plot_spectrum(freq[400:500], fzp[400:500]*RAD2PM, annotate=False)
        #plt.title('NSets: %d' % len(nlist))
        fh = open(fn, 'w')
        fh.write(str(len(nlist))+'\n')
        for f, p in zip(freq, fzp):
            fh.write('%f, %f\n' % (f, p))
        fh.close()

def read_results(fn):
    fh = open(fn)
    lines = fh.readlines()
    nsets = int(lines[0].strip())
    freq = np.array([float(item.split(',')[0]) for item in lines[1:]])
    pist = np.array([float(item.split(',')[1].strip()) for item in lines[1:]])
    return nsets, freq, pist

def plot_results(nsets, freq, pist):
        plt.ion()
        plt.clf()
        plot_spectrum(freq[400:500], pist[400:500]*RAD2PM, annotate=False)
        plt.title('NSets: %d' % nsets)


def avg_all_spec(dpath, root, nlist, outfile=None):
    '''
    Given a list of integers, plot on the same plot with vertical
    offsets, the corresponding piston spectra.
    '''

    freqarr = np.arange(1250) / 10.
    avg_zp = np.zeros(2500)
    avg_psd = np.zeros(2500)
    window = np.kaiser(2500, 8)
    counter = 0
    for num in nlist:
        fn = 'zernikes.txt'
        valid, z = loadz(os.path.join(dpath, root % num, 'results', fn))
        if not valid:
            continue
        zp = temporal_unwrap(z[:, 0])
        avg_psd += np.abs(nf.fft(zp*window))**2
        avg_zp += zp
        counter += 1
    avg_zp = avg_zp / counter
    avg_psd = avg_psd / counter
    fzp = np.abs(nf.fft(avg_zp*window))[:len(z)//2]
    if outfile:
        f = open(outfile+'_amp.txt', 'w')
        for val in fzp:
            f.write(str(val)+'\n')
        f.close()
        f = open(outfile+'_psd.txt', 'w')
        for val in avg_psd:
            f.write(str(val)+'\n')
        f.close()
    return freqarr, fzp


def subset(indir, outdir, shape):
    '''
    Read a .dat file and extract a subset and write the subset to
    a new directory.
    '''
    flistin = glob.glob(os.path.join(indir, 'frame_*.dat'))
    flistin.sort()
    h, d = hdio.read(flistin[0])
    inshape = d.shape
    xdiff = inshape[1] - shape[0]
    ydiff = inshape[0] - shape[1]
    xoff = xdiff / 2
    yoff = ydiff / 2
    for i, f in enumerate(flistin):
        print(i)
        h, d = hdio.read(f)
        h['DX'] = shape[0]
        h['DY'] = shape[1]
        newd = d[yoff:yoff+shape[1], xoff:xoff+shape[0]].astype(np.uint8)
        hic._write_dat(i, h, newd.tostring(), outdir)


def get_complex_vals(dpath, root, nlist, freqlist):
    vals = np.zeros((len(nlist), len(freqlist)), dtype=np.complex64)
    window = np.kaiser(2500, 8)
    for i, num in enumerate(nlist):
        fn = 'zernikes.txt'
        z = loadz(os.path.join(dpath, root % num, 'results', fn))
        zp = temporal_unwrap(z[:, 0])
        fzp = nf.fft(zp*window)
        vals[i] = fzp[freqlist]
    return vals

RAD2PM = 92.4898

def plot_noise_hist(cvals):
    # scale values to picometers
    # scvals = 632.699988 cvals / (np.pi * 4)
    fcvals = cvals.copy()
    fcvals.shape = fcvals.shape[0]*fcvals.shape[1]
    rval = fcvals.real * RAD2PM
    ival = fcvals.imag * RAD2PM
    range = (-5 * RAD2PM, 5 * RAD2PM)
    rhist = np.histogram(rval, bins=600, range=range)
    ihist = np.histogram(ival, bins=600, range=range)
    plt.ion()
    plt.clf()
    plt.plot(rhist[1][:-1], rhist[0])
    plt.plot(ihist[1][:-1], ihist[0])
    plt.plot(rhist[1][:-1], gaussian(600, 42)*190, linewidth='2')
    plt.title("Histogram of real/imaginary FFT background values\n(between 40 & 45 Hz)")
    plt.xlabel('Intensity (picometer)')
    plt.legend(['real component', 'imaginary component', 'gaussian (std=64.7 pm)'])

def get_time(path):
    f = open(path,'rb')
    fs = np.fromstring(f.read(36),  dtype=np.uint32)
    ftime = fs[5]*3600. + fs[6]*60 + fs[7] + fs[8]/1000000.
    print(fs)
    print(fs[5], fs[6], fs[7], fs[8])
    return ftime
 
def collect():
    flist = glob.glob('*/000_results/zernikes.txt')
    nset = 0
    fo = open('collected_results%d.txt' % nset,'w')
    for i, f in enumerate(flist):
        z = loadz(f)[1][:,0]
        fo.write('>>>>>>>>\n')
        fo.write('%d %s\n' % (i, f))
        for j in range(1000):
            fo.write('%f\n' % (z[j],))
        if (i % 200) == 0 and i != 0:
            nset += 1
            fo.close()
            fo = open('collected_results%d.txt' % nset,'w')
    fo.close()

def calc_modulation(phasecube, mask):
    '''
    Compute the modulation fraction.
    This is essentially the amplitude of the fringe divided by the 
    average image intensity
    '''
    m0, m90, m180, m270 = phasecube
    Nm = m270 - m90
    Dm = m0   - m180
    average_intensity = \
        (phasecube.sum(axis=0)*mask).sum()/mask.astype(np.int32).sum()
    mod_amp_im = np.sqrt(Nm**2 + Dm**2)/2
    ave_mod = \
        mod_amp_im[mask.astype(np.bool_)].sum()/mask.astype(np.int32).sum()
    return ave_mod/average_intensity

def diagnostics(path, maskset):
    '''
    Compute the standard deviation in a speckle measurement
    '''
    mask = maskset.mask
    rawrefpix = fdio.read(os.path.join(path, '0.rawframe'))
    refcube = combine.separate_pixelated(rawrefpix)
    rawpix = fdio.read(os.path.join(path, '1.rawframe'))
    phasecube = combine.separate_pixelated(rawpix)
    diffim = doit_speckle(os.path.join(path,'1.rawframe'), maskset, refcube, 0, output=False)[0]
    mdiffim = diffim[maskset.mask]
    sat = (rawpix > 250).sum()
    b0, b90, b180, b270 = refcube
    m0, m90, m180, m270 = phasecube
    Nm = m270 - m90
    Dm = m0   - m180
    Nb = b270 - b90
    Db = b0   - b180
    numerator = Nm*Db - Dm*Nb
    denominator = Dm*Db + Nm*Nb
    nmaskpoints = mask.astype(np.int32).sum()
    average_intensity = ((refcube + phasecube).sum(axis=0)*mask).sum() / (2 * nmaskpoints)
    modulation = np.sqrt(numerator**2 + denominator**2) / 2
    avg_mod = (modulation * mask).sum() / nmaskpoints
    return mdiffim.std(), sat, avg_mod / average_intensity

def sdiff(maskset):
    refcube = combine.separate_pixelated(fdio.read(os.path.join('.','0.rawframe')))
    diffim = doit_speckle(os.path.join('.','1.rawframe'), maskset, refcube, 0, output=False)[0]
    return diffim
     

def monitor_noise(maskset):
    while True:
        dlist = glob.glob('2018*')
        if len(dlist) > 1:
            path = dlist[-2]
            std, sat, mod = diagnostics(path, maskset)
            print(std)
        time.sleep(60)    
    
def gather_results(directory):
    """
    Collect the results in completed zernikes.txt files and export them to a file after
    averaging
    """
    fl = glob.glob(os.path.join(path_prefix, 
                                directory, '201*/000_results/zernikes.txt'))
    dl = glob.glob(os.path.join(path_prefix, 
                                directory, '201*'))
    print('number of data runs:', len(dl))
    print('number of results files found:', len(fl))
    ffl = [item for item in fl if len(loadz(item)[1])==1000]
    ncr = len(ffl)
    print('number of complete results files found:', ncr)
    zarr = np.zeros((ncr, 1000, 10))
    i = 0

    for i, f in enumerate(ffl):
        zarr[i] = loadz(f)[1]
    uarr = zarr.copy()
    for i in range(ncr):
        uarr[i, :, 0] = temporal_unwrap(zarr[i, :, 0])
    return uarr

def average_timeseries(timeseries):
    ncr = timeseries.shape[0]
    uavg = timeseries.sum(axis=0)/ncr
    return uavg

def compute_average(directory, save=True):
    uarr = gather_results(directory)
    uavg = average_timeseries(uarr)
    if save:
        np.savetxt(os.path.join(path_prefix, directory, directory+'.uavg'), uavg)
        # np.savetxt(os.path.join(path_prefix, directory, directory+'.uarr'), uarr)
    else:
        return uavg, uarr

def compute_rms(uarr, freqset=None):
    if freqset is None:
        freqset = ((33., 35.6), (36.4, 40.2))
    freq = np.arange(1000)/5.
    mask = np.zeros((1000,), dtype=np.bool_)
    for freqrange in freqset:
        mask = mask | ((freq > freqrange[0]) & (freq < freqrange[1]))
    window = np.kaiser(1000, 8)
    window.shape = (1, 1000, 1)
    fuarr = nf.fft(window * uarr, axis=1)
    selu = fuarr[:, mask, 0]
    real_rms = selu.real.std()
    imag_rms = selu.imag.std()
    return real_rms, imag_rms

def transform(timeseries):
    window = np.kaiser(1000, 8)
    if len(timeseries.shape) > 1:
        new_window_shape = window.shape + (1,) * len(time.series.shape) - 1
        window.shape = new_window_shape
    fr = np.abs(nf.fft(window * timeseries, axis=0)[:500])
    freq = np.arange(500)/5.
    return freq, fr

def gather_diagnostics(directory, maskset, range=None):
    """
    Compute a diagnostic for each run including:

    spatial RMS
    number of saturated pixels
    percentage modulation
    """
    dl = glob.glob(os.path.join(path_prefix, directory, '201*'))
    if range is not None:
        dl = dl[range[0]:range[1]]
    else:
        dl = dl[-50:]
    nruns = len(dl)
    rms = np.zeros((nruns,))
    nsat = np.zeros((nruns,), dtype=np.int32)
    mod = 0 * rms
    print('%d runs to process' % nruns)
    ff = open('diagnostics.txt','w')
    for i, d in enumerate(dl):
        try:
            rms[i], nsat[i], mod[i] = diagnostics(d, maskset)
            print("%d    %5.3f    %d %5.2f" % (i, rms[i], nsat[i], mod[i]))
            ff.write("%d    %5.3f    %d %5.2f\n" % (i, rms[i], nsat[i], mod[i]))
        except ValueError:
            rms[i], nsat[i], mod[i] = np.nan, 0, np.nan
            print("error encountered at %d" % (i,))
    ff.close()
    return rms, nsat, mod

    
   


mask480 = hsio.read('/server42/vol2/perry/mask480.saf')[1]
mask240 = hsio.read('/server42/vol2/perry/mask240.saf')[1]
mask120 = hsio.read('/server42/vol2/perry/mask120.saf')[1]
mask60 = hsio.read('/server42/vol2/perry/mask60.saf')[1]
mask30 = hsio.read('/server42/vol2/perry/mask30.saf')[1]
mask15 = hsio.read('/server42/vol2/perry/mask15.saf')[1]
maskref = hsio.read('/server42/vol2/perry/maskref.saf')[1]
maskbot = hsio.read('/server42/vol2/perry/maskbot.saf')[1]
maskmid = hsio.read('/server42/vol2/perry/maskmid.saf')[1]
masktop = hsio.read('/server42/vol2/perry/masktop.saf')[1]

ms480 = MaskSet(mask480, center=(240, 240), radius=180)
ms240 = MaskSet(mask240, center=(240, 240), radius=120)
ms120 = MaskSet(mask120, center=(240, 240), radius=60)
ms60 = MaskSet(mask60, center=(240, 240), radius=30)
ms30 = MaskSet(mask30, center=(240, 240), radius=15)
ms15 = MaskSet(mask15, center=(240, 240), radius=7.5)
msref = MaskSet(maskref, center=(356,345), radius=33)
msbot = MaskSet(maskbot, center=(227,345), radius=33)
msmid = MaskSet(maskmid, center=(227,230), radius=33)
mstop = MaskSet(masktop, center=(227,100), radius=33)

