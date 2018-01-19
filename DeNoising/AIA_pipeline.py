import os
import fnmatch
import sunpy.map as map
import sunpy.instr.aia as aia
import astropy.units as u
from astropy.io import fits
import numpy as np
from skimage import restoration
from sunpy.image.coalignment import mapcube_coalign_by_match_template
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

class CubeProcess:
    def __init__(self, wavelength, directory=None):
        """
        All of the routines needed to process and AIA image through the DeNoising pipeline.
        (We can slot actual de-noising routines in as needed)

        :param wavelength: aia wavelength directory to process
        """
        self.wavelength = wavelength * u.AA
        self.get_files(directory)
        self.get_psf()

    def get_files(self, directory=None):

        if directory:
            self.dir = directory
        else:
            directory = {94 * u.AA: '/Volumes/DataDisk/AIA/094/',
                        131 * u.AA: '/Volumes/DataDisk/AIA/131/',
                        171 * u.AA: '/Volumes/DataDisk/AIA/171/',
                        193 * u.AA: '/Volumes/DataDisk/AIA/193/',
                        211 * u.AA: '/Volumes/DataDisk/AIA/211/',
                        304 * u.AA: '/Volumes/DataDisk/AIA/304/',
                        335 * u.AA: '/Volumes/DataDisk/AIA/335/'}

            self.dir = directory[self.wavelength]

        self.filelist = fnmatch.filter(os.listdir(self.dir), '*aia*'+str(np.int(self.wavelength.value))+'*.fits')


    def read_aia(self, ii):

        self.full_map = map.Map(self.dir + self.filelist[ii])

    def get_psf(self, origin='LMSAL'):

        if origin == 'LMSAL':
            psf_files = {94 * u.AA: '/Users/mskirk/data/AIA/Point Spread Function/LMSAL/AIA_94_PSF.fits',
                         131 * u.AA: '/Users/mskirk/data/AIA/Point Spread Function/LMSAL/AIA_131_PSF.fits',
                         171 * u.AA: '/Users/mskirk/data/AIA/Point Spread Function/LMSAL/AIA_171_PSF.fits',
                         193 * u.AA: '/Users/mskirk/data/AIA/Point Spread Function/LMSAL/AIA_193_PSF.fits',
                         211 * u.AA: '/Users/mskirk/data/AIA/Point Spread Function/LMSAL/AIA_211_PSF.fits',
                         304 * u.AA: '/Users/mskirk/data/AIA/Point Spread Function/LMSAL/AIA_304_PSF.fits',
                         335 * u.AA: '/Users/mskirk/data/AIA/Point Spread Function/LMSAL/AIA_335_PSF.fits'}

            self.psf = fits.getdata(psf_files[self.wavelength])

        if origin == 'SWRI':
            psf_files = {94 * u.AA: '/Users/mskirk/data/AIA/Point Spread Function/SWRI/PSF94A.fits',
                         131 * u.AA: '/Users/mskirk/data/AIA/Point Spread Function/SWRI/PSF131A.fits',
                         171 * u.AA: '/Users/mskirk/data/AIA/Point Spread Function/SWRI/PSF171A.fits',
                         193 * u.AA: '/Users/mskirk/data/AIA/Point Spread Function/SWRI/PSF193A.fits',
                         211 * u.AA: '/Users/mskirk/data/AIA/Point Spread Function/SWRI/PSF211A.fits',
                         304 * u.AA: '/Users/mskirk/data/AIA/Point Spread Function/SWRI/PSF304A.fits',
                         335 * u.AA: '/Users/mskirk/data/AIA/Point Spread Function/SWRI/PSF335A.fits'}

            self.psf = np.pad(fits.getdata(psf_files[self.wavelength]),(1847,1848), 'constant', constant_values=(0,0))

    def deconvolve_psf(self, method='WH'):

        self.full_map.meta['lvl_num'] = 1.4

        if method == 'WH':
            self.full_map.data = restoration.unsupervised_wiener(self.full_map.data.astype('float64'), self.psf.astype('float64'), clip=False)[0]

        if method == 'RL_SSW':
            # should be equivalent to the IDL AIA_DECONVOLVE_RICHARDSONLUCY() routine but it isn't
            # Not working...
            image = self.full_map.data.astype(np.float)
            im_deconv = np.copy(self.full_map.data.astype(np.float))
            psf = self.psf.astype(np.float)
            psf_mirror = psf[::-1, ::-1] # to make the correlation easier
            psfnorm = fftconvolve(psf, np.ones_like(psf), 'same')

            for _ in range(25):
                relative_blur = image / fftconvolve(psf, im_deconv, 'same')
                im_deconv *= fftconvolve(relative_blur, psf_mirror, 'same')/psfnorm

            self.full_map.data=np.abs(im_deconv)

        if method == 'RL':
            # Not working...
            self.full_map.data = restoration.richardson_lucy(self.full_map.data.astype('float64'), self.psf.astype('float64'), iterations=25, clip=False)

    def region_cutout(self):
        self.aia_promote()

        ji_xrange = [-539.16847142, -115.07568342] * u.arcsec
        ji_yrange = [-116.93790609, 113.73012591] * u.arcsec

        self.submap = self.full_map.submap(ji_xrange, ji_yrange)

    def aia_promote(self):
        if self.full_map.processing_level == 1.4:
            self.full_map = aia.aiaprep(self.full_map)
            self.full_map.meta['lvl_num'] = 1.6
        else:
            self.full_map = aia.aiaprep(self.full_map)
            self.full_map.meta['lvl_num'] = 1.5

    def image_save(self, outfile, svdir, filenm):

        outfile.save(svdir+filenm, filetype='fits', clobber=True)

    def png_save(self, outfile, svdir, filenm):

        outfile.plot()
        plt.savefig(svdir+filenm,  bbox_inches='tight')

    def im_coalign(self):

        self.mapcube = mapcube_coalign_by_match_template(self.mapcube, layer_index=np.round(len(self.mapcube)/2))

class CubeAnalysis:
    def __init__(self, wavelength, directory=None):
        """
        All of the routines needed to analyse AIA image cubes.

        :param wavelength: aia wavelength directory to process
        """
        self.wavelength = wavelength * u.AA
        self.get_files(directory)

    def get_files(self, directory=None):

        if directory:
            self.dir = directory
        else:
            directory = {94 * u.AA: '/Volumes/DataDisk/AIA/094/',
                        131 * u.AA: '/Volumes/DataDisk/AIA/131/',
                        171 * u.AA: '/Volumes/DataDisk/AIA/171/',
                        193 * u.AA: '/Volumes/DataDisk/AIA/193/',
                        211 * u.AA: '/Volumes/DataDisk/AIA/211/',
                        304 * u.AA: '/Volumes/DataDisk/AIA/304/',
                        335 * u.AA: '/Volumes/DataDisk/AIA/335/'}

            self.dir = directory[self.wavelength]

        self.filelist = fnmatch.filter(os.listdir(self.dir), '*aia*'+str(np.int(self.wavelength.value))+'*.fits')

    def make_cube(self):
        self.mapcube = sunpy.map.Map(self.dir+'*.fits', cube=True)

    def read_image(self, ii):

        self.map = map.Map(self.dir + self.filelist[ii])

    def cube_ssim(self, comp_dir):

