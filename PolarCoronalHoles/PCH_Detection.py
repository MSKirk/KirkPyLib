from sunpy import map
import numpy as np
import os
import PCH_Tools
import astropy.units as u
from skimage import exposure, morphology


'''
Detection of polar coronal holes given a directory of images. 
A pure python version reproducing the following IDL routines:
    chole_*_run.pro
    chole_*_year.pro
    chole_*_year2hrot.pro
    chole_*_data.pro
    chole_mark.pro
    chole_mask.pro
    chole_series.pro
    chole_area.pro
    what_image.pro
    chole_series_area.pro
    **_image_check.pro
    new_center.pro
'''


class PCH_Detection:

    def __init__(self, image_dir):
        self.dir = os.path.abspath(image_dir)

    def recenter_data(theta, rho):
        # Recenter polar data for fitting.
        return (theta,rho)


    def chole_mask(self, map, factor=0.5):
        # To isolate the lim of the sun to study coronal holes
        # returns a binary array of the outer lim of the sun

        #Class check
        if not isinstance(map, sunpy.map.GenericMap):
            raise ValueError('Input needs to be an sunpy map object.')

        # Bad image check
        if np.max(map.data) < 1:
            map.mask = np.ones_like(map.data)

        # EUVI Wavelet adjustment
        if np.max(map.data) < 100:
            map.data *= map.data

        # Range Clipping
        map.data[map.data > 10000] = 10000

        map.data[map.data < 0] = 0

        rsun = np.array([map.rsun_obs.to('deg').value])
        rsun_pix = np.array([map.wcs.all_world2pix(rsun, rsun, 0)[0] - map.wcs.all_world2pix(0, 0, 0)[0],
                             map.wcs.all_world2pix(rsun, rsun, 0)[1] - map.wcs.all_world2pix(0, 0, 0)[1]])

        # EUVI Wavelet adjustment
        if map.detector == 'EUVI':
            if map.wavelength > 211*u.AA:
                   map.mask = exposure.equalize_hist(map.data, mask=np.logical_not(PCH_Tools.annulus_mask(map.data.shape, (0,0), rsun_pix, center=map.wcs.wcs.crpix)))
            else:
                map.mask = exposure.equalize_hist(map.data)
        else:
            map.mask = np.copy(map.data)

        # Found through experiment...
        if map.detector == 'AIA':
            if map.wavelength == 193 * u.AA:
                factor = 0.30
            if map.wavelength == 171 * u.AA:
                factor = 0.62
            if map.wavelength == 304 * u.AA:
                factor = 0.15
        if map.detector == 'EIT':
            if map.wavelength == 195 * u.AA:
                factor = 0.27
            if map.wavelength == 171 * u.AA:
                factor = 0.37
            if map.wavelength == 304 * u.AA:
                factor = 0.14
            if map.wavelength == 284 * u.AA:
                factor = 0.22
        if map.detector == 'EUVI':
            if map.wavelength == 195 * u.AA:
                factor = 0.53
            if map.wavelength == 171 * u.AA:
                factor = 0.55
            if map.wavelength == 304 * u.AA:
                factor = 0.35
            if map.wavelength == 284 * u.AA:
                factor = 0.22
        if map.detector == 'SWAP':
            if map.wavelength == 174 * u.AA:
                factor = 0.55

        # Creating a kernel for the morphological transforms
        if map.wavelength == 304 * u.AA:
            structelem = morphology.disk(np.average(rsun_pix) * 0.007)
        else:
            structelem = morphology.disk(np.average(rsun_pix) * 0.004)

        # First morphological pass...
        if map.wavelength == 304 * u.AA:
            map.mask = morphology.opening(morphology.closing(map.mask, selem=structelem))
        else:
            map.mask = morphology.closing(morphology.opening(map.mask, selem=structelem))

        # Masking off limb structures and Second morphological pass...
        map.mask = morphology.opening(map.mask * PCH_Tools.annulus_mask(map.data.shape, (0,0), rsun_pix, center=map.wcs.wcs.crpix), selem=structelem)

        # Extracting holes...
        thresh = PCH_Tools.hist_percent(map.mask[np.nonzero(map.mask)], factor, number_of_bins=1000.)
        map.mask = np.where(map.mask >= thresh, map.mask, 0)



