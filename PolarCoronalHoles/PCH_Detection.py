from sunpy import map
import numpy as np
import os
import PCH_Tools
import astropy.units as u
from skimage import exposure

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

        if map.detector == 'EUVI':
            if map.wavelength > 211*u.AA:
                rsun = np.array([map.rsun_obs.to('deg').value])
                rsun_pix = np.array([map.wcs.all_world2pix(rsun,rsun,0)[0]-map.wcs.all_world2pix(0,0,0)[0], map.wcs.all_world2pix(rsun,rsun,0)[1]-map.wcs.all_world2pix(0,0,0)[1]])
                map.mask = exposure.equalize_hist(map.data, mask=PCH_Tools.annulus_mask(map.data.shape, (0,0), rsun_pix, center=map.wcs.wcs.crpix)