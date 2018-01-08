from sunpy import map
import numpy as np
import os

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
            map.mask = np.zeros_like(map.data)

        # EUVI Wavelet adjustment
        if np.max(map.data) < 100:
            map.data *= map.data

        # Range Clipping
        map.data[map.data > 10000] = 10000

        map.data[map.data < 0] = 0

