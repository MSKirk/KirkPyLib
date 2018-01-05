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
'''


class PCH_Detection:

    def __init__(self, image_dir):
        self.dir = os.path.abspath(image_dir)

