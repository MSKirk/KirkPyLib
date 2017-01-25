
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import fnmatch
import astropy.units as u
from astropy.time import Time
from sunpy.time import parse_time
from astropy.table import Table
import pandas as pd
import DataCenterImport as DC

class DCAnalysis:
    def __init__(self, directory):
        '''
        :param directory: the directory in which the google data is located
        '''
        self.dir=directory
        self.dat=DC.DataCenter(directory)

    def site_ave(self):


        error_ave = np.array([np.mean(self.dat.error_set['rate'][self.dat.error_set['loc_id'] == idnum]) for idnum in
                             self.dat.site_info['ID']])

        error_std = np.array([np.std(self.dat.error_set['rate'][self.dat.error_set['loc_id'] == idnum]) for idnum in
                             self.dat.site_info['ID']])

        tbegin = [np.min(dat.error_set.index[dat.error_set['loc_id'] == idnum]) for idnum in dat.site_info['ID']]

        tend = [np.max(dat.error_set.index[dat.error_set['loc_id'] == idnum]) for idnum in dat.site_info['ID']]
