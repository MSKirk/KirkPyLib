# Need to add in wavelength pair investegation
# Telescope 1: [304,94]
# Telescope 2: [171]
# Telescope 3: [211,193]
# Telescope 4: [335,131]

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from scipy import ndimage
import pyupset as pyu
import matplotlib.pyplot as plt

class Spikes_Stats:
    def __init__(self, directory):
        # Root directory of the Spikes Database: e.g. '/Volumes/BigSolar/Filtered_Spikes'
        self.dir = os.path.abspath(directory)

        # Read in the expected DB file
        self.spikes_db = pd.read_hdf(self.dir + '/Table_SpikesDB.h5', 'table')
        # Gropuing in TIme
        self.db_groups()

        # Number of Coincident spikes needed for a positive detection.
        # !!!!! Set for 2 or more coincidence hits. Change for more stringent parameters. !!!!!
        self.n_co_spikes = 2.

    def db_groups(self):
        # group the spikes into 12 second intervals
        self.spikes_db['TimeGroup'] = self.spikes_db.groupby(pd.Grouper(key='YMDTime', freq='12s')).ngroup()

    def spike_space_grouping(self,subset):
        # count through the coincident spikes and label them sequentially...
        # 3x3 structuring element with connectivity 2
        struct = ndimage.generate_binary_structure(2, 2)

        spike_filter = np.zeros((4096, 4096))

        for spike_path in subset.Path:
            spike_filter += ndimage.binary_dilation(np.int_(self.spikes_to_image(spike_path)[0,:,:] > 0), structure=struct).astype(spike_filter.dtype)

        labeled_array, num_features = ndimage.measurements.label(np.int_(spike_filter > (self.n_co_spikes - 1)))
        return labeled_array

    def spikes_dataframe_gen(self, n_sample_groups=0):

        if not n_sample_groups:
            n_sample_groups = len(self.spikes_db.TimeGroup.unique())

        sample_groups = np.random.choice(self.spikes_db.TimeGroup.unique(), size=n_sample_groups)
        self.spikes_df = pd.DataFrame(columns={'MJDTime','YMDTime','Wavelength','Pix_i','Pix_j','Int','TimeGroup','Im_int','SpaceGroup'})

        # For each group in our sample, read in the fits files information
        # Concat into a large dataframe
        for group in sample_groups:
            # might want to move spacial grouping to SpikeVectors...
            subset = self.spikes_db.loc[self.spikes_db['TimeGroup'] == group]

            self.spacegroup = self.spike_space_grouping(subset)

            for db_index in subset.index:
                temp_df = pd.DataFrame(columns={'MJDTime','YMDTime','Wavelength','Pix_i','Pix_j','Int','TimeGroup','Im_int','SpaceGroup'})
                raw_spikes = fits.open(self.spikes_db.Path[db_index])[0].data

                temp_df.Pix_j = raw_spikes[0] % 4096
                temp_df.Pix_i = np.int_((raw_spikes[0] - raw_spikes[0] % 4096)/4096)
                temp_df.Int = raw_spikes[1]
                temp_df.Im_int = raw_spikes[2]

                temp_df.TimeGroup = self.spikes_db.TimeGroup[db_index]
                temp_df.MJDTime = np.repeat(self.time_gen(self.spikes_db.Path[db_index]).mjd, len(temp_df.Pix_i))
                temp_df.YMDTime = np.repeat(self.time_gen(self.spikes_db.Path[db_index]).datetime, len(temp_df.Pix_i))
                temp_df.Wavelength = np.repeat(self.wave_gen(self.spikes_db.Path[db_index]), len(temp_df.Pix_i))
                temp_df.SpaceGroup = self.spacegroup[temp_df.Pix_i, temp_df.Pix_j]

                self.spikes_df = pd.concat([self.spikes_df,temp_df], ignore_index=True)

    def time_gen(self, filename):
        # This corresponds to the t_obs keyword value from the associated AIA image.
        datetime = [filename.split('/')[-1].split('Z')[0]]
        return Time(datetime, format='isot', scale='utc')

    def wave_gen(self, filename):
        return np.asarray([filename.split(':')[-1].split('_')[1].split('.')[0]], dtype='uint16')

    def spikes_to_image(self, filename):
        # read a spike file and return an image with the spikes
        # Raw_spikes are a (3,N) int32 array with:
        #   [0,N] pixel location in vector format
        #   [1,N] pixel value before despike calibration
        #   [2,N] pixel value at level 1 image after despike
        #
        # Returns a [2,4096,4096] array with [0,:,:] with the spike value, and [1,:,:] with the level 1 value

        raw_spikes = fits.open(filename)[0].data
        spike_vector = np.zeros((4096, 4096), dtype='int32').flatten()
        lev1_vector = np.zeros_like(spike_vector)

        spike_vector[raw_spikes[0,:]] = raw_spikes[1,:]
        lev1_vector[raw_spikes[0,:]] = raw_spikes[2,:]

        return np.stack((spike_vector.reshape((4096, 4096)),lev1_vector.reshape((4096, 4096))))

    def pyupset_format(self):

        return {'94': self.spikes_df.query('Wavelength == 94'),'131': self.spikes_df.query('Wavelength == 131'),
               '171': self.spikes_df.query('Wavelength == 171'),'193': self.spikes_df.query('Wavelength == 193'),
               '211': self.spikes_df.query('Wavelength == 211'),'304': self.spikes_df.query('Wavelength == 304'),
               '335': self.spikes_df.query('Wavelength == 335')}

    def upset_plots_gen(self):

            self.spikes_dataframe_gen(n_sample_groups=50)
            ups = self.pyupset_format()

            plt.rc('font', size=12)
            pyu.plot(ups, unique_keys=['SpaceGroup', 'TimeGroup'], inters_degree_bounds=(2, 2), sort_by='size')
            plt.title('Pairwise Spike Coincidences', {'fontsize': 18, 'fontweight':'bold'})
            plt.savefig('/Users/mskirk/Documents/Conferences/AGU 2017/2-way.png')

            plt.rc('font', size=12)
            pyu.plot(ups, unique_keys=['SpaceGroup', 'TimeGroup'], inters_degree_bounds=(2, 2), sort_by='size',
                    query=[('304','94'),('211','193'),('335','131')])
            plt.title('Pairwise Spike Coincidences', {'fontsize': 18, 'fontweight':'bold'})
            plt.savefig('/Users/mskirk/Documents/Conferences/AGU 2017/2-way_c.png')

            plt.rc('font', size=12)
            pyu.plot(ups, unique_keys=['SpaceGroup', 'TimeGroup'], inters_degree_bounds=(3, 3), sort_by='size')
            plt.title('3-Way Spike Coincidences', {'fontsize': 18, 'fontweight':'bold'})
            plt.savefig('/Users/mskirk/Documents/Conferences/AGU 2017/3-way.png')

            plt.rc('font', size=12)
            pyu.plot(ups, unique_keys=['SpaceGroup', 'TimeGroup'], inters_degree_bounds=(4, 4), sort_by='size')
            plt.title('4-way Spike Coincidences', {'fontsize': 18, 'fontweight':'bold'})
            plt.savefig('/Users/mskirk/Documents/Conferences/AGU 2017/4-way.png')

            plt.rc('font', size=12)
            pyu.plot(ups, unique_keys=['SpaceGroup', 'TimeGroup'], inters_degree_bounds=(5, 7), sort_by='degree',
                     query=[('304', '94','211', '193', '335', '131','171')])
            plt.title('5, 6, and 7-way Spike Coincidences', {'fontsize': 18, 'fontweight':'bold'})
            plt.savefig('/Users/mskirk/Documents/Conferences/AGU 2017/567-way.png')
