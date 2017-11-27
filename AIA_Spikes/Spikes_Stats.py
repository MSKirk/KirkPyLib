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


class Spikes_Stats:
    def __init__(self, directory):
        # Root directory of the Spikes Database: e.g. '/Volumes/BigSolar/Filtered_Spikes'
        self.dir = os.path.abspath(directory)

        # Read in the expected DB file
        self.spikes_db = pd.read_hdf(self.dir + '/Table_SpikesDB.h5', 'table')
        self.db_groups()

    def db_groups(self):
        # group the spikes into 12 second intervals
        self.spikes_db['GroupNumber'] = self.spikes_db.groupby(pd.Grouper(key='YMDTime', freq='12s')).ngroup()

    def spikes_dataframe_gen(self, n_sample_groups=False):

        if n_sample_groups == False:
            n_sample_groups = len(self.spikes_db.GroupNumber.unique())

        sample_groups = np.random.choice(self.spikes_db.GroupNumber.unique(), size=n_sample_groups)
        self.spikes_df = pd.DataFrame(columns={'MJDTime','YMDTime','Wavelength','Pix_i','Pix_j','Int','GroupNumber','Im_int'})

        # For each group in our sample, read in the fits files information
        for group in sample_groups:
            for db_index in self.spikes_db.loc[self.spikes_db['GroupNumber'] == group].index:
                temp_df = pd.DataFrame(columns={'MJDTime','YMDTime','Wavelength','Pix_i','Pix_j','Int','GroupNumber','Im_int'})
                raw_spikes = fits.open(self.spikes_db.Path[db_index])[0].data

                temp_df.Pix_i = raw_spikes[0] % 4096
                temp_df.Pix_j = np.int_((raw_spikes[0] - raw_spikes[0] % 4096)/4096)
                temp_df.Int = raw_spikes[1]
                temp_df.Im_int = raw_spikes[2]

                temp_df.GroupNumber = self.spikes_db.GroupNumber[db_index]
                temp_df.MJDTime = np.repeat(self.time_gen(self.spikes_db.Path[db_index]).mjd, len(temp_df.Pix_i))
                temp_df.YMDTime = np.repeat(self.time_gen(self.spikes_db.Path[db_index]).datetime, len(temp_df.Pix_i))
                temp_df.Wavelength = np.repeat(self.wave_gen(self.spikes_db.Path[db_index]), len(temp_df.Pix_i))

                self.spikes_df = pd.concat([self.spikes_df,temp_df], ignore_index=True)

    def time_gen(self, filename):
        # This corresponds to the t_obs keyword value from the associated AIA image.
        datetime = [filename.split('/')[-1].split('Z')[0]]
        return Time(datetime, format='isot', scale='utc')

    def wave_gen(self, filename):
        return np.asarray([filename.split(':')[-1].split('_')[1].split('.')[0]], dtype='uint16')

