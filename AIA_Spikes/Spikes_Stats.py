# Need to add in wavelength pair investegation
# Telescope 1: [304,94]
# Telescope 2: [171]
# Telescope 3: [211,193]
# Telescope 4: [335,131]

import os
import numpy as np
import pandas as pd
from astropy.io import fits

class Spikes_Stats:
    def __init__(self, directory):
        # Root directory of the Spikes Database: e.g. '/Volumes/BigSolar/Filtered_Spikes'
        self.dir = os.path.abspath(directory)

        # Read in the expected DB file
        self.spikes_db = pd.read_hdf(self.dir + '/Table_SpikesDB.h5', 'table')
        self.db_groups()

        # Number of Groups to sample from
        self.n_sample_groups = len(self.spikes_db.GroupNumber.unique())

    def db_groups(self):
        # group the spikes into 12 second intervals
        self.spikes_db['GroupNumber'] = self.spikes_db.groupby(pd.Grouper(key='YMDTime', freq='12s')).ngroup()

    def spikes_dataframe_gen(self,n_sample_groups=self.n_sample_groups):

        sample_groups = np.random.choice(self.spikes_db.GroupNumber.unique(), size=n_sample_groups)
        self.spikes_df =

        for group in sample_groups:
