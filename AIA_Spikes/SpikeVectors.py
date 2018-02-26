import os
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import ndimage


def spikes_to_image(spike_file):
    # read a spike file and return an image with the spikes
    # Raw_spikes are a (3,N) int32 array with:
    #   [0,N] pixel location in vector format
    #   [1,N] pixel value before despike calibration
    #   [2,N] pixel value at level 1 image after despike
    #
    # Returns a [2,4096,4096] array with [0,:,:] with the spike value, and [1,:,:] with the level 1 value

    raw_spikes = fits.open(spike_file)[1].data
    spike_vector = np.zeros((4096, 4096), dtype='int32').flatten()
    lev1_vector = np.zeros_like(spike_vector)

    spike_vector[raw_spikes[0, :]] = raw_spikes[1, :]
    lev1_vector[raw_spikes[0, :]] = raw_spikes[2, :]

    return np.stack((spike_vector.reshape((4096, 4096)), lev1_vector.reshape((4096, 4096))))


class Sort_Spikes:
    def __init__(self, directory, count_index=0):
        # This isn't fast enough yet....
        # Bench marking is putting the processing rate at about 1 file per sec. (
        #
        # Root directory of the Database: e.g. '/Volumes/BigSolar/AIA_Spikes'
        self.dir = os.path.abspath(directory)
        self.count_index = int(count_index)

        # Number of Coincident spikes needed for a positive detection.
        # !!!!! Set for 2 or more coincidence hits. Change for more stringent parameters. !!!!!
        self.n_co_spikes = 2.

        # Read in the expected DB file
        self.spikes_db = pd.HDFStore(self.dir+'/Table_SpikesDB.h5')

        # Segment the DB into 12s groups (a full cycle of wavelengths)
        self.db_groups()

        # Generate filtered spike files
        self.good_spike_db_gen()

    def db_groups(self):
        # group the spikes into 12 second intervals

        if '/GroupNumber' not in self.spikes_db.keys():
            self.spikes_db.put('GroupNumber', self.spikes_db.get('YMDTime').groupby(pd.Grouper(key='YMDTime', freq='12s')).ngroup())

    def good_spike_filter(self, subset):
        # 3x3 structuring element with connectivity 2
        struct = ndimage.generate_binary_structure(2, 2)

        spike_filter = np.zeros((4096, 4096))

        for spike_path in subset.Path:
            spike_filter += ndimage.binary_dilation((spikes_to_image(self.dir+spike_path.decode('UTF-8'))[0, :, :] > 0), structure=struct).astype(spike_filter.dtype)

        return spike_filter > (self.n_co_spikes - 1)

    def filter_spike_file_rename(self, old_filename):
        # Return a modified filename of the filtered spike files.
        new_name = 'filtered'+str(int(self.n_co_spikes))
        return old_filename.replace('spikes', new_name).replace('AIA_Spikes', 'Filtered_Spikes')

    def good_spike_db_gen(self):
        # Create a new file database with only good spikes.
        # The file name is identical to the generating file except it is .filtered.fits

        group_numbers = self.spikes_db.get('GroupNumber')
        paths = self.spikes_db.get('Path')

        for count, group_number in enumerate(group_numbers.unique()):

            # Selecting every 8th file for crappy multi threading
            if count % 8 == self.count_index:
                subset = paths.loc[group_numbers == group_number]

                # should create a file checking to see if it already exists.

                spike_filter = self.good_spike_filter(subset)

                for ind_num in subset.index:
                    good_spikes_vector = (spikes_to_image(subset.Path[ind_num])[0,:,:]*spike_filter).flatten()
                    good_spikes_lev1 = (spikes_to_image(subset.Path[ind_num])[1,:,:]*spike_filter).flatten()
                    good_spikes_index = np.where((good_spikes_vector > 0))[0]

                    # Need to compress the output CompImageHDU maybe?
                    hdu = fits.PrimaryHDU(np.stack((good_spikes_index,good_spikes_vector[good_spikes_index],good_spikes_lev1[good_spikes_index])))
                    hdu.writeto(self.filter_spike_file_rename(subset.Path[ind_num]), overwrite=True)

                print('Group number '+str(group_number)+' is complete.')

    def plot_example(self):
        # Plotting an example of a spikes image
        sp_im = spikes_to_image('/Volumes/BigSolar/AIA_Spikes/2010/06/04/2010-06-04T16:00:23.07Z_0131.spikes.fits')
        plt.imshow(sp_im[0, :, :] > 1, cmap='binary_r')
        plt.axis('off')
        plt.title('All Spikes Removed from an AIA 131 Image')