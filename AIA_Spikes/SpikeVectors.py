import os
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import ndimage
import matplotlib.pyplot as plt


def spikes_to_image(spike_file):
    # read a spike file and return an image with the spikes
    # Raw_spikes are a (3,N) int32 array with:
    #   [0,N] pixel location in vector format
    #   [1,N] pixel value before despike calibration
    #   [2,N] pixel value at level 1 image after despike
    #
    # Returns a [2,4096,4096] array with [0,:,:] with the spike value, and [1,:,:] with the level 1 value

    raw_spikes = fits.open(spike_file)[-1].data
    spike_vector = np.zeros((4096, 4096), dtype='int32').flatten()
    lev1_vector = np.zeros_like(spike_vector)

    spike_vector[raw_spikes[0, :]] = raw_spikes[1, :]
    lev1_vector[raw_spikes[0, :]] = raw_spikes[2, :]

    return np.stack((spike_vector.reshape((4096, 4096)), lev1_vector.reshape((4096, 4096))))


def image_to_spikes(images):
    # Take an [2, :, :] image numpy array and write it like a spike file
    # returns an astropy fits hdu object

    spikes_index = np.where((images[0, :, :].flatten() > 0))[0]

    hdu = fits.PrimaryHDU(np.stack((spikes_index, images[0, :, :].flatten()[spikes_index], images[1, :, :].flatten()[spikes_index])).astype('int32'))

    return hdu


def plot_example():
    # Plotting an example of a spikes image
    sp_im = spikes_to_image('/Volumes/BigSolar/AIA_Spikes/2010/06/04/2010-06-04T16:00:23.07Z_0131.spikes.fits')
    plt.imshow(sp_im[0, :, :] > 1, cmap='binary_r')
    plt.axis('off')
    plt.title('All Spikes Removed from an AIA 131 Image')


class Sort_Spikes:
    def __init__(self, directory, count_index=0, end_group=None):
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
        self.spikes_db = pd.HDFStore(os.path.join(self.dir,'Table_SpikesDB.h5'))

        # Segment the DB into 12s groups (a full cycle of wavelengths)
        self.db_groups()

        # Generate filtered spike files
        self.good_spike_db_gen(terminus=end_group)

    def db_groups(self):
        # group the spikes into 12 second intervals

        if '/GroupNumber' not in self.spikes_db.keys():
            self.spikes_db.put('GroupNumber', self.spikes_db.get('YMDTime').groupby(pd.Grouper(key='YMDTime', freq='12s')).ngroup())

    def good_spike_filter(self, subset):
        # Returns a masked array for coincident spikes; 3x3 structuring element with connectivity 2
        # Also creates a list of spike_images for each wavelength

        struct = ndimage.generate_binary_structure(2, 2)

        spike_filter = np.zeros((4096, 4096))

        self.sp_im = []

        for spike_path in subset.Path:
            self.sp_im += [spikes_to_image(self.dir+spike_path.decode('UTF-8'))]
            spike_filter += ndimage.binary_dilation((self.sp_im[-1][0, :, :] > 0), structure=struct).astype(spike_filter.dtype)

        return spike_filter > (self.n_co_spikes - 1)

    def filter_spike_file_rename(self, old_filename):
        # Return a modified filename of the filtered spike files.
        new_name = 'filtered'+str(int(self.n_co_spikes))
        return old_filename.replace('spikes', new_name).replace('AIA_Spikes', 'Filtered_Spikes')

    def good_spike_db_gen(self, terminus=float('inf')):
        # Create a new files with only good spikes.
        # The file name is identical to the generating file except it is .filtered.fits

        group_numbers = self.spikes_db.get('GroupNumber')
        paths = self.spikes_db.get('Path')

        for count, group_number in enumerate(group_numbers.unique()):
            # Exit the loop early.
            if group_number > terminus:
                self.spikes_db.close()
                group_numbers = None
                paths = None
                break

            # Selecting every 8th file for crappy multi threading
            if count % 8 == self.count_index:
                subset = paths.loc[group_numbers == group_number]

                # should create a file checking to see if it already exists.

                spike_filter = self.good_spike_filter(subset)

                # filtering spikes and saving files.
                for ind_num in subset.index:
                    good_spikes_vector = (self.sp_im[ind_num-subset.index.min()][0,:,:]*spike_filter).flatten()
                    good_spikes_lev1 = (self.sp_im[ind_num-subset.index.min()][1,:,:]*spike_filter).flatten()
                    good_spikes_index = np.where((good_spikes_vector > 0))[0]

                    hdu = fits.PrimaryHDU(np.stack((good_spikes_index, good_spikes_vector[good_spikes_index], good_spikes_lev1[good_spikes_index])).astype('int32'))
                    hdu.writeto(self.filter_spike_file_rename(os.path.join(self.dir, subset.Path[ind_num].decode('UTF-8'))), overwrite=True)

                print('Group number '+str(group_number)+' is complete.')

        self.spikes_db.close()
        group_numbers = None
        paths = None