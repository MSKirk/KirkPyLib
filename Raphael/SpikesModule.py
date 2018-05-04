import os
import multiprocessing as mp
import pandas as pd
import numpy as np
from astropy.io import fits
import fitsio


class SpikesLookup:
    def __init__(self, db_filepath, data_dir, outputdir, n_co_spikes=2.):
        # Address of dataframe .hdf5 file and directory of fits files
        self.db_filepath = db_filepath
        self.data_dir = data_dir
        # Output directory where the filtered spikes fits files will be written.
        self.output_dir = outputdir
        # Open the data base as a store.
        self.spikes_db = pd.HDFStore(db_filepath)
        # Number of coincidental spikes
        self.n_co_spikes = 2
        # Get the dataframe "Path" containing the list of paths of the fits files, set index to the GroupNumber Series
        paths_df = self.spikes_db.get('Path').set_index(self.spikes_db.get('GroupNumber'))
        # Convert group number and paths to numpy arrays for faster lookup than pandas queries (x1000 speed-up, crazy)
        self.npgroups = paths_df.Path.index.values
        self.nppaths = paths_df.Path.values
        # Get unique values of groups (ugroups), get associated indices (uinds) and counts for each group (ugroupc)
        self.ugroups, self.uinds, self.ugroupc = np.unique(self.npgroups, return_index=True, return_counts=True)
        # Get list of 1D and 2D coordinates for a 4096 x 4096 array
        nx = 4096
        ny = 4096
        shape = [ny, nx]
        coords_1d = np.arange(nx * ny)
        coordy, coordx = np.unravel_index(coords_1d, shape)
        coords2d = np.array([coordy, coordx])
        # Create a look-up table of the 2D coordinates of all the pixels and their 8 neighrest neighbour.
        # These are very big arrays meant to be shared over the different workers
        # 8 nearest neighbour relative coordinates in 2D
        coords_8nb = np.array([[0, 0], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1]])
        # Dimensions [neighbour index, x and y axis, pixel index]
        coords2d_8nb = np.clip(coords2d[np.newaxis, ...] + coords_8nb[..., np.newaxis], 0, 4095)
        # Convert the above to 1D coordinates.
        self.index_8nb = np.array([coords2d_8nb[i, 0, :] * shape[1] + coords2d_8nb[i, 1, :] for i in range(len(coords_8nb))],
                                  dtype='int32', order='C')

        # For sharing the above across workers:
        # Best solution (2nd answer from EelkeSpaak) at:
        # https://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing
        #
        # Alternatives:
        # https://stackoverflow.com/questions/659865/multiprocessing-sharing-a-large-read-only-object-between-processes
        # https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html

    def get_filepaths(self, group_index):
        """
        Get a decoded list of files associated with the same given group/index from the dataframe.

        :param group_index: unique group number associated with the list of spikes files to process
        :return: List of strings of decoded absolute files paths
        """

        path_index = self.uinds[group_index]
        count = self.ugroupc[group_index]
        paths = [ os.path.join(self.data_dir, fpath.decode('UTF-8')) for fpath in self.nppaths[path_index:path_index + count]]
        return paths


def accumulate_spikes(spikes):
    """
    Cumulate the spikes from the group using 8-connectivity.
    Spikes that falls within the 8 nearest neighbours of a given pixel pile up in a list of maximum 12 different wavelengths.
    Finally, get the locations of the pixels where at least 2 spikes have cumulated within the 8 neirest neighbours.

    :param spikes: List of spikes coordinates within the group.
    :return: 1D coordinates of the pixels were hit by at least 2 spikes within the 8-connectivity neihbourhood.
    """

    # Initial set of the group-cumulated set.
    # MUST take the "unique" ones. If I don't do that, there will be a bias in the count,
    # as if a single wavelength was hit twice at the same photosite, which does not make sense, and it would
    # be seen as if it was hit on 2 different wavelengths at the same pixel.
    # This replicates the effect of applying the binary dilation in the image-processing-based version of the algorithm
    # where if 2 different pixels in the same image shares the same nearest neighbour, they become part of the
    # same dilated 1-valued set. The fact that they are 1-valued is what I replicate here with the use of "unique".
    cumulated_spikes_coords = np.unique(lut.index_8nb[:, spikes[0][0, :]].ravel())

    # Make each spike coordinates spawn an additional set of 8 nearest neighbours coordinates,
    # and accumulate it across the images of the group (accumulation = concatenation).
    for raw_spikes in spikes[1:]:
        cumulated_spikes_coords = np.concatenate([cumulated_spikes_coords, np.unique(lut.index_8nb[:, raw_spikes[0, :]].ravel())])

    # Get the distribution of these coordinates:
    # actual 1D coordinate values associated with how many times it appears in the group-cumulated set
    # Faster implementation than np.bincount() discussed in
    # https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-vector
    # in comment from @wehrdo:
    # np.bincount(arr) returns an array as large as the largest element in arr,
    # so a small array with a large range would create an excessively large array
    #
    # Conclusion => Use np.unique() with return_counts = True instead of np.bincount()
    (unique_values, counts) = np.unique(cumulated_spikes_coords, return_counts=True)  # 35 ms
    # binc = np.bincount(cumul_spikes) # 50 ms

    # Keep only the coordinates that appear in the group-cumulated set at least "n_co_spikes" times.
    filtered_1d_coords = unique_values[counts >= lut.n_co_spikes]  # 1 ms
    # filtered_1d_coords = np.where(binc >= 2)[0]  # 20 ms

    return filtered_1d_coords


def create_good_spikes(good_1d_coords, spikes):
    """ Same as "create_good_spikes_w" but do not write fits files to disk.

    Build a mask out of the array of 1D coordinates of selected pixels.
    The mask is used to retrieve the associated pixel intensities.
    Do not write the filtered data as fits files. This is for testing and benchmarking only.

    :param good_1d_coords: array of selected 1D coordinates.
    :param spikes: List of spikes data [1D coordinates, intensity before despiking, intensity after despiking]
    :return: None
    """
    for i in range(len(spikes)):
        good_spikes_mask = np.isin(spikes[i][0, :], good_1d_coords)
        _ = fits.PrimaryHDU(np.stack((spikes[i][0, good_spikes_mask], spikes[i][1, good_spikes_mask], spikes[i][2, good_spikes_mask])))


def create_good_spikes_w(good_1d_coords, spikes, paths):
    """  Same as "create_good_spikes" + write fits files to disk.

    Build a mask out of the array of 1D coordinates of selected pixels.
    The mask is used to retrieve the associated pixel intensities.
    Write the filtered data as fits files.

    :param good_1d_coords: array of selected 1D coordinates.
    :param spikes: List of spikes data [1D coordinates, intensity before despiking, intensity after despiking]
    :param paths: Array of original file paths used for creating the new file paths of the filtered fits files.
    :return: None
    """
    for i in range(len(spikes)):
        good_spikes_mask = np.isin(spikes[i][0, :], good_1d_coords)
        hdu = fits.PrimaryHDU(np.stack((spikes[i][0, good_spikes_mask], spikes[i][1, good_spikes_mask], spikes[i][2, good_spikes_mask])))
        # Name of filtered fits files
        new_name = os.path.join(lut.output_dir, os.path.basename(paths[i]).replace('spikes', 'filtered%d'%lut.n_co_spikes))
        # TODO: Consider adding the relative path of these new files into the database
        hdu.writeto(new_name, overwrite=True)


def process_spikes_io(group_index):
    """
    Spikes processing for one group and write the filtered data back to disk.

    :param group_index: group number associated with a series of up to 12 spikes files.
    :return: group_index. Not really necessary but useful for debugging parallel processing
    """
    fpaths = lut.get_filepaths(group_index)
    spikes = [fitsio.read(path) for path in fpaths]
    good_1d_coords = accumulate_spikes(spikes)
    create_good_spikes_w(good_1d_coords, spikes, fpaths)
    return group_index


def process_spikes_noWrite(group_index):
    """
    Spikes processing for one group but do not write the filtered data back to disk.

    :param group_index: group number associated with a series of up to 12 spikes files.
    :return: group_index. Not really necessary but useful for debugging parallel processing
    """
    fpaths = lut.get_filepaths(group_index)
    spikes = [fitsio.read(path) for path in fpaths]
    good_1d_coords = accumulate_spikes(spikes)
    create_good_spikes(good_1d_coords, spikes)
    return group_index


def process_map_IO(groups, spikes_lookup):
    """
    Main processing / single cpu, looping the spikes processing over the group.
    Write fits files to disk.

    :param groups: list of group numbers to process
    :param spikes_lookup: lookup table containing the pixels coordinates and all nearest neighbours
    :return: None
    """
    global lut
    lut = spikes_lookup
    _ = list(map(process_spikes_io, groups))
    return None


def process_map_noWrite(groups, spikes_lookup):
    """
    Same as process_map_IO() without writing fits files.

    :param groups: list of group numbers to process
    :param spikes_lookup: lookup table containing the pixels coordinates and all nearest neighbours
    :return: None
    """
    global lut
    lut = spikes_lookup

    _ = list(map(process_spikes_noWrite, groups))
    return None


def multiprocess_IO(groups, spikes_lookup, nworkers=4):
    """
    Main processing run in parallel, looping the spikes processing over groups distributed across workers.

    :param groups: list of group numbers to process
    :param spikes_lookup: lookup table containing the pixels coordinates and all nearest neighbours
    :param nworkers: number of parallel cores involved in this processing.
    :return:  List of groups executed by the workers. They do not need to be in the same order as input.
    """
    global lut
    lut = spikes_lookup
    pool = mp.Pool(processes = nworkers)
    results = [pool.apply_async(process_spikes_io, args=(group,)) for group in groups]
    results = [p.get() for p in results]
    #results = pool.map(process_spikes_io, groups)
    return results
