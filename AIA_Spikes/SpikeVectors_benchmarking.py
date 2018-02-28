from datetime import datetime
from AIA_Spikes import SpikeVectors as sv
import os
from astropy.io import fits
import numpy as np

def io_benchmark(directory):
    direc = os.path.abspath(directory)
    ffiles = [f for f in os.listdir(direc) if os.path.isfile(os.path.join(direc, f))]
    times = []

    for f in ffiles:
        if f.endswith('.fits'):
            startTime = datetime.now()
            sp_image = sv.spikes_to_image(direc+'/'+f)
            hdu = sv.image_to_spikes(sp_image)
            hdu.writeto(direc+'/'+f, overwrite=True)
            times = times + [datetime.now() - startTime]

    print('Over', len(ffiles)-1, 'files,')
    print('It takes an average of', np.mean(times).total_seconds(), ' seconds for file I/O.')


def filesize_benchmark():

    filtered_dir = '/Volumes/BigSolar/Filtered_Spikes/2010/05/13'
    unfiltered_dir = '/Volumes/BigSolar/AIA_Spikes/2010/05/13'

    ffiles = [f for f in os.listdir(filtered_dir) if os.path.isfile(os.path.join(filtered_dir, f))]
    ff_size = [os.path.getsize(filtered_dir + '/' + f) for f in ffiles]

    ufiles = [f for f in os.listdir(unfiltered_dir) if os.path.isfile(os.path.join(unfiltered_dir, f))]
    uf_size = [os.path.getsize(unfiltered_dir + '/' + f) for f in ufiles]

    redux_ratio = np.sum(ff_size[1:])/np.sum(uf_size[1:len(ff_size)])

    print('Over', len(ff_size)-1, 'files,')
    print('filtered files are', redux_ratio,'% the size of the original.')


def sort_spikes_benchmark():
    startTime = datetime.now()
    sv.Sort_Spikes('/Volumes/BigSolar/AIA_Spikes', end_group=24)
    timedif = datetime.now()-startTime

    print('Processing over', 24 % 8, 'groups:')
    print('Total processing time', timedif.total_seconds(), ' seconds;')
    print('Averaging', timedif.total_seconds()/((24 % 8) * 7), ' seconds per spike file.')