from datetime import datetime
from AIA_Spikes import SpikeVectors as sv
import os
import pandas as pd
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

    print('Over %0.0f files.' %(len(ffiles)-1))
    print('It takes an average of %0.1f seconds for file I/O.' %np.mean(times).total_seconds())


def filesize_benchmark():

    filtered_dir = '/Volumes/BigSolar/Filtered_Spikes/2010/05/13'
    unfiltered_dir = '/Volumes/BigSolar/AIA_Spikes/2010/05/13'

    ffiles = [f for f in os.listdir(filtered_dir) if os.path.isfile(os.path.join(filtered_dir, f))]
    ff_size = [os.path.getsize(filtered_dir + '/' + f) for f in ffiles]

    ufiles = [f for f in os.listdir(unfiltered_dir) if os.path.isfile(os.path.join(unfiltered_dir, f))]
    uf_size = [os.path.getsize(unfiltered_dir + '/' + f) for f in ufiles]

    redux_ratio = np.sum(ff_size[1:])/np.sum(uf_size[1:len(ff_size)])

    print('Over %0.0f files' %(len(ff_size)-1))
    print('filtered files are %0.3f times the size of the original.' %float(redux_ratio))


def sort_spikes_benchmark():
    startTime = datetime.now()
    sv.Sort_Spikes('/Volumes/BigSolar/AIA_Spikes', end_group=24)
    timedif = datetime.now()-startTime
    ngroups = (24/8)+1

    print('Processing over %0.0f groups:' %ngroups)
    print('Total processing time %0.1f seconds;' %timedif.total_seconds())
    print('Averaging %0.1f seconds per spike file (not counting initialization).' %((timedif.total_seconds()-61.3)/(ngroups * 7)))


def initialization_benchmark():
    startTime = datetime.now()
    spikes_db = pd.HDFStore('/Volumes/BigSolar/AIA_Spikes/Table_SpikesDB.h5')
    group_numbers = spikes_db.get('GroupNumber')
    paths = spikes_db.get('Path')
    spikes_db.close()

    group_numbers = None
    paths = None
    timedif = datetime.now() - startTime

    print('Total initialization time: %0.1f seconds;' %timedif.total_seconds())


