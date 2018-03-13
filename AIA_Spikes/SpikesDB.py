import pandas as pd
from astropy.time import Time
import os, pathlib
import numpy as np
import glob
import sys


def query(question):
    yes = {'yes', 'y', 'ye', ''}
    no = {'no', 'n'}

    choice = input(question).lower()
    if choice in yes:
        return True
    elif choice in no:
        return False
    else:
        sys.stdout.write("Please respond with 'yes' or 'no'")


class SpikesDB:

    def __init__(self, db_location):
        self.dir = os.path.abspath(db_location)
        self.n_files = sum([len(files) for r, d, files in os.walk(self.dir)])

        if query('Build database for '+str(self.n_files)+' files?'):
            self.file_list_gen()
            self.time_gen()
            self.wave_gen()
            self.db_gen()

        print('Spikes database file stored in: '+os.path.join(self.dir, 'Table_SpikesDB.h5'))

    def db_gen(self):

        # Time in JD | Time in YMD | Wavelength | file path | file size

        store = pd.HDFStore(self.dir+'/Table_SpikesDB.h5')
    
        store.put('Path', pd.DataFrame(data={'Path': self.fullfilelist}).astype('|S60'))
        store.put('MJDTime', pd.DataFrame(data={'MJDTime': self.time_object.mjd}))
        store.put('YMDTime', pd.DataFrame(data={'YMDTime': self.time_object.datetime}))
        store.put('Wavelength', pd.DataFrame(data={'Wavelength': self.wave}))
        store.put('Size', pd.DataFrame(data={'Size': self.filesize}))
        
        store.close()

    def file_list_gen(self):
        self.filesize = np.zeros(self.n_files, dtype='uint32')
        self.fullfilelist = ['' for xx in range(self.n_files)]

        # search for a file list and extract the file size
        for ii, filename in enumerate(glob.iglob(os.path.join(self.dir, '**/20*.*.fits'), recursive=True)):
            self.filesize[ii] = os.path.getsize(filename)
            self.fullfilelist[ii] = os.path.join(*pathlib.Path(filename).parts[-4:])

        # set up a filter for null results
        self.filesize = self.filesize[[jj for jj,kk in enumerate(self.fullfilelist) if kk != '']]
        self.fullfilelist = list(filter(None, self.fullfilelist))

    def time_gen(self):
        # This corresponds to the t_obs keyword value from the associated AIA image.
        datetime = [ii.split('/')[-1].split('Z')[0] for ii in self.fullfilelist]
        self.time_object = Time(datetime, format='isot', scale='utc')

    def wave_gen(self):
        self.wave = np.asarray([ii.split(':')[-1].split('_')[1].split('.')[0] for ii in self.fullfilelist], dtype='uint16')

