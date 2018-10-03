import astropy.units as u
from sunpy.net import Fido, jsoc, attrs as a
import os
import numpy as np
import random
import fnmatch
import time


def aia_pch_data_download(rootpath='', waves=[171,193,304]):

    client = jsoc.JSOCClient()
    blank_results = Fido.search(a.Time('2030/1/1', '2030/1/2'), a.Instrument('aia'))

    for wave in waves:

        if not rootpath:
            path = os.path.abspath(os.path.curdir)
        else:
            path = os.path.abspath(rootpath)+'/'+np.str(wave)

        results = blank_results

        while len(results.table) == 0:
            results = client.search(a.jsoc.Time('2016/1/1', '2018/6/30'), a.jsoc.Notify('michael.s.kirk@nasa.gov'),
                                  a.jsoc.Series('aia.lev1_euv_12s'), a.jsoc.Wavelength(wave * u.angstrom),
                                  a.Sample(3 * u.hour))

        expected_number = 7210.
        neededfileindex = []

#
#        if (len(results.table) / expected_number) > 0.98:
#            while requests.status != 0:
#                requests = client.request_data(results)
#                if requests.status != 0:
#                    time.sleep(60*5.)



        if (len(results.table)/expected_number) > 0.98:
            file_order = random.sample(range(len(results.table)), len(results.table))

            for ii, file_number in enumerate(file_order):
                downloaded_file = Fido.fetch(results[0, file_number], path=path+'/{file}')
                print(int((ii/len(results.table))*10000.)/100.)

                if not downloaded_file:
                    print('Error in file number '+np.str(file_number))
                    neededfileindex += [file_number]

        if len(results.table) == len(fnmatch.filter(os.listdir(path), '*.fits')):
            print('Download between 2016/1/1 and 2018/6/30 successful')
        else:
            print('Expected '+np.str(len(results.table))+' files.')
            print('Got '+np.str(len(fnmatch.filter(os.listdir(path), '*.fits')))+' files.')

            with open(path+'/'+np.str(wave)+'NeededFileIndex_16-18.txt', 'w') as f:
                for item in neededfileindex:
                    f.write("%s\n" % item)

        # ----------------------------------------------------

        results = blank_results

        while len(results.table) == 0:
            results = client.search(a.jsoc.Time('2010/1/1', '2015/12/31'), a.jsoc.Notify('michael.s.kirk@nasa.gov'),
                                  a.jsoc.Series('aia.lev1_euv_12s'), a.jsoc.Wavelength(wave * u.angstrom),
                                  a.Sample(3 * u.hour))

        expected_number = 16456.
        neededfileindex = []

        if (len(results.table)/expected_number) > 0.98:
            file_order = random.sample(range(len(results.table)), len(results.table))

            for ii, file_number in enumerate(file_order):
                downloaded_file = Fido.fetch(results[0, file_number], path=path+'/{file}')
                print(int((ii/len(results.table))*10000.)/100.)

                if not downloaded_file:
                    print('Error in file number ' + np.str(file_number))
                    neededfileindex += [file_number]

        if len(results.table) == len(fnmatch.filter(os.listdir(path), '*.fits')):
            print('Download between 2010/1/1 and 2015/6/30 successful')
        else:
            print('Expected '+np.str(len(results.table))+' files.')
            print('Got '+np.str(len(fnmatch.filter(os.listdir(path), '*.fits')))+' files.')

            with open(path+'/'+np.str(wave)+'NeededFileIndex_10-15.txt', 'w') as f:
                for item in neededfileindex:
                    f.write("%s\n" % item)