import astropy.units as u
from sunpy.net import Fido, attrs as a
import os
import numpy as np
import random
import fnmatch


def aia_pch_data_download(rootpath='', waves=[171,193,304]):

    blank_results = Fido.search(a.Time('2030/1/1', '2030/1/2'), a.Instrument('aia'))

    for wave in waves:

        if not rootpath:
            path = os.path.abspath(os.path.curdir)
        else:
            path = os.path.abspath(rootpath)+'/'+np.str(wave)

        results = blank_results

        while results.file_num == 0:
            results = Fido.search(a.Time('2016/1/1', '2018/6/30'), a.Instrument('aia'), a.Wavelength(wave * u.angstrom), a.vso.Sample(3 * u.hour))

        expected_number = 7210.
        neededfileindex = []

        if (results.file_num/expected_number) > 0.98:
            file_order = random.sample(range(results.file_num), results.file_num)

            for ii, file_number in enumerate(file_order):
                downloaded_file = Fido.fetch(results[0, file_number], path=path+'/{file}')
                print(int((ii/results.file_num)*10000.)/100.)

                if not downloaded_file:
                    print('Error in file number '+np.str(file_number))
                    neededfileindex += [file_number]

        if results.file_num == len(fnmatch.filter(os.listdir(path), '*.fits')):
            print('Download between 2016/1/1 and 2018/6/30 successful')
        else:
            print('Expected '+np.str(results.file_num)+' files.')
            print('Got '+np.str(len(fnmatch.filter(os.listdir(path), '*.fits')))+' files.')

            with open(path+'/'+np.str(wave)+'NeededFileIndex_16-18.txt', 'w') as f:
                for item in neededfileindex:
                    f.write("%s\n" % item)

        # ----------------------------------------------------

        results = blank_results

        while results.file_num == 0:
            results = Fido.search(a.Time('2010/1/1', '2015/12/31'), a.Instrument('aia'), a.Wavelength(wave * u.angstrom), a.vso.Sample(3 * u.hour))

        expected_number = 16456.
        neededfileindex = []

        if (results.file_num/expected_number) > 0.98:
            file_order = random.sample(range(results.file_num), results.file_num)

            for ii, file_number in enumerate(file_order):
                downloaded_file = Fido.fetch(results[0, file_number], path=path+'/{file}')
                print(int((ii/results.file_num)*10000.)/100.)

                if not downloaded_file:
                    print('Error in file number ' + np.str(file_number))
                    neededfileindex += [file_number]

        if results.file_num == len(fnmatch.filter(os.listdir(path), '*.fits')):
            print('Download between 2010/1/1 and 2015/6/30 successful')
        else:
            print('Expected '+np.str(results.file_num)+' files.')
            print('Got '+np.str(len(fnmatch.filter(os.listdir(path), '*.fits')))+' files.')

            with open(path+'/'+np.str(wave)+'NeededFileIndex_10-15.txt', 'w') as f:
                for item in neededfileindex:
                    f.write("%s\n" % item)