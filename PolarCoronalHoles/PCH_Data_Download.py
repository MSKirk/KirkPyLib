import astropy.units as u
from sunpy.net import Fido, attrs as a
import os
import numpy as np


def aia_pch_data_download(rootpath=''):

    if not rootpath:
        path = os.path.abspath(os.path.curdir)
    else:
        path = os.path.abspath(rootpath)

    blank_results = Fido.search(a.Time('2030/1/1', '2030/1/2'), a.Instrument('aia'))

    for waves in [171,193,304]:
        results = blank_results

        while results.file_num == 0:
            results = Fido.search(a.Time('2016/1/1', '2018/6/30'), a.Instrument('aia'), a.Wavelength(waves * u.angstrom), a.vso.Sample(3 * u.hour))

        downloaded_files = Fido.fetch(results, path=path+'/'+np.str(waves)+'/{file}')

        if results.file_num == len(downloaded_files):
            print('Download between 2016/1/1 and 2018/6/30 successful')
        else:
            print('Expected '+np.str(results.file_num)+' files.')
            print('Got '+np.str(len(downloaded_files))+' files.')

            with open(path+'/'+np.str(waves)+'filelist_16-18.txt', 'w') as f:
                for item in downloaded_files:
                    f.write("%s\n" % item)

        results = blank_results

        while results.file_num == 0:
            results = Fido.search(a.Time('2010/1/1', '2015/12/31'), a.Instrument('aia'), a.Wavelength(waves * u.angstrom), a.vso.Sample(3 * u.hour))

        downloaded_files = Fido.fetch(results, path=path+'/'+np.str(waves)+'/{file}')

        ## check downloaded files...
        if results.file_num == len(downloaded_files):
            print('Download between 2010/1/1 and 2015/6/30 successful')
        else:
            print('Expected '+np.str(results.file_num)+' files.')
            print('Got '+np.str(len(downloaded_files))+' files.')

            with open(path+'/'+np.str(waves)+'filelist_10-15.txt', 'w') as f:
                for item in downloaded_files:
                    f.write("%s\n" % item)
