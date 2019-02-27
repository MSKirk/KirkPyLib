import astropy.units as u
from sunpy.net import Fido, jsoc, attrs as a
from sunpy.time import parse_time
import datetime
from dateutil.rrule import rrule, MONTHLY
import os
import numpy as np
import fnmatch


def aia_pch_data_download(rootpath='', waves=[171, 193, 211, 304]):

    blank_results = Fido.search(a.jsoc.Time('2030/1/1', '2030/1/2'), a.jsoc.Series('aia.lev1_euv_12s'))

    base = parse_time('2010/06/01 00:00:00')
    ending = parse_time('2019/01/01 00:00:00')

    date_list = [dt for dt in rrule(MONTHLY, dtstart=base, until=ending)]

    for wave in waves:
        for ii in range(0, len(date_list)-1):

            if not rootpath:
                save_dir = os.path.abspath(os.path.curdir)
            else:
                save_dir = os.path.abspath(rootpath)

            directories = np.str(wave)+date_list[ii].strftime("/%Y/%m")
            save_path = os.path.join(save_dir, directories)
            os.makedirs(save_path, exist_ok=True)

            results = blank_results

            while results.file_num == 0:
                results = Fido.search(a.jsoc.Time(date_list[ii].strftime('%Y/%m/%d'), date_list[ii+1].strftime('%Y/%m/%d')),
                                      a.jsoc.Notify('michael.s.kirk@nasa.gov'), a.jsoc.Series('aia.lev1_euv_12s'),
                                      a.jsoc.Wavelength(wave * u.angstrom), a.Sample(3 * u.hour))

            expected_number = 9.
            neededfileindex = []

            if (results.file_num/expected_number) > 0.50:

                downloaded_file = Fido.fetch(results, path=save_path+'/{file}')

                if not downloaded_file:
                    print('Error in ' + date_list[ii].strftime('%Y/%m/%d') + ' ' + np.str(wave))
                    neededfileindex += [date_list[ii]]

            if results.file_num == len(fnmatch.filter(os.listdir(save_path), '*.fits')):
                print('Download between '+date_list[ii].strftime('%Y/%m/%d')+' and ' + date_list[ii+1].strftime('%Y/%m/%d') + ' successful')
            else:
                print('Expected '+np.str(results.file_num)+' files.')
                print('Got '+np.str(len(fnmatch.filter(os.listdir(save_path), '*.fits')))+' files.')

                with open(save_dir+'/'+np.str(wave)+'NeededFileIndex_'+date_list[ii].strftime('%Y')+'.txt', 'w') as f:
                    for item in neededfileindex:
                        f.write("%s\n" % item)

