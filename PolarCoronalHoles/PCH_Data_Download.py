import astropy.units as u
from sunpy.net import Fido, jsoc, attrs as a
from sunpy.time import parse_time
import datetime
from dateutil.rrule import rrule, MONTHLY
import os
import numpy as np
import fnmatch

from bs4 import BeautifulSoup 
from urllib import request, parse


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
                        
def euvi_pch_data_download(rootpath=''):
    # Crawl through and scrape the EUVI wavelet images

    url_head = 'http://sd-www.jhuapl.edu/secchi/wavelets/fits/'

    resp = request.urlopen(url_head)
    soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'))
    subs = [link.text for link in soup.find_all('a', href=True) if link.text.endswith('/')]        
    
    url_subdir1 = [parse.urljoin(url_head, sub_dir) for sub_dir in subs]
    
    # Continue recursively crawling until a full list has been generated
    for subdir1 in url_subdir1:
        resp = request.urlopen(subdir1)
        soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'))
        subs = [link.text for link in soup.find_all('a', href=True) if link.text.endswith('/')]
    
        url_subdir2 = [parse.urljoin(subdir1, sub_dir) for sub_dir in subs]

        for subdir2 in url_subdir2:
            resp = request.urlopen(subdir2)
            soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'))
            subs = [link.text for link in soup.find_all('a', href=True) if link.text.endswith('/')]
    
            url_subdir3 = [parse.urljoin(subdir2, sub_dir) for sub_dir in subs]

            for subdir3 in url_subdir3:
                resp = request.urlopen(subdir3)
                soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'))
                subs = [link.text for link in soup.find_all('a', href=True) if link.text.endswith('.fits.gz')]
    
                image_url = [parse.urljoin(subdir3, sub_dir) for sub_dir in subs]

                return image_url
                # create a path from removing the url_head
    			# download each image

