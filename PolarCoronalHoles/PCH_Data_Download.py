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


def euvi_pch_data_download(rootpath='', start_date='2007/05/01', end_date='2019/01/01'):
    # Crawl through and scrape the EUVI wavelet images

    url_head = 'http://sd-www.jhuapl.edu/secchi/wavelets/fits/'
    start_date = parse_time(start_date)
    end_date = parse_time(end_date)

    resp = request.urlopen(url_head)
    soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'))
    subs = [link.text for link in soup.find_all('a', href=True) if link.text.endswith('/')]        
    
    url_subdir1 = [parse.urljoin(url_head, sub_dir) for sub_dir in subs]

    if not rootpath:
        save_dir = os.path.abspath(os.path.curdir)
    else:
        save_dir = os.path.abspath(rootpath)

    #  crawling until a full list has been generated
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
                subs = []
                resp = request.urlopen(subdir3)
                soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'))
                subs = [link.text for link in soup.find_all('a', href=True) if link.text.endswith('.fts.gz')]

                if len(subs) > 1:
                    image_url = [parse.urljoin(subdir3, sub_dir) for sub_dir in subs]
                    save_path = [os.path.join(save_dir, subdir3.split('fits/')[1], path) for path in subs]

                    image_times = [datetime_from_euvi_filename(filepath) for filepath in save_path]

                    # grab every 4 hours
                    dt = list(np.logical_not([np.mod((time - image_times[0]).seconds, 14400) for time in image_times]))

                    if np.sum(dt) < 6:
                        dt2 = list(np.logical_not([np.mod((time - image_times[1]).seconds, 14400) for time in image_times]))
                        if len(dt2) > len(dt):
                            dt = dt2
                    
                    st = [tt >= start_date for tt in image_times]
                    et = [tt <= end_date for tt in image_times]
                    goodness = [(aa and bb and cc) for aa, bb, cc in zip(dt,st,et)]

                    if np.sum(goodness):
                        os.makedirs(os.path.join(save_dir, subdir3.split('fits/')[1]), exist_ok=True)
                        
                    # download each image
                    for good_image, image_loc, image_destination in zip(goodness, image_url, save_path):
                        if good_image:
                            request.urlretrieve(image_loc, image_destination)

                    print('Downloaded EUVI ' + wavelength_from_euvi_filename(save_path[0]) + 'images for ' +
                          image_times[0].strftime('%Y-%m-%d'))
                else:
                    print('Too few images detected in: ', subdir3)


def datetime_from_euvi_filename(filepath):
    basename = os.path.basename(filepath)[0:15]
    file_time_str = basename[:4] + '-' + basename[4:6] + '-' + basename[6:8] + 'T' + basename[9:11] + ':' + \
                    basename[11:13] + ':' + basename[13:15]
    file_datetime = parse_time(file_time_str)
    return file_datetime

def wavelength_from_euvi_filename(filepath):
    return os.path.basename(filepath)[16:19]



