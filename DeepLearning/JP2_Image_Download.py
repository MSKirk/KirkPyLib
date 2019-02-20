from sunpy.net import hek, helioviewer
from sunpy.time import parse_time
from sunpy.coordinates import frames
from sunpy.map import Map
import sunpy.io.jp2

import astropy.units as u
from astropy.coordinates import SkyCoord

import numpy as np

import datetime
import os


class Jp2ImageDownload:

    def __init__(self, save_dir='', full_image_set=False, tstart='2012/05/30 23:59:59', tend='2012/05/31 23:59:59'):

        if full_image_set:
            tstart = '2010/05/31 23:59:59'
            tend = '2018/12/31 23:59:59'

        if save_dir:
            self.save_dir = os.path.abspath(save_dir)
        else:
            self.save_dir = os.path.abspath(os.curdir)

        self.date_format = '%Y/%m/%d %H:%M:%S'

        self.tstart = datetime.datetime.strptime(tstart, self.date_format)
        self.tend = datetime.datetime.strptime(tend, self.date_format)

        self.date_list = self.gen_date_list()
        self.date_string_list = [tt.strftime(self.date_format) for tt in self.date_list]

    def download_images(self):

        for ii, download_date in enumerate(self.date_list):
            directories = download_date.strftime("%Y/%m/%d")
            save_path = os.path.join(self.save_dir, directories)
            os.makedirs(save_path, exist_ok=True)

            self.get_feature_images(self.date_string_list[ii + 1], self.date_string_list[ii], save_path)

        return True

    def get_all_sdo_images(self, time_in, save_path=''):
        # Get a complete set of the SDO images in AIA and HMI for a given time

        hv = helioviewer.HelioviewerClient()

        wavelnths = ['1600', '1700', '094', '131', '171', '193', '211', '304', '335']
        measurement = ['continuum', 'magnetogram']

        for wav in wavelnths:
            aia_filepath = hv.download_jp2(time_in, observatory='SDO', instrument='AIA', detector='AIA', measurement=wav,
                                       directory=save_path, overwrite=True)

        for measure in measurement:
            hmi_filepath = hv.download_jp2(time_in, observatory='SDO', instrument='HMI', detector='HMI',
                                       measurement=measure, directory=save_path, overwrite=True)

        return aia_filepath

    def get_feature_images(self, time_start, time_end, save_dir):

        client = hek.HEKClient()
        result = client.search(hek.attrs.Time(time_start, time_end), hek.attrs.FRM.Name == 'SPoCA')  # CH and AR
        result += client.search(hek.attrs.Time(time_start, time_end), hek.attrs.FRM.Name == 'EGSO_SFC')  # SS

        times = list(set([elem["event_starttime"] for elem in result]))
        times.sort()

        ch = [elem for elem in result if elem['event_type'] == 'CH']
        ar = [elem for elem in result if elem['event_type'] == 'AR']
        ss = [elem for elem in result if elem['event_type'] == 'SS']

        for time_in in times:
            image_file = self.get_all_sdo_images(time_in, save_path=save_dir)
            ch_mask = self.gen_feature_mask(time_in, [elem for elem in ch if elem['event_starttime'] == time_in], image_file)
            ar_mask = self.gen_feature_mask(time_in, [elem for elem in ar if elem['event_starttime'] == time_in], image_file)
            ss_mask = self.gen_feature_mask(time_in, [elem for elem in ss if elem['event_starttime'] == time_in], image_file)


    def gen_date_list(self):

        time_between = self.tend - self.tstart
        return [self.tend - datetime.timedelta(days=dd) for dd in range(0, time_between.days + 1)]

    def gen_feature_mask(self, feature_time, feature_list, image_filepath):

        aia_map = Map(image_filepath)

        mask = np.zeros([4096,4096])

        for feature in feature_list:
            p1 = feature["hpc_boundcc"][9:-2]
            p2 = p1.split(',')
            p3 = [v.split(" ") for v in p2]
            feature_date = parse_time(feature_time)

            feature_boundary = SkyCoord([(float(v[0]), float(v[1])) * u.arcsec for v in p3], obstime=feature_date,
                                        frame=frames.Helioprojective)

            pixel_contour = feature_boundary.to_pixel(aia_map.wcs)




        return mask

