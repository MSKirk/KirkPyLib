from sunpy.net import hek, helioviewer
import datetime
import os


class JP2ImageDownload:

    def __init__(self, save_dir='', full_image_set=False, tstart='2011/05/30 23:59:59', tend='2011/05/31 23:59:59'):

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

        self.download_images()

    def download_images(self):

        for ii, download_date in enumerate(self.date_list[:-1]):
            directories = download_date.strftime("%Y/%m/%d")
            save_path = os.path.join(self.save_dir, directories)
            os.makedirs(save_path, exist_ok=True)

            self.get_spoca_images(self.date_string_list[ii + 1], self.date_string_list[ii], save_path)
            self.get_sunspot_images(self.date_string_list[ii + 1], self.date_string_list[ii], save_path)

        print('Images downloaded into '+self.save_dir)

    def get_all_sdo_images(self, time_in, save_path=''):
        # Get a complete set of the SDO images in AIA and HMI for a given time

        hv = helioviewer.HelioviewerClient()

        wavelnths = ['094', '131', '171', '193', '211', '304', '335', '1600', '1700']
        measurement = ['continuum', 'magnetogram']

        for wav in wavelnths:
            filepath = hv.download_jp2(time_in, observatory='SDO', instrument='AIA', detector='AIA', measurement=wav,
                                       directory=save_path, overwrite=True)

        for measure in measurement:
            filepath = hv.download_jp2(time_in, observatory='SDO', instrument='HMI', detector='HMI',
                                       measurement=measure, directory=save_path, overwrite=True)

    def get_spoca_images(self, time_start, time_end, save_dir):

        client = hek.HEKClient()
        result = client.search(hek.attrs.Time(time_start, time_end), hek.attrs.FRM.Name == 'SPoCA')

        times = list(set([elem["event_starttime"] for elem in result]))

        for time_in in times:
            self.get_all_sdo_images(time_in, save_path=save_dir)

    def get_sunspot_images(self, time_start, time_end, save_dir):

        client = hek.HEKClient()
        result = client.search(hek.attrs.Time(time_start, time_end), hek.attrs.FRM.Name == 'EGSO_SFC')

        times = list(set([elem["event_starttime"] for elem in result]))

        for time_in in times:
            self.get_all_sdo_images(time_in, save_path=save_dir)

    def gen_date_list(self):

        time_between = self.tend - self.tstart
        return [self.tend - datetime.timedelta(days=dd) for dd in range(0, time_between.days + 1)]
