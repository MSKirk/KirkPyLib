

import os
import fnmatch
import astropy.units as u
from astropy.table import Table
import pandas as pd
from datetime import timedelta

class DataCenter:
    def __init__(self, directory):
        '''
        Reads and provides data sets for analysis:
            site_info : table of information for each site
            nmdb_set : Neutron Monitor Database
            error_set : Error rate of data center
            wx_set : weather info for each data center
            goes_set : P7 high energy particle flux information
        :param directory: the directory in which the google data is located
        '''

        self.dir=directory
        self.get_site_info()
        self.parse_nmdb()
        self.parse_errorrate()
        self.parse_wx()
        self.parse_goes()

    def get_site_info(self):

        # Charleston Merge: ID = 400
        # The Dalles Merge: ID = 401
        # Kotka (Finland) Merge: ID = 402

        ids = [194,185,184,165,78,77,76,75,73,67,66,65,64,63,61,60,59,58,57,56,55,54,52,47,44,43,42,32,28,26,400,401,402]
        lat = [53.3136,33.0644,35.8936,60.5393,37.3926,37.3926,37.3926,37.3926,22.9174,2.9174,37.4202,-23.4943,48.1434,50.1279,52.5382,53.4374,36.2427,50.4633,41.2197,33.0738,35.8996,33.775,33.7497,42.2437,45.6316,45.6324,24.8484,33.7471,53.3237,38.9484,33.0516,45.6320,60.5393] * u.deg
        lon = [-6.4477,-80.0413,-81.5457,27.1171,-122.082,-122.082,-122.082,-122.082,114.6618,101.6618,-122.0734,-46.8112,11.5563,8.601,13.2379,6.8626,-95.3212,3.8708,-95.8618,-80.0384,-81.5466,-84.4194,-84.5848,-83.7298,-121.1994,-121.2023,121.1824,-84.579,-6.34,-77.3261,-80.0399,-121.2009,27.1171] * u.deg
        maglat = [56.2858,42.8342,45.6182,57.5542,43.3428,43.3428,43.3428,43.3428,13.1173,-6.8993,43.3712,-14.5334,48.2347,50.6556,52.1837,54.1332,45.2162,51.7892,50.1256,42.8437,45.6242,43.3981,43.3653,51.8828,51.5543,51.5547,15.2356,43.363,56.2773,48.7753,42.8390,51.5545,57.5542] * u.deg
        maglon = [79.348,-8.7662,-10.6343,115.2884,-56.4318,-56.4318,-56.4318,-56.4318,-173.3442,174.0037,-56.4294,24.1336,95.013,92.9406,98.4647,92.7506,-26.5076,88.436,-27.8747,-8.7633,-10.6357,-13.8077,-13.9948,-13.6545,-57.8303,-57.8337,-167.261,-13.988,79.4643,-5.8425,-8.7648,-57.8320,115.2884] * u.deg
        name = ['Ireland','SouthCarolina','NorthCarolina','Finland','California','California','California','California','China','Malaysia','California','Brazil','Munich','Frankfurt','Berlin','Netherlands','Oklahoma','Belgium','Iowa','SouthCarolina','NorthCarolina','Georgia','Georgia','Michigan','Oregon','Oregon','Taiwan','Georgia','Ireland','Virginia','Charleston','TheDalles','Kotka']
        altitude = [81.38,13.72,331.93,12.50,24.38,24.38,24.38,24.38,107.29,49.38,3.96,781.81,520.29,104.24,35.97,2.44,192.02,28.65,297.79,8.84,331.93,285.29,258.47,254.81,30.48,30.48,285.29,235.92,44.20,109.12,11.28,30.48,12.5] * u.m

        self.site_info = Table([ids,lat,lon,maglat,maglon,name,altitude],names=('ID','Lat','Lon','MagLat','MagLon','Site','Altitude'), meta={'name':'site_info'})

    def parse_nmdb(self):
        self.nmdb_set = pd.DataFrame()
        self.nmdb_filelist = fnmatch.filter(os.listdir(self.dir), '20*nmdb*.txt')

        for file in self.nmdb_filelist:
            self.nmdb_set = pd.concat([self.nmdb_set, pd.read_csv(os.path.join(self.dir, file), index_col=0, delimiter=';', na_values='   null')])

        self.nmdb_set.index = pd.to_datetime(self.nmdb_set.index, utc=True)
        self.nmdb_set = self.nmdb_set.sort_index()

    def parse_errorrate(self):
        self.error_set = pd.DataFrame()

        self.google_filelist = fnmatch.filter(os.listdir(self.dir), 'Norm*.txt')

        for file in self.google_filelist:
            self.error_set = pd.concat([self.error_set, pd.read_csv(os.path.join(self.dir, file), index_col=1, delimiter='\s+')])

        # Time index is in microseconds.
        self.error_set.index = pd.to_datetime(self.error_set.index, unit='us', utc=True)
        self.error_set = self.error_set.sort_index()

    def parse_wx(self):
        self.wx_set = pd.DataFrame()

        self.wx_filelist = fnmatch.filter(os.listdir(self.dir), 'WX_*.txt')

        for file in self.wx_filelist:
            self.wx_set = pd.concat([self.wx_set, pd.read_csv(os.path.join(self.dir, file), index_col=0, delimiter='\s+',usecols=[0]+list(range(10,13))+[21,22], na_values='nan')])

        # set index to noon local time
        self.wx_set.index = pd.to_datetime(self.wx_set.index)+timedelta(hours=12)

        # adjust for time zone and convert to UTC
        self.wx_set.index = pd.to_datetime([(self.wx_set.index[ii].tz_localize(self.wx_set['GMTOffset'][ii])).tz_convert('UTC') for ii in range(self.wx_set.shape[0])], utc=True)

        self.wx_set = self.wx_set.sort_index()

    def parse_goes(self):
        self.goes_set = pd.DataFrame()

        self.goes_filelist = fnmatch.filter(os.listdir(self.dir), 'g15*.csv')

        for file in self.goes_filelist:
            self.goes_set = pd.concat([self.goes_set, pd.read_csv(os.path.join(self.dir, file), index_col=0, delimiter=',', comment='#',usecols=[0]+list(range(49,57)))])

        self.goes_set.index = pd.to_datetime(self.goes_set.index, utc=True)
