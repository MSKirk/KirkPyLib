
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import fnmatch
from astropy.time import Time, TimeDelta
from astropy.table import Table
import astropy.units as u
import pandas as pd
from gatspy import periodic
from scipy import signal
from sklearn.utils import resample
import scipy.signal as sg
from datetime import datetime
from multiprocessing import Pool

import PCH_Tools

class PCH:
    def __init__(self, directory, interval='11D'):

        self.directory = directory
        self.filelist = fnmatch.filter(os.listdir(self.directory), '*PCH_Detections*.csv')
        self.list_of_obj = []  # initialize a null list

        for file in self.filelist:
            self.list_of_obj += [self.pch_data_parse(os.path.join(self.directory, file))]

        self.pch_obj = self.combine_pch_obj(self.list_of_obj, interval=interval)
        
        self.northern = self.pch_obj.StartLat > 0
        self.southern = self.pch_obj.StartLat < 0
        
        self.confidence_obj(confidence=0.99, interval=interval)

        self.area_of_all_measures()

        plt.ion()

        #self.S_Cent_periodogram()

    def plot_everything(self):

        self.area_periodogram()
        self.centroid_periodogram()
        self.series_plot()
        self.S_Cent_periodogram()
        self.N_Cent_periodogram()

    def pch_data_parse(self, file_path):
        """
        :param file_path: full path of the file to readt
        :return: Parsed PCH data
        """

#        # Obsolete. Preserved for old IDL parsing.
#        if os.path.basename(file_path).split('.')[1] == 'txt':
#            pch_obj = np.loadtxt(file_path, skiprows=3, dtype={'names': ('Harvey_Rotation', 'North_Size', 'North_Spread',
#                                                                         'South_Size', 'South_Spread', 'N_Cent_Lon',
#                                                                         'N_Cent_CoLat', 'S_Cent_Lon', 'S_Cent_CoLat'),
#                                                               'formats': ('f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f')})
#
#           pch_obj = pd.DataFrame(data=pch_obj, index=parse_time(self.HarRot2JD(pch_obj['Harvey_Rotation'], duplicates=False)))
#
#            # Create data masks for non-measurements - less than 0 or unphysical.
#            pch_obj['North_Size'] = pch_obj['North_Size'].mask(pch_obj['North_Size'] < 0)
#            pch_obj['North_Spread'] = pch_obj['North_Spread'].mask(pch_obj['North_Size'] < 0)
#            pch_obj['N_Cent_Lon'] = pch_obj['N_Cent_Lon'].mask(pch_obj['North_Size'] < 0)
#            pch_obj['N_Cent_CoLat'] = pch_obj['N_Cent_CoLat'].mask(pch_obj['North_Size'] < 0)
#
#            pch_obj['South_Size'] = pch_obj['South_Size'].mask(pch_obj['South_Size'] < 0)
#            pch_obj['South_Spread'] = pch_obj['South_Spread'].mask(pch_obj['South_Size'] < 0)
#            pch_obj['S_Cent_Lon'] = pch_obj['S_Cent_Lon'].mask(pch_obj['South_Size'] < 0)
#            pch_obj['S_Cent_CoLat'] = pch_obj['S_Cent_CoLat'].mask(pch_obj['South_Size'] < 0)

            # NAN hunting - periodogram does not deal with nans so well.
#
#            pch_obj = pch_obj.dropna(subset=['North_Size'])
#
#            pch_obj.loc[pch_obj['North_Spread'] > pch_obj['North_Size'], 'North_Spread'] = \
#                pch_obj.loc[pch_obj['North_Spread'] > pch_obj['North_Size'], 'North_Size']
#
#            pch_obj.loc[pch_obj['South_Spread'] > pch_obj['South_Size'], 'South_Spread'] = \
#                pch_obj.loc[pch_obj['South_Spread'] > pch_obj['South_Size'], 'South_Size']
#
#            # Adding explicit Filters for periodogram
#            name = ''
#            name = name.join(file_path.split('/')[-1].split('_Area')[0].split('_'))
#            pch_obj['Filter'] = np.repeat(name[0]+name[-4]+name[-1], pch_obj['North_Size'].size)
#
#            pch_obj['Hole_Separation'] = self.haversine(pch_obj['N_Cent_Lon'], pch_obj['N_Cent_CoLat']-90.,
#                                                        pch_obj['S_Cent_Lon'],pch_obj['S_Cent_CoLat']-90.)

        if os.path.basename(file_path).split('.')[1] == 'csv':
            table = Table.read(file_path, format='ascii.ecsv')
            pch_obj = table.to_pandas()

            pch_obj = pch_obj.dropna(subset=['Area'])
            pch_obj['Filter'] = np.repeat(os.path.basename(file_path).split('_')[0], pch_obj['StartLat'].size)

            datetime = [Time(ii).to_datetime() for ii in pch_obj['Date']]
            pch_obj['DateTime'] = datetime
            pch_obj = pch_obj.set_index('DateTime')

        else:
            print('File type not yet supported.')
            pch_obj = []

        return pch_obj

    def centroid_separation(self, data_key):

        # convert form co-lat to lat (north & south are inverted... but it results in the same answer)
        return haversine(self.data_sets[data_key]['N_Cent_Lon'], self.data_sets[data_key]['N_Cent_CoLat']-90.,
                              self.data_sets[data_key]['S_Cent_Lon'],self.data_sets[data_key]['S_Cent_CoLat']-90.)

    def peek(self):

        sns.set(style="darkgrid")

        north = self.pch_obj.Area[self.northern]
        south = self.pch_obj.Area[self.southern]

        plt.figure()
        plt.title(self.pch_obj.Filter[0], loc='left')

        plt.subplot(2,1,1)
        sns.lineplot(x=north.index.round('11D'), y=north, ci='sd', n_boot=1000, estimator='median')
        plt.title('Northern Polar Coronal Hole')
        plt.ylabel('Fractional Area')
        plt.xlabel('')
        plt.ylim(0, 0.08)

        plt.subplot(2, 1, 2)
        sns.lineplot(x=south.index.round('11D'), y=south, ci='sd', n_boot=1000, estimator='median')
        plt.title('Southern Polar Coronal Hole')
        plt.ylabel('Fractional Area')
        plt.xlabel('')
        plt.ylim(0, 0.08)

        plt.show()

    def area_periodogram(self):

        set_time, north_pch, north_dpch, north_filts, south_pch, south_dpch, south_filts = self._assemble_pch_set()

        jd_time = self.HarRot2JD(set_time)

        # Complete the LS for each data set, nyquist freq is 66 days
        periods = np.linspace(1, 900, 9000)
        model = periodic.NaiveMultiband(BaseModel=periodic.LombScargle)
        model.optimizer.period_range = (120, 900)
        # errors must be non zero and defined
        model.fit(jd_time.mjd, (north_pch), (north_dpch)+1e-6, north_filts)
        P = model.scores(periods)

        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(5, 2, left=0.07, right=0.95, bottom=0.15,
                          wspace=0.1, hspace=0.6)
        ax = [fig.add_subplot(gs[:, 0]),
              fig.add_subplot(gs[:-2, 1]),
              fig.add_subplot(gs[-2:, 1])]

        for sets in self.data_names:
            mask = (north_filts == self.filter_keys[sets])
            ax[0].errorbar((jd_time.mjd[mask]-jd_time.mjd[0])/(jd_time.mjd[-1]-jd_time.mjd[0]), north_pch[mask],
                           north_dpch[mask], fmt='.', label=sets)
        labels = np.around(jd_time.decimalyear[np.int32(np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9999])*jd_time.size)],
                           decimals=2)
        ax[0].set_ylim(0,0.09)
        ax[0].legend(loc='upper left', fontsize=12, ncol=4)
        ax[0].set_title('Northern Polar Coronal Hole Area',  fontsize=14)
        ax[0].set_xticklabels(labels)
        ax[0].set_xlabel('Year')
        ax[0].set_ylabel('Fractional Solar Surface Area')

        for ii, sets in enumerate(self.data_names):
            offset = 12. - ii
            ax[1].plot(periods, P[self.filter_keys[sets]] + offset, lw=1)
            ax[1].text(95, 0.8 + offset, sets, fontsize=10, ha='right', va='top')
        ax[1].set_title('Standard Periodogram in Each Measurement', fontsize=14)
        ax[1].yaxis.set_major_formatter(plt.NullFormatter())
        ax[1].xaxis.set_major_formatter(plt.NullFormatter())
        ax[1].set_ylabel('power + offset')

        LS_multi = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=1)
        LS_multi.optimizer.period_range = (120, 900)
        LS_multi.fit(jd_time.mjd, (north_pch), (north_dpch)+1e-6, north_filts)
        P_multi = LS_multi.periodogram(periods)
        ax[2].plot(periods, P_multi, lw=1, color='gray')

        ax[2].set_title('Multiband Periodogram', fontsize=12)
        ax[2].set_yticks([0, 0.2, 0.4])
        ax[2].set_ylim(0, 0.5)
        ax[2].yaxis.set_major_formatter(plt.NullFormatter())
        ax[2].set_xlabel('Period (days)')
        ax[2].set_ylabel('power')

        plt.savefig('/Users/mskirk/Desktop/PCH_North_periodogram.png')

        # SOUTH PCH

        periods = np.linspace(1, 900, 9000)
        model = periodic.NaiveMultiband(BaseModel=periodic.LombScargle)
        model.optimizer.period_range = (120, 900)
        # errors must be non zero and defined
        model.fit(jd_time.mjd, (south_pch), (south_dpch)+1e-6, south_filts)
        P = model.scores(periods)

        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(5, 2, left=0.07, right=0.95, bottom=0.15,
                          wspace=0.1, hspace=0.6)
        ax = [fig.add_subplot(gs[:, 0]),
              fig.add_subplot(gs[:-2, 1]),
              fig.add_subplot(gs[-2:, 1])]

        for sets in self.data_names:
            mask = (south_filts == self.filter_keys[sets])
            ax[0].errorbar((jd_time.mjd[mask]-jd_time.mjd[0])/(jd_time.mjd[-1]-jd_time.mjd[0]), south_pch[mask],
                           south_dpch[mask], fmt='.', label=sets)
        labels = np.around(jd_time.decimalyear[np.int32(np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9999])*jd_time.size)],
                           decimals=2)
        ax[0].set_ylim(0,0.09)
        ax[0].legend(loc='upper left', fontsize=12, ncol=4)
        ax[0].set_title('Southern Polar Coronal Hole Area',  fontsize=14)
        ax[0].set_xticklabels(labels)
        ax[0].set_xlabel('Year')
        ax[0].set_ylabel('Fractional Solar Surface Area')

        for ii, sets in enumerate(self.data_names):
            offset = 12. - ii
            ax[1].plot(periods, P[self.filter_keys[sets]] + offset, lw=1)
            ax[1].text(95, 0.8 + offset, sets, fontsize=10, ha='right', va='top')
        ax[1].set_title('Standard Periodogram in Each Measurement', fontsize=14)
        ax[1].yaxis.set_major_formatter(plt.NullFormatter())
        ax[1].xaxis.set_major_formatter(plt.NullFormatter())
        ax[1].set_ylabel('power + offset')

        LS_multi = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=1)
        LS_multi.optimizer.period_range = (120, 900)
        LS_multi.fit(jd_time.mjd, (south_pch), (south_dpch)+1e-6, south_filts)
        P_multi = LS_multi.periodogram(periods)
        ax[2].plot(periods, P_multi, lw=1, color='gray')

        ax[2].set_title('Multiband Periodogram', fontsize=12)
        ax[2].set_yticks([0, 0.2, 0.4])
        ax[2].set_ylim(0, 0.5)
        ax[2].yaxis.set_major_formatter(plt.NullFormatter())
        ax[2].set_xlabel('Period (days)')
        ax[2].set_ylabel('power')

        plt.savefig('/Users/mskirk/Desktop/PCH_South_periodogram.png')

    def series_plot(self):

        self.regrid_time()

        self.north_pch_series = np.empty((len(self.data_names),self.time_grid.size))
        self.south_pch_series = np.empty_like(self.north_pch_series)
        self.hole_separation_series = np.empty_like(self.north_pch_series)

        fig = plt.figure(figsize=(20, 10), tight_layout=True)

        for ii, data_key in enumerate(self.data_names):
            self.north_pch_series[ii,:], self.south_pch_series[ii,:] = self.regrid_series(data_key)
            self.hole_separation_series[ii,:] = self.regrid_series(data_key, centroid=True)

        plt.subplot(2, 1, 1)
        ax1 = sns.tsplot(data=self.north_pch_series,time=self.time_grid.decimalyear, estimator=np.nanmedian)

        plt.subplot(2, 1, 2)
        ax2 = sns.tsplot(data=self.south_pch_series,time=self.time_grid.decimalyear, estimator=np.nanmedian,
                         color='darkorchid')

        ax1.set_title('Polar Coronal Hole Size', fontsize=14, fontweight='bold')
        ax1.set_ylim(0,0.07)
        ax1.set_ylabel('Fractional Surface Area', fontsize=13)
        ax1.text(2010.1, 0.061, 'Northern Polar Hole', fontsize=13, ha='right', va='top', fontweight='bold')

        ax2.set_xlabel('Year', fontsize=13)
        ax2.set_ylim(0, 0.07)
        ax2.set_ylabel('Fractional Surface Area', fontsize=13)
        ax2.text(2010.1, 0.061, 'Southern Polar Hole', fontsize=13, ha='right', va='top', fontweight='bold')

        plt.savefig('/Users/mskirk/Desktop/PCH_Series.png')

        fig = plt.figure(figsize=(20, 10), tight_layout=True)
        ax3 = sns.tsplot(data=self.hole_separation_series, time=self.time_grid.decimalyear, estimator=np.nanmedian,
                         color='firebrick')
        trend = pd.Series(np.nanmedian(self.hole_separation_series, axis=0))
        trend_interp = trend.interpolate(method='linear')
        trend_line = signal.savgol_filter(trend_interp, 301, 5)
        plt.plot(self.time_grid.decimalyear, trend_line, color='k', lw=4)

        ax3.set_title('Polar Coronal Separation Angle', fontsize=14, fontweight='bold')
        ax3.set_ylim(155,180)
        ax3.set_ylabel('Separation Angle [degrees]', fontsize=13)
        ax3.text(2012.6, 166, '180° represents symmetrical holes', fontsize=13, ha='right', va='top', fontweight='bold')

        plt.savefig('/Users/mskirk/Desktop/PCH_Separation_Series.png')

    def centroid_periodogram(self):
        set_time, north_colat, north_lon, north_filts, south_colat, south_lon, south_filts, hole_separation = \
            self._assemble_pch_set(cent_series=True)

        jd_time = self.HarRot2JD(set_time)

        # Complete the LS for each data set, nyquist freq is 66 days
        periods = np.linspace(1, 900, 9000)
        model = periodic.NaiveMultiband(BaseModel=periodic.LombScargle)
        model.optimizer.period_range = (33, 900)
        # errors must be non zero and defined
        model.fit(jd_time.mjd, (hole_separation), (np.sqrt(180.-hole_separation))+1e-6,
                  north_filts)
        P = model.scores(periods)

        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(5, 2, left=0.07, right=0.95, bottom=0.15,
                          wspace=0.1, hspace=0.6)
        ax = [fig.add_subplot(gs[:, 0]),
              fig.add_subplot(gs[:-2, 1]),
              fig.add_subplot(gs[-2:, 1])]

        for sets in self.data_names:
            mask = (north_filts == self.filter_keys[sets])
            ax[0].plot((jd_time.mjd[mask]-jd_time.mjd[0])/(jd_time.mjd[-1]-jd_time.mjd[0]), hole_separation[mask],'.',
                       label=sets, lw=1)
        labels = np.around(jd_time.decimalyear[np.int32(np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9999])*jd_time.size)],
                           decimals=2)
        ax[0].set_ylim(150,190)
        ax[0].legend(loc='upper left', fontsize=12, ncol=4)
        ax[0].set_title('Angular Separation Between Polar Holes',  fontsize=14)
        ax[0].set_xticklabels(labels)
        ax[0].set_xlabel('Year')
        ax[0].set_ylabel('Separation [Degrees]')

        for ii, sets in enumerate(self.data_names):
            offset = 12. - ii
            ax[1].plot(periods, P[self.filter_keys[sets]] + offset, lw=1)
            ax[1].text(95, 0.8 + offset, sets, fontsize=10, ha='right', va='top')
        ax[1].set_title('Standard Periodogram in Each Measurement', fontsize=14)
        ax[1].yaxis.set_major_formatter(plt.NullFormatter())
        ax[1].xaxis.set_major_formatter(plt.NullFormatter())
        ax[1].set_ylabel('power + offset')

        LS_multi = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=1)
        LS_multi.optimizer.period_range = (33, 900)
        LS_multi.fit(jd_time.mjd, (hole_separation), (np.sqrt(180.-hole_separation))+1e-6,
                     north_filts)
        P_multi = LS_multi.periodogram(periods)
        ax[2].plot(periods, P_multi, lw=1, color='gray')

        ax[2].set_title('Multiband Periodogram', fontsize=12)
        ax[2].set_yticks([0, 0.2, 0.4])
        ax[2].set_ylim(0, 1)
        ax[2].yaxis.set_major_formatter(plt.NullFormatter())
        ax[2].set_xlabel('Period (days)')
        ax[2].set_ylabel('power')

        plt.savefig('/Users/mskirk/Desktop/PCH_Separation_periodogram.png')

    def N_Cent_periodogram(self):
        set_time, north_colat, north_lon, north_filts, south_colat, south_lon, south_filts, hole_separation = \
            self._assemble_pch_set(cent_series=True)

        jd_time = self.HarRot2JD(set_time)

        # Complete the LS for each data set, nyquist freq is 66 days
        periods = np.linspace(1, 900, 9000)
        model = periodic.NaiveMultiband(BaseModel=periodic.LombScargle)
        model.optimizer.period_range = (33, 900)
        # errors must be non zero and defined
        model.fit(jd_time.mjd, (north_colat), (np.sqrt(north_colat)) + 1e-6,
                  north_filts)
        P = model.scores(periods)

        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(5, 2, left=0.07, right=0.95, bottom=0.15,
                          wspace=0.1, hspace=0.6)
        ax = [fig.add_subplot(gs[:, 0]),
              fig.add_subplot(gs[:-2, 1]),
              fig.add_subplot(gs[-2:, 1])]

        for sets in self.data_names:
            mask = (north_filts == self.filter_keys[sets])
            ax[0].plot((jd_time.mjd[mask] - jd_time.mjd[0]) / (jd_time.mjd[-1] - jd_time.mjd[0]), north_colat[mask], '.',
                       label=sets, lw=1)
        labels = np.around(jd_time.decimalyear[np.int32(np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9999]) * jd_time.size)],
                           decimals=2)
        # ax[0].set_ylim(0, 40)
        ax[0].legend(loc='upper left', fontsize=12, ncol=4)
        ax[0].set_title('Northern Co-Latitude', fontsize=14)
        ax[0].set_xticklabels(labels)
        ax[0].set_xlabel('Year')
        ax[0].set_ylabel('Location [Degrees]')

        for ii, sets in enumerate(self.data_names):
            offset = 12. - ii
            ax[1].plot(periods, P[self.filter_keys[sets]] + offset, lw=1)
            ax[1].text(95, 0.8 + offset, sets, fontsize=10, ha='right', va='top')
        ax[1].set_title('Standard Periodogram in Each Measurement', fontsize=14)
        ax[1].yaxis.set_major_formatter(plt.NullFormatter())
        ax[1].xaxis.set_major_formatter(plt.NullFormatter())
        ax[1].set_ylabel('power + offset')

        LS_multi = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=1)
        LS_multi.optimizer.period_range = (33, 900)
        LS_multi.fit(jd_time.mjd, (north_colat), (np.sqrt(north_colat)) + 1e-6,
                     north_filts)
        P_multi = LS_multi.periodogram(periods)
        ax[2].plot(periods, P_multi, lw=1, color='gray')

        ax[2].set_title('Multiband Periodogram', fontsize=12)
        ax[2].set_yticks([0, 0.2, 0.4])
        ax[2].set_ylim(0, 0.5)
        ax[2].yaxis.set_major_formatter(plt.NullFormatter())
        ax[2].set_xlabel('Period (days)')
        ax[2].set_ylabel('power')

        plt.savefig('/Users/mskirk/Desktop/N_CoLat_periodogram.png')

        # Lon

        # Complete the LS for each data set, nyquist freq is 66 days
        periods = np.linspace(1, 900, 9000)
        model = periodic.NaiveMultiband(BaseModel=periodic.LombScargle)
        model.optimizer.period_range = (33, 900)
        # errors must be non zero and defined
        model.fit(jd_time.mjd, (np.sin(np.deg2rad(north_lon))), (np.sqrt(abs(np.sin(np.deg2rad(north_lon))))) + 1e-6,
                  north_filts)
        P = model.scores(periods)

        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(5, 2, left=0.07, right=0.95, bottom=0.15,
                          wspace=0.1, hspace=0.6)
        ax = [fig.add_subplot(gs[:, 0]),
              fig.add_subplot(gs[:-2, 1]),
              fig.add_subplot(gs[-2:, 1])]

        for sets in self.data_names:
            mask = (north_filts == self.filter_keys[sets])
            ax[0].plot((jd_time.mjd[mask] - jd_time.mjd[0]) / (jd_time.mjd[-1] - jd_time.mjd[0]), np.sin(np.deg2rad(north_lon))[mask],
                       '.', label=sets, lw=1)
        labels = np.around(jd_time.decimalyear[np.int32(np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9999]) * jd_time.size)],
                           decimals=2)
        # ax[0].set_ylim(0, 40)
        ax[0].legend(loc='upper left', fontsize=12, ncol=4)
        ax[0].set_title('Northern Co-Latitude', fontsize=14)
        ax[0].set_xticklabels(labels)
        ax[0].set_xlabel('Year')
        ax[0].set_ylabel('Location [Degrees]')

        for ii, sets in enumerate(self.data_names):
            offset = 12. - ii
            ax[1].plot(periods, P[self.filter_keys[sets]] + offset, lw=1)
            ax[1].text(95, 0.8 + offset, sets, fontsize=10, ha='right', va='top')
        ax[1].set_title('Standard Periodogram in Each Measurement', fontsize=14)
        ax[1].yaxis.set_major_formatter(plt.NullFormatter())
        ax[1].xaxis.set_major_formatter(plt.NullFormatter())
        ax[1].set_ylabel('power + offset')

        LS_multi = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=1)
        LS_multi.optimizer.period_range = (33, 900)
        LS_multi.fit(jd_time.mjd, (np.sin(np.deg2rad(north_lon))), (np.sqrt(abs(np.sin(np.deg2rad(north_lon))))) + 1e-6,
                     north_filts)
        P_multi = LS_multi.periodogram(periods)
        ax[2].plot(periods, P_multi, lw=1, color='gray')

        ax[2].set_title('Multiband Periodogram', fontsize=12)
        ax[2].set_yticks([0, 0.2, 0.4])
        ax[2].set_ylim(0, 0.5)
        ax[2].yaxis.set_major_formatter(plt.NullFormatter())
        ax[2].set_xlabel('Period (days)')
        ax[2].set_ylabel('power')

        plt.savefig('/Users/mskirk/Desktop/N_Lon_periodogram.png')

    def S_Cent_periodogram(self):
        set_time, north_colat, north_lon, north_filts, south_colat, south_lon, south_filts, hole_separation = \
            self._assemble_pch_set(cent_series=True)

        jd_time = HarRot2JD(set_time)

        # Complete the LS for each data set, nyquist freq is 66 days
        periods = np.linspace(1, 900, 9000)
        model = periodic.NaiveMultiband(BaseModel=periodic.LombScargle)
        model.optimizer.period_range = (33, 900)
        # errors must be non zero and defined
        model.fit(jd_time.mjd, (south_colat), (np.sqrt(180.-south_colat)) + 1e-6,
                  south_filts)
        P = model.scores(periods)

        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(5, 2, left=0.07, right=0.95, bottom=0.15,
                          wspace=0.1, hspace=0.6)
        ax = [fig.add_subplot(gs[:, 0]),
              fig.add_subplot(gs[:-2, 1]),
              fig.add_subplot(gs[-2:, 1])]

        for sets in self.data_names:
            mask = (south_filts == self.filter_keys[sets])
            ax[0].plot((jd_time.mjd[mask] - jd_time.mjd[0]) / (jd_time.mjd[-1] - jd_time.mjd[0]), south_colat[mask], '.',
                       label=sets, lw=1)
        labels = np.around(jd_time.decimalyear[np.int32(np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9999]) * jd_time.size)],
                           decimals=2)
        # ax[0].set_ylim(0, 40)
        ax[0].legend(loc='upper left', fontsize=12, ncol=4)
        ax[0].set_title('Southern Co-Latitude', fontsize=14)
        ax[0].set_xticklabels(labels)
        ax[0].set_xlabel('Year')
        ax[0].set_ylabel('Location [Degrees]')

        for ii, sets in enumerate(self.data_names):
            offset = 12. - ii
            ax[1].plot(periods, P[self.filter_keys[sets]] + offset, lw=1)
            ax[1].text(95, 0.8 + offset, sets, fontsize=10, ha='right', va='top')
        ax[1].set_title('Standard Periodogram in Each Measurement', fontsize=14)
        ax[1].yaxis.set_major_formatter(plt.NullFormatter())
        ax[1].xaxis.set_major_formatter(plt.NullFormatter())
        ax[1].set_ylabel('power + offset')

        LS_multi = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=1)
        LS_multi.optimizer.period_range = (33, 900)
        LS_multi.fit(jd_time.mjd, (south_colat), (np.sqrt(180.-south_colat)) + 1e-6,
                     south_filts)
        P_multi = LS_multi.periodogram(periods)
        ax[2].plot(periods, P_multi, lw=1, color='gray')

        ax[2].set_title('Multiband Periodogram', fontsize=12)
        ax[2].set_yticks([0, 0.2, 0.4])
        ax[2].set_ylim(0, 0.5)
        ax[2].yaxis.set_major_formatter(plt.NullFormatter())
        ax[2].set_xlabel('Period (days)')
        ax[2].set_ylabel('power')

        plt.savefig('/Users/mskirk/Desktop/S_CoLat_periodogram.png')

        # Lon

        # Complete the LS for each data set, nyquist freq is 66 days
        periods = np.linspace(1, 900, 9000)
        model = periodic.NaiveMultiband(BaseModel=periodic.LombScargle)
        model.optimizer.period_range = (33, 900)
        # errors must be non zero and defined
        model.fit(jd_time.mjd, (np.sin(np.deg2rad(south_lon))), (np.sqrt(abs(np.sin(np.deg2rad(south_lon))))) + 1e-6,
                  south_filts)
        P = model.scores(periods)

        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(5, 2, left=0.07, right=0.95, bottom=0.15,
                          wspace=0.1, hspace=0.6)
        ax = [fig.add_subplot(gs[:, 0]),
              fig.add_subplot(gs[:-2, 1]),
              fig.add_subplot(gs[-2:, 1])]

        for sets in self.data_names:
            mask = (south_filts == self.filter_keys[sets])
            ax[0].plot((jd_time.mjd[mask] - jd_time.mjd[0]) / (jd_time.mjd[-1] - jd_time.mjd[0]), np.sin(np.deg2rad(south_lon))[mask],
                       '.', label=sets, lw=1)
        labels = np.around(jd_time.decimalyear[np.int32(np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9999]) * jd_time.size)],
                           decimals=2)
        # ax[0].set_ylim(0, 40)
        ax[0].legend(loc='upper left', fontsize=12, ncol=4)
        ax[0].set_title('Southern Co-Latitude', fontsize=14)
        ax[0].set_xticklabels(labels)
        ax[0].set_xlabel('Year')
        ax[0].set_ylabel('Location [Degrees]')

        for ii, sets in enumerate(self.data_names):
            offset = 12. - ii
            ax[1].plot(periods, P[self.filter_keys[sets]] + offset, lw=1)
            ax[1].text(95, 0.8 + offset, sets, fontsize=10, ha='right', va='top')
        ax[1].set_title('Standard Periodogram in Each Measurement', fontsize=14)
        ax[1].yaxis.set_major_formatter(plt.NullFormatter())
        ax[1].xaxis.set_major_formatter(plt.NullFormatter())
        ax[1].set_ylabel('power + offset')

        LS_multi = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=1)
        LS_multi.optimizer.period_range = (33, 900)
        LS_multi.fit(jd_time.mjd, (np.sin(np.deg2rad(south_lon))), (np.sqrt(abs(np.sin(np.deg2rad(south_lon))))) + 1e-6,
                     south_filts)
        P_multi = LS_multi.periodogram(periods)
        ax[2].plot(periods, P_multi, lw=1, color='gray')

        ax[2].set_title('Multiband Periodogram', fontsize=12)
        ax[2].set_yticks([0, 0.2, 0.4])
        ax[2].set_ylim(0, 0.5)
        ax[2].yaxis.set_major_formatter(plt.NullFormatter())
        ax[2].set_xlabel('Period (days)')
        ax[2].set_ylabel('power')

        plt.savefig('/Users/mskirk/Desktop/S_Lon_periodogram.png')

    def combine_pch_obj(self, list_of_obj, interval='1D'):

        pchobj = pd.concat(list_of_obj).sort_index()

        self.north_area = pchobj.Area[pchobj.StartLat > 0].resample(interval).median()
        self.south_area = pchobj.Area[pchobj.StartLat < 0].resample(interval).median()

        self.north_fit = pchobj.Fit[pchobj.StartLat > 0].resample(interval).median()
        self.south_fit = pchobj.Fit[pchobj.StartLat < 0].resample(interval).median()

        self.north_centroid = pchobj[['Center_lat', 'Center_lon']][pchobj.StartLat > 0].resample(interval).median()
        self.south_centroid = pchobj[['Center_lat', 'Center_lon']][pchobj.StartLat < 0].resample(interval).median()

        self.jd_time = Time(pchobj.index.tolist(), format='datetime').jd

        return pchobj

    def confidence_obj(self, confidence=0.98, interval='11D'):

        c_obj = pd.DataFrame(index=self.pch_obj.resample(interval).median().index, columns=['north_area_high', 'north_area_low', 'south_area_high',
                                                                'south_area_low', 'north_fit_high', 'north_fit_low',
                                                                'south_fit_high', 'south_fit_low', 'north_CentLat_high',
                                                                'north_CentLat_low', 'south_CentLat_high',
                                                                'south_CentLat_low', 'north_CentLon_high',
                                                                'north_CentLon_low', 'south_CentLon_high',
                                                                'south_CentLon_low'])

        # ------- North Area ---------
        comb = pd.concat([self.pch_obj.Area[self.northern], self.pch_obj.Area_max[self.northern], self.pch_obj.Area_min[self.northern]]).sort_index()
        ci = series_bootstrap(comb, interval=interval, iterations=5000, confidence=confidence, statistic=np.nanmedian)

        ci.lower = ci.lower.fillna(value=comb)
        ci.upper = ci.upper.fillna(value=comb)

        c_obj['north_area_high'] = ci.upper.resample(interval).median()
        c_obj['north_area_low'] = ci.lower.resample(interval).median()

        # ------- South Area ---------
        comb = pd.concat([self.pch_obj.Area[self.southern], self.pch_obj.Area_max[self.southern], self.pch_obj.Area_min[self.southern]]).sort_index()
        ci = series_bootstrap(comb, interval=interval, iterations=5000, confidence=confidence, statistic=np.nanmedian)

        ci.lower = ci.lower.fillna(value=comb)
        ci.upper = ci.upper.fillna(value=comb)

        c_obj['south_area_high'] = ci.upper.resample(interval).median()
        c_obj['south_area_low'] = ci.lower.resample(interval).median()

        # ------- North Perimeter Fit ---------
        comb = pd.concat([self.pch_obj.Fit[self.northern], self.pch_obj.Fit_max[self.northern], self.pch_obj.Fit_min[self.northern]]).sort_index()
        ci = series_bootstrap(comb, interval=interval, iterations=5000, confidence=confidence, statistic=np.nanmedian)

        ci.lower = ci.lower.fillna(value=comb)
        ci.upper = ci.upper.fillna(value=comb)

        c_obj['north_fit_high'] = ci.upper.resample(interval).median()
        c_obj['north_fit_low'] = ci.lower.resample(interval).median()

        # ------- South Perimeter Fit ---------
        comb = pd.concat([self.pch_obj.Fit[self.southern], self.pch_obj.Fit_max[self.southern], self.pch_obj.Fit_min[self.southern]]).sort_index()
        ci = series_bootstrap(comb, interval=interval, iterations=5000, confidence=confidence, statistic=np.nanmedian)

        ci.lower = ci.lower.fillna(value=comb)
        ci.upper = ci.upper.fillna(value=comb)

        c_obj['south_fit_high'] = ci.upper.resample(interval).median()
        c_obj['south_fit_low'] = ci.lower.resample(interval).median()

        # ------- North Centroid Lat ---------

        ci = series_bootstrap(self.pch_obj.Center_lat[self.northern], interval=interval, iterations=5000, confidence=confidence, statistic=np.nanmedian)

        ci.lower = ci.lower.fillna(value=self.pch_obj.Center_lat[self.northern])
        ci.upper = ci.upper.fillna(value=self.pch_obj.Center_lat[self.northern])

        c_obj['north_CentLat_high'] = ci.upper.resample(interval).median()
        c_obj['north_CentLat_low'] = ci.lower.resample(interval).median()

        # ------- South Centroid Lat---------
        ci = series_bootstrap(self.pch_obj.Center_lat[self.southern], interval=interval, iterations=5000, confidence=confidence, statistic=np.nanmedian)

        ci.lower = ci.lower.fillna(value=self.pch_obj.Center_lat[self.southern])
        ci.upper = ci.upper.fillna(value=self.pch_obj.Center_lat[self.southern])

        c_obj['south_CentLat_high'] = ci.upper.resample(interval).median()
        c_obj['south_CentLat_low'] = ci.lower.resample(interval).median()

        # ------- North Centroid Lon ---------

        ci = series_bootstrap(self.pch_obj.Center_lon[self.northern], interval=interval, iterations=5000, confidence=confidence, statistic=np.nanmedian)

        ci.lower = ci.lower.fillna(value=self.pch_obj.Center_lon[self.northern])
        ci.upper = ci.upper.fillna(value=self.pch_obj.Center_lon[self.northern])

        c_obj['north_CentLon_high'] = ci.upper.resample(interval).median()
        c_obj['north_CentLon_low'] = ci.lower.resample(interval).median()

        # ------- South Centroid Lon---------
        ci = series_bootstrap(self.pch_obj.Center_lon[self.southern], interval=interval, iterations=5000, confidence=confidence, statistic=np.nanmedian)

        ci.lower = ci.lower.fillna(value=self.pch_obj.Center_lon[self.southern])
        ci.upper = ci.upper.fillna(value=self.pch_obj.Center_lon[self.southern])

        c_obj['south_CentLon_high'] = ci.upper.resample(interval).median()
        c_obj['south_CentLon_low'] = ci.lower.resample(interval).median()

        self.confidence = c_obj

    def confidence_obj_parallel(self, confidence=0.98, interval='11D'):

        c_obj = pd.DataFrame(index=self.pch_obj.resample(interval).median().index, columns=['north_area_high', 'north_area_low', 'south_area_high',
                                                                'south_area_low', 'north_fit_high', 'north_fit_low',
                                                                'south_fit_high', 'south_fit_low', 'north_CentLat_high',
                                                                'north_CentLat_low', 'south_CentLat_high',
                                                                'south_CentLat_low', 'north_CentLon_high',
                                                                'north_CentLon_low', 'south_CentLon_high',
                                                                'south_CentLon_low'])

        pool = Pool()

        # params = [interval=interval, iterations=5000, confidence=confidence, statistic=np.nanmedian]

        # n_area_args = [pd.concat([self.pch_obj.Area[self.northern], self.pch_obj.Area_max[self.northern], self.pch_obj.Area_min[self.northern]]).sort_index(), ]


    def make_afrl_table(self, save_dir='', interval='1D'):

        afrl_table = Table()

        sample = self.pch_obj.resample(interval).median()
        errors = self.confidence.resample(interval).pad()
        sample = sample[errors.index[0] : errors.index[-1]]
        northern = self.pch_obj.StartLat > 0
        southern = self.pch_obj.StartLat < 0

        afrl_table['Date'] = sample.index
        afrl_table['Julian Day'] = Time(sample.index).jd

        afrl_table['North Lat'] = self.pch_obj.Fit[northern].resample(interval).median()[errors.index[0]: errors.index[-1]]
        afrl_table['South Lat'] = self.pch_obj.Fit[southern].resample(interval).median()[errors.index[0]: errors.index[-1]]
        afrl_table['North Lat CI Low'] = errors.north_fit_low.values
        afrl_table['North Lat CI High'] = errors.north_fit_high.values
        afrl_table['South Lat CI Low'] = errors.south_fit_low.values
        afrl_table['South Lat CI High'] = errors.south_fit_high.values


        afrl_table['North Area'] = self.pch_obj.Area[northern].resample(interval).median()[errors.index[0]: errors.index[-1]]
        afrl_table['South Area'] = self.pch_obj.Area[southern].resample(interval).median()[errors.index[0]: errors.index[-1]]
        afrl_table['North Area CI Low'] = errors.north_area_low.values
        afrl_table['North Area CI High'] = errors.north_area_high.values
        afrl_table['South Area CI Low'] = errors.south_area_low.values
        afrl_table['South Area CI High'] = errors.south_area_high.values

        write_file = os.path.join(os.path.abspath(save_dir), 'PolarCH_Table.ecsv')

        afrl_table.write(write_file, format='ascii.ecsv')

    def area_of_all_measures(self):
        self.all_area = pd.DataFrame(index=self.pch_obj.index, columns=['Area', 'Area_min', 'Area_max', 'Fit', 'Fit_min',
                                                                   'Fit_max', 'Center_lat', 'Center_lon'])

        self.fits_area = pd.DataFrame(index=self.pch_obj.index, columns=['Area', 'Area_min', 'Area_max', 'Fit', 'Fit_min',
                                                                   'Fit_max', 'Center_lat', 'Center_lon'])

        # Adding in Area Calculation each point with one HR previous measurements for all wavelengths.
        area = []
        fit = []
        center = []

        fit_area = []
        fit_fit = []
        fit_center = []

        for ii, h_rot in enumerate(self.pch_obj['Harvey_Rotation']):
            if self.pch_obj.iloc[ii]['StartLat'] > 0:
                northern = True
            else:
                northern = False
            ar, ft, cm = generic_hole_area(self.pch_obj, h_rot, northern=northern)
            f_ar, f_ft, f_cm = generic_hole_area(self.pch_obj, h_rot, northern=northern, use_fit=True)

            area = area + [ar]
            fit = fit + [ft]
            center = center + [cm]

            fit_area = fit_area + [f_ar]
            fit_fit = fit_fit + [f_ft]
            fit_center = fit_center + [f_cm]

            print(ii)

        center = np.vstack([arr[0:2] for arr in center])
        fit_center = np.vstack([arr[0:2] for arr in fit_center])

        self.all_area['Area'] = np.asarray(area)[:, 1]
        self.all_area['Area_min'] = np.asarray(area)[:, 0]
        self.all_area['Area_max'] = np.asarray(area)[:, 2]

        self.all_area['Fit'] = np.asarray(fit)[:, 1] * u.deg
        self.all_area['Fit_min'] = np.asarray(fit)[:, 0] * u.deg
        self.all_area['Fit_max'] = np.asarray(fit)[:, 2] * u.deg

        self.all_area['Center_lat'] = center[:, 1] * u.deg
        self.all_area['Center_lon'] = center[:, 0] * u.deg

        self.fits_area['Area'] = np.asarray(fit_area)[:, 1]
        self.fits_area['Area_min'] = np.asarray(fit_area)[:, 0]
        self.fits_area['Area_max'] = np.asarray(fit_area)[:, 2]

        self.fits_area['Fit'] = np.asarray(fit_fit)[:, 1] * u.deg
        self.fits_area['Fit_min'] = np.asarray(fit_fit)[:, 0] * u.deg
        self.fits_area['Fit_max'] = np.asarray(fit_fit)[:, 2] * u.deg

        self.fits_area['Center_lat'] = fit_center[:, 1] * u.deg
        self.fits_area['Center_lon'] = fit_center[:, 0] * u.deg


#    def stack_series(self, data_key):
#        self.north_pch = np.stack((self.data_sets[data_key]['North_Size']-self.data_sets[data_key]['North_Spread'],
#                                self.data_sets[data_key]['North_Size'],
#                                self.data_sets[data_key]['North_Size']+self.data_sets[data_key]['North_Spread']))
#
#        self.south_pch = np.stack((self.data_sets[data_key]['South_Size'] - self.data_sets[data_key]['South_Spread'],
#                              self.data_sets[data_key]['South_Size'],
#                              self.data_sets[data_key]['South_Size'] + self.data_sets[data_key]['South_Spread']))
#    def regrid_time(self):
#        jd_time = self.HarRot2JD(self._assemble_pch_set(time_only=True))
#
#        # one step per day
#        tsteps = np.floor((jd_time.mjd[-1] - jd_time.mjd[0]))
#
#        self.time_grid = Time(np.linspace(jd_time.mjd[0], jd_time.mjd[-1], tsteps), format='mjd')
#
#    def regrid_series(self, data_key, centroid=False):
#
#        # regularize across all data sets
#
#        self.regrid_time()
#
#        north_series = np.empty_like(self.time_grid.mjd)
#        north_series[:] = np.NAN
#        south_series = np.empty_like(self.time_grid.mjd)
#        south_series[:] = np.NAN
#        center_series = np.empty_like(self.time_grid.mjd)
#        center_series[:] = np.NAN
#
#        cent_sep = self.centroid_separation(data_key)#
#
#        tmp_time = self.HarRot2JD(self.data_sets[data_key]['Harvey_Rotation'])
#        for ii, time_step in enumerate(self.time_grid.mjd[:-1]):
#            north_series[ii] = np.mean(self.data_sets[data_key]['North_Size']
#                                            [(tmp_time.mjd >= self.time_grid.mjd[ii]) *
#                                             (tmp_time.mjd < self.time_grid.mjd[ii + 1])])
#            south_series[ii] = np.mean(self.data_sets[data_key]['South_Size']
#                                            [(tmp_time.mjd >= self.time_grid.mjd[ii]) *
#                                             (tmp_time.mjd < self.time_grid.mjd[ii + 1])])
#            center_series[ii] = np.mean(cent_sep[(tmp_time.mjd >= self.time_grid.mjd[ii]) *
#                                                 (tmp_time.mjd < self.time_grid.mjd[ii + 1])])
#
#        if centroid:
#            return center_series
#        else:
#            return north_series, south_series
#
#    def _assemble_pch_set(self, time_only=False, cent_series=False):
#
#        set_size = 0
#        for kk in self.data_sets:
#            set_size += self.data_sets[kk]['Harvey_Rotation'].size
#
#        north_pch = np.zeros(set_size)
#        north_dpch = np.zeros_like(north_pch)
#        north_filts = np.empty_like(north_pch, dtype='<U3')
#        north_colat = np.zeros_like(north_pch)
#        north_lon = np.zeros_like(north_pch)
#
#        set_time = np.zeros_like(north_pch)
#        hole_separation = np.zeros_like(north_pch)
#
#        south_pch = np.zeros_like(north_pch)
#        south_dpch = np.zeros_like(north_pch)
#        south_filts = np.empty_like(north_pch, dtype='<U3')
#        south_colat = np.zeros_like(north_pch)
#        south_lon = np.zeros_like(north_pch)#
#
#
#        bookmark = 0.
#        for sets in self.data_sets:
#            north_pch[bookmark:bookmark+self.data_sets[sets]['North_Size'].size] = self.data_sets[sets]['North_Size']
#            north_dpch[bookmark:bookmark + self.data_sets[sets]['North_Spread'].size] = \
#                self.data_sets[sets]['North_Spread']
#            north_filts[bookmark:bookmark + self.data_sets[sets]['North_Size'].size] = \
#                np.repeat(self.filter_keys[sets], self.data_sets[sets]['North_Size'].size)
#            north_colat[bookmark:bookmark + self.data_sets[sets]['North_Size'].size] = \
#                self.data_sets[sets]['N_Cent_CoLat']
#            north_lon[bookmark:bookmark + self.data_sets[sets]['North_Size'].size] = \
#                self.data_sets[sets]['N_Cent_Lon']
#
#            south_pch[bookmark:bookmark+self.data_sets[sets]['South_Size'].size] = self.data_sets[sets]['South_Size']
#            south_dpch[bookmark:bookmark + self.data_sets[sets]['South_Spread'].size] = \
#                self.data_sets[sets]['South_Spread']
#            south_filts[bookmark:bookmark + self.data_sets[sets]['South_Size'].size] = \
#                np.repeat(self.filter_keys[sets], self.data_sets[sets]['South_Size'].size)
#            south_colat[bookmark:bookmark + self.data_sets[sets]['South_Size'].size] = \
#                self.data_sets[sets]['S_Cent_CoLat']
#            south_lon[bookmark:bookmark + self.data_sets[sets]['South_Size'].size] = \
#                self.data_sets[sets]['S_Cent_Lon']
#
#            hole_separation[bookmark:bookmark + self.data_sets[sets]['Harvey_Rotation'].size] = \
#                self.centroid_separation(sets)
#            set_time[bookmark:bookmark + self.data_sets[sets]['Harvey_Rotation'].size] = \
#                self.data_sets[sets]['Harvey_Rotation']
#            bookmark += self.data_sets[sets]['Harvey_Rotation'].size
#
#        new_order = np.argsort(set_time)
#
#        if time_only:
#            return set_time[new_order]
#        elif cent_series:
#            return set_time[new_order], north_colat[new_order], north_lon[new_order], north_filts[new_order], \
#                   south_colat[new_order], south_lon[new_order], south_filts[new_order], hole_separation[new_order]
#        else:
#            return set_time[new_order], north_pch[new_order], north_dpch[new_order], north_filts[new_order], \
#                   south_pch[new_order], south_dpch[new_order], south_filts[new_order]
#


def haversine(lon1, lat1, lon2, lat2, degrees=True):
    """
    Calculate the great circle distance between two points
    on a sphere (default in decimal degrees)
    """

    if degrees: # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = np.deg2rad([lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 1  # Normalized Radius - can change this to physical units later...

    if degrees:
        return np.rad2deg(c * r)
    else:
        return c * r


def harrot2jd(hr, duplicates=True):
    """

    :param hr: Harvey Rotation Number
                Harvey Rotations are 33 day rotations where hr = 0 ::= Jan 4, 1900
    :return: astropy.time Time object
    """

    jd = (((hr - 1.) * 360.) / (360. / 33.)) + 2415023.5

    # If duplicates = false and the time stamp repeats, add a minute to the later one - account for rounding errors.

    if not duplicates:
        for iteration, time_stamp in enumerate(jd):
            if not np.sum(np.unique(jd, return_index=True)[1] == iteration):
                jd[iteration] = time_stamp+(1./(24. * 60.))

    return Time(jd, format='jd')


def bootstrap(dataset, confidence=0.95, iterations=10000, sample_size=None, statistic=np.median):
    """
    Bootstrap the confidence intervals for a given sample of a population
    and a statistic.
    Args:
        dataset: A list of values, each a sample from an unknown population
        confidence: The confidence value (a float between 0 and 1.0)
        iterations: The number of iterations of resampling to perform
        sample_size: The sample size for each of the resampled (0 to 1.0
                     for 0 to 100% of the original data size)
        statistic: The statistic to use. This must be a function that accepts
                   a list of values and returns a single value.
    Returns:
        Returns the upper and lower values of the confidence interval.

    Reference:
        Open Journal of Statistics, Vol.07 No.03(2017), Article ID:76758,17 pages, 10.4236/ojs.2017.73029
    """
    stats = list()
    if not sample_size:
        sample_size = 1 / np.sqrt(len(dataset))
    n_size = int(len(dataset) * sample_size)

    for _ in range(iterations):
        # Sample (with replacement) from the given dataset
        sample = resample(dataset, n_samples=n_size, replace=True)
        # Calculate user-defined statistic and store it
        stat = statistic(sample)
        stats.append(stat)

    # Sort the array of per-sample statistics and cut off ends
    ostats = sorted(stats)
    lval = np.nanpercentile(ostats, ((1 - confidence) / 2) * 100)
    uval = np.nanpercentile(ostats, (confidence + ((1 - confidence) / 2)) * 100)

    return lval, uval


def series_bootstrap(series, interval='1D', statistic=np.median, delta=False, **kwargs):
    ci = pd.DataFrame(index=series.index, columns=['lower', 'upper'])
    groups = list()
    lower = list()
    upper = list()

    for g, data in series.groupby(series.index.floor(interval)):
        groups.append(g)

        low, high = bootstrap(data.values, statistic=statistic, **kwargs)

        if np.isfinite(low):
            lower.append(low)
        else:
            lower.append(statistic(data.values))
        if np.isfinite(high):
            upper.append(high)
        else:
            upper.append(statistic(data.values))

    for ii in range(len(groups)-2):
        if delta:
            ci.loc[groups[ii]:groups[ii + 1], 'upper'] = upper[ii] - series.resample(interval).median()[ii]

            if lower[ii] <= 0:
                ci.loc[groups[ii]:groups[ii + 1], 'lower'] = 0
            else:
                ci.loc[groups[ii]:groups[ii + 1], 'lower'] = lower[ii] - series.resample(interval).median()[ii]
        else:
            ci.loc[groups[ii]:groups[ii + 1], 'lower'] = lower[ii]
            ci.loc[groups[ii]:groups[ii + 1], 'upper'] = upper[ii]

    return ci


def read_wso_data(filename):
    raw_wso = pd.read_csv(filename, sep='\s+', header=0)

    wso = pd.DataFrame(index=pd.Series([datetime.strptime(dd, '%Y:%m:%d_%Hh:%Mm:%Ss') for dd in raw_wso.Date]))

    wso['North'] = pd.to_numeric(raw_wso.North.str.replace('N', '')).values
    wso['South'] = pd.to_numeric(raw_wso.South.str.replace('S', '')).values
    wso['Average'] = pd.to_numeric(raw_wso.Avg.str.replace('Avg', '')).values
    wso['NorthFilter'] = pd.to_numeric(raw_wso.N_filt.str.replace('Nf', '')).values
    wso['SouthFilter'] = pd.to_numeric(raw_wso.S_filt.str.replace('Sf', '')).values
    wso['AverageFilter'] = pd.to_numeric(raw_wso.Avg_filt.str.replace('Avgf', '')).values

    # The data is already reported at a 10 Day cadence, this just tells pandas about it
    return wso.resample('10D').mean()


def year_butter_filter(series, period_low=0.82, period_high=1.1, filter_type='bandstop', Series=False, show_spectrum=False):

    # peroids low and high are in years
    # pd series must have a consistent frequency defined

    fs = 1/TimeDelta(series.index.freq.delta).to(u.yr)

    b, a = sg.butter(6, [period_low, period_high], btype=filter_type, fs=fs.value)

    if show_spectrum:
        f, Pxx_spec = signal.periodogram(sg.filtfilt(b, a, series.interpolate('linear')), fs.value, 'flattop', scaling='spectrum')
        plt.semilogy(f, np.sqrt(Pxx_spec))
        plt.show()

    if Series:
        return pd.Series(sg.filtfilt(b, a, series.interpolate('linear')), index=series.index)

    return sg.filtfilt(b, a, series.interpolate('linear'))


def low_butter_filter(series, period_low=0.82, filter_type='bandpass', Series=False, show_spectrum=False):

    # low pass filter
    # peroids low are in years
    # pd series must have a consistent frequency defined

    fs = 1/TimeDelta(series.index.freq.delta).to(u.yr)

    b, a = sg.butter(6, period_low, btype=filter_type, fs=fs.value)

    if show_spectrum:
        f, Pxx_spec = signal.periodogram(sg.filtfilt(b, a, series.interpolate('linear')), fs.value, 'flattop', scaling='spectrum')
        plt.semilogy(f, np.sqrt(Pxx_spec))
        plt.show()

    if Series:
        return pd.Series(sg.filtfilt(b, a, series.interpolate('linear')), index=series.index)

    return sg.filtfilt(b, a, series.interpolate('linear'))


def generic_hole_area(detection_df, h_rotation_number, northern=True, use_fit=False):
        # Returns the area as a fraction of the total solar surface area
        # Returns the location of the perimeter fit for the given h_rotation_number

        begin = np.min(np.where(detection_df['Harvey_Rotation'] > (h_rotation_number - 1)))
        end = np.max(np.where(detection_df['Harvey_Rotation'] == h_rotation_number))

        if use_fit:
            if northern:
                # A northern hole with Arclength Filter for eliminating small holes but not zeros
                index_measurements = np.where((detection_df[begin:end]['Fit'] > 0) & np.logical_not(
                    detection_df[begin:end]['ArcLength'] < 3.0))
            else:
                # A southern hole with Arclength Filter for eliminating small holes
                index_measurements = np.where((detection_df[begin:end]['Fit'] < 0) & np.logical_not(
                    detection_df[begin:end]['ArcLength'] < 3.0))

            index_measurements = np.array(index_measurements).squeeze() + begin

            # Filters for incomplete hole measurements: at least 10 points and half a h-rotation needs to be defined
            if index_measurements.size < 10:
                return np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array(
                    [np.nan, np.nan, np.nan])

            elif detection_df['Harvey_Rotation'][index_measurements[-1]] - detection_df['Harvey_Rotation'][
                index_measurements[0]] < 0.5:
                return np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array(
                    [np.nan, np.nan, np.nan])

            else:
                lons = detection_df.iloc[index_measurements]['H_StartLon'].values * u.deg
                lats = detection_df.iloc[index_measurements]['StartLat'].values * u.deg
                errors = np.asarray(1 / detection_df.iloc[index_measurements]['Quality'])

        else:
            if northern:
                # A northern hole with Arclength Filter for eliminating small holes but not zeros
                index_measurements = np.where((detection_df[begin:end]['StartLat'] > 0) & np.logical_not(
                    detection_df[begin:end]['ArcLength'] < 3.0))
            else:
                # A southern hole with Arclength Filter for eliminating small holes
                index_measurements = np.where((detection_df[begin:end]['StartLat'] < 0) & np.logical_not(
                    detection_df[begin:end]['ArcLength'] < 3.0))

            index_measurements = np.array(index_measurements).squeeze() + begin

            # Filters for incomplete hole measurements: at least 10 points and half a h-rotation needs to be defined
            if index_measurements.size < 10:
                return np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])

            elif detection_df['Harvey_Rotation'][index_measurements[-1]] - detection_df['Harvey_Rotation'][index_measurements[0]] < 0.5:
                return np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])

            else:
                lons = np.concatenate(np.vstack([detection_df.iloc[index_measurements]['H_StartLon'].values,
                                                 detection_df.iloc[index_measurements]['H_EndLon'].values])) * u.deg
                lats = np.concatenate(np.vstack([detection_df.iloc[index_measurements]['StartLat'].values,
                                                 detection_df.iloc[index_measurements]['EndLat'].values])) * u.deg
                errors = np.concatenate(np.vstack([np.asarray(1/detection_df.iloc[index_measurements]['Quality']),
                                                   np.asarray(1/detection_df.iloc[index_measurements]['Quality'])]))

        perimeter_length = np.zeros(6) * u.rad
        fit_location = np.zeros(6) * u.rad
        hole_area = np.zeros(6)

        # Centroid offset for better fitting results
        if northern:
            offset_coords = np.transpose(np.asarray([lons, (90 * u.deg) - lats]))
        else:
            offset_coords = np.transpose(np.asarray([lons, (90 * u.deg) + lats]))

        offset_cm = PCH_Tools.center_of_mass(offset_coords, mass=1/errors) * u.deg
        offset_lons = np.mod(offset_coords[:, 0] * u.deg - offset_cm[0], 360* u.deg)
        offset_lats = offset_coords[:, 1] * u.deg - offset_cm[1]

        for ii, degrees in enumerate([4, 5, 6, 7, 8, 9]):
            try:
                hole_fit = PCH_Tools.trigfit(np.deg2rad(offset_lons), np.deg2rad(offset_lats), degree=degrees,
                                             sigma=errors)

                # Lambert cylindrical equal-area projection to find the area using the composite trapezoidal rule
                # And removing centroid offset
                if northern:
                    lamb_x = np.deg2rad(np.arange(0, 360, 0.01) * u.deg)
                    lamb_y = np.sin(
                        (np.pi * 0.5) - hole_fit['fitfunc'](lamb_x.value) - np.deg2rad(offset_cm[1]).value) * u.rad

                    lamb_x = np.deg2rad(np.arange(0, 360, 0.01) * u.deg + offset_cm[0])
                    fit_location[ii] = np.rad2deg((np.pi * 0.5) - hole_fit['fitfunc'](np.deg2rad(
                        PCH_Tools.get_harvey_lon(PCH_Tools.hrot2date(h_rotation_number)) - offset_cm[
                            0]).value) - np.deg2rad(offset_cm[1]).value) * u.deg
                else:
                    lamb_x = np.deg2rad(np.arange(0, 360, 0.01) * u.deg)
                    lamb_y = np.sin(
                        hole_fit['fitfunc'](lamb_x.value) - (np.pi * 0.5) + np.deg2rad(offset_cm[1]).value) * u.rad

                    lamb_x = np.deg2rad(np.arange(0, 360, 0.01) * u.deg + offset_cm[0])
                    fit_location[ii] = np.rad2deg(hole_fit['fitfunc'](np.deg2rad(
                        PCH_Tools.get_harvey_lon(PCH_Tools.hrot2date(h_rotation_number)) - offset_cm[0]).value) - (
                                                              np.pi * 0.5) + np.deg2rad(offset_cm[1]).value) * u.deg

                perimeter_length[ii] = PCH_Tools.curve_length(lamb_x, lamb_y)

                if northern:
                    hole_area[ii] = (2 * np.pi) - np.trapz(lamb_y, x=lamb_x).value
                else:
                    hole_area[ii] = (2 * np.pi) + np.trapz(lamb_y, x=lamb_x).value

            except RuntimeError:
                hole_area[ii] = np.nan
                perimeter_length[ii] = np.inf * u.rad
                fit_location[ii] = np.nan

        # allowing for a 5% perimeter deviation off of a circle
        good_areas = hole_area[np.where((perimeter_length / (2*np.pi*u.rad)) -1 < 0.05)]
        good_fits = fit_location[np.where((perimeter_length / (2*np.pi*u.rad)) -1 < 0.05)]

        # A sphere is 4π steradians in surface area
        if good_areas.size > 0:
            percent_hole_area = (np.nanmin(good_areas) / (4 * np.pi), np.nanmean(good_areas) / (4 * np.pi), np.nanmax(good_areas) / (4 * np.pi))
            # in degrees
            hole_perimeter_location = (np.rad2deg(np.nanmin(good_fits)).value, np.rad2deg(np.nanmean(good_fits)).value, np.rad2deg(np.nanmax(good_fits)).value)
        else:
            percent_hole_area = (np.nan, np.nan, np.nan)
            hole_perimeter_location = np.array([np.nan, np.nan, np.nan])

        # From co-lat to lat

        if northern:
            if offset_cm[1] < 0:
                offset_cm[1] = (90 * u.deg) + offset_cm[1]
                offset_cm[0] = np.mod(offset_cm[0]-180*u.deg, 360* u.deg)
            else:
                offset_cm[1] = (90 * u.deg) - offset_cm[1]
        else:
            if offset_cm[1] > 0:
                offset_cm[1] = (-90 * u.deg) + offset_cm[1]
            else:
                offset_cm[1] = (-90 * u.deg) - offset_cm[1]
                offset_cm[0] = np.mod(offset_cm[0]-180*u.deg, 360* u.deg)

        # Tuples of shape (Min, Mean, Max)
        return np.asarray(percent_hole_area), np.asarray(hole_perimeter_location), np.asarray(offset_cm)
