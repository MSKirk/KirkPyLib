
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import fnmatch
from astropy.time import Time
from sunpy.time import parse_time
import pandas as pd
from gatspy import periodic
from scipy import signal


class PCH:
    def __init__(self, directory):

        self.directory = directory
        self.filelist = fnmatch.filter(os.listdir(self.directory), '*Area*.txt')
        self.data_sets = pd.DataFrame()  # initialize a null dictionary
        self.filter_keys = {}

        for file in self.filelist:
            name = ''
            name = name.join(file.split('_Area')[0].split('_'))
            self.data_sets = pd.concat([self.data_sets, self.pch_data_parse(os.path.join(self.directory, file))])
            self.filter_keys[name] = name[0]+name[-4]+name[-1]

        self.data_names = np.sort([k for k in self.filter_keys])
        self.data_sets =self.data_sets.sort_index()

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

        pch_obj = np.loadtxt(file_path, skiprows=3, dtype={'names': ('Harvey_Rotation', 'North_Size', 'North_Spread',
                                                                     'South_Size', 'South_Spread', 'N_Cent_Lon',
                                                                     'N_Cent_CoLat', 'S_Cent_Lon', 'S_Cent_CoLat'),
                                                           'formats': ('f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f')})

        pch_obj = pd.DataFrame(data=pch_obj, index=parse_time(self.HarRot2JD(pch_obj['Harvey_Rotation'], duplicates=False)))

        # Create data masks for non-measurements - less than 0 or unphysical.
        pch_obj['North_Size'] = pch_obj['North_Size'].mask(pch_obj['North_Size'] < 0)
        pch_obj['North_Spread'] = pch_obj['North_Spread'].mask(pch_obj['North_Size'] < 0)
        pch_obj['N_Cent_Lon'] = pch_obj['N_Cent_Lon'].mask(pch_obj['North_Size'] < 0)
        pch_obj['N_Cent_CoLat'] = pch_obj['N_Cent_CoLat'].mask(pch_obj['North_Size'] < 0)

        pch_obj['South_Size'] = pch_obj['South_Size'].mask(pch_obj['South_Size'] < 0)
        pch_obj['South_Spread'] = pch_obj['South_Spread'].mask(pch_obj['South_Size'] < 0)
        pch_obj['S_Cent_Lon'] = pch_obj['S_Cent_Lon'].mask(pch_obj['South_Size'] < 0)
        pch_obj['S_Cent_CoLat'] = pch_obj['S_Cent_CoLat'].mask(pch_obj['South_Size'] < 0)

        # NAN hunting - periodogram does not deal with nans so well.

        pch_obj = pch_obj.dropna(subset=['North_Size'])

        pch_obj.loc[pch_obj['North_Spread'] > pch_obj['North_Size'], 'North_Spread'] = \
            pch_obj.loc[pch_obj['North_Spread'] > pch_obj['North_Size'], 'North_Size']

        pch_obj.loc[pch_obj['South_Spread'] > pch_obj['South_Size'], 'South_Spread'] = \
            pch_obj.loc[pch_obj['South_Spread'] > pch_obj['South_Size'], 'South_Size']
        
        # Adding explicit Filters for periodogram
        name = ''
        name = name.join(file_path.split('/')[-1].split('_Area')[0].split('_'))
        pch_obj['Filter'] = np.repeat(name[0]+name[-4]+name[-1], pch_obj['North_Size'].size)

        pch_obj['Hole_Separation'] = self.haversine(pch_obj['N_Cent_Lon'], pch_obj['N_Cent_CoLat']-90.,
                                                    pch_obj['S_Cent_Lon'],pch_obj['S_Cent_CoLat']-90.)

        return pch_obj

    def HarRot2JD(self, hr, duplicates=True):
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

    def haversine(self, lon1, lat1, lon2, lat2, degrees=True):
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

    def centroid_separation(self, data_key):

        # convert form co-lat to lat (north & south are inverted... but it results in the same answer)
        return self.haversine(self.data_sets[data_key]['N_Cent_Lon'], self.data_sets[data_key]['N_Cent_CoLat']-90.,
                              self.data_sets[data_key]['S_Cent_Lon'],self.data_sets[data_key]['S_Cent_CoLat']-90.)

    def stack_series(self, data_key):
        self.north_pch = np.stack((self.data_sets[data_key]['North_Size']-self.data_sets[data_key]['North_Spread'],
                                self.data_sets[data_key]['North_Size'],
                                self.data_sets[data_key]['North_Size']+self.data_sets[data_key]['North_Spread']))

        self.south_pch = np.stack((self.data_sets[data_key]['South_Size'] - self.data_sets[data_key]['South_Spread'],
                              self.data_sets[data_key]['South_Size'],
                              self.data_sets[data_key]['South_Size'] + self.data_sets[data_key]['South_Spread']))
    def regrid_time(self):
        jd_time = self.HarRot2JD(self._assemble_pch_set(time_only=True))

        # one step per day
        tsteps = np.floor((jd_time.mjd[-1] - jd_time.mjd[0]))

        self.time_grid = Time(np.linspace(jd_time.mjd[0], jd_time.mjd[-1], tsteps), format='mjd')

    def regrid_series(self, data_key, centroid=False):

        # regularize across all data sets

        self.regrid_time()

        north_series = np.empty_like(self.time_grid.mjd)
        north_series[:] = np.NAN
        south_series = np.empty_like(self.time_grid.mjd)
        south_series[:] = np.NAN
        center_series = np.empty_like(self.time_grid.mjd)
        center_series[:] = np.NAN

        cent_sep = self.centroid_separation(data_key)

        tmp_time = self.HarRot2JD(self.data_sets[data_key]['Harvey_Rotation'])
        for ii, time_step in enumerate(self.time_grid.mjd[:-1]):
            north_series[ii] = np.mean(self.data_sets[data_key]['North_Size']
                                            [(tmp_time.mjd >= self.time_grid.mjd[ii]) *
                                             (tmp_time.mjd < self.time_grid.mjd[ii + 1])])
            south_series[ii] = np.mean(self.data_sets[data_key]['South_Size']
                                            [(tmp_time.mjd >= self.time_grid.mjd[ii]) *
                                             (tmp_time.mjd < self.time_grid.mjd[ii + 1])])
            center_series[ii] = np.mean(cent_sep[(tmp_time.mjd >= self.time_grid.mjd[ii]) *
                                                 (tmp_time.mjd < self.time_grid.mjd[ii + 1])])

        if centroid:
            return center_series
        else:
            return north_series, south_series

    def peek(self, data_key):

        sns.set(style="darkgrid")

        self.time = self.HarRot2JD(self.data_sets[data_key]['Harvey_Rotation'])

        self.stack_series(data_key)

        plt.figure()
        plt.title('Polar Coronal Hole Area')

        plt.subplot(2,1,1)
        ax1 = sns.tsplot(data=self.north_pch, time=self.time.mjd, value="North PCH Area")

        plt.subplot(2, 1, 2)
        ax2 = sns.tsplot(data=self.south_pch, time=self.time.mjd, value="South PCH Area")

        plt.show()

    def _assemble_pch_set(self, time_only=False, cent_series=False):

        set_size = 0
        for kk in self.data_sets:
            set_size += self.data_sets[kk]['Harvey_Rotation'].size

        north_pch = np.zeros(set_size)
        north_dpch = np.zeros_like(north_pch)
        north_filts = np.empty_like(north_pch, dtype='<U3')
        north_colat = np.zeros_like(north_pch)
        north_lon = np.zeros_like(north_pch)

        set_time = np.zeros_like(north_pch)
        hole_separation = np.zeros_like(north_pch)
        
        south_pch = np.zeros_like(north_pch)
        south_dpch = np.zeros_like(north_pch)
        south_filts = np.empty_like(north_pch, dtype='<U3')
        south_colat = np.zeros_like(north_pch)
        south_lon = np.zeros_like(north_pch)


        bookmark = 0.
        for sets in self.data_sets:
            north_pch[bookmark:bookmark+self.data_sets[sets]['North_Size'].size] = self.data_sets[sets]['North_Size']
            north_dpch[bookmark:bookmark + self.data_sets[sets]['North_Spread'].size] = \
                self.data_sets[sets]['North_Spread']
            north_filts[bookmark:bookmark + self.data_sets[sets]['North_Size'].size] = \
                np.repeat(self.filter_keys[sets], self.data_sets[sets]['North_Size'].size)
            north_colat[bookmark:bookmark + self.data_sets[sets]['North_Size'].size] = \
                self.data_sets[sets]['N_Cent_CoLat']
            north_lon[bookmark:bookmark + self.data_sets[sets]['North_Size'].size] = \
                self.data_sets[sets]['N_Cent_Lon']

            south_pch[bookmark:bookmark+self.data_sets[sets]['South_Size'].size] = self.data_sets[sets]['South_Size']
            south_dpch[bookmark:bookmark + self.data_sets[sets]['South_Spread'].size] = \
                self.data_sets[sets]['South_Spread']
            south_filts[bookmark:bookmark + self.data_sets[sets]['South_Size'].size] = \
                np.repeat(self.filter_keys[sets], self.data_sets[sets]['South_Size'].size)
            south_colat[bookmark:bookmark + self.data_sets[sets]['South_Size'].size] = \
                self.data_sets[sets]['S_Cent_CoLat']
            south_lon[bookmark:bookmark + self.data_sets[sets]['South_Size'].size] = \
                self.data_sets[sets]['S_Cent_Lon']

            hole_separation[bookmark:bookmark + self.data_sets[sets]['Harvey_Rotation'].size] = \
                self.centroid_separation(sets)
            set_time[bookmark:bookmark + self.data_sets[sets]['Harvey_Rotation'].size] = \
                self.data_sets[sets]['Harvey_Rotation']
            bookmark += self.data_sets[sets]['Harvey_Rotation'].size

        new_order = np.argsort(set_time)

        if time_only:
            return set_time[new_order]
        elif cent_series:
            return set_time[new_order], north_colat[new_order], north_lon[new_order], north_filts[new_order], \
                   south_colat[new_order], south_lon[new_order], south_filts[new_order], hole_separation[new_order]
        else:
            return set_time[new_order], north_pch[new_order], north_dpch[new_order], north_filts[new_order], \
                   south_pch[new_order], south_dpch[new_order], south_filts[new_order]


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

        jd_time = self.HarRot2JD(set_time)

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

