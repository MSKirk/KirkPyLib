


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import fnmatch
from astropy.time import Time
import pandas as pd
from gatspy import periodic


class PCH:
    def __init__(self, directory):

        self.directory = directory
        self.filelist = fnmatch.filter(os.listdir(self.directory), '*Area*.txt')
        self.data_sets = {}  # initialize a null dictionary
        self.filter_keys = {}

        for file in self.filelist:
            name = ''
            name = name.join(file.split('_Area')[0].split('_'))
            self.data_sets[name] = self.pch_data_parse(os.path.join(self.directory, file))
            self.filter_keys[name] = name[0]+name[-4]+name[-1]

        self.data_names = np.sort([k for k in self.data_sets])

        plt.ion()

    def pch_data_parse(self, file_path, df=True):
        """
        :param file_path: full path of the file to read
        :param df: If true, returns a pandas data frame, otherwise an NP object
        :return: Parsed PCH data
        """

        pch_obj = np.loadtxt(file_path, skiprows=3, dtype={'names': ('Harvey_Rotation', 'North_Size', 'North_Spread',
                                                                     'South_Size', 'South_Spread', 'N_Cent_Lon',
                                                                     'N_Cent_CoLat', 'S_Cent_Lon', 'S_Cent_CoLat'),
                                                           'formats': ('f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f')})

        pch_obj = pd.DataFrame(data=pch_obj)

        # Create data masks for non-measurements - less than 0 or unphysical.
        pch_obj['North_Size'] = pch_obj['North_Size'].mask(pch_obj['North_Size'] < 0)
        pch_obj['North_Spread'] = pch_obj['North_Spread'].mask(pch_obj['North_Size'] < 0)
        pch_obj['North_Spread'] = pch_obj['North_Spread'].mask(pch_obj['North_Size'] < pch_obj['North_Spread'])
        pch_obj['N_Cent_Lon'] = pch_obj['N_Cent_Lon'].mask(pch_obj['North_Size'] < 0)
        pch_obj['N_Cent_CoLat'] = pch_obj['N_Cent_CoLat'].mask(pch_obj['North_Size'] < 0)

        pch_obj['South_Size'] = pch_obj['South_Size'].mask(pch_obj['South_Size'] < 0)
        pch_obj['South_Spread'] = pch_obj['South_Spread'].mask(pch_obj['South_Size'] < 0)
        pch_obj['South_Spread'] = pch_obj['South_Spread'].mask(pch_obj['South_Size'] < pch_obj['South_Spread'])
        pch_obj['S_Cent_Lon'] = pch_obj['S_Cent_Lon'].mask(pch_obj['South_Size'] < 0)
        pch_obj['S_Cent_CoLat'] = pch_obj['S_Cent_CoLat'].mask(pch_obj['South_Size'] < 0)

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
                    jd[iteration]=time_stamp+(1./(24. * 60.))

        return Time(jd, format='jd')

    def stack_series(self, data_key):
        self.north_pch = np.stack((self.data_sets[data_key]['North_Size']-self.data_sets[data_key]['North_Spread'],
                                self.data_sets[data_key]['North_Size'],
                                self.data_sets[data_key]['North_Size']+self.data_sets[data_key]['North_Spread']))

        self.south_pch = np.stack((self.data_sets[data_key]['South_Size'] - self.data_sets[data_key]['South_Spread'],
                              self.data_sets[data_key]['South_Size'],
                              self.data_sets[data_key]['South_Size'] + self.data_sets[data_key]['South_Spread']))
    def regrid_time(self):
        jd_time = self.HarRot2JD(self._assemble_pch_set(time_only=True))

        # two steps per day
        tsteps = np.floor((jd_time.mjd[-1] - jd_time.mjd[0]) * 2)

        self.time_grid = Time(np.linspace(jd_time.mjd[0], jd_time.mjd[-1], tsteps), format='mjd')

    def regrid_series(self, data_key):

        # regularize across all data sets

        self.regrid_time()

        north_series = np.empty_like(self.time_grid.mjd)
        north_series[:] = np.NAN
        south_series = np.empty_like(self.time_grid.mjd)
        south_series[:] = np.NAN


        tmp_time = self.HarRot2JD(self.data_sets[data_key]['Harvey_Rotation'])
        for ii, time_step in enumerate(self.time_grid.mjd[:-1]):
            north_series[ii] = np.mean(self.data_sets[data_key]['North_Size']
                                            [(tmp_time.mjd >= self.time_grid.mjd[ii]) *
                                             (tmp_time.mjd < self.time_grid.mjd[ii + 1])])
            south_series[ii] = np.mean(self.data_sets[data_key]['South_Size']
                                            [(tmp_time.mjd >= self.time_grid.mjd[ii]) *
                                             (tmp_time.mjd < self.time_grid.mjd[ii + 1])])
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

    def _assemble_pch_set(self, time_only=False):

        set_size = 0
        for kk in self.data_sets:
            set_size += self.data_sets[kk]['Harvey_Rotation'].size

        north_pch = np.zeros(set_size)
        north_dpch = np.zeros_like(north_pch)
        north_filts = np.empty_like(north_pch, dtype='<U3')

        set_time = np.zeros_like(north_pch)
        
        south_pch = np.zeros_like(north_pch)
        south_dpch = np.zeros_like(north_pch)
        south_filts = np.empty_like(north_pch, dtype='<U3')

        bookmark = 0.
        for sets in self.data_sets:
            north_pch[bookmark:bookmark+self.data_sets[sets]['North_Size'].size] = self.data_sets[sets]['North_Size']
            north_dpch[bookmark:bookmark + self.data_sets[sets]['North_Spread'].size] = \
                self.data_sets[sets]['North_Spread']
            north_filts[bookmark:bookmark + self.data_sets[sets]['North_Size'].size] = \
                np.repeat(self.filter_keys[sets], self.data_sets[sets]['North_Size'].size)
            
            south_pch[bookmark:bookmark+self.data_sets[sets]['South_Size'].size] = self.data_sets[sets]['South_Size']
            south_dpch[bookmark:bookmark + self.data_sets[sets]['South_Spread'].size] = \
                self.data_sets[sets]['South_Spread']
            south_filts[bookmark:bookmark + self.data_sets[sets]['South_Size'].size] = \
                np.repeat(self.filter_keys[sets], self.data_sets[sets]['South_Size'].size)

            set_time[bookmark:bookmark + self.data_sets[sets]['Harvey_Rotation'].size] = \
                self.data_sets[sets]['Harvey_Rotation']
            bookmark += self.data_sets[sets]['Harvey_Rotation'].size

        new_order = np.argsort(set_time)

        if time_only:
            return set_time[new_order]
        else:
            return set_time[new_order], north_pch[new_order], north_dpch[new_order], north_filts[new_order], \
                   south_pch[new_order], south_dpch[new_order], south_filts[new_order]

    def periodogram(self):

        set_time, north_pch, north_dpch, north_filts, south_pch, south_dpch, south_filts = self._assemble_pch_set()

        jd_time = self.HarRot2JD(set_time)

        # Complete the LS for each data set, nyquist freq is 66 days
        periods = np.linspace(33, 900, 6000)
        model = periodic.NaiveMultiband(BaseModel=periodic.LombScargle)
        model.optimizer.period_range = (120, 1000)
        # errors must be non zero and defined
        model.fit(jd_time.mjd, np.nan_to_num(north_pch), np.nan_to_num(north_dpch)+1e-6, north_filts)
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
            ax[1].text(20, 0.8 + offset, sets, fontsize=10, ha='right', va='top')
        ax[1].set_title('Standard Periodogram in Each Measurement', fontsize=14)
        ax[1].yaxis.set_major_formatter(plt.NullFormatter())
        ax[1].xaxis.set_major_formatter(plt.NullFormatter())
        ax[1].set_ylabel('power + offset')

        LS_multi = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=1)
        LS_multi.optimizer.period_range = (120, 1000)
        LS_multi.fit(jd_time.mjd, np.nan_to_num(north_pch), np.nan_to_num(north_dpch)+1e-6, north_filts)
        P_multi = LS_multi.periodogram(periods)
        ax[2].plot(periods, P_multi, lw=1, color='gray')

        ax[2].set_title('Multiband Periodogram', fontsize=12)
        ax[2].set_yticks([0, 0.2, 0.4])
        ax[2].set_ylim(0, 0.5)
        ax[2].yaxis.set_major_formatter(plt.NullFormatter())
        ax[2].set_xlabel('Period (days)')
        ax[2].set_ylabel('power')

        plt.savefig('/Users/mskirk/Desktop/PCH_North_perodogram.png')

        # SOUTH PCH

        periods = np.linspace(33, 900, 6000)
        model = periodic.NaiveMultiband(BaseModel=periodic.LombScargle)
        model.optimizer.period_range = (120, 1000)
        # errors must be non zero and defined
        model.fit(jd_time.mjd, np.nan_to_num(south_pch), np.nan_to_num(south_dpch)+1e-6, south_filts)
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
            ax[1].text(20, 0.8 + offset, sets, fontsize=10, ha='right', va='top')
        ax[1].set_title('Standard Periodogram in Each Measurement', fontsize=14)
        ax[1].yaxis.set_major_formatter(plt.NullFormatter())
        ax[1].xaxis.set_major_formatter(plt.NullFormatter())
        ax[1].set_ylabel('power + offset')

        LS_multi = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=1)
        LS_multi.optimizer.period_range = (120, 1000)
        LS_multi.fit(jd_time.mjd, np.nan_to_num(south_pch), np.nan_to_num(south_dpch)+1e-6, south_filts)
        P_multi = LS_multi.periodogram(periods)
        ax[2].plot(periods, P_multi, lw=1, color='gray')

        ax[2].set_title('Multiband Periodogram', fontsize=12)
        ax[2].set_yticks([0, 0.2, 0.4])
        ax[2].set_ylim(0, 0.5)
        ax[2].yaxis.set_major_formatter(plt.NullFormatter())
        ax[2].set_xlabel('Period (days)')
        ax[2].set_ylabel('power')

        plt.savefig('/Users/mskirk/Desktop/PCH_South_perodogram.png')

    def series_plot(self):

        self.regrid_time()

        self.north_pch_series = np.empty((len(self.data_names),self.time_grid.size))
        self.south_pch_series = np.empty_like(self.north_pch_series)

        fig = plt.figure(figsize=(20, 10), tight_layout=True)

        for ii, data_key in enumerate(self.data_names):
            self.north_pch_series[ii,:], self.south_pch_series[ii,:] = self.regrid_series(data_key)

        plt.subplot(2, 1, 1)
        ax1 = sns.tsplot(data=self.north_pch_series,time=self.time_grid.decimalyear, estimator=np.nanmedian)

        plt.subplot(2, 1, 2)
        ax2 = sns.tsplot(data=self.south_pch_series,time=self.time_grid.decimalyear, estimator=np.nanmedian,
                         color='darkorchid')

        ax1.set_title('Polar Coronal Hole Size', fontsize=14, fontweight='bold')
        ax1.set_ylim(0,0.07)
        ax1.set_ylabel('Fractional Surface Area', fontsize=13)
        ax1.text(2010.1, 0.061, 'Norther Polar Hole', fontsize=13, ha='right', va='top', fontweight='bold')

        ax2.set_xlabel('Year', fontsize=13)
        ax2.set_ylim(0, 0.07)
        ax2.set_ylabel('Fractional Surface Area', fontsize=13)
        ax2.text(2010.1, 0.061, 'Southern Polar Hole', fontsize=13, ha='right', va='top', fontweight='bold')

        plt.savefig('/Users/mskirk/Desktop/PCH_Series.png')