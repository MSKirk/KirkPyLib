import pandas as pd
from astropy.time import Time
import astropy.units as u
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np
import datetime

def read_ccmc_model(filename):
    col_names = [['Time', 'R', 'Lat', 'Lon', 'V_r', 'V_lon', 'V_lat', 'B_r', 'B_lon', 'B_lat', 'N', 'T', 'E_r', 'E_lon',
                  'E_lat', 'V', 'B', 'P_ram', 'BP'],
                 ['u.d', 'u.au', 'u.deg', 'u.deg', 'u.km / u.s', 'u.km / u.s', 'u.km / u.s', 'u.nT', 'u.nT', 'u.nT', 'u.cm ** -3', 'u.K',
                  'u.mV / u.m', 'u.mV / u.m', 'u.mV / u.m', 'u.km / u.s', 'u.nT', 'u.nPa', '']]

    tuples = list(zip(*col_names))
    col_index = pd.MultiIndex.from_tuples(tuples, names=['measurement', 'units'])

    missing_val = -1.09951e+12

    file = open(filename, 'r')
    for line in file:
        if 'Start Date' in line:
            st_dt=line

    start_date = Time(st_dt[st_dt.find(':')+2:-1].replace('/', '-').replace('  ', ' '))

    df = pd.read_csv(filename, names=col_index, na_values=missing_val , delimiter='\s+', comment='#')

    time_stamp = pd.Series([(start_date + (t * eval(df.Time.keys()[0]))).datetime for t in df.Time[df.Time.keys()[0]]])

    df.set_index(time_stamp)

    # to access the unit information for, e.g., Time:
    # import astropy.units as u
    # eval(df.Time.keys()[0])
    # df.Time[df.Time.keys()[0]][0]*eval(df.Time.keys()[0])

    return df


def elbow_plot_kmeans(dataframe):
    # dropping Time from clustering
    array_data = dataframe.drop('Time', axis=1).as_matrix()
    Nc = range(1, 20)
    distortions = []
    score = []

    kmeans = [KMeans(init='k-means++', n_clusters=i, n_init=150) for i in Nc]

    for ii in range(len(kmeans)):
        kmmodel = kmeans[ii].fit(array_data)
        score.append(kmmodel.score(array_data))
        distortions.append(sum(np.min(cdist(array_data, kmmodel.cluster_centers_, 'euclidean'), axis=1)) / array_data.shape[0])

    plt.plot(Nc, np.gradient(score) / np.gradient(score).max())
    plt.plot(Nc, np.array(distortions) / np.array(distortions).max())
    plt.xlabel('Number of Clusters')
    plt.ylabel('Group Distortion Score')
    plt.title('Elbow Curve')
    plt.show()


def read_omni_data(filename):
    # to read a omni asc text file from the archive

    # See ftp://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/hroformat.txt for more information

    col_names = [['Year', 'Day', 'Hour', 'Minute', 'ID_IMF', 'ID_SWPlasma', 'N_points_IMF', 'N_points_Plasma',
                 'Percent_interp', 'Timeshift', 'RMS_Timeshift', 'RMS_Phase', 'Delta_Time', 'B_magnitude', 'Bx_GSE_GSM',
                 'By_GSE', 'Bz_GSE', 'By_GSM', 'Bz_GSM', 'RMS_SD_B', 'RMS_SD_field', 'Flow_speed', 'Vx_GSE', 'Vy_GSE',
                 'Vz_GSE', 'Proton_Density', 'Temp', 'Flow_pressure', 'E_field', 'Plasma_beta', 'Alfven_mach_num',
                 'X_GSE', 'Y_GSE', 'Z_GSE', 'BSN_Xgse', 'BSN_Ygse', 'BSN_Zgse', 'AE_index', 'AL_index', 'AU_index',
                 'SYM/D_index', 'SYM/H_index', 'ASY/D_index', 'ASY/H_index', 'PC_N_index', 'Mag_mach_num'],

                 ['u.yr', 'u.d', 'u.h', 'u.min', '', '', '', '', 'u.percent', 'u.s', 'u.s', '', 'u.s', 'u.nT', 'u.nT',
                  'u.nT', 'u.nT', 'u.nT', 'u.nT', 'u.nT', 'u.nT', 'u.km/u.s', 'u.km/u.s', 'u.km/u.s', 'u.km/u.s',
                  'u.cm**-3', 'u.K', 'u.nPa', 'u.mV/u.m', '', '', '', '', '', '', '', '', 'u.nT', 'u.nT', 'u.nT','u.nT',
                  'u.nT', 'u.nT', 'u.nT', '', '']]

    tuples = list(zip(*col_names))
    col_index = pd.MultiIndex.from_tuples(tuples, names=['measurement', 'units'])

    df = pd.read_csv(filename, names=col_index, delimiter='\s+')

    # Cleaning up NaN values for each measurement

    nan_values = [['Year', 'Day', 'Hour', 'Minute', 'ID_IMF', 'ID_SWPlasma', 'N_points_IMF', 'N_points_Plasma',
                 'Percent_interp', 'Timeshift', 'RMS_Timeshift', 'RMS_Phase', 'Delta_Time', 'B_magnitude', 'Bx_GSE_GSM',
                 'By_GSE', 'Bz_GSE', 'By_GSM', 'Bz_GSM', 'RMS_SD_B', 'RMS_SD_field', 'Flow_speed', 'Vx_GSE', 'Vy_GSE',
                 'Vz_GSE', 'Proton_Density', 'Temp', 'Flow_pressure', 'E_field', 'Plasma_beta', 'Alfven_mach_num',
                 'X_GSE', 'Y_GSE', 'Z_GSE', 'BSN_Xgse', 'BSN_Ygse', 'BSN_Zgse', 'AE_index', 'AL_index', 'AU_index',
                 'SYM/D_index', 'SYM/H_index', 'ASY/D_index', 'ASY/H_index', 'PC_N_index', 'Mag_mach_num'],

                  [9999, 999, 99, 99, 99, 99, 999, 999, 999, 999999, 999999, 99.99, 999999, 9999.99, 9999.99, 9999.99,
                   9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 99999.9, 99999.9, 99999.9, 99999.9, 999.99, 9999999.,
                   99.99, 999.99, 999.99, 999.9, 9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 9999.99, 999, 999, 99, 99,
                   99, 99, 99, 9.99, 99.9]]

    nan_dic = dict(zip(*nan_values))

    for key in df.keys():
        df[key[0]] = df[key[0]].replace(nan_dic[key[0]], np.nan)

    time_stamp = pd.Series([datetime.datetime(int(df['Year']['u.yr'][ii]), 1, 1) +
                            datetime.timedelta(int(df['Day']['u.d'][ii]) - 1, hours=int(df['Hour']['u.h'][ii]),
                                               minutes=int(df['Minute']['u.min'][ii])) for ii in df.index])

    df.set_index(time_stamp)

    return df