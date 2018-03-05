import pandas as pd
from astropy.time import Time


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