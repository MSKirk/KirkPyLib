import pandas as pd
import astropy.units as u
from astropy.time import Time


def read_ccmc_model(filename):
    col_names = ['Time', 'R', 'Lat', 'Lon', 'V_r', 'V_lon', 'V_lat', 'B_r', 'B_lon', 'B_lat', 'N', 'T', 'E_r', 'E_lon', 'E_lat', 'V', 'B', 'P_ram', 'BP']
    units = [u.d, u.au, u.deg, u.deg, u.km/u.s, u.km/u.s, u.km/u.s, u.nT, u.nT, u.nT, u.cm**-3, u.K, u.mV/u.m, u.mV/u.m, u.mV/u.m, u.km/u.s, u.nT, u.nPa, ]
    missing_val = -1.09951e+12

    file = open(filename, 'r')
    for line in file:
        if 'Start Date' in line:
            st_dt=line
        else:
            st_dt = None

    start_date = Time(st_dt[st_dt.find(':')+2:-1].replace('/', '-').replace('  ', ' '))

    df = pd.read_csv(filename, names=col_names, na_values=missing_val , delimiter='\s+', comment='#')

    time_stamp = pd.Series([((t*u.d)+start_date).datetime for t in df.Time])

    df.set_index(time_stamp)

