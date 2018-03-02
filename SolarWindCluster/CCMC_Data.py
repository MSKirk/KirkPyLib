import pandas as pd
import astropy.units as u


def read_ccmc_model(filename):
   col_names = ['Time', 'R', 'Lat', 'Lon', 'V_r', 'V_lon', 'V_lat', 'B_r', 'B_lon', 'B_lat', 'N', 'T', 'E_r', 'E_lon', 'E_lat', 'V', 'B', 'P_ram', 'BP']
   units = [u.d, u.au, u.deg, u.deg, u.km/u.s, u.km/u.s, u.km/u.s, u.nt, u.nt, u.nt, u.cm**-3, u.K, u.mV/u.m, u.mV/u.m, u.mV/u.m, u.km/u.s, u.nt, u.nPa, ]
   missing_val = -1.09951e+12
