
import pandas as pd
import io
import requests
from bs4 import BeautifulSoup


def aia_eff_area(wavelnth, target_time, ratio=True, filename=None):

    if filename:
        response_table = filename
    else:
        url = 'https://hesperia.gsfc.nasa.gov/ssw/sdo/aia/response/'
        soup = BeautifulSoup(requests.get(url).text, 'html.parser')
        all_versions = [node.get('href') for node in soup.find_all('a') if
                        node.get('href').endswith('_response_table.txt')]
        table_url = url + sorted([table_files for table_files in all_versions if table_files.startswith('aia_V')])[-1]
        tbl = requests.get(table_url).content
        response_table = io.StringIO(tbl.decode('utf-8'))

    dat = pd.read_csv(response_table, sep='\s+',
                      parse_dates=[1], infer_datetime_format=True, index_col=1)

    eff_area_series = dat[dat.WAVELNTH == wavelnth].EFF_AREA

    launch_value = eff_area_series[eff_area_series.index.min()]
    current_estimate = eff_area_series[eff_area_series.index.max()]

    eff_area_series = eff_area_series.reindex(pd.to_datetime(list(eff_area_series.index.values) +
                                                             [pd.to_datetime('2030-05-01 00:00:00.000')]))
    eff_area_series[-1] = current_estimate

    if ratio:
        return time_interpolate(eff_area_series, target_time)/launch_value
    else:
        return time_interpolate(eff_area_series, target_time)


def time_interpolate(ts, target_time):
    ts1 = ts.sort_index()
    b = (ts1.index > target_time).argmax()  # index of first entry after target
    s = ts1.iloc[b-1:b+1]
    # Insert empty value at target time.
    s = s.reindex(pd.to_datetime(list(s.index.values) + [pd.to_datetime(target_time)]))
    return s.interpolate('time').loc[target_time]
