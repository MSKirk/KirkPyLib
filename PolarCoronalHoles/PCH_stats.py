import PCH_Tools
import astropy.stats.circstats as cs
from scipy.stats import circmean, circstd
import time
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
import pandas as pd
import numpy as np


def filter_select(pch_obj, filter):
    return pch_obj.where(pch_obj.Filter == filter).dropna(how='all')


def one_hr_select(pch_obj, hr_start):
    return pch_obj.where((pch_obj.Harvey_Rotation >= hr_start) & (pch_obj.Harvey_Rotation < (hr_start + 1.))).dropna(
        how='all')


def sph_center_of_mass(lats, lons, **kwargs):
    weights = kwargs.get('weights', 1)

    rect_coords = PCH_Tools.coord_spher2rec(np.ones_like(lats.value), lats, lons)
    rect_center = PCH_Tools.center_of_mass(np.transpose(np.array(rect_coords)), mass=weights)

    sphere_center = PCH_Tools.coord_rec2spher(rect_center[0], rect_center[1], rect_center[2])

    return sphere_center


def circular_rebinning(pch_obj, binsize=5):
    # Bin size in degrees; split into N and S

    north = pch_obj.where(pch_obj.StartLat > 0).dropna(how='all')
    south = pch_obj.where(pch_obj.StartLat < 0).dropna(how='all')

    # Aggregation Rules
    n_startagg = {
        'H_StartLon': {
            'Mean': lambda x: circmean(x, low=0, high=360),
            'Std': lambda x: circstd(x, low=0, high=360)},
        'StartLat': {
            'Mean': lambda x: circmean(x, low=-90, high=90),
            'Std': lambda x: circstd(x, low=-90, high=90)}
    }
    n_endagg = {
        'H_EndLon': {
            'Mean': lambda x: circmean(x, low=0, high=360),
            'Std': lambda x: circstd(x, low=0, high=360)},
        'EndLat': {
            'Mean': lambda x: circmean(x, low=-90, high=90),
            'Std': lambda x: circstd(x, low=-90, high=90)}
    }
    s_endagg = {
        'H_EndLon': {
            'Mean': lambda x: circmean(x, low=0, high=360),
            'Std': lambda x: circstd(x, low=0, high=360)},
        'EndLat': {
            'Mean': lambda x: circmean(x, low=-90, high=90),
            'Std': lambda x: circstd(x, low=-90, high=90)}
    }
    s_startagg = {
        'H_StartLon': {
            'Mean': lambda x: circmean(x, low=0, high=360),
            'Std': lambda x: circstd(x, low=0, high=360)},
        'StartLat': {
            'Mean': lambda x: circmean(x, low=-90, high=90),
            'Std': lambda x: circstd(x, low=-90, high=90)}
    }

    north_end = north.groupby(north['H_EndLon'].apply(lambda x: np.round(x / binsize))).agg(n_endagg)
    north_end.columns = ["_".join(x) for x in north_end.columns.ravel()]

    north_start = north.groupby(north['H_StartLon'].apply(lambda x: np.round(x / binsize))).agg(n_startagg)
    north_start.columns = ["_".join(x) for x in north_start.columns.ravel()]

    south_end = south.groupby(south['H_EndLon'].apply(lambda x: np.round(x / binsize))).agg(s_endagg)
    south_end.columns = ["_".join(x) for x in south_end.columns.ravel()]

    south_start = south.groupby(south['H_StartLon'].apply(lambda x: np.round(x / binsize))).agg(s_startagg)
    south_start.columns = ["_".join(x) for x in south_start.columns.ravel()]

    northern = pd.concat([north_start, north_end], join='outer', axis=1)
    southern = pd.concat([south_start, south_end], join='outer', axis=1)

    # Clean up of nan values
    values = {'H_StartLon_Std': 0, 'StartLat_Std': 0, 'H_EndLon_Std': 0, 'EndLat_Std': 0}
    northern = northern.fillna(value=values).fillna(method='ffill').fillna(method='bfill')
    southern = southern.fillna(value=values).fillna(method='ffill').fillna(method='bfill')

    bin_stats = {'N_Lon_Mean': pd.Series(
        circmean([northern.H_EndLon_Mean.values, northern.H_StartLon_Mean.values], axis=0, low=0, high=360),
        index=northern.index),
                 'N_Lon_Std': np.sqrt(northern.H_StartLon_Std ** 2 + northern.H_EndLon_Std ** 2),
                 'N_Lat_Mean': pd.Series(
                     circmean([northern.EndLat_Mean.values, northern.StartLat_Mean.values], axis=0, low=-90, high=90),
                     index=northern.index),
                 'N_Lat_Std': np.sqrt(northern.StartLat_Std ** 2 + northern.EndLat_Std ** 2),
                 'S_Lon_Mean': pd.Series(
                     circmean([southern.H_EndLon_Mean.values, southern.H_StartLon_Mean.values], axis=0, low=0,
                              high=360), index=southern.index),
                 'S_Lon_Std': np.sqrt(southern.H_StartLon_Std ** 2 + southern.H_EndLon_Std ** 2),
                 'S_Lat_Mean': pd.Series(
                     circmean([southern.EndLat_Mean.values, southern.StartLat_Mean.values], axis=0, low=-90, high=90),
                     index=southern.index),
                 'S_Lat_Std': np.sqrt(southern.StartLat_Std ** 2 + southern.EndLat_Std ** 2)
                 }

    return pd.DataFrame(data=bin_stats)


def aggregation_rebinning(hole_stats, binsize=5):
    # Aggregate mission dataframes like those returned by chole_stats

    # Aggregation Rules
    n_agg = {
        'N_lon_mean': {
            '': lambda x: circmean(x, low=0, high=360)},
        'N_lon_std': {
            '': lambda x: np.sqrt(np.nansum(x ** 2))},
        'N_lat_mean': {
            '': lambda x: circmean(x, low=-90, high=90)},
        'N_lat_lower': {
            '': lambda x: np.sqrt(np.nansum(x ** 2))},
        'N_lat_upper': {
            '': lambda x: np.sqrt(np.nansum(x ** 2))},
        'N_mean_area': {
            'agg': 'mean'},
    }

    s_agg = {
        'S_lon_mean': {
            '': lambda x: circmean(x, low=0, high=360)},
        'S_lon_std': {
            '': lambda x: np.sqrt(np.nansum(x ** 2))},
        'S_lat_mean': {
            '': lambda x: circmean(x, low=-90, high=90)},
        'S_lat_lower': {
            '': lambda x: np.sqrt(np.nansum(x ** 2))},
        'S_lat_upper': {
            '': lambda x: np.sqrt(np.nansum(x ** 2))},
        'S_mean_area': {
            'agg': 'mean'},
    }

    northern = hole_stats.groupby(hole_stats['N_lon_mean'].apply(lambda x: np.round(x / binsize))).agg(n_agg)
    northern.columns = ["".join(x) for x in northern.columns.ravel()]

    southern = hole_stats.groupby(hole_stats['S_lon_mean'].apply(lambda x: np.round(x / binsize))).agg(s_agg)
    southern.columns = ["".join(x) for x in southern.columns.ravel()]

    bin_stats = pd.concat([northern, southern], join='outer', axis=1)

    values = {'S_lon_std': 0, 'N_lon_std': 0, 'S_lat_lower': 0, 'S_lat_upper': 0, 'N_lat_lower': 0, 'N_lat_upper': 0}
    bin_stats = bin_stats.fillna(value=values).fillna(method='ffill').fillna(method='bfill')

    return bin_stats


def areaint(lats, lons):
    assert lats.size == lons.size, 'List of latitudes and longitudes are different sizes.'

    if isinstance(lats.iloc[0], u.Quantity):
        lat = np.array([lat.value for lat in lats] + [lats.iloc[0].value]) * u.deg
        lon = np.array([lon.value for lon in lons] + [lons.iloc[0].value]) * u.deg
    else:
        lat = np.append(lats, lats.iloc[0]) * u.deg
        lon = np.append(lons, lons.iloc[0]) * u.deg

    if lat.max() > 0:
        northern = True

    # Get colatitude (a measure of surface distance as an angle) and
    # azimuth of each point in segment from the center of mass.

    _, center_lat, center_lon = sph_center_of_mass(lat[:-1], lon[:-1])

    # force centroid at the N or S pole
    # if northern:
    #    center_lat = 90 * u.deg
    # else:
    #   center_lat = -90 * u.deg
    # center_lon = 0 * u.deg

    colat = np.array([distance(center_lon.to(u.deg), center_lat.to(u.deg), longi, latit).to(u.deg).value
                      for latit, longi in zip(lat, lon)]) * u.deg
    az = np.array([azimuth(center_lon.to(u.deg), center_lat.to(u.deg), longi, latit).to(u.deg).value
                   for latit, longi in zip(lat, lon)]) * u.deg

    # Calculate step sizes, taking the complementary angle where needed
    daz = np.diff(az).to(u.rad)
    daz[np.where(daz > 180 * u.deg)] -= 360. * u.deg
    daz[np.where(daz < -180 * u.deg)] += 360. * u.deg

    # Determine average surface distance for each step
    deltas = np.diff(colat) / 2.
    colats = colat[0:-1] + deltas

    # Integral over azimuth is 1-cos(colatitudes)
    integrands = (1 - np.cos(colats)) * daz

    # Integrate and return the answer as a fraction of the unit sphere.
    # Note that the sum of the integrands will include a part of 4pi.

    return np.abs(np.nansum(integrands)) / (4 * np.pi * u.rad), center_lat, center_lon


def distance(lon1, lat1, lon2, lat2):
    return np.arccos((np.sin(lat1) * np.sin(lat2)) + (np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)))


def azimuth(lon1, lat1, lon2, lat2, ratio=1):
    # Compute the bearing for either a spherical or elliptical geoid.
    # Note that for a sphere, ratio = 1, par1 = lat1, par2 = lat2 and part4 = 0.
    # Ratio = Semiminor/semimajor (b/a)
    # This forumula can be shown to be equivalent for a sphere to
    # J. P. Snyder,  "Map Projections - A Working Manual,"  US Geological
    # Survey Professional Paper 1395, US Government Printing Office,
    # Washington, DC, 1987,  pp. 29-32.

    ratio = ratio ** 2.

    part1 = np.cos(lat2) * np.sin(lon2 - lon1)
    part2 = ratio * np.cos(lat1) * np.sin(lat2)
    part3 = np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    part4 = (1 - ratio) * np.sin(lat1) * np.cos(lat2) * np.cos(lat1) / np.cos(lat2)

    bearing = np.arctan2(part1, part2 - part3 + part4)

    # Identify those cases where a pole is a starting point or a destination

    # if np.isclose(lat2.to(u.rad).value, (np.pi/2), rtol=1.7453292519943295e-08):
    #    bearing = (np.pi/2 * u.rad).to(lat2.unit)  # South pole ends

    # if np.isclose(lat2.to(u.rad).value, (np.pi / 2), rtol=1.7453292519943295e-08):
    #        bearing = -np.pi/2  # north pole ends

    # if np.isclose(lat1.to(u.rad).value, (-np.pi / 2), rtol=1.7453292519943295e-08):
    #    bearing = -np.pi/2  # south pole starts

    # if np.isclose(lat1.to(u.rad).value, (np.pi/2), rtol=1.7453292519943295e-08):
    #    bearing = (np.pi/2 * u.rad).to(lat1.unit)  # north pole starts

    return bearing


def chole_stats(pch_obj, wav_filter, binsize=5, sigma=1):
    # Sigma : number of standard deviations
    # pch_obj : the entire detection object
    # wav_filter : the mission filter, e.g. 'EIT171'

    if wav_filter not in pch_obj.Filter.unique():
        print('Please specify a correct filter.')
        return pd.DataFrame()

    measurements = filter_select(pch_obj, wav_filter)

    hole_stats = pd.DataFrame(index=measurements.Harvey_Rotation.unique(), columns=['Harvey_Rotation',
                                                                                    'Date', 'N_mean_area',
                                                                                    'S_mean_area', 'N_min_area',
                                                                                    'S_min_area', 'N_max_area',
                                                                                    'S_max_area', 'N_center_lat',
                                                                                    'N_center_lon', 'S_center_lat',
                                                                                    'S_center_lon',
                                                                                    'N_lat_mean', 'N_lat_upper',
                                                                                    'N_lat_lower', 'N_lon_mean',
                                                                                    'N_lon_std',
                                                                                    'S_lat_mean', 'S_lat_lower',
                                                                                    'S_lat_upper', 'S_lon_mean',
                                                                                    'S_lon_std'])

    for hr in measurements.Harvey_Rotation.unique():
        one_rot = one_hr_select(measurements, hr)

        binned = circular_rebinning(one_rot, binsize=binsize)

        h_loc = np.argmin(np.abs((binned.index.values * binsize) - one_rot.Harvey_Longitude.iloc[0]))
        hole_stats['N_lon_mean'][hr] = binned.N_Lon_Mean.iloc[h_loc] * u.deg
        hole_stats['N_lon_std'][hr] = binned.N_Lon_Std.iloc[h_loc] * u.deg
        hole_stats['N_lat_mean'][hr] = binned.N_Lat_Mean.iloc[h_loc] * u.deg
        hole_stats['N_lat_upper'][hr] = ((binned.N_Lat_Mean.iloc[h_loc] + binned.N_Lat_Std.iloc[h_loc]).clip(50, 90) -
                                         binned.N_Lat_Mean.iloc[h_loc]) * u.deg
        hole_stats['N_lat_lower'][hr] = ((binned.N_Lat_Mean.iloc[h_loc] - binned.N_Lat_Std.iloc[h_loc]).clip(50, 90) -
                                         binned.N_Lat_Mean.iloc[h_loc]) * u.deg
        hole_stats['S_lon_mean'][hr] = binned.S_Lon_Mean.iloc[h_loc] * u.deg
        hole_stats['S_lon_std'][hr] = binned.S_Lon_Std.iloc[h_loc] * u.deg
        hole_stats['S_lat_mean'][hr] = binned.S_Lat_Mean.iloc[h_loc] * u.deg
        hole_stats['S_lat_upper'][hr] = ((binned.S_Lat_Mean.iloc[h_loc] - binned.S_Lat_Std.iloc[h_loc]).clip(-90, -50) -
                                         binned.S_Lat_Mean.iloc[h_loc]) * u.deg
        hole_stats['S_lat_lower'][hr] = ((binned.S_Lat_Mean.iloc[h_loc] + binned.S_Lat_Std.iloc[h_loc]).clip(-90, -50) -
                                         binned.S_Lat_Mean.iloc[h_loc]) * u.deg

        hole_stats['N_mean_area'][hr], hole_stats['N_center_lat'][hr], hole_stats['N_center_lon'][hr] = areaint(
            binned.N_Lat_Mean, binned.N_Lon_Mean)
        hole_stats['N_max_area'][hr], _, _ = areaint(
            (binned.N_Lat_Mean - (sigma * binned.N_Lat_Std)).clip(50, 90) * u.deg, binned.N_Lon_Mean * u.deg)
        hole_stats['N_min_area'][hr], _, _ = areaint(
            (binned.N_Lat_Mean * u.deg + (sigma * binned.N_Lat_Std * u.deg)).clip(50, 90) * u.deg,
            binned.N_Lon_Mean * u.deg)

        hole_stats['S_mean_area'][hr], hole_stats['S_center_lat'][hr], hole_stats['S_center_lon'][hr] = areaint(
            binned.S_Lat_Mean, binned.S_Lon_Mean)
        hole_stats['S_max_area'][hr], _, _ = areaint(
            (binned.S_Lat_Mean - (sigma * binned.S_Lat_Std)).clip(-90, -50) * u.deg, binned.S_Lon_Mean * u.deg)
        hole_stats['S_min_area'][hr], _, _ = areaint(
            (binned.S_Lat_Mean + (sigma * binned.S_Lat_Std)).clip(-90, -50) * u.deg, binned.S_Lon_Mean * u.deg)

        hole_stats['Date'][hr] = PCH_Tools.hrot2date(hr).iso
        hole_stats['Harvey_Rotation'][hr] = hr
        print(np.int(((hr - measurements.Harvey_Rotation[0]) / (
                    measurements.Harvey_Rotation[-1] - measurements.Harvey_Rotation[0])) * 100.))

    hole_stats['Filter'] = wav_filter

    return hole_stats


def combine_stats(pch_dfs, sigma=1, binsize=5):
    # pch_dfs = a list of dataframes to combine

    all_stats = pd.concat(pch_dfs).sort_index()

    hole_stats = pd.DataFrame(index=all_stats.Harvey_Rotation.unique(), columns=['Harvey_Rotation',
                                                                                 'Date', 'N_mean_area', 'S_mean_area',
                                                                                 'N_min_area', 'S_min_area',
                                                                                 'N_max_area', 'S_max_area',
                                                                                 'N_center_lat', 'N_center_lon',
                                                                                 'S_center_lat', 'S_center_lon',
                                                                                 'N_lat_mean', 'N_lat_upper',
                                                                                 'N_lat_lower', 'N_lon_mean',
                                                                                 'N_lon_std', 'S_lat_mean',
                                                                                 'S_lat_lower', 'S_lat_upper',
                                                                                 'S_lon_mean', 'S_lon_std',
                                                                                 'N_mean_area_agg', 'S_mean_area_agg'])

    for hr in all_stats.Harvey_Rotation.unique():
        one_rot = one_hr_select(all_stats, hr)

        binned = aggregation_rebinning(remove_quantities(one_rot), binsize=binsize)

        h_loc = np.argmin(
            np.abs((binned.index.values * binsize) - PCH_Tools.get_harvey_lon(Time(one_rot.Date.iloc[0])).value))
        hole_stats['N_lon_mean'][hr] = binned.N_lon_mean.iloc[h_loc] * u.deg
        hole_stats['N_lon_std'][hr] = binned.N_lon_std.iloc[h_loc] * u.deg
        hole_stats['N_lat_mean'][hr] = binned.N_lat_mean.iloc[h_loc] * u.deg
        hole_stats['N_lat_upper'][hr] = binned.N_lat_upper.iloc[h_loc] * u.deg
        hole_stats['N_lat_lower'][hr] = binned.N_lat_lower.iloc[h_loc] * u.deg
        hole_stats['N_mean_area_agg'][hr] = binned.N_mean_areaagg.iloc[h_loc]

        hole_stats['S_lon_mean'][hr] = binned.S_lon_mean.iloc[h_loc] * u.deg
        hole_stats['S_lon_std'][hr] = binned.S_lon_std.iloc[h_loc] * u.deg
        hole_stats['S_lat_mean'][hr] = binned.S_lat_mean.iloc[h_loc] * u.deg
        hole_stats['S_lat_upper'][hr] = binned.S_lat_upper.iloc[h_loc] * u.deg
        hole_stats['S_lat_lower'][hr] = binned.S_lat_lower.iloc[h_loc] * u.deg
        hole_stats['S_mean_area_agg'][hr] = binned.S_mean_areaagg.iloc[h_loc]

        hole_stats['N_mean_area'][hr], hole_stats['N_center_lat'][hr], hole_stats['N_center_lon'][hr] = areaint(
            binned.N_lat_mean, binned.N_lon_mean)
        hole_stats['N_max_area'][hr], _, _ = areaint(
            (binned.N_lat_mean + (sigma * binned.N_lat_upper)).clip(50, 90) * u.deg,
            binned.N_lon_mean * u.deg)
        hole_stats['N_min_area'][hr], _, _ = areaint(
            (binned.N_lat_mean + (sigma * binned.N_lat_lower)).clip(50, 90) * u.deg,
            binned.N_lon_mean * u.deg)

        hole_stats['S_mean_area'][hr], hole_stats['S_center_lat'][hr], hole_stats['S_center_lon'][hr] = areaint(
            binned.S_lat_mean, binned.S_lon_mean)
        hole_stats['S_max_area'][hr], _, _ = areaint(
            (binned.S_lat_mean + (sigma * binned.S_lat_upper)).clip(-90, -50) * u.deg,
            binned.S_lon_mean * u.deg)
        hole_stats['S_min_area'][hr], _, _ = areaint(
            (binned.S_lat_mean + (sigma * binned.S_lat_lower)).clip(-90, -50) * u.deg,
            binned.S_lon_mean * u.deg)

        hole_stats['Date'][hr] = PCH_Tools.hrot2date(hr).iso
        hole_stats['Harvey_Rotation'][hr] = hr
        print(np.int(((hr - all_stats.Harvey_Rotation.iloc[0]) / (
                all_stats.Harvey_Rotation.iloc[-1] - all_stats.Harvey_Rotation.iloc[0])) * 100.))

    return hole_stats


def df_chole_stats_hem(pch_df, binsize=5, sigma=1.0, wave_filter='AIA171', northern=True, window_size='33D'):
    # **** anything that is _calc is a more expensive opperation ***

    # Processing time
    tstart = time.time()

    # Northern Mean, Upper, and Lower ---------------------------------------------------
    df_mean, df_upper, df_lower = df_pre_process(pch_df, northern=northern, wave_filter=wave_filter, sigma=sigma,
                                                 binsize=binsize, window_size='33D')

    # Center of Mass Calculation *** df_CoM_calc *** is the expensive function
    com_mean = df_CoM_calc(df_mean, window_size=window_size)
    com_upper = df_CoM_calc(df_upper, window_size=window_size)
    com_lower = df_CoM_calc(df_lower, window_size=window_size)

    df_mean = pd.concat([df_mean, com_mean], axis=1)
    df_upper = pd.concat([df_upper, com_upper], axis=1)
    df_lower = pd.concat([df_lower, com_lower], axis=1)

    df_mean, df_upper, df_lower = df_colat_az(df_mean, df_upper, df_lower)

    # Area Calculation  *** df_area_calc *** is the expensive function
    mean_area = df_area_calc(df_mean[['Lon', 'colat_rad', 'az_rad']], window_size=window_size)
    upper_area = df_area_calc(df_upper[['Lon', 'colat_rad', 'az_rad']], window_size=window_size)
    lower_area = df_area_calc(df_lower[['Lon', 'colat_rad', 'az_rad']], window_size=window_size)

    if northern:
        df_hem = pd.concat([mean_area.rename('N_mean_area'),
                            com_mean.rename(columns={'c_lat': 'N_mean_lat', 'c_lon': 'N_mean_lon'}),
                            upper_area.rename('N_upper_area'),
                            com_upper.rename(columns={'c_lat': 'N_upper_lat', 'c_lon': 'N_upper_lon'}),
                            lower_area.rename('N_lower_area'),
                            com_lower.rename(columns={'c_lat': 'N_lower_lat', 'c_lon': 'N_lower_lon'})], axis=1)

    else:
        df_hem = pd.concat([mean_area.rename('S_mean_area'),
                            com_mean.rename(columns={'c_lat': 'S_mean_lat', 'c_lon': 'S_mean_lon'}),
                            upper_area.rename('S_upper_area'),
                            com_upper.rename(columns={'c_lat': 'S_upper_lat', 'c_lon': 'S_upper_lon'}),
                            lower_area.rename('S_lower_area'),
                            com_lower.rename(columns={'c_lat': 'S_lower_lat', 'c_lon': 'S_lower_lon'})], axis=1)

    # # ---- Mean Only... to save time as well as upper and lower don't yield consistent results ----
    # df_mean, _df_up, _df_lo = df_pre_process(pch_df, northern=northern, wave_filter=wave_filter, sigma=sigma,
    #                                          binsize=binsize, window_size=window_size)
    #
    # # Center of Mass Calculation *** df_CoM_calc *** is the expensive function
    # com_mean = df_CoM_calc(df_mean, window_size=window_size)
    #
    # df_mean = pd.concat([df_mean, com_mean], axis=1)
    #
    # df_mean['colat_rad'] = df_colat(df_mean, ref_lat=df_mean.c_lat, ref_lon=df_mean.c_lon)
    # df_mean['az_rad'] = df_azimuth(df_mean, ref_lat=df_mean.c_lat, ref_lon=df_mean.c_lon)
    #
    # # Area Calculation  *** df_area_calc *** is the expensive function
    # mean_area = df_area_calc(df_mean[['Lon', 'colat_rad', 'az_rad']], window_size=window_size)
    #
    # if northern:
    #     df_hem = pd.concat([mean_area.rename('N_mean_area'),
    #                         com_mean.rename(columns={'c_lat': 'N_mean_lat', 'c_lon': 'N_mean_lon'})], axis=1)
    #
    # else:
    #     df_hem = pd.concat([mean_area.rename('S_mean_area'),
    #                         com_mean.rename(columns={'c_lat': 'S_mean_lat', 'c_lon': 'S_mean_lon'})], axis=1)
    # # End Mean Calc.

    elapsed_time = time.time() - tstart
    print('Compute time for {:s} - northern={:s} : {:1.0f} sec ({:1.1f} min)'.format(wave_filter, str(northern),
                                                                                     elapsed_time, elapsed_time / 60))

    return df_hem


def df_concat_stats_hem(pch_df, binsize=5, sigma=1.0, northern=True, window_size='33D'):
    # **** anything that is _calc is a more expensive opperation ***

    # Processing time
    tstart = time.time()

    resampled_dfs = dict()

    # # Mean, Upper, and Lower ---------------------------------------------------
    # for wave_filter in pch_df.Filter.unique():
    #     df_mean, df_upper, df_lower = df_pre_process(pch_df, northern=northern, wave_filter=wave_filter, sigma=sigma,
    #                                              binsize=binsize, window_size=window_size, resample=True)
    #
    #     resampled_dfs[wave_filter] = [df_mean, df_upper, df_lower]
    #
    # df_mean = pch_dict_concat(resampled_dfs, index=0).sort_index()
    # df_mean.index = df_mean.index.rename('DateTime')
    # df_upper = pch_dict_concat(resampled_dfs, index=1).sort_index()
    # df_upper.index = df_upper.index.rename('DateTime')
    # df_lower = pch_dict_concat(resampled_dfs, index=2).sort_index()
    # df_lower.index = df_lower.index.rename('DateTime')
    #
    # # Center of Mass Calculation *** df_CoM_calc *** is the expensive function
    # com_mean = df_CoM_calc(df_mean, window_size=window_size)
    # com_upper = df_CoM_calc(df_upper, window_size=window_size)
    # com_lower = df_CoM_calc(df_lower, window_size=window_size)
    #
    # df_mean = pd.concat([df_mean, com_mean], axis=1)
    # df_upper = pd.concat([df_upper, com_upper], axis=1)
    # df_lower = pd.concat([df_lower, com_lower], axis=1)
    #
    # df_mean, df_upper, df_lower = df_colat_az(df_mean, df_upper, df_lower)
    #
    # # Area Calculation  *** df_area_calc *** is the expensive function
    # mean_area = df_area_calc(df_mean[['Lon', 'colat_rad', 'az_rad']], window_size=window_size)
    # upper_area = df_area_calc(df_upper[['Lon', 'colat_rad', 'az_rad']], window_size=window_size)
    # lower_area = df_area_calc(df_lower[['Lon', 'colat_rad', 'az_rad']], window_size=window_size)
    #
    # if northern:
    #     df_hem = pd.concat([mean_area.rename('N_mean_area'),
    #                         com_mean.rename(columns={'c_lat': 'N_mean_lat', 'c_lon': 'N_mean_lon'}),
    #                         upper_area.rename('N_upper_area'),
    #                         com_upper.rename(columns={'c_lat': 'N_upper_lat', 'c_lon': 'N_upper_lon'}),
    #                         lower_area.rename('N_lower_area'),
    #                         com_lower.rename(columns={'c_lat': 'N_lower_lat', 'c_lon': 'N_lower_lon'})], axis=1)
    #
    # else:
    #     df_hem = pd.concat([mean_area.rename('S_mean_area'),
    #                         com_mean.rename(columns={'c_lat': 'S_mean_lat', 'c_lon': 'S_mean_lon'}),
    #                         upper_area.rename('S_upper_area'),
    #                         com_upper.rename(columns={'c_lat': 'S_upper_lat', 'c_lon': 'S_upper_lon'}),
    #                         lower_area.rename('S_lower_area'),
    #                         com_lower.rename(columns={'c_lat': 'S_lower_lat', 'c_lon': 'S_lower_lon'})], axis=1)

    # # ---- Mean Only... to save time as well as upper and lower don't yield consistent results ----
    for wave_filter in pch_df.Filter.unique():
        df_mean, _df_up, _df_lo = df_pre_process(pch_df, northern=northern, wave_filter=wave_filter, sigma=sigma,
                                                 binsize=binsize, window_size='11D', resample=True)

        resampled_dfs[wave_filter] = [df_mean]

    df_mean = pch_dict_concat(resampled_dfs, index=0).sort_index()
    df_mean.index = df_mean.index.rename('DateTime')
    df_mean = df_mean.groupby('bin').resample('11D').median()[['Lat', 'Lon']].dropna(how='all').reset_index().set_index(
        ['DateTime']).sort_index()

    # Center of Mass Calculation *** df_CoM_calc *** is the expensive function
    com_mean = df_CoM_calc(df_mean, window_size=window_size)

    df_mean = pd.concat([df_mean, com_mean], axis=1)

    df_mean['colat_rad'] = df_colat(df_mean, ref_lat=df_mean.c_lat, ref_lon=df_mean.c_lon)
    df_mean['az_rad'] = df_azimuth(df_mean, ref_lat=df_mean.c_lat, ref_lon=df_mean.c_lon)

    # Area Calculation  *** df_area_calc *** is the expensive function
    mean_area = df_area_calc(df_mean[['Lon', 'colat_rad', 'az_rad']], window_size=window_size)

    if northern:
        df_hem = pd.concat([mean_area.rename('N_mean_area'),
                            com_mean.rename(columns={'c_lat': 'N_mean_lat', 'c_lon': 'N_mean_lon'})], axis=1)

    else:
        df_hem = pd.concat([mean_area.rename('S_mean_area'),
                            com_mean.rename(columns={'c_lat': 'S_mean_lat', 'c_lon': 'S_mean_lon'})], axis=1)
    # # End Mean Calc.

    elapsed_time = time.time() - tstart
    print('Compute time for northern={:s} : {:1.0f} sec ({:1.1f} min)'.format(str(northern), elapsed_time, elapsed_time / 60))

    return df_hem


def mean_hole_coords(pch_df, date, binsize=5, window_size='33D', sigma=1, northern=True, fit_method='spline'):
    #date = date string or list of date strings

    # Processing time
    tstart = time.time()

    resampled_dfs = dict()

    begin = (pd.to_datetime(date) - pd.to_timedelta('16.5D'))
    end = (pd.to_datetime(date) + pd.to_timedelta('16.5D'))

    # Mean, Upper, and Lower ---------------------------------------------------
    for wave_filter in pch_df.Filter.unique():
        df_mean, df_upper, df_lower = df_pre_process(pch_df, northern=northern, wave_filter=wave_filter, sigma=sigma,
                                                 binsize=binsize, window_size=window_size, resample=True)

        resampled_dfs[wave_filter] = [df_mean, df_upper, df_lower]

    df_mean = pch_dict_concat(resampled_dfs, index=0).sort_index()[begin: end].sort_values('Lon')
    s_mean = pd.Series(data=df_mean.Lat.values, index=df_mean.Lon.values)
    df_upper = pch_dict_concat(resampled_dfs, index=1).sort_index()[begin: end].sort_values('Lon')
    s_upper = pd.Series(data=df_upper.Lat.values, index=df_upper.Lon.values)
    df_lower = pch_dict_concat(resampled_dfs, index=2).sort_index()[begin: end].sort_values('Lon')
    s_lower = pd.Series(data=df_lower.Lat.values, index=df_lower.Lon.values)

    lons = PCH_Tools.get_harvey_lon(Time(date)).value

    s_mean = s_mean.append(pd.Series(index=[PCH_Tools.get_harvey_lon(Time(date)).value])).sort_index().interpolate(method=fit_method)
    s_upper = s_upper.append(pd.Series(index=[PCH_Tools.get_harvey_lon(Time(date)).value])).sort_index().interpolate(method=fit_method)
    s_lower = s_lower.append(pd.Series(index=[PCH_Tools.get_harvey_lon(Time(date)).value])).sort_index().interpolate(method=fit_method)

    return s_mean[lons], s_upper[lons], s_lower[lons]


def pch_dict_concat(pch_dict, index=0):

    hem_df = pd.DataFrame(columns=pch_dict[next(iter(pch_dict))][index].columns)
    for df_key in pch_dict:
        hem_df = pd.concat([hem_df, pch_dict[df_key][index]])

    return hem_df


def df_chole_stats(binsize=5, sigma=1.0, wav_filter='AIA171'):
    # **** anything with "***" is a more expensive opperation ***
    global pch_df
    # ***
    n_df = df_chole_stats_hem(pch_df, northern=True, binsize=binsize, sigma=sigma, wav_filter=wav_filter)
    # ***
    s_df = df_chole_stats_hem(pch_df, northern=False, binsize=binsize, sigma=sigma, wav_filter=wav_filter)

    # Return both in list (makes it easier to collect afterwards from parallel processing)
    return [n_df, s_df]


def df_pre_process(pch_df, northern=True, resample=False, **kwargs):
    pch_df['bin'] = np.floor(pch_df.Lon / kwargs.get('binsize', 5))

    if northern:
        df_mean = pch_df[(pch_df.Lat > 0)].groupby(['bin', 'Filter'])[['Lat', 'Lon']].rolling(kwargs.get('window_size'),
                                                                                              win_type='boxcar').median().xs(
            kwargs.get('wave_filter'), level=1).reset_index().set_index(['DateTime']).sort_index()
        df_std = pch_df[(pch_df.Lat > 0)].groupby(['bin', 'Filter'])[['Lat', 'Lon']].rolling(kwargs.get('window_size'),
                                                                                             win_type='boxcar').std().xs(
            kwargs.get('wave_filter'), level=1).reset_index().set_index(['DateTime']).sort_index()

    else:
        df_mean = pch_df[(pch_df.Lat < 0)].groupby(['bin', 'Filter'])[['Lat', 'Lon']].rolling(kwargs.get('window_size'),
                                                                                              win_type='boxcar').median().xs(
            kwargs.get('wave_filter'), level=1).reset_index().set_index(['DateTime']).sort_index()
        df_std = pch_df[(pch_df.Lat < 0)].groupby(['bin', 'Filter'])[['Lat', 'Lon']].rolling(kwargs.get('window_size'),
                                                                                             win_type='boxcar').std().xs(
            kwargs.get('wave_filter'), level=1).reset_index().set_index(['DateTime']).sort_index()

    if resample:
        df_mean = df_mean.groupby('bin').resample('1D').median()[['Lat', 'Lon']].dropna(how='all').reset_index().set_index(['DateTime']).sort_index()

        df_std = df_std.groupby('bin').resample('1D').std()[['Lat', 'Lon']].dropna(how='all').reset_index().set_index(['DateTime']).sort_index()

    df_upper = df_mean + (kwargs.get('sigma', 1) * df_std).drop(columns='bin')
    df_upper.Lat[df_upper.Lat > 90] = 90
    df_upper.Lat[df_upper.Lat < 50] = 50
    df_upper.Lon[df_upper.Lat >= 360] -= 360
    df_upper.Lon[df_upper.Lat <= 0] += 360

    df_lower = df_mean - (kwargs.get('sigma', 1) * df_std).drop(columns='bin')
    df_lower.Lat[df_lower.Lat > 90] = 90
    df_lower.Lat[df_lower.Lat < 50] = 50
    df_lower.Lon[df_lower.Lat >= 360] -= 360
    df_lower.Lon[df_lower.Lat <= 0] += 360

    return df_mean, df_upper, df_lower


def df_colat_az(df_mean, df_upper, df_lower):
    # Define Colat and azimuth in radians
    df_mean['colat_rad'] = df_colat(df_mean, ref_lat=df_mean.c_lat, ref_lon=df_mean.c_lon)
    df_mean['az_rad'] = df_azimuth(df_mean, ref_lat=df_mean.c_lat, ref_lon=df_mean.c_lon)

    df_upper['colat_rad'] = df_colat(df_upper, ref_lat=df_upper.c_lat, ref_lon=df_upper.c_lon)
    df_upper['az_rad'] = df_azimuth(df_upper, ref_lat=df_upper.c_lat, ref_lon=df_upper.c_lon)

    df_lower['colat_rad'] = df_colat(df_lower, ref_lat=df_lower.c_lat, ref_lon=df_lower.c_lon)
    df_lower['az_rad'] = df_azimuth(df_lower, ref_lat=df_lower.c_lat, ref_lon=df_lower.c_lon)

    return df_mean, df_upper, df_lower


def df_area_calc(df, window_size='33D'):
    df_series = df.reset_index().set_index(['DateTime']).sort_index()
    df_series['idx'] = range(len(df_series))

    return df_series.idx.rolling(window_size).apply(lambda x: _area_apply(x, df_series), raw=True)


def _area_apply(elems, mydf):
    # From raphael
    # elems is what's passed by the "x" variable of the lambda function
    # it is a list of whatever indices will be in the '33D' window and that are used
    # to reference the rows of the database (i.e. here the dataframe “mydf”)

    df_series = mydf.iloc[elems]
    print('Percent Complete: ', "{:3.1f}".format(np.round((np.min(elems)*1000.)/mydf.shape[0])*0.1))

    daz = np.diff(df_series.sort_values(by=['Lon'])['az_rad'])
    daz[np.where(daz > np.deg2rad(180))] -= np.deg2rad(360.)
    daz[np.where(daz < np.deg2rad(-180))] += np.deg2rad(360.)
    #
    colat_sorted = df_series.sort_values(by=['Lon'])['colat_rad']
    deltas = np.diff(colat_sorted) / 2.
    colats = colat_sorted[0:-1] + deltas
    integrands = (1 - np.cos(colats)) * daz

    return np.abs(integrands.sum()) / (4 * np.pi)


def df_CoM_calc_mike(df, window_size='33D'):
    df_series = df.reset_index().set_index(['DateTime']).sort_index()
    df_series['idx'] = range(len(df_series))

    com_df = pd.DataFrame()
    com_df['c_lat'] = df_series.idx.rolling(window_size).apply(lambda x: _center_lat_apply(x, df_series), raw=True)
    com_df['c_lon'] = df_series.idx.rolling(window_size).apply(lambda x: _center_lon_apply(x, df_series), raw=True)

    return com_df


def _center_lat_apply_mike(elems, mydf, **kwargs):
    df_series = mydf.iloc[elems]
    weights = kwargs.get('weights', 1)
    rr = 1

    xx = rr * np.cos(np.deg2rad(df_series.Lat)) * np.cos(np.deg2rad(df_series.Lon))
    yy = rr * np.cos(np.deg2rad(df_series.Lat)) * np.sin(np.deg2rad(df_series.Lon))
    zz = rr * np.sin(np.deg2rad(df_series.Lat))

    # xx_center technically is xx.sum()/np.sum(weights)

    return np.arctan(zz.mean() / np.sqrt(xx.mean() ** 2 + yy.mean() ** 2))


def _center_lon_apply_mike(elems, mydf, **kwargs):
    df_series = mydf.iloc[elems]
    weights = kwargs.get('weights', 1)
    rr = 1

    xx = rr * np.cos(np.deg2rad(df_series.Lat)) * np.cos(np.deg2rad(df_series.Lon))
    yy = rr * np.cos(np.deg2rad(df_series.Lat)) * np.sin(np.deg2rad(df_series.Lon))

    # xx_center technically is xx.sum()/np.sum(weights)

    return np.arctan(yy.mean() / xx.mean())


def df_CoM_calc(df, window_size='33D'):
    df_series = df.reset_index().set_index(['DateTime']).sort_index()
    df_series['idx'] = range(len(df_series))

    rr = 1

    # Even if you planned to use a rolling-window-dependent "weights", You still don't need to process
    # the coordinate conversion inside the rolling window as you'd otherwise calculate the same many times.
    # In fact, for a  given rolling window of say, 100 elements, you end up calculating them 99x more than you need...
    # This completely inflated your compute time.
    # Instead, calculate them here once and pass them also as a look-up variable.
    # Also I suggest to create these coordinates as new tables in your DF as you seem to use them a lot.
    # This way it will be contained in the df_series too and if you  have to weight them, if need be, this will
    # only be a cheap multiplication.
    cos_lat = rr * np.cos(np.deg2rad(df_series.Lat))
    xx = cos_lat * np.cos(np.deg2rad(df_series.Lon))
    yy = cos_lat * np.sin(np.deg2rad(df_series.Lon))
    zz = rr * np.sin(np.deg2rad(df_series.Lat))

    com_df = pd.DataFrame()
    com_df['c_lat'] = df_series.idx.rolling(window_size).apply(lambda x: _center_lat_apply(x, (xx, yy, zz)), raw=True)

    com_df['c_lon'] = df_series.idx.rolling(window_size).apply(lambda x: _center_lon_apply(x, (xx, yy, zz)), raw=True)

    com_df.c_lon[com_df.c_lon < 0] += 2*np.pi
    com_df.c_lon[com_df.c_lon > 2*np.pi] -= 2*np.pi

    return com_df


def _center_lon_apply(elems, coords, **kwargs):
    xx, yy, zz = coords[0].iloc[elems], coords[1].iloc[elems], coords[2].iloc[elems]

    return np.arctan(yy.mean() / xx.mean())


def _center_lat_apply(elems, coords, **kwargs):
    xx, yy, zz = coords[0].iloc[elems], coords[1].iloc[elems], coords[2].iloc[elems]

    return np.arctan(zz.mean() / np.sqrt(xx.mean() ** 2 + yy.mean() ** 2))


def df_colat(df, ref_lat=np.deg2rad(90), ref_lon=np.deg2rad(0)):
    # Returns in Radians

    return np.arccos((np.sin(ref_lat) * np.sin(np.deg2rad(df.Lat))) + (np.cos(ref_lat) * np.cos(
        np.deg2rad(df.Lat)) * np.cos(np.deg2rad(df.Lon) - ref_lon)))


def df_azimuth(df, ref_lat=np.deg2rad(90), ref_lon=np.deg2rad(0)):
    # Returns in Radians
    # elipsoidal ratio
    ratio = 1

    ratio = ratio ** 2.
    part1 = np.cos(np.deg2rad(df.Lat)) * np.sin(np.deg2rad(df.Lon) - ref_lon)
    part2 = ratio * np.cos(ref_lat) * np.sin(np.deg2rad(df.Lat))
    part3 = np.sin(ref_lat) * np.cos(np.deg2rad(df.Lat)) * np.cos(np.deg2rad(df.Lon) - ref_lon)
    part4 = (1 - ratio) * np.sin(ref_lat) * np.cos(np.deg2rad(df.Lat)) * np.cos(ref_lat) / np.cos(np.deg2rad(df.Lat))

    return np.arctan2(part1, part2 - part3 + part4)


def remove_quantities(df):
    # Return a dataframe without quantiies

    for key in df.keys():
        if isinstance(df[key].iloc[0], u.Quantity):
            df[key] = np.array([itm.value for itm in df[key]])

    return df


def test_run_area_calc(mydf, window_size='33D'):
    tstart = time.time()
    res = dfs.idx.rolling(window_size).apply(lambda x: _area_apply_raphael(x, mydf))
    elapsed_time = time.time() - tstart
    print('Compute time: {:1.0f} sec ({:1.1f} min)'.format(elapsed_time, elapsed_time / 60))
    return res


def read_all_pch_df():
    # Hardcoded shortcut

    aia171 = pd.read_pickle('/Users/mskirk/data/PCH_Project/AIA171.pkl')
    aia193 = pd.read_pickle('/Users/mskirk/data/PCH_Project/AIA193.pkl')
    aia211 = pd.read_pickle('/Users/mskirk/data/PCH_Project/AIA211.pkl')
    aia304 = pd.read_pickle('/Users/mskirk/data/PCH_Project/AIA304.pkl')

    eit171 = pd.read_pickle('/Users/mskirk/data/PCH_Project/EIT171.pkl')
    eit195 = pd.read_pickle('/Users/mskirk/data/PCH_Project/EIT195.pkl')
    eit304 = pd.read_pickle('/Users/mskirk/data/PCH_Project/EIT304.pkl')

    euvi171 = pd.read_pickle('/Users/mskirk/data/PCH_Project/EUVI171.pkl')
    euvi195 = pd.read_pickle('/Users/mskirk/data/PCH_Project/EUVI195.pkl')
    euvi304 = pd.read_pickle('/Users/mskirk/data/PCH_Project/EUVI304.pkl')

    swap174 = pd.read_pickle('/Users/mskirk/data/PCH_Project/swap174.pkl')

    return aia171, aia193, aia211, aia304, eit171, eit195, eit304, euvi171, euvi195, euvi304, swap174
