import astropy.stats.circstats as cs
import matplotlib.pyplot as plt
import astropy.units as u
import pandas as pd
import numpy as np

def filter_select(pch_obj, filter):
    return pch_obj.where(pch_obj.Filter == filter).dropna(how='all')


def one_hr_select(pch_obj, hr_start):
    return pch_obj.where((pch_obj.Harvey_Rotation >= hr_start) & (pch_obj.Harvey_Rotation < (hr_start+1.))).dropna(how='all')


def circular_rebinning(pch_obj, binsize):
    # Bin size in degrees

    lon_bins = pd.DataFrame(index=np.arange(0, 360, binsize), columns=['N_Lon_Mean', 'N_Lon_Var', 'N_Lat_Mean', 'N_Lat_Var',
                                                                       'S_Lon_Mean', 'S_Lon_Var', 'S_Lat_Mean', 'S_Lat_Var'])

    for lon_bin in np.arange(0, 360, binsize):
        # Northern Coordinates
        start_bin = pch_obj.where((pch_obj.H_StartLon >= lon_bin) & (pch_obj.H_StartLon <= (lon_bin+binsize)) & (pch_obj.StartLat > 0)).dropna(how='all')
        end_bin = pch_obj.where((pch_obj.H_EndLon >= lon_bin) & (pch_obj.H_EndLon <= (lon_bin+binsize)) & (pch_obj.StartLat > 0)).dropna(how='all')

        lon_bins.N_Lon_Mean[lon_bin] = cs.circmean(np.concatenate([start_bin.H_StartLon.values, end_bin.H_EndLon.values]) * u.deg,
                                            weights=np.concatenate([start_bin.Quality.values, end_bin.Quality.values]))

        lon_bins.N_Lon_Var[lon_bin] = cs.circvar(np.concatenate([start_bin.H_StartLon.values, end_bin.H_EndLon.values]) * u.deg,
                                            weights=np.concatenate([start_bin.Quality.values, end_bin.Quality.values]))

        lon_bins.N_Lat_Mean[lon_bin] = cs.circmean(np.concatenate([start_bin.StartLat.values, end_bin.EndLat.values]) * u.deg,
                                            weights=np.concatenate([start_bin.Quality.values, end_bin.Quality.values]))

        lon_bins.N_Lat_Var[lon_bin] = cs.circvar(np.concatenate([start_bin.StartLat.values, end_bin.EndLat.values]) * u.deg,
                                            weights=np.concatenate([start_bin.Quality.values, end_bin.Quality.values]))

        # Southern Coordinates
        start_bin = pch_obj.where((pch_obj.H_StartLon >= lon_bin) & (pch_obj.H_StartLon <= (lon_bin+binsize)) & (pch_obj.StartLat < 0)).dropna(how='all')
        end_bin = pch_obj.where((pch_obj.H_EndLon >= lon_bin) & (pch_obj.H_EndLon <= (lon_bin+binsize)) & (pch_obj.StartLat < 0)).dropna(how='all')

        lon_bins.S_Lon_Mean[lon_bin] = cs.circmean(np.concatenate([start_bin.H_StartLon.values, end_bin.H_EndLon.values]) * u.deg,
                                            weights=np.concatenate([start_bin.Quality.values, end_bin.Quality.values]))

        lon_bins.S_Lon_Var[lon_bin] = cs.circvar(np.concatenate([start_bin.H_StartLon.values, end_bin.H_EndLon.values]) * u.deg,
                                            weights=np.concatenate([start_bin.Quality.values, end_bin.Quality.values]))

        lon_bins.S_Lat_Mean[lon_bin] = cs.circmean(np.concatenate([start_bin.StartLat.values, end_bin.EndLat.values]) * u.deg,
                                            weights=np.concatenate([start_bin.Quality.values, end_bin.Quality.values]))

        lon_bins.S_Lat_Var[lon_bin] = cs.circvar(np.concatenate([start_bin.StartLat.values, end_bin.EndLat.values]) * u.deg,
                                            weights=np.concatenate([start_bin.Quality.values, end_bin.Quality.values]))

    return lon_bins


def areaint(lats, lons):

    assert lats.size == lons.size, 'List of latitudes and longitudes are different sizes.'

    lat = np.array([lat.value for lat in lats]+[lats[0].value])*u.deg
    lon = np.array([lon.value for lon in lons]+[lons[0].value])*u.deg

    if lat.max() > 0:
        northern = True

    # Get colatitude (a measure of surface distance as an angle) and
    # azimuth of each point in segment from the pole

    if northern:
        colat = np.array([distance(0 * u.deg, 90 * u.deg, longi, latit).to(u.deg).value for latit, longi in zip(lat,lon)]) * u.deg
        az = np.array([azimuth(0 * u.deg, 90 * u.deg, longi, latit).to(u.deg).value for latit, longi in zip(lat,lon)]) * u.deg
    else:
        colat = np.array([distance(0 * u.deg, -90 * u.deg, longi, latit).to(u.deg).value for latit, longi in zip(lat,lon)]) * u.deg
        az = np.array([azimuth(0 * u.deg, -90 * u.deg, longi, latit).to(u.deg).value for latit, longi in zip(lat,lon)]) * u.deg

    # Change sign of last az component to avoid backwards counting
    az[-1] *= -1.
    # Calcluate step sizes
    daz = np.diff(az).to(u.rad)

    # Determine average surface distance for each step
    deltas = np.diff(colat) / 2.
    colats = colat[0:-1]+deltas

    # Integral over azimuth is 1-cos(colatitudes)
    integrands = (1 - np.cos(colats)) * daz

    # Integrate and return the answer as a fraction of the unit sphere.
    # Note that the sum of the integrands will include a partor of 4pi.

    return np.abs(np.sum(integrands)) / (4 * np.pi * u.rad)


def distance(lon1, lat1, lon2, lat2):
    return np.arccos((np.sin(lat1) * np.sin(lat2)) + (np.cos(lat1) * np.cos(lat2) * np.cos(lon2-lon1)))


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


    #if np.isclose(lat2.to(u.rad).value, (np.pi/2), rtol=1.7453292519943295e-08):
    #    bearing = (np.pi/2 * u.rad).to(lat2.unit)  # South pole ends

    #if np.isclose(lat2.to(u.rad).value, (np.pi / 2), rtol=1.7453292519943295e-08):
    #        bearing = -np.pi/2  # north pole ends

    #if np.isclose(lat1.to(u.rad).value, (-np.pi / 2), rtol=1.7453292519943295e-08):
    #    bearing = -np.pi/2  # south pole starts

    #if np.isclose(lat1.to(u.rad).value, (np.pi/2), rtol=1.7453292519943295e-08):
    #    bearing = (np.pi/2 * u.rad).to(lat1.unit)  # north pole starts

    return bearing
