from PolarCoronalHoles import PCH_Tools
import astropy.stats.circstats as cs
import matplotlib.pyplot as plt
import astropy.units as u
import pandas as pd
import numpy as np


def filter_select(pch_obj, filter):
    return pch_obj.where(pch_obj.Filter == filter).dropna(how='all')


def one_hr_select(pch_obj, hr_start):
    return pch_obj.where((pch_obj.Harvey_Rotation >= hr_start) & (pch_obj.Harvey_Rotation < (hr_start+1.))).dropna(how='all')


def sph_center_of_mass(lats, lons, **kwargs):

    weights = kwargs.get('weights', 1)

    rect_coords = PCH_Tools.coord_spher2rec(np.ones_like(lats.value), lats, lons)
    rect_center = PCH_Tools.center_of_mass(np.transpose(np.array(rect_coords)), mass=weights)

    sphere_center = PCH_Tools.coord_rec2spher(rect_center[0], rect_center[1], rect_center[2])

    return sphere_center


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
    # azimuth of each point in segment from the center of mass.

    _, center_lat, center_lon = sph_center_of_mass(lat[:-1], lon[:-1])

    # force centroid at the N or S pole
    #if northern:
    #    center_lat = 90 * u.deg
    #else:
    #   center_lat = -90 * u.deg
    #center_lon = 0 * u.deg

    colat = np.array([distance(center_lon.to(u.deg), center_lat.to(u.deg), longi, latit).to(u.deg).value
                      for latit, longi in zip(lat,lon)]) * u.deg
    az = np.array([azimuth(center_lon.to(u.deg), center_lat.to(u.deg), longi, latit).to(u.deg).value
                   for latit, longi in zip(lat,lon)]) * u.deg

    # Calculate step sizes, taking the complementary angle where needed
    daz = np.diff(az).to(u.rad)
    daz[np.where(daz > 180 * u.deg)] -= 360. * u.deg
    daz[np.where(daz < -180 * u.deg)] += 360. * u.deg

    # Determine average surface distance for each step
    deltas = np.diff(colat) / 2.
    colats = colat[0:-1]+deltas

    # Integral over azimuth is 1-cos(colatitudes)
    integrands = (1 - np.cos(colats)) * daz

    # Integrate and return the answer as a fraction of the unit sphere.
    # Note that the sum of the integrands will include a part of 4pi.

    return np.abs(np.nansum(integrands)) / (4 * np.pi * u.rad), center_lat, center_lon


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


def chole_stats(pch_obj, filter, binsize=10):

    if filter not in pch_obj.Filter.unique():
        print('Please specify a correct filter.')
        return pd.DataFrame()

    measurements = filter_select(pch_obj, filter)

    hole_stats = pd.DataFrame(index=measurements.Harvey_Rotation.unique(), columns=['Harvey_Rotation',
                                        'Date', 'N_mean_area', 'S_mean_area', 'N_min_area', 'S_min_area', 'N_max_area',
                                        'S_max_area', 'N_center_lat', 'N_center_lon', 'S_center_lat', 'S_center_lon'])

    for hr in measurements.Harvey_Rotation:
        one_rot = one_hr_select(measurements, hr)

        binned = circular_rebinning(one_rot, binsize=binsize)

        hole_stats['N_mean_area'][hr], hole_stats['N_center_lat'][hr], hole_stats['N_center_lon'][hr] = areaint(binned.N_Lat_Mean, binned.N_Lon_Mean)
        hole_stats['N_max_area'][hr], _, _ = areaint(binned.N_Lat_Mean - binned.N_Lat_Var * u.deg, binned.N_Lon_Mean)
        hole_stats['N_min_area'][hr], _, _ = areaint(binned.N_Lat_Mean + binned.N_Lat_Var * u.deg, binned.N_Lon_Mean)
        
        hole_stats['S_mean_area'][hr], hole_stats['S_center_lat'][hr], hole_stats['S_center_lon'][hr] = areaint(binned.S_Lat_Mean, binned.S_Lon_Mean)
        hole_stats['S_max_area'][hr], _, _ = areaint(binned.S_Lat_Mean - binned.S_Lat_Var * u.deg, binned.S_Lon_Mean)
        hole_stats['S_min_area'][hr], _, _ = areaint(binned.S_Lat_Mean + binned.S_Lat_Var * u.deg, binned.S_Lon_Mean)

        hole_stats['Date'][hr] = PCH_Tools.hrot2date(hr).iso
        hole_stats['Harvey_Rotation'][hr] = hr

    return hole_stats


