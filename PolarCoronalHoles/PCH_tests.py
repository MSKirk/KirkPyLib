from PolarCoronalHoles import PCH_Tools, PCH_Detection, PCH_stats, PCH_series
from sunpy import map
import sunpy.data.sample
import numpy as np
from skimage import measure
from sunpy.coordinates.utils import GreatArc
import matplotlib.pyplot as plt
from astropy import units as u
import pandas as pd

test_map = map.Map(sunpy.data.sample.AIA_171_IMAGE)


def assertEquals(var1, var2):
    if var1 == var2:
        return True
    else:
        return False


def test_rsun_pix():
    pixels = PCH_Detection.rsun_pix(test_map)

    if assertEquals(np.sum(pixels), 786.95568651283929):
        print('Pass')
    else:
        print('rsun_pix fails')


def test_pch_mask():
    PCH_Detection.pch_mask(test_map)

    if assertEquals(test_map.mask.dtype, 'bool'):
        print('Pass')
    else:
        print('pch_mask mask datatype invalid')

    if assertEquals(np.sum(test_map.mask), 1046296):
        print('Pass')
    else:
        print('pch_mask mask check sum fail')


def test_pick_hole_extremes():

    PCH_Detection.pch_mask(test_map)
    holes = measure.label(np.logical_not(test_map.mask).astype(int), connectivity=1, background=0)

    plot_map = map.Map(sunpy.data.sample.AIA_171_IMAGE)

    fig = plt.figure()
    ax = plt.subplot(projection=plot_map)
    plot_map.plot(axes=ax)

    for r_number in range(1, np.max(holes)+1, 1):
        hole_coords = test_map.pixel_to_world(np.where(holes == r_number)[1] * u.pixel,
                                                np.where(holes == r_number)[0] * u.pixel, 0)
        hole_start, hole_end = PCH_Detection.pick_hole_extremes(hole_coords)
        great_arc = GreatArc(hole_start, hole_end)
        ax.plot_coord(great_arc.coordinates(), color='c')
        print(hole_start, hole_end)

    plt.show()


def test_areaint():
    # Test for a square 1/12 of a sphere
    lats = pd.Series(np.concatenate([np.arange(-90, 91, 1), np.arange(89, -90, -1)]))
    lons = pd.Series(np.concatenate([np.zeros(181), 30 * np.ones(179)]))

    np.isclose(PCH_stats.areaint(lats, lons)[0].value, 1/12., atol=1e-4)


def test_chole_area():
    PCH_Detection.pch_mask(test_map)


def test_hole_area():
    start_date = '2010-07-30'
    end_date = '2010-10-04'

    pch_obj = pd.read_pickle('/Users/mskirk/OneDrive - NASA/PCH_data/pch_obj.pkl')[start_date:end_date]
    pch_obj = pch_obj[pch_obj.StartLat <0]

    test_obj = pd.DataFrame()
    test_obj['Lat'] = pd.concat([pch_obj.StartLat, pch_obj.EndLat])
    test_obj['Lon'] = pd.concat([pch_obj.H_StartLon, pch_obj.H_EndLon])
    test_obj['Filter'] = pd.concat([pch_obj.Filter, pch_obj.Filter])
    test_obj['HRotation'] = pd.concat([pch_obj.Harvey_Rotation, pch_obj.Harvey_Rotation])

    test_obj = test_obj.sort_index()
    windowsize = '11D'
    deg_bins = 5

    mean_hole_area = []
    for hr in pch_obj.Harvey_Rotation[int(pch_obj.shape[0]/2):]:
        mean_hole_area += [PCH_series.generic_hole_area(pch_obj, hr, northern=False)[0][1]]

    mean_area_series = pd.Series(data=mean_hole_area, index= pch_obj.index[int(pch_obj.shape[0]/2):])
    test_obj2 = PCH_stats.df_pre_process(test_obj, northern=False, window_size=windowsize, wave_filter='EIT171', binsize=deg_bins)[0]
    #test_obj2 = PCH_stats.df_concat_stats_hem(test_obj, binsize=5, sigma=1.0, northern=False, window_size='11D')

    plt.plot(mean_area_series)
    plt.plot(test_obj2.S_mean_area)


def test_area_algorithms():

    lons = np.arange(0,360,1)
    lats = (np.random.randint(10, size=360) + 70)

    old = old_area(lats*u.deg, lons*u.deg)[1]

    s_lats = pd.Series(lats)
    s_lons = pd.Series(lons)

    new = PCH_stats.areaint(s_lats, s_lons)[0].value

    np.testing.assert_approx_equal(old,new,significant=3,err_msg='Actual = Old; Desired = New',verbose=True)


def old_area(lats, lons, northern=True):
    h_rotation_number = 0
    offset_cm = np.array([0,0])*u.deg

    perimeter_length = np.zeros(6) * u.rad
    fit_location = np.zeros(6) * u.rad
    hole_area = np.zeros(6)

    for ii, degrees in enumerate([4, 5, 6, 7, 8, 9]):
        try:
            hole_fit = PCH_Tools.trigfit(np.deg2rad(lons), np.deg2rad(lats), degree=degrees)

            # Lambert cylindrical equal-area projection to find the area using the composite trapezoidal rule
            # And removing centroid offset
            if northern:
                lamb_x = np.deg2rad(np.arange(0, 360, 0.01) * u.deg)
                lamb_y = np.sin(
                    (np.pi * 0.5) - hole_fit['fitfunc'](lamb_x.value) - np.deg2rad(offset_cm[1]).value) * u.rad

                lamb_x = np.deg2rad(np.arange(0, 360, 0.01) * u.deg + offset_cm[0])
                fit_location[ii] = np.rad2deg((np.pi * 0.5) - hole_fit['fitfunc'](np.deg2rad(
                    PCH_Tools.get_harvey_lon(PCH_Tools.hrot2date(h_rotation_number)) - offset_cm[
                        0]).value) - np.deg2rad(offset_cm[1]).value) * u.deg
            else:
                lamb_x = np.deg2rad(np.arange(0, 360, 0.01) * u.deg)
                lamb_y = np.sin(
                    hole_fit['fitfunc'](lamb_x.value) - (np.pi * 0.5) + np.deg2rad(offset_cm[1]).value) * u.rad

                lamb_x = np.deg2rad(np.arange(0, 360, 0.01) * u.deg + offset_cm[0])
                fit_location[ii] = np.rad2deg(hole_fit['fitfunc'](np.deg2rad(
                    PCH_Tools.get_harvey_lon(PCH_Tools.hrot2date(h_rotation_number)) - offset_cm[0]).value) - (
                                                      np.pi * 0.5) + np.deg2rad(offset_cm[1]).value) * u.deg

            perimeter_length[ii] = PCH_Tools.curve_length(lamb_x, lamb_y)

            if northern:
                hole_area[ii] = (2 * np.pi) - np.trapz(lamb_y, x=lamb_x).value
            else:
                hole_area[ii] = (2 * np.pi) + np.trapz(lamb_y, x=lamb_x).value

        except RuntimeError:
            hole_area[ii] = np.nan
            perimeter_length[ii] = np.inf * u.rad
            fit_location[ii] = np.nan

    # allowing for a 5% perimeter deviation off of a circle
    good_areas = hole_area[np.where((perimeter_length / (2 * np.pi * u.rad)) - 1 < 0.05)]
    good_fits = fit_location[np.where((perimeter_length / (2 * np.pi * u.rad)) - 1 < 0.05)]

    # A sphere is 4π steradians in surface area
    if good_areas.size > 0:
        percent_hole_area = (
        np.nanmin(good_areas) / (4 * np.pi), np.nanmean(good_areas) / (4 * np.pi), np.nanmax(good_areas) / (4 * np.pi))
        # in degrees
        hole_perimeter_location = (np.rad2deg(np.nanmin(good_fits)).value, np.rad2deg(np.nanmean(good_fits)).value,
                                   np.rad2deg(np.nanmax(good_fits)).value)
    else:
        percent_hole_area = (np.nan, np.nan, np.nan)
        hole_perimeter_location = np.array([np.nan, np.nan, np.nan])

    return percent_hole_area


def test_preprocessing():
    start_date = '2010-08-29'
    end_date = '2010-10-01'

    pch_obj = pd.read_pickle('/Users/mskirk/OneDrive - NASA/PCH_data/pch_obj.pkl')[start_date:end_date]

    test_obj = pd.DataFrame()
    test_obj['Lat'] = pd.concat([pch_obj.StartLat, pch_obj.EndLat])
    test_obj['Lon'] = pd.concat([pch_obj.H_StartLon, pch_obj.H_EndLon])
    test_obj['Filter'] = pd.concat([pch_obj.Filter, pch_obj.Filter])
    test_obj['HRotation'] = pd.concat([pch_obj.Harvey_Rotation, pch_obj.Harvey_Rotation])

    test_obj = test_obj.sort_index()

    windowsize = '11D'
    deg_bins = 5

    EIT171 = PCH_stats.df_pre_process(test_obj, northern=False, window_size=windowsize, wave_filter='EIT171', binsize=deg_bins)[0]
    EUVI171 = PCH_stats.df_pre_process(test_obj, northern=False, window_size=windowsize, wave_filter='EUVI171', binsize=deg_bins)[0]
    AIA171 = PCH_stats.df_pre_process(test_obj, northern=False, window_size=windowsize, wave_filter='AIA171', binsize=deg_bins)[0]
    SWAP174 = PCH_stats.df_pre_process(test_obj, northern=False, window_size=windowsize, wave_filter='SWAP174', binsize=deg_bins)[0]

    eit171_p = pd.Series(data=EIT171.Lat.values, index=EIT171.Lon.values).sort_index()
    euvi171_p = pd.Series(data=EUVI171.Lat.values, index=EUVI171.Lon.values).sort_index()
    aia171_p = pd.Series(data=AIA171.Lat.values, index=AIA171.Lon.values).sort_index()
    swap174_p = pd.Series(data=SWAP174.Lat.values, index=SWAP174.Lon.values).sort_index()

    EIT171 = PCH_stats.df_pre_process(test_obj, northern=False, window_size=windowsize, wave_filter='EIT171', resample=True, binsize=deg_bins)[0]
    EUVI171 = PCH_stats.df_pre_process(test_obj, northern=False, window_size=windowsize, wave_filter='EUVI171', resample=True, binsize=deg_bins)[0]
    AIA171 = PCH_stats.df_pre_process(test_obj, northern=False, window_size=windowsize, wave_filter='AIA171', resample=True, binsize=deg_bins)[0]
    SWAP174 = PCH_stats.df_pre_process(test_obj, northern=False, window_size=windowsize, wave_filter='SWAP174', resample=True, binsize=deg_bins)[0]

    eit171_r = pd.Series(data=EIT171.Lat.values, index=EIT171.Lon.values).sort_index()
    euvi171_r = pd.Series(data=EUVI171.Lat.values, index=EUVI171.Lon.values).sort_index()
    aia171_r = pd.Series(data=AIA171.Lat.values, index=AIA171.Lon.values).sort_index()
    swap174_r = pd.Series(data=SWAP174.Lat.values, index=SWAP174.Lon.values).sort_index()

    EIT171 = test_obj[(test_obj.Filter == 'EIT171') & (test_obj.Lat < 0)]
    EUVI171 = test_obj[(test_obj.Filter == 'EUVI171') & (test_obj.Lat < 0)]
    AIA171 = test_obj[(test_obj.Filter == 'AIA171') & (test_obj.Lat < 0)]
    SWAP174 = test_obj[(test_obj.Filter == 'SWAP174') & (test_obj.Lat < 0)]

    eit171 = pd.Series(data=EIT171.Lat.values, index=EIT171.Lon.values).sort_index()
    euvi171 = pd.Series(data=EUVI171.Lat.values, index=EUVI171.Lon.values).sort_index()
    aia171 = pd.Series(data=AIA171.Lat.values, index=AIA171.Lon.values).sort_index()
    swap174 = pd.Series(data=SWAP174.Lat.values, index=SWAP174.Lon.values).sort_index()

    plt.subplot(4,1,1)
    plt.plot(eit171, '.')
    plt.plot(eit171_p)
    plt.plot(eit171_r, 'g')

    plt.subplot(4,1,2)
    plt.plot(euvi171, '.')
    plt.plot(euvi171_p)
    plt.plot(euvi171_r, 'g')

    plt.subplot(4,1,3)
    plt.plot(aia171, '.')
    plt.plot(aia171_p)
    plt.plot(aia171_r, 'g')

    plt.subplot(4,1,4)
    plt.plot(swap174, '.')
    plt.plot(swap174_p)
    plt.plot(swap174_r, 'g')
    plt.tight_layout()


def series_area_calc(euv_series):
    # EUV Series with index= lon, data = lat

    lats = pd.Series(euv_series.values)
    lons = pd.Series(euv_series.index.values)

    area = PCH_stats.areaint(lats, lons)

    return area[0].value

# if northern:
#     offset_coords = np.transpose(np.asarray([lons, (90 * u.deg) - lats]))
# else:
#     offset_coords = np.transpose(np.asarray([lons, (90 * u.deg) + lats]))
#
# offset = PCH_Tools.center_of_mass(offset_coords, mass=1 / errors)
# offset_lons = offset_coords[:, 0] - offset[0]
# offset_lats = offset_coords[:, 1] - offset[1]
#
# hole_fit = PCH_Tools.trigfit(np.deg2rad(lons), np.deg2rad(lats), degree=6, sigma=errors)
# co_hole_fit = PCH_Tools.trigfit(np.deg2rad(offset_coords[:,0])*u.rad, np.deg2rad(offset_coords[:,1])*u.rad, degree=6, sigma=errors)
# offset_hole_fit = PCH_Tools.trigfit(np.deg2rad(offset_lons)*u.rad, np.deg2rad(offset_lats)*u.rad, degree=6, sigma=errors)
#
#
# fit_yy =  hole_fit['fitfunc'](np.deg2rad(lons).value)
# co_fit_yy = co_hole_fit['fitfunc'](np.deg2rad(offset_coords[:,0]))
# offset_fit_yy = offset_hole_fit['fitfunc'](np.deg2rad(offset_lons))
#
#
# plt.plot(np.deg2rad(lons).value, fit_yy - np.deg2rad(lats).value, '.')
# plt.plot(np.deg2rad(offset_coords[:,0]), co_fit_yy - np.deg2rad(offset_coords[:,1]), '.')
# plt.plot(np.deg2rad(offset_lons), offset_fit_yy - np.deg2rad(offset_lats), '.')
#
# plt.plot(np.deg2rad(lons).value, fit_yy, '.')
# plt.plot(np.deg2rad(offset_lons+np.deg2rad(offset[0])), offset_fit_yy - (np.pi *0.5)+ np.deg2rad(offset[1]), '.')
