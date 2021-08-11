import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from skimage import exposure
import datetime
import seaborn as sns
from sunpy.map import Map
from aiapy.calibrate import register, update_pointing
from sunpy import coordinates
import astropy.units as u
from astropy.coordinates import SkyCoord
import glob
import random
from scipy import fftpack
from scipy.signal import savgol_filter

pch_obj = pd.read_pickle('/Users/mskirk/data/PCH_Project/pch_stats_dic_swap.pkl')


def latitude_range_plot(pch_obj):
    """

    :param fitler_df_list: a list of two data frames of North and South data like from pch_stats_dic.pkl
    :return:
    """
    fig, (ax, ax1) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
    sns.set_palette('Paired')

    swap_title = r'$\mathrm{SWAP \ 174}\AA$'
    aia_title = r'$\mathrm{AIA \ 171}\AA$'

    # only accurate to a 1 day resolution.
    north = pch_obj['AIA171'][0].resample('1D').median()
    south = pch_obj['AIA171'][1].resample('1D').median()

    smoothing_window = '67D'

    n_smooth = savgol_smooth(smoothing_window, np.rad2deg(north.N_upper_lat - north.N_lower_lat)
                             .interpolate(method='linear'))
    s_smooth = savgol_smooth(smoothing_window, np.rad2deg(south.S_lower_lat - south.S_upper_lat)
                             .interpolate(method='linear'))

    nmean = np.rad2deg(np.mean(north.N_upper_lat - north.N_lower_lat))
    nstd = np.rad2deg(np.std(north.N_upper_lat - north.N_lower_lat))
    smean = np.rad2deg(np.mean(south.S_lower_lat - south.S_upper_lat))
    sstd = np.rad2deg(np.std(south.S_lower_lat - south.S_upper_lat))

    line, = ax.plot(n_smooth, label='Northern PCH', color='C1')
    line2, = ax.plot(s_smooth, label='Southern PCH', color='C9')

    ax.fill_between(n_smooth.index, n_smooth+nstd, n_smooth-nstd, alpha=0.6, color="C0")
    ax.fill_between(s_smooth.index, s_smooth+sstd, s_smooth-sstd, alpha=0.6, color="C8")
    ax.set_ylim(0.0, 20)
    ax.set_xlim(datetime.date(2010, 1, 1), datetime.date(2020, 7, 1))
    ax.set_ylabel('Measured Latitude Range [deg]', fontsize=12)
    ax.set_title(aia_title, fontsize=14)
    ax.legend(fontsize=14, loc=2)

    textstr = '\n'.join((
        r'$\mathrm{Northern \ mean}=%.2f \degree$' % (nmean, ),
        r'$\mathrm{Northern \ }\sigma=%.2f \degree$' % (nstd, ),
        r'$\mathrm{Southern \ mean}=%.2f \degree$' % (smean, ),
        r'$\mathrm{Southern \ }\sigma=%.2f \degree$' % (sstd, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.8, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # only accurate to a 1 day resolution.
    north = pch_obj['SWAP174'][0].resample('1D').median()
    south = pch_obj['SWAP174'][1].resample('1D').median()

    n_smooth = savgol_smooth(smoothing_window, np.rad2deg(north.N_upper_lat - north.N_lower_lat)
                             .interpolate(method='linear'))
    s_smooth = savgol_smooth(smoothing_window, np.rad2deg(south.S_lower_lat - south.S_upper_lat)
                             .interpolate(method='linear'))

    nmean = np.rad2deg(np.mean(north.N_upper_lat - north.N_lower_lat))
    nstd = np.rad2deg(np.std(north.N_upper_lat - north.N_lower_lat))
    smean = np.rad2deg(np.mean(south.S_lower_lat - south.S_upper_lat))
    sstd = np.rad2deg(np.std(south.S_lower_lat - south.S_upper_lat))

    line, = ax1.plot(n_smooth,  label='Northern PCH', color='C3')
    line2, = ax1.plot(s_smooth, label='Southern PCH', color='C7')

    ax1.fill_between(n_smooth.index, n_smooth+nstd, n_smooth-nstd, alpha=0.6, color="C2")
    ax1.fill_between(s_smooth.index, s_smooth+sstd, s_smooth-sstd, alpha=0.6, color="C6")
    ax1.set_ylim(0.0, 20)
    ax1.set_xlim(datetime.date(2010, 1, 1), datetime.date(2020, 7, 1))
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Measured Latitude Range [deg]', fontsize=12)
    ax1.set_title(swap_title, fontsize=14)
    ax1.legend(fontsize=14, loc=2)

    textstr = '\n'.join((
        r'$\mathrm{Northern \ mean}=%.2f \degree$' % (nmean, ),
        r'$\mathrm{Northern \ }\sigma=%.2f \degree$' % (nstd, ),
        r'$\mathrm{Southern \ mean}=%.2f \degree$' % (smean, ),
        r'$\mathrm{Southern \ }\sigma=%.2f \degree$' % (sstd, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax1.text(0.8, 0.95, textstr, transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()


def area_plot(pch_obj, smoothing='16.5D'):
    fig, (ax, ax1) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

    sns.set_palette('Paired')

    swap_title = r'$\mathrm{SWAP \ 174}\AA$'
    aia_title = r'$\mathrm{AIA \ 171}\AA$'

    ax.set_title('Northern Polar Coronal Hole', fontsize=14)
    ax.set_ylabel('Fractional Surface Area', fontsize=12)
    ax.set_ylim(0, 0.1)
    ax.set_xlim(datetime.date(2010, 1, 1), datetime.date(2020, 7, 1))
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.plot(pch_obj['AIA171'][0].N_mean_area.resample(smoothing).median(), label=aia_title, color="C1")
    ax.fill_between(pch_obj['AIA171'][0].N_mean_area.resample(smoothing).median().index,
                     pch_obj['AIA171'][0].N_lower_area.resample(smoothing).median(),
                     pch_obj['AIA171'][0].N_upper_area.resample(smoothing).median(), alpha=0.6, color="C0")

    ax.plot(pch_obj['SWAP174'][0].N_mean_area.resample(smoothing).median(), label=swap_title, color="C3")
    ax.fill_between(pch_obj['SWAP174'][0].N_mean_area.resample(smoothing).median().index,
                 pch_obj['SWAP174'][0].N_lower_area.resample(smoothing).median(),
                 pch_obj['SWAP174'][0].N_upper_area.resample(smoothing).median(), alpha=0.6, color="C2")

    ax.legend(fontsize=14, loc=4)

    ax1.set_title('Southern Polar Coronal Hole', fontsize=14)
    ax1.set_ylabel('Fractional Surface Area', fontsize=12)
    ax1.set_ylim(0, 0.1)
    ax1.set_xlim(datetime.date(2010, 1, 1), datetime.date(2020, 7, 1))
    ax1.set_xlabel('Year', fontsize=12)
    plt.plot(pch_obj['AIA171'][1].S_mean_area.resample(smoothing).median(), label=aia_title, color="C9")
    plt.fill_between(pch_obj['AIA171'][1].S_mean_area.resample(smoothing).median().index,
                     pch_obj['AIA171'][1].S_lower_area.resample(smoothing).median(),
                     pch_obj['AIA171'][1].S_upper_area.resample(smoothing).median(), alpha=0.6, color="C8")

    plt.plot(pch_obj['SWAP174'][1].S_mean_area.resample(smoothing).median(), label=swap_title, color="C7")
    plt.fill_between(pch_obj['SWAP174'][1].S_mean_area.resample(smoothing).median().index,
                     pch_obj['SWAP174'][1].S_lower_area.resample(smoothing).median(),
                     pch_obj['SWAP174'][1].S_upper_area.resample(smoothing).median(), alpha=0.6, color="C6")

    ax1.legend(fontsize=14, loc=4)
    plt.tight_layout()
    plt.show()


def latitude_difference_plot(pch_obj):

    sns.set_palette('Paired')

    swap_north = pch_obj['SWAP174'][0].resample('1D').median()
    swap_south = pch_obj['SWAP174'][1].resample('1D').median()

    aia_north = pch_obj['AIA171'][0].resample('1D').median()
    aia_south = pch_obj['AIA171'][1].resample('1D').median()

    alpha = (swap_north.N_mean_lat - aia_north.N_mean_lat)
    beta = (swap_south.S_mean_lat - aia_south.S_mean_lat)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

    # Negative = Bigger hole in Swap
    # Positive = Bigger hole in AIA

    ax1.hlines(np.rad2deg(alpha.mean()), datetime.date(2010, 1, 1), datetime.date(2021, 1, 1), color='darkgray')
    sns.lineplot(x=alpha.index.round('33D'), y=np.rad2deg(alpha), ci='sd', n_boot=1000, estimator='median',
                 color='C5', ax=ax1)
    ax1.set_ylabel('Degrees Latitude', fontsize=12)
    ax1.set_title('North PCH Latitude Difference: SWAP - AIA', fontsize=14)
    ax1.set_ylim(-5, 12.5)
    ax1.set_xlim(datetime.date(2010, 1, 1), datetime.date(2020, 6, 1))

    textstr = 'Larger PCH in AIA'
    props = dict(boxstyle='round', facecolor='silver', alpha=0.3)
    ax1.text(0.75, 0.85, textstr, transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    textstr = 'Larger PCH in SWAP'
    props = dict(boxstyle='round', facecolor='silver', alpha=0.3)
    ax1.text(0.75, 0.1, textstr, transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    ax2.set_ylabel('Degrees Latitude', fontsize=12)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_title('South PCH Latitude Difference: SWAP - AIA', fontsize=14)
    ax2.set_ylim(-15, 7.5)
    ax2.set_xlim(datetime.date(2010, 1, 1), datetime.date(2020, 6, 1))

    ax2.hlines(np.rad2deg(beta.mean()), datetime.date(2010, 1, 1), datetime.date(2021, 1, 1), color='darkgray')
    sns.lineplot(x=beta.index.round('33D'), y=np.rad2deg(beta), ci='sd', n_boot=1000, estimator='median',
                 color='C11', ax=ax2)
    textstr = 'Larger PCH in SWAP'
    props = dict(boxstyle='round', facecolor='silver', alpha=0.3)
    ax2.text(0.07, 0.92, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    textstr = 'Larger PCH in AIA'
    props = dict(boxstyle='round', facecolor='silver', alpha=0.3)
    ax2.text(0.07, 0.12, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

    print(f'Mean northern offset: {np.rad2deg(alpha.mean())} degrees')
    print(f'Mean southern offset: {np.rad2deg(beta.mean())} degrees')


def overplot_aia_coords(aiaimg, pch_obj):

    plt.rcParams.update({'font.size': 16})
    aia = Map(aiaimg)

    if aia.fits_header['lvl_num'] == 1:
        aia2 = update_pointing(aia)
        aia = register(aia2)

    aia_north = pch_obj['AIA171'][0]
    #aia_north = pch_obj['AIA171'][0].resample('0.5D').median()

    n_lat = aia_north[np.abs(pd.to_datetime(aia_north.index).date - aia.date.to_datetime().date()) <=
                      datetime.timedelta(days=8.25)].N_lower_lat.values * u.rad
    n_lon = aia_north[np.abs(pd.to_datetime(aia_north.index).date - aia.date.to_datetime().date()) <=
                      datetime.timedelta(days=8.25)].N_mean_lon.values * u.rad
    times = aia_north[np.abs(pd.to_datetime(aia_north.index).date - aia.date.to_datetime().date()) <=
                      datetime.timedelta(days=8.25)].index
    n_pts = coordinates.HeliographicStonyhurst(n_lon, n_lat, obstime=times)
    n_pts_hpc = n_pts.transform_to(aia.coordinate_frame)

    #fov = 800 * u.arcsec
    #mid_pt = n_pts_hpc[int(np.floor((len(n_pts_hpc)-1)/2))]
    #bottom_left = SkyCoord(mid_pt.Tx - fov/2, mid_pt.Ty - fov/2, frame=aia.coordinate_frame)
    #smap = aia.submap(bottom_left, width=fov, height=fov)


    #ax = plt.subplot(projection=smap)
    #smap.plot()
    #smap.draw_limb()
    #ax.grid(False)
    #ax.plot_coord(n_pts_hpc, 'x', color='deepskyblue', label='North PCH')
    #plt.legend()
    #plt.show()

    #------------------------------------

    #aia_south = pch_obj['AIA171'][1].resample('0.5D').median()
    aia_south = pch_obj['AIA171'][1]
    s_lat = aia_south[np.abs(pd.to_datetime(aia_south.index).date - aia.date.to_datetime().date()) <=
                      datetime.timedelta(days=8.25)].S_lower_lat.values * u.rad
    s_lon = aia_south[np.abs(pd.to_datetime(aia_south.index).date - aia.date.to_datetime().date()) <=
                      datetime.timedelta(days=8.25)].S_mean_lon.values * u.rad
    times = aia_south[np.abs(pd.to_datetime(aia_south.index).date - aia.date.to_datetime().date()) <=
                      datetime.timedelta(days=8.25)].index
    s_pts = coordinates.HeliographicStonyhurst(s_lon, s_lat, obstime=times)
    s_pts_hpc = s_pts.transform_to(aia.coordinate_frame)

    fov = 800 * u.arcsec
    mid_pt = s_pts_hpc[int(np.floor((len(s_pts_hpc)-1)/2))]
    bottom_left = SkyCoord(mid_pt.Tx - fov/2, mid_pt.Ty - fov/2, frame=aia.coordinate_frame)
    smap = aia.submap(bottom_left, width=fov, height=fov)
    smap= aia

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=smap)
    smap.plot()
    ax.grid(False)
    ax.plot_coord(n_pts_hpc, 'x', color='cornflowerblue', label='North PCH')
    ax.plot_coord(s_pts_hpc, 'x', color='rebeccapurple', label='South PCH')
    plt.legend()
    plt.savefig('/Users/mskirk/Desktop/SWAP Paper Plots/PCH_AIA171_'+str(aia.date.to_datetime().date())+'.png')
    # plt.show()


def overplot_swap_coords(swapimg, pch_obj):

    plt.rcParams.update({'font.size': 16})
    swap = Map(swapimg)

    swap_north = pch_obj['SWAP174'][0]
    #swap_north = pch_obj['SWAP174'][0].resample('0.5D').median()

    n_lat = swap_north[np.abs(pd.to_datetime(swap_north.index).date - swap.date.to_datetime().date()) <=
                      datetime.timedelta(days=8.25)].N_lower_lat.values * u.rad
    n_lon = swap_north[np.abs(pd.to_datetime(swap_north.index).date - swap.date.to_datetime().date()) <=
                      datetime.timedelta(days=8.25)].N_mean_lon.values * u.rad
    times = swap_north[np.abs(pd.to_datetime(swap_north.index).date - swap.date.to_datetime().date()) <=
                      datetime.timedelta(days=8.25)].index
    n_pts = coordinates.HeliographicStonyhurst(n_lon, n_lat, obstime=times)
    n_pts_hpc = n_pts.transform_to(swap.coordinate_frame)

    #fov = 800 * u.arcsec
    #mid_pt = n_pts_hpc[int(np.floor((len(n_pts_hpc)-1)/2))]
    #bottom_left = SkyCoord(mid_pt.Tx - fov/2, mid_pt.Ty - fov/2, frame=swap.coordinate_frame)
    #smap = swap.submap(bottom_left, width=fov, height=fov)


    #ax = plt.subplot(projection=smap)
    #smap.plot()
    #smap.draw_limb()
    #ax.grid(False)
    #ax.plot_coord(n_pts_hpc, 'x', color='deepskyblue', label='North PCH')
    #plt.legend()
    #plt.show()

    #------------------------------------

    #swap_south = pch_obj['SWAP174'][1].resample('0.5D').median()
    swap_south = pch_obj['SWAP174'][1]
    s_lat = swap_south[np.abs(pd.to_datetime(swap_south.index).date - swap.date.to_datetime().date()) <=
                      datetime.timedelta(days=8.25)].S_lower_lat.values * u.rad
    s_lon = swap_south[np.abs(pd.to_datetime(swap_south.index).date - swap.date.to_datetime().date()) <=
                      datetime.timedelta(days=8.25)].S_mean_lon.values * u.rad
    times = swap_south[np.abs(pd.to_datetime(swap_south.index).date - swap.date.to_datetime().date()) <=
                      datetime.timedelta(days=8.25)].index
    s_pts = coordinates.HeliographicStonyhurst(s_lon, s_lat, obstime=times)
    s_pts_hpc = s_pts.transform_to(swap.coordinate_frame)

    fov = 2454 * u.arcsec
    mid_pt = s_pts_hpc[int(np.floor((len(s_pts_hpc)-1)/2))]
    #bottom_left = SkyCoord(mid_pt.Tx - fov/2, mid_pt.Ty - fov/2, frame=swap.coordinate_frame)
    bottom_left = SkyCoord(-fov/2, -fov/2, frame=swap.coordinate_frame)
    smap = swap.submap(bottom_left, width=fov, height=fov)

    #Contrast Adjustment
    p2, p99 = np.percentile(smap.data, (2, 99))
    img_rescale = exposure.rescale_intensity(smap.data, in_range=(p2, p99))
    smap.data[:] = img_rescale

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=smap)
    smap.plot()
    ax.grid(False)
    ax.plot_coord(n_pts_hpc, 'x', color='cornflowerblue', label='North PCH')
    ax.plot_coord(s_pts_hpc, 'x', color='rebeccapurple', label='South PCH')
    plt.legend()
    plt.savefig('/Users/mskirk/Desktop/SWAP Paper Plots/PCH_SWAP174_'+str(swap.date.to_datetime().date())+'.png')
    # plt.show()


def generate_plot_example(number=10, pch_obj=pch_obj):
    aia_fls = glob.glob('/Volumes/CoronalHole/AIA_lev15/171/*/*/*')
    swap_fls = glob.glob('/Volumes/CoronalHole/SWAP/*/*/*')

    for ii in range(number):
        test_img = random.choice(swap_fls)
        overplot_swap_coords(test_img, pch_obj)

    for ii in range(number):
        test_img = random.choice(aia_fls)
        overplot_aia_coords(test_img, pch_obj)


def freq_to_window(freq_string, series_freq):
    # convert a series freqency to a window size.
    # e.g. '10D' into 100 elements. Returns an int size of window.
    return int(pd.to_timedelta(freq_string)/pd.to_timedelta(series_freq))


def savgol_smooth(freq, df, poly_degree=3):

    window = freq_to_window(freq, df.index.freq)

    try:
        df_smoothed = pd.DataFrame(savgol_filter(df, window, poly_degree, axis=0), columns=df.columns, index=df.index)
    except AttributeError:
        df_smoothed = pd.Series(savgol_filter(df, window, poly_degree, axis=0), index=df.index)

    return df_smoothed


def series_powerspectrum(series):
    series.fillna(value=series.mean(), inplace=True)
    series = series * np.hamming(series.size)
    temp_fft = fftpack.fft(series.values)
    temp_psd = np.abs(temp_fft) ** 2
    duration = (series.index[-1] -series.index[0]).days + (series.index[-1] -series.index[0]).seconds/86400
    fftfreq = fftpack.fftfreq(len(temp_psd), 1. / duration)
    i = fftfreq > 0
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(fftfreq[i], 10 * np.log10(temp_psd[i]))
    ax.set_xlabel('Frequency (days)')
    ax.set_ylabel('PSD (dB)')
    plt.show()
