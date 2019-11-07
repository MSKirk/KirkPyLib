import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import scipy.stats as stats
import pandas as pd
import PCH_series
import PCH_Tools
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
from matplotlib import gridspec
import seaborn as sns
from astropy.time import Time
from datetime import timedelta


def TESS_2018_plot():
    ll = np.load('/Users/mskirk/data/PCH Project/Example_LatLon.npy')
    lats = np.radians(ll[0])
    lons = np.radians(ll[1])
    test_lons = np.radians(np.arange(0,360,0.01))

    test3 = np.random.choice(range(len(lats)), 100, replace=False)
    test4 = np.random.choice(range(len(lats)), 100, replace=False)
    test5 = np.random.choice(range(len(lats)), 100, replace=False)
    test6 = np.random.choice(range(len(lats)), 100, replace=False)
    test7 = np.random.choice(range(len(lats)), 100, replace=False)
    test8 = np.random.choice(range(len(lats)), 100, replace=False)

    hole_fit3 = PCH_Tools.trigfit((lons[test3] * u.rad), (lats[test3] * u.rad), degree=7)
    hole_fit4 = PCH_Tools.trigfit((lons[test4] * u.rad), (lats[test4] * u.rad), degree=7)
    hole_fit5 = PCH_Tools.trigfit((lons[test5] * u.rad), (lats[test5] * u.rad), degree=7)
    hole_fit6 = PCH_Tools.trigfit((lons[test6] * u.rad), (lats[test6] * u.rad), degree=7)
    hole_fit7 = PCH_Tools.trigfit((lons[test7] * u.rad), (lats[test7] * u.rad), degree=7)
    hole_fit8 = PCH_Tools.trigfit((lons[test8] * u.rad), (lats[test8] * u.rad), degree=7)

    plt.plot(np.degrees(test_lons), np.sin(hole_fit3['fitfunc'](test_lons)), color='r', linewidth=8)
    plt.plot(np.degrees(test_lons), np.sin(hole_fit4['fitfunc'](test_lons)), color='m', linewidth=8)
    plt.plot(np.degrees(test_lons), np.sin(hole_fit5['fitfunc'](test_lons)), color='orange', linewidth=8)
    plt.plot(np.degrees(test_lons), np.sin(hole_fit6['fitfunc'](test_lons)), color='g', linewidth=8)
    plt.plot(np.degrees(test_lons), np.sin(hole_fit7['fitfunc'](test_lons)), color='b', linewidth=8)
    plt.plot(np.degrees(test_lons), np.sin(hole_fit8['fitfunc'](test_lons)), color='c', linewidth=8)

    plt.plot(np.degrees(lons[test3]), np.sin(lats[test3]), '.', markersize=12, color='r')
    plt.plot(np.degrees(lons[test4]), np.sin(lats[test4]), '.', markersize=12, color='m')
    plt.plot(np.degrees(lons[test5]), np.sin(lats[test5]), '.', markersize=12, color='orange')
    plt.plot(np.degrees(lons[test6]), np.sin(lats[test6]), '.', markersize=12, color='g')
    plt.plot(np.degrees(lons[test7]), np.sin(lats[test7]), '.', markersize=12, color='b')
    plt.plot(np.degrees(lons[test8]), np.sin(lats[test8]), '.', markersize=12, color='c')

    plt.ylim([0.7,1])
    plt.xlim([0,360])
    plt.ylabel('Sine Latitude', fontweight='bold')
    plt.xlabel('Degrees Longitude', fontweight='bold')

    fittest = np.concatenate([[np.sin(hole_fit3['fitfunc'](test_lons))],
                              [np.sin(hole_fit4['fitfunc'](test_lons))],
                              [np.sin(hole_fit5['fitfunc'](test_lons))],
                              [np.sin(hole_fit6['fitfunc'](test_lons))],
                              [np.sin(hole_fit7['fitfunc'](test_lons))],
                              [np.sin(hole_fit8['fitfunc'](test_lons))]])

    ax = sns.tsplot(data=fittest, time=np.degrees(test_lons), ci=[90,95,99], linewidth=8)

    plt.gcf().clear()

    # ---------------------
    testbed = np.concatenate([test3,test4,test5,test6,test7,test8])

    ind = np.sort(testbed)

    test_lats2 = np.sin(lats[ind])
    test_lons2 = lons[ind]

    number_of_bins = 50

    bin_median, bin_edges, binnumber = stats.binned_statistic(test_lons2, test_lats2, statistic='median', bins=number_of_bins)

    a = [np.where(binnumber == ii)[0] for ii in range(1,number_of_bins+1)]
    lens = [len(aa) for aa in a]
    binned_lats = np.zeros([max(lens),number_of_bins])

    for jj, ar in enumerate(a):
        binned_lats[:,jj] = np.nanmean(test_lats2[ar])
        binned_lats[0:lens[jj], jj] = test_lats2[ar]

    ax = sns.tsplot(data=binned_lats, time=np.degrees(bin_edges[0:number_of_bins]), err_style='boot_traces', n_boot=800, color='m')


def area_series_plot()

    all_area = pd.read_pickle('/Users/mskirk/data/PCH_Project/all_area.pkl')
    north = pd.concat([all_area.Area[all_area.Center_lat > 0], all_area.Area_max[all_area.Center_lat > 0],
                       all_area.Area_min[all_area.Center_lat > 0]]).sort_index()
    south = pd.concat([all_area.Area[all_area.Center_lat < 0], all_area.Area_max[all_area.Center_lat < 0],
                       all_area.Area_min[all_area.Center_lat < 0]]).sort_index()

    north_ci = PCH_series.series_bootstrap(north, interval='360D', delta=True, confidence=0.997).fillna(0)

    n_div = (north_ci.upper - north_ci.lower)/2

    south_ci = PCH_series.series_bootstrap(south, interval='360D', delta=True, confidence=0.997).fillna(0)

    s_div = (south_ci.upper - south_ci.lower)/2

    wso = PCH_series.read_wso_data('/Users/mskirk/data/PCH_Project/WSO_PolarField.txt')

    sns.set(style="darkgrid")

    fig = plt.figure(figsize=(8, 4))
    outer_grid = gridspec.GridSpec(2, 1, wspace=0.0, hspace=0.0,height_ratios=[20, 1])
    ax = fig.add_subplot(outer_grid[0])
    plt.plot(north.resample('11D').median())
    plt.fill_between(north_ci.resample('11D').median().index, (north -n_div).resample('11D').median(), (north + n_div).resample('11D').median(), alpha=0.5)
    ax.set_title('Northern Polar Coronal Hole')
    ax.set_ylabel('Fractional Area')
    ax.set_ylim(0, 0.08)
    ax.set_xlim(all_area.index[0] - timedelta(days=180), all_area.index[-1] + timedelta(days=180))
    ax.set_xlabel('')
    ax.set_xticklabels([])


    ax1 = fig.add_subplot(outer_grid[1])
    inxval = mdates.date2num(wso.index.to_pydatetime())
    y = np.zeros_like(inxval)
    points = np.array([inxval, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = ListedColormap(['r', 'y'])
    norm = BoundaryNorm([wso.NorthFilter.min(), 0, wso.NorthFilter.max()], cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(wso.NorthFilter)
    lc.set_linewidth(4)
    line = ax1.add_collection(lc)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    monthFmt = mdates.DateFormatter("%Y")
    ax1.xaxis.set_major_formatter(monthFmt)
    ax1.autoscale_view()
    ax1.set_ylim(-0.1, 0.1)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.set_aspect(1000)
    t_min = np.floor(Time(all_area.index[0]).jd - 1721424.5 -180)
    t_max = np.floor(Time(all_area.index[-1]).jd - 1721424.5 +180)
    ax1.set_xlim(t_min, t_max)



def confetti_plot():
    comp = pch.pch_obj['2012-10-01':'2012-11-04']
    comp = comp[comp.StartLat < 0]

    plt.plot(comp.Harvey_Longitude[comp.Filter == 'EIT171'], comp.StartLat[comp.Filter == 'EIT171'], '.', color='b')
    plt.plot(comp.Harvey_Longitude[comp.Filter == 'EIT171'], comp.EndLat[comp.Filter == 'EIT171'], '.', color='b')

    plt.plot(comp.Harvey_Longitude[comp.Filter == 'AIA171'], comp.StartLat[comp.Filter == 'AIA171'], '.', color='g')
    plt.plot(comp.Harvey_Longitude[comp.Filter == 'AIA171'], comp.EndLat[comp.Filter == 'AIA171'], '.', color='g')

    plt.plot(comp.Harvey_Longitude[comp.Filter == 'EUVI171'], comp.StartLat[comp.Filter == 'EUVI171'], '.', color='r')
    plt.plot(comp.Harvey_Longitude[comp.Filter == 'EUVI171'], comp.EndLat[comp.Filter == 'EUVI171'], '.', color='r')

    plt.plot(comp.Harvey_Longitude[comp.Filter == 'SWAP174'], comp.StartLat[comp.Filter == 'SWAP174'], '.', color='c')
    plt.plot(comp.Harvey_Longitude[comp.Filter == 'SWAP174'], comp.EndLat[comp.Filter == 'SWAP174'], '.', color='c')

    comp = pch.pch_obj['2017-01-04':'2017-02-04']
    comp = comp[comp.StartLat < 0]

    plt.plot(comp.Harvey_Longitude[comp.Filter == 'EIT171'], comp.StartLat[comp.Filter == 'EIT171'], '.', color='b')
    plt.plot(comp.Harvey_Longitude[comp.Filter == 'EIT171'], comp.EndLat[comp.Filter == 'EIT171'], '.', color='b')

    plt.plot(comp.Harvey_Longitude[comp.Filter == 'AIA171'], comp.StartLat[comp.Filter == 'AIA171'], '.', color='g')
    plt.plot(comp.Harvey_Longitude[comp.Filter == 'AIA171'], comp.EndLat[comp.Filter == 'AIA171'], '.', color='g')

    plt.plot(comp.Harvey_Longitude[comp.Filter == 'EUVI171'], comp.StartLat[comp.Filter == 'EUVI171'], '.', color='r')
    plt.plot(comp.Harvey_Longitude[comp.Filter == 'EUVI171'], comp.EndLat[comp.Filter == 'EUVI171'], '.', color='r')

    plt.plot(comp.Harvey_Longitude[comp.Filter == 'SWAP174'], comp.StartLat[comp.Filter == 'SWAP174'], '.', color='c')
    plt.plot(comp.Harvey_Longitude[comp.Filter == 'SWAP174'], comp.EndLat[comp.Filter == 'SWAP174'], '.', color='c')


