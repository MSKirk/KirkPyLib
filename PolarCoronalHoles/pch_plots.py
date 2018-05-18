import numpy as np
import matplotlib.pyplot as plt
from PolarCoronalHoles import PCH_Tools
import astropy.units as u
import seaborn as sns
import scipy.stats as stats

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

