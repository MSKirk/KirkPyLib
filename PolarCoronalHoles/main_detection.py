import glob, os, time
from multiprocessing import Pool
import warnings
import numpy as np
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.table import vstack
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import Longitude
from astropy.io import ascii
from PCH_Detection import run_detection, hole_area
import PCH_Tools


# Uncomment which dataset to run
# >> python main_detection.py

if __name__ == '__main__':
    warnings.simplefilter('ignore', category=AstropyDeprecationWarning)

    # AIA
    #all_files = list(set([os.path.dirname(p) for p in glob.glob("/Volumes/CoronalHole/AIA_lev15/*/*/*")]))

    # EUVI
    #all_files = list(set([os.path.dirname(p) for p in glob.glob("/Volumes/CoronalHole/EUVI/*/*")]))

    # SWAP
    all_files = list(set([os.path.abspath(p) for p in glob.glob("/Volumes/CoronalHole/SWAP/*/*/*")]))

    # EIT
    #all_files = list(set([os.path.dirname(p) for p in glob.glob("/Volumes/CoronalHole/EIT_lev1/*/*")]))

    all_files.sort()
    #all_files = all_files[0:500]

    tstart = time.time()
    nprocesses = 26 # IO Bound
    with Pool(nprocesses) as p:
        pts_pool = p.map(run_detection, all_files)

    # concat detections
    point_detections = vstack(pts_pool)
    
    # Clean up
    try:
        point_detections.remove_row(np.where(point_detections['Date'] == Time('1900-01-04'))[0][0])
    except IndexError:
        pass

    # Add Harvey Rotation
    point_detections['Harvey_Rotation'] = [PCH_Tools.date2hrot(date, fractional=True) for date in
                                               point_detections['Date']]
    point_detections['Harvey_Longitude'] = np.squeeze(
        [PCH_Tools.get_harvey_lon(date) for date in point_detections['Date']])
    point_detections['H_StartLon'] = Longitude((np.squeeze(np.asarray(point_detections['Harvey_Longitude']))
                                                    + np.array(point_detections['StartLon'])) * u.deg)
    point_detections['H_EndLon'] = Longitude((np.squeeze(np.asarray(point_detections['Harvey_Longitude']))
                                                  + np.array(point_detections['EndLon'])) * u.deg)
    point_detections.sort('Harvey_Rotation')

    area = []
    fit = []
    center = []
    for ii, h_rot in enumerate(point_detections['Harvey_Rotation']):
        if point_detections[ii]['StartLat'] > 0:
            northern = True
        else:
            northern = False
        ar, ft, cm = hole_area(point_detections, h_rot, northern=northern)
        area = area + [ar]
        fit = fit + [ft]
        center = center + [cm]
    center = np.vstack([arr[0:2] for arr in center])
    
    point_detections['Area'] = np.asarray(area)[:, 1]
    point_detections['Area_min'] = np.asarray(area)[:, 0]
    point_detections['Area_max'] = np.asarray(area)[:, 2]

    point_detections['Fit'] = np.asarray(fit)[:, 1] * u.deg
    point_detections['Fit_min'] = np.asarray(fit)[:, 0] * u.deg
    point_detections['Fit_max'] = np.asarray(fit)[:, 2] * u.deg

    point_detections['Center_lat'] = center[:, 1] * u.deg
    point_detections['Center_lon'] = center[:, 0] * u.deg

    all_ints = ['AIA', 'SWAP', 'EUVI', 'EIT']
    all_waves = ['171', '193', '195', '211', '304', 'STACKED']

    ints_string = [ints for ints in all_ints if ints in point_detections['FileName'][0].upper()][0]

    wave_string = [wv for wv in all_waves if wv in all_files[0].upper()][0]
    if wave_string == 'STACKED':
        wave_string = '174'

    date_str = str(point_detections['Date'][0].datetime.year) + '-' + str(point_detections['Date'][-1].datetime.year)

    save_filename = '/Users/mskirk/data/PCH_Project/' + ints_string + wave_string + "PCH_Detections" + date_str + '.csv'

    ascii.write(point_detections, save_filename, format='ecsv', overwrite=True)

    elapsed_time = time.time() - tstart
    print('Compute time: {:1.0f} sec ({:1.1f} min)'.format(elapsed_time, elapsed_time / 60))
