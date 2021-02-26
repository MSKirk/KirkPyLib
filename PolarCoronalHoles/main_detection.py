import glob, os, time
from multiprocessing import Pool
import warnings
import numpy as np
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.table import vstack, Table, join
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import Longitude
from astropy.io import ascii
from PCH_Detection import run_detection, hole_area_parallel
import PCH_Tools


# Uncomment which dataset to run
# >> python main_detection.py

def detect_hole(all_files):

    nprocesses = 26  # IO Bound
    with Pool(nprocesses) as p:
        pts_pool = p.map(run_detection, all_files)

    # concat detections
    point_detections = Table(vstack(pts_pool))

    # Clean up
    point_detections.remove_rows(np.where(point_detections['Date'] <= Time('1950-01-04'))[0])

    return point_detections


def calc_area(point_detections):
    tstart = time.time()

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

    nprocesses = 26

    def run_hole_area(index):
        area_dict = hole_area_parallel(point_detections, index)
        return area_dict

    with Pool(nprocesses) as p:
        area_dict_pool = p.map(run_hole_area, list(np.arange(len(point_detections['Harvey_Rotation']))))

    area_table = pool_to_table(area_dict_pool)
    area_table.sort('Index')

    elapsed_time = time.time() - tstart
    print('Hole area compute time: {:1.0f} sec ({:1.1f} min)'.format(elapsed_time, elapsed_time / 60))

    return join(point_detections, area_table)


def add_harvey(point_detections):
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

    return point_detections


def save_table(all_files, point_detections):
    all_ints = ['AIA', 'SWAP', 'EUVI', 'EIT']
    all_waves = ['171', '193', '195', '211', '304', 'STACKED']

    ints_string = [ints for ints in all_ints if ints in all_files[0].upper()][0]

    wave_string = [wv for wv in all_waves if wv in all_files[0].upper()][0]
    if wave_string == 'STACKED':
        wave_string = '174'

    date_str = str(point_detections['Date'][0].datetime.year) + '-' + str(point_detections['Date'][-1].datetime.year)

    save_filename = '/Users/mskirk/data/PCH_Project/' + ints_string + wave_string + "PCH_Detections" + date_str + '.csv'

    ascii.write(point_detections, save_filename, format='ecsv', overwrite=True)


def pool_to_table(area_dict_pool):
    area_table = Table([[pts_dic['Index'] for pts_dic in area_dict_pool],
                        [pts_dic['area'][1] for pts_dic in area_dict_pool]], names=('Index', 'Area'))

    area_table.add_column([pts_dic['area'][0] for pts_dic in area_dict_pool], name='Area_min')
    area_table.add_column([pts_dic['area'][2] for pts_dic in area_dict_pool], name='Area_max')

    area_table.add_column([pts_dic['fit'][1] * u.deg for pts_dic in area_dict_pool], name='Fit')
    area_table.add_column([pts_dic['fit'][0] * u.deg for pts_dic in area_dict_pool], name='Fit_min')
    area_table.add_column([pts_dic['fit'][2] * u.deg for pts_dic in area_dict_pool], name='Fit_max')

    area_table.add_column([pts_dic['center'][1] for pts_dic in area_dict_pool], name='Center_lat')
    area_table.add_column([pts_dic['center'][0] for pts_dic in area_dict_pool], name='Center_lon')

    return area_table


if __name__ == '__main__':
    warnings.simplefilter('ignore', category=AstropyDeprecationWarning)

    AIA_dirs = ['/Volumes/CoronalHole/AIA_lev15/171/*/*/*', '/Volumes/CoronalHole/AIA_lev15/193/*/*/*',
                '/Volumes/CoronalHole/AIA_lev15/211/*/*/*', '/Volumes/CoronalHole/AIA_lev15/304/*/*/*']
    EUVI_dirs = ['/Volumes/CoronalHole/EUVI/171*/*', '/Volumes/CoronalHole/EUVI/195*/*',
                 '/Volumes/CoronalHole/EUVI/304*/*']
    SWAP_dirs = ['/Volumes/CoronalHole/SWAP/*/*/*']
    EIT_dirs = ['/Volumes/CoronalHole/EIT_lev1/171/*', '/Volumes/CoronalHole/EIT_lev1/195/*',
                '/Volumes/CoronalHole/EIT_lev1/304/*']

    all_dirs = AIA_dirs + EUVI_dirs + SWAP_dirs + EIT_dirs
    all_dirs = ['/Volumes/CoronalHole/AIA_lev15/171/*/*/*'] + SWAP_dirs

    for inst_dir in all_dirs:
        all_files = list(set([os.path.abspath(p) for p in glob.glob(inst_dir)]))
        all_files.sort()

        if not all_files:
            raise IOError('No files in input directory')

        # all_files = all_files[0:1000]
        tstart = time.time()

        point_detections = detect_hole(all_files)
        point_detections = add_harvey(point_detections)
        point_detections.sort('Harvey_Rotation')

        point_detections.add_column(np.arange(len(point_detections)), name='Index')

        ascii.write(point_detections, '/Users/mskirk/data/PCH_Project/Temp.csv', format='ecsv', overwrite=True)

        nprocesses = 26

        index_list = list(point_detections['Index'])

        with Pool(nprocesses) as p:
            area_dict_pool = p.map(hole_area_parallel, index_list)

        area_table = pool_to_table(area_dict_pool)
        area_table.sort('Index')

        elapsed_time = time.time() - tstart
        print('Hole area compute time: {:1.0f} sec ({:1.1f} min)'.format(elapsed_time, elapsed_time / 60))

        save_table(all_files, join(point_detections, area_table))


# if __name__ == '__main__':
#
#     inst_dir = '/Volumes/CoronalHole/AIA_lev15/171/*/*/*'
#     all_files = list(set([os.path.abspath(p) for p in glob.glob(inst_dir)]))
#
#     point_detections = Table(ascii.read('/Users/mskirk/data/PCH_Project/Temp.csv'))
#     nprocesses = 26
#
#     index_list = list(np.arange(len(point_detections['Harvey_Rotation'])))
#
#     tstart = time.time()
#
#     with Pool(nprocesses) as p:
#         area_dict_pool = p.map(hole_area_parallel, index_list)
#
#     area_table = pool_to_table(area_dict_pool)
#     area_table.sort('Index')
#
#     elapsed_time = time.time() - tstart
#     print('Hole area compute time: {:1.0f} sec ({:1.1f} min)'.format(elapsed_time, elapsed_time / 60))
#
#     save_table(all_files, join(point_detections, area_table))
