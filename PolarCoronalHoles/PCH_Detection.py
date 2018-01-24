import sunpy
from sunpy import map
import numpy as np
import os
from PolarCoronalHoles import PCH_Tools
import astropy.units as u
from skimage import exposure, morphology, measure
from sunpy.coordinates.utils import GreatArc
from astropy.table import Table, join
from astropy.time import Time
from astropy.coordinates import Longitude
from astropy.io import ascii
import warnings

'''
Detection of polar coronal holes given a directory of images. 
A pure python version reproducing the following IDL routines:
    chole_*_run.pro
    chole_*_year.pro
    chole_*_year2hrot.pro
    chole_*_data.pro
    chole_mark.pro
    chole_mask.pro
    chole_series.pro
    chole_area.pro
    what_image.pro
    chole_series_area.pro
    **_image_check.pro
    new_center.pro
'''


def rsun_pix(inmap):
    # Returns rsun in pixel units
    rsun = np.array([inmap.rsun_obs.to('deg').value])

    return np.array([inmap.wcs.all_world2pix(rsun, rsun, 0)[0] - inmap.wcs.all_world2pix(0, 0, 0)[0],
                     inmap.wcs.all_world2pix(rsun, rsun, 0)[1] - inmap.wcs.all_world2pix(0, 0, 0)[1]])


def pch_mask(map, factor=0.5):
    # To isolate the lim of the sun to study coronal holes
    # returns a binary array of the outer lim of the sun

    # Class check
    if not isinstance(map, sunpy.map.GenericMap):
        raise ValueError('Input needs to be an sunpy map object.')

    # Bad image check
    if np.max(map.data) < 1:
        map.mask = np.ones_like(map.data)

    # EUVI Wavelet adjustment
    if np.max(map.data) < 100:
        map.data *= map.data

    # Range Clipping
    map.data[map.data > 10000] = 10000

    map.data[map.data < 0] = 0

    rsun_in_pix = rsun_pix(map)

    # EUVI Wavelet adjustment
    if map.detector == 'EUVI':
        if map.wavelength > 211*u.AA:
               map.mask = exposure.equalize_hist(map.data, mask=np.logical_not(PCH_Tools.annulus_mask(map.data.shape, (0,0), rsun_in_pix, center=map.wcs.wcs.crpix)))
        else:
            map.mask = exposure.equalize_hist(map.data)
    else:
        map.mask = np.copy(map.data)

    # Found through experiment...
    if map.detector == 'AIA':
        if map.wavelength == 193 * u.AA:
            factor = 0.30
        if map.wavelength == 171 * u.AA:
            factor = 0.62
        if map.wavelength == 304 * u.AA:
            factor = 0.15
    if map.detector == 'EIT':
        if map.wavelength == 195 * u.AA:
            factor = 0.27
        if map.wavelength == 171 * u.AA:
            factor = 0.37
        if map.wavelength == 304 * u.AA:
            factor = 0.14
        if map.wavelength == 284 * u.AA:
            factor = 0.22
    if map.detector == 'EUVI':
        if map.wavelength == 195 * u.AA:
            factor = 0.53
        if map.wavelength == 171 * u.AA:
            factor = 0.55
        if map.wavelength == 304 * u.AA:
            factor = 0.35
        if map.wavelength == 284 * u.AA:
            factor = 0.22
    if map.detector == 'SWAP':
        if map.wavelength == 174 * u.AA:
            factor = 0.55

    # Creating a kernel for the morphological transforms
    if map.wavelength == 304 * u.AA:
        structelem = morphology.disk(np.round(np.average(rsun_in_pix) * 0.007))
    else:
        structelem = morphology.disk(np.round(np.average(rsun_in_pix) * 0.004))

    # First morphological pass...
    if map.wavelength == 304 * u.AA:
        map.mask = morphology.opening(morphology.closing(map.mask, selem=structelem))
    else:
        map.mask = morphology.closing(morphology.opening(map.mask, selem=structelem))

    # Masking off limb structures and Second morphological pass...
    map.mask = morphology.opening(map.mask * PCH_Tools.annulus_mask(map.data.shape, (0,0), rsun_in_pix, center=map.wcs.wcs.crpix), selem=structelem)

    # Extracting holes...
    thresh = PCH_Tools.hist_percent(map.mask[np.nonzero(map.mask)], factor, number_of_bins=1000)
    map.mask = np.where(map.mask >= thresh, map.mask, 0)

    # Extracting annulus
    map.mask[PCH_Tools.annulus_mask(map.data.shape, rsun_in_pix*0.965, rsun_in_pix*0.995, center=map.wcs.wcs.crpix) == False] = np.nan

    # Filter for hole size scaled to resolution of the image

    regions = measure.label(np.logical_not(map.mask).astype(int), connectivity=1, background=0)

    if np.max(regions) > 0:
        for r_number in range(1,np.max(regions),1):
            if np.where(regions == r_number)[0].size < (0.4 * map.mask.shape[0]):
                regions[np.where(regions == r_number)] = 0
        regions[np.where(regions >= 1)] = 1
    map.mask = np.logical_not(regions)


def pch_quality(masked_map, hole_start, hole_end, n_hole_pixels):
    # hole_start and hole_end are tuples of (lat, lon)
    # returns a fractional hole quality between 0 and 1
    # 1 is a very well defined hole, 0 is an undefined hole (not actually possible)

    arc_width = rsun_pix(masked_map)[0] * (0.995 - 0.965)
    degree_sep = GreatArc(hole_start, hole_end).inner_angle.to(u.deg)
    arc_length = 2 * np.pi * rsun_pix(masked_map)[0] * u.pix * (degree_sep[-1]/(360. * u.deg))
    quality_ratio = n_hole_pixels / (arc_length * arc_width)

    return quality_ratio


def pick_hole_extremes(hole_coordinates):

    inner_angles = np.zeros([4,4])

    hole_lat_max = np.where(hole_coordinates.heliographic_stonyhurst.lat == hole_coordinates.heliographic_stonyhurst.lat.max())[0]
    hole_lat_min = np.where(hole_coordinates.heliographic_stonyhurst.lat == hole_coordinates.heliographic_stonyhurst.lat.min())[0]
    hole_lon_max = np.where(hole_coordinates.heliographic_stonyhurst.lon == hole_coordinates.heliographic_stonyhurst.lon.max())[0]
    hole_lon_min = np.where(hole_coordinates.heliographic_stonyhurst.lon == hole_coordinates.heliographic_stonyhurst.lon.min())[0]

    test_coords = [hole_coordinates[hole_lat_max][0], hole_coordinates[hole_lat_min][0],
                   hole_coordinates[hole_lon_max][0], hole_coordinates[hole_lon_min][0]]

    for ii, point1 in enumerate(test_coords):
        for jj, point2 in enumerate(test_coords):
            if ii > jj:
                inner_angles[ii,jj] = GreatArc(point1,point2).inner_angle.value

    return [test_coords[np.where(inner_angles == inner_angles.max())[0][0]], test_coords[np.where(inner_angles == inner_angles.max())[1][0]]]


def pch_mark(masked_map):
    # Marks the edge of a polar hole on a map and returns the stonyhurst heliographic coordinates
    # Returns an astropy table of the ['StartLat','StartLon','EndLat','EndLon','Quality']
    # for the edge points of the holes detected
    # This is a change from the IDL chole_mark in that it doesn't assume just one north and south hole.

    # Class check
    if not isinstance(masked_map, sunpy.map.GenericMap):
        raise ValueError('Input needs to be an sunpy map object.')

    # Check if mask assigned
    if (np.nanmax(masked_map.mask) <= 0) or (np.nanmin(masked_map.mask) != 0):
        return Table(names=('StartLat','StartLon','EndLat','EndLon','Quality'))

    holes = measure.label(np.logical_not(masked_map.mask).astype(int), connectivity=1, background=0)

    edge_points = Table(names=('StartLat', 'StartLon', 'EndLat', 'EndLon', 'ArcLength', 'Quality'))

    for r_number in range(1, np.max(holes)+1, 1):

        hole_coords = masked_map.pixel_to_world(np.where(holes == r_number)[1]*u.pixel, np.where(holes == r_number)[0] * u.pixel,0)

        # filtering points below 50 deg lat, i.e. not polar any more
        hole_coords = hole_coords[np.where(np.abs(hole_coords.heliographic_stonyhurst.lat) >= 50 * u.deg)]

        if hole_coords.shape[0] > 0:
            pts = pick_hole_extremes(hole_coords)
            edge_points.add_row((pts[0].heliographic_stonyhurst.lat, pts[0].heliographic_stonyhurst.lon,
                                 pts[1].heliographic_stonyhurst.lat, pts[1].heliographic_stonyhurst.lon,
                                 GreatArc(pts, pts).inner_angle.value.to(u.deg),
                                 pch_quality(masked_map, pts[0], pts[1], np.where(holes == r_number)[0].size * u.pix)))

    if not len(edge_points):
        edge_points.add_row((np.nan, np.nan, np.nan, np.nan, np.nan, 0))

    return edge_points


def image_integrity_check(inmap):
    # Runs through image checks to make sure the integrity of the full disk image
    # Looks to make sure the input data is prepped for processing

    if not isinstance(inmap, sunpy.map.GenericMap):
        raise ValueError('Input needs to be an sunpy map object.')

    # Size check
    good_image = (inmap.data.shape >= (1024,1024))

    if inmap.detector == 'AIA':
        if 'QUALITYV0' in inmap.meta:
            good_image = False

        if 'MISSVALS' in inmap.data:
            good_image = False

        if inmap.meta['Quality'] not in [2097152, 1073741824, 1073741828, 1075838976]:
            good_image = False

    if inmap.detector == 'EIT':
        if inmap.meta['OBJECT'] == 'partial FOV':
            good_image = False

    if inmap.detector == 'EUVI':
        if (len(np.where(inmap.data < 0)[0])/len(inmap.data.nonzero()[0])) > 0.75:
            good_image = False

    if inmap.detector == 'SWAP':
        if inmap.meta['LEVEL'] != 11:
            good_image = False

    return good_image


class PCH_Detection:

    def __init__(self, image_dir):

        # Quiet things a bit
        warnings.simplefilter('ignore', UserWarning)

        self.dir = os.path.abspath(image_dir)

        if 'efz' in os.listdir(self.dir)[-10]:
            self.files = [file for file in os.listdir(self.dir) if file.startswith('efz')]

        if '.fts' in os.listdir(self.dir)[-10]:
            self.files = [file for file in os.listdir(self.dir) if file.endswith('.fts')]

        if '.fits' in os.listdir(self.dir)[-10]:
            self.files = [file for file in os.listdir(self.dir) if file.startswith('.fits')]

        self.point_detection = Table([[0], [0], [0], [0], [0], [0], [''], Time('1900-01-04')],
                                     names=('StartLat', 'StartLon', 'EndLat', 'EndLon', 'ArcLength', 'Quality', 'FileName', 'Date'),
                                     meta={'name': 'Coordinate_Detections'})

        for ii, image_file in enumerate(self.files):
            solar_image = sunpy.map.Map(image_file)

            self.detector = solar_image.detector

            if ii == 0:
                self.begin_date = solar_image.date.year

            if ii == len(self.files)-1:
                self.end_date = solar_image.date.year

            if image_integrity_check(solar_image):
                pch_mask(solar_image)
                pts = pch_mark(solar_image)

                pts['FileName'] = [self.dir+'/'+image_file] * len(pts)
                pts['Date'] = [Time(solar_image.date)] * len(pts)

                self.point_detection = join(pts, self.point_detection, join_type='outer')

        self.point_detection.remove_row(0)
        self.add_harvey_coordinates()

    def add_harvey_coordinates(self):
        # Modifies the point detection to add in harvey lon.

        self.point_detection['Harvey_Rotation'] = [PCH_Tools.date2hrot(date, fractional=True) for date in self.point_detection['Date']]
        harvey_lon = np.array([PCH_Tools.get_harvey_lon(date) for date in self.point_detection['Date']]) * u.deg
        self.point_detection["H_StartLon"] = Longitude(harvey_lon + np.array(self.point_detection['StartLon']) * u.deg)
        self.point_detection["H_EndLon"] = Longitude(harvey_lon + np.array(self.point_detection['EndLon']) * u.deg)

    def write_table(self, table, write_dir=''):

        if self.begin_date == self.end_date:
            date_string = str(self.begin_date)
        else:
            date_string = str(self.begin_date)+'-'+str(self.end_date)

        if write_dir == '':
            write_dir = self.dir

        write_file = write_dir+'/'+self.detector+'_'+table.meta['name']+date_string+'.csv'

        ascii.write(table, format='ecsv', filename=write_file)

