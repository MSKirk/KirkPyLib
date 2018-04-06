import sunpy
from sunpy.map import Map 
import numpy as np
import os
from pathlib import Path
from PolarCoronalHoles import PCH_Tools
import astropy.units as u
from skimage import exposure, morphology, measure
from sunpy.coordinates.utils import GreatArc
from astropy.table import Table, join
from astropy.time import Time
from astropy.coordinates import Longitude
from astropy.io import ascii, fits
from astropy.utils.exceptions import AstropyWarning
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


def pch_mask(mask_map, factor=0.5):
    # To isolate the lim of the sun to study coronal holes
    # returns a binary array of the outer lim of the sun

    # Class check
    if not isinstance(mask_map, sunpy.map.GenericMap):
        raise ValueError('Input needs to be an sunpy map object.')

    # Bad image check
    if np.max(mask_map.data) < 1:
        mask_map.mask = np.ones_like(mask_map.data)
    else:
        # EUVI Wavelet adjustment
        if np.max(mask_map.data) < 100:
            mask_map.data[:, :] *= mask_map.data

        # Range Clipping
        mask_map.data[mask_map.data > 10000] = 10000

        mask_map.data[mask_map.data < 0] = 0

        rsun_in_pix = rsun_pix(mask_map)

        # EUVI Wavelet adjustment
        if mask_map.detector == 'EUVI':
            if mask_map.wavelength > 211*u.AA:
                   mask_map.mask = exposure.equalize_hist(mask_map.data, mask=np.logical_not(PCH_Tools.annulus_mask(mask_map.data.shape, (0,0), rsun_in_pix, center=mask_map.wcs.wcs.crpix)))
            else:
                mask_map.mask = exposure.equalize_hist(mask_map.data)
        else:
            mask_map.mask = np.copy(mask_map.data).astype('float')

        # Found through experiment...
        if mask_map.detector == 'AIA':
            if mask_map.wavelength == 193 * u.AA:
                factor = 0.30
            if mask_map.wavelength == 171 * u.AA:
                factor = 0.62
            if mask_map.wavelength == 304 * u.AA:
                factor = 0.15
        if mask_map.detector == 'EIT':
            if mask_map.wavelength == 195 * u.AA:
                factor = 0.27
            if mask_map.wavelength == 171 * u.AA:
                factor = 0.37
            if mask_map.wavelength == 304 * u.AA:
                factor = 0.14
            if mask_map.wavelength == 284 * u.AA:
                factor = 0.22
        if mask_map.detector == 'EUVI':
            if mask_map.wavelength == 195 * u.AA:
                factor = 0.53
            if mask_map.wavelength == 171 * u.AA:
                factor = 0.55
            if mask_map.wavelength == 304 * u.AA:
                factor = 0.35
            if mask_map.wavelength == 284 * u.AA:
                factor = 0.22
        if mask_map.detector == 'SWAP':
            if mask_map.wavelength == 174 * u.AA:
                factor = 0.55

        # Creating a kernel for the morphological transforms
        if mask_map.wavelength == 304 * u.AA:
            structelem = morphology.disk(np.round(np.average(rsun_in_pix) * 0.007))
        else:
            structelem = morphology.disk(np.round(np.average(rsun_in_pix) * 0.004))

        # First morphological pass...
        if mask_map.wavelength == 304 * u.AA:
            mask_map.mask = morphology.opening(morphology.closing(mask_map.mask, selem=structelem))
        else:
            mask_map.mask = morphology.closing(morphology.opening(mask_map.mask, selem=structelem))

        # Masking off limb structures and Second morphological pass...
        mask_map.mask = morphology.opening(mask_map.mask * PCH_Tools.annulus_mask(mask_map.data.shape, (0,0), rsun_in_pix, center=mask_map.wcs.wcs.crpix), selem=structelem)

        # Extracting holes...
        thresh = PCH_Tools.hist_percent(mask_map.mask[np.nonzero(mask_map.mask)], factor, number_of_bins=1000)
        mask_map.mask = np.where(mask_map.mask >= thresh, mask_map.mask, 0)

        # Extracting annulus
        mask_map.mask[PCH_Tools.annulus_mask(mask_map.data.shape, rsun_in_pix*0.965, rsun_in_pix*0.995, center=mask_map.wcs.wcs.crpix) == False] = np.nan

        # Filter for hole size scaled to resolution of the image

        regions = measure.label(np.logical_not(mask_map.mask).astype(int), connectivity=1, background=0)

        if np.max(regions) > 0:
            for r_number in range(1,np.max(regions),1):
                if np.where(regions == r_number)[0].size < (0.4 * mask_map.mask.shape[0]):
                    regions[np.where(regions == r_number)] = 0
            regions[np.where(regions >= 1)] = 1
        mask_map.mask = np.logical_not(regions)


def pch_quality(masked_map, hole_start, hole_end, n_hole_pixels):
    # hole_start and hole_end are tuples of (lat, lon)
    # returns a fractional hole quality between 0 and 1
    # 1 is a very well defined hole, 0 is an undefined hole (not actually possible)

    # something is not working here - continually returns values over 1

    arc_width = rsun_pix(masked_map)[0] * (0.995 - 0.965)
    degree_sep = GreatArc(hole_start, hole_end).inner_angle.to(u.deg)
    arc_length = 2 * np.pi * rsun_pix(masked_map)[0] * u.pix * (degree_sep/(360. * u.deg))
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

    edge_points = Table(names=('StartLat', 'StartLon', 'EndLat', 'EndLon', 'ArcLength', 'Quality'))

    # Check if mask assigned
    if masked_map.mask.all() or not masked_map.mask.any():
        return edge_points

    holes = measure.label(np.logical_not(masked_map.mask).astype(int), connectivity=1, background=0)

    for r_number in range(1, holes.max()+1, 1):

        hole_xx = np.where(holes == r_number)[0] * u.pixel
        hole_yy = np.where(holes == r_number)[1] * u.pixel

        # Double check Size Filter
        if len(hole_xx) > 10:

            hole_coords = masked_map.pixel_to_world(hole_yy, hole_xx,0)

            # filtering points below 50 deg lat, i.e. not polar any more

            only_polar_holes = np.where(np.abs(hole_coords.heliographic_stonyhurst.lat) >= 50 * u.deg)

            if only_polar_holes[0].size > 0:
                hole_coords = hole_coords[only_polar_holes]
                pts = pick_hole_extremes(hole_coords)
                edge_points.add_row((pts[0].heliographic_stonyhurst.lat, pts[0].heliographic_stonyhurst.lon,
                                    pts[1].heliographic_stonyhurst.lat, pts[1].heliographic_stonyhurst.lon,
                                    GreatArc(pts[0], pts[1]).inner_angle.to(u.deg),
                                    pch_quality(masked_map, pts[0], pts[1], np.where(holes == r_number)[0].size * u.pix)))
                print(edge_points)
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

        if inmap.meta['OBJECT'] == 'Dark':
            good_image = False

    if (len(np.where(inmap.data <= 0)[0])/(float(len(np.where(inmap.data > 0)[0]))+0.1)) > 0.75:
        good_image = False

    if inmap.detector == 'SWAP':
        if inmap.meta['LEVEL'] != 11:
            good_image = False

    if inmap.data.max() < 1:
        good_image = False

    return good_image


def file_integrity_check(infile):
    # Makes sure it can be fed to the Map
    warnings.simplefilter('ignore', AstropyWarning)

    file_path = Path(infile)

    if file_path.is_file():
        hdu1 = fits.open(file_path)
        hdu1[0].verify('fix')

        # SWAP Stacked image fix
        if 'CTYPE1' not in hdu1[0].header:
            hdu1[0].header['CTYPE1'] = 'HPLN-TAN'

        if 'CTYPE2' not in hdu1[0].header:
            hdu1[0].header['CTYPE2'] = 'HPLT-TAN'
            hdu1.writeto(file_path, overwrite=True)

        if 'NAXIS3' in hdu1[0].header:
            return False
        if type(hdu1[0].header['CDELT1']) != float:
            return False
        if type(hdu1[0].header['CDELT2']) != float:
            return False
        else:
            return True
    else:
        return False


class PCH_Detection:

    def __init__(self, image_dir):

        # Quiet things a bit
        warnings.simplefilter('ignore', UserWarning)

        self.dir = os.path.abspath(image_dir)

        self.files = []

        for root, dirnames, filenames in os.walk(image_dir):
            for filename in filenames:
                if filename.endswith(('.fts', '.fits') or filename.startswith('efz')):
                    self.files.append(os.path.join(root, filename))

        self.point_detection = Table([[0], [0], [0], [0], [0], [0], [''], [Time('1900-01-04')]],
                                     names=('StartLat', 'StartLon', 'EndLat', 'EndLon', 'ArcLength', 'Quality', 'FileName', 'Date'),
                                     meta={'name': 'Coordinate_Detections'})

        for ii, image_file in enumerate(self.files):

            if file_integrity_check(image_file):
                solar_image = sunpy.map.Map(image_file)

                print(image_file)

                self.detector = solar_image.detector

                if ii == 0:
                    self.begin_date = solar_image.date.year

                if ii == len(self.files)-1:
                    self.end_date = solar_image.date.year

                if image_integrity_check(solar_image):
                    pch_mask(solar_image)
                    pts = pch_mark(solar_image)

                    if len(pts) > 0:
                        pts['FileName'] = [image_file] * len(pts)
                        pts['Date'] = [Time(solar_image.date)] * len(pts)

                        self.point_detection = join(pts, self.point_detection, join_type='outer')

        self.point_detection.remove_row(np.where(self.point_detection['ArcLength'] == 0)[0][0])
        self.add_harvey_coordinates()
        self.point_detection.sort(['Harvey_Rotation'])

        # Adding in Area Calculation each point with one HR previous measurements
        self.point_detection['Area'] = [hole_area(h_rot)for h_rot in self.point_detection['Harvey_Rotation']]

    def hole_area(self, h_rotation_number):
        begin = np.min(np.where(self.point_detection['Harvey_Rotation'] > (h_rotation_number - 1)))
        end = np.max(np.where(self.point_detection['Harvey_Rotation'] == h_rotation_number))

        if self.point_detection[h_rotation_number]['StartLat'] > 0:
            # A northern hole
            index_measurements = np.where(self.point_detection[begin:end]['StartLat'] > 0)
        else:
            # A southern hole
            index_measurements = np.where(self.point_detection[begin:end]['StartLat'] < 0)

        if len(index_measurements[0]) < 10:
            return np.nan
        else:
            lons = np.concatenate([self.point_detection[index_measurements]['H_StartLon'].data.data,
                                   self.point_detection[index_measurements]['H_EndLon'].data.data])
            lats = np.concatenate([self.point_detection[index_measurements]['StartLat'].data.data,
                                   self.point_detection[index_measurements]['EndLat'].data.data])


            hole_fit = PCH_Tools.trigfit(lons, lats, degree=6)

            # Need fit with several degrees
            # Need to define error in trigfit
            # Need to off set center of mass
            # Need to Find area within curve


    def add_harvey_coordinates(self):
        # Modifies the point detection to add in harvey lon.

        self.point_detection['Harvey_Rotation'] = [PCH_Tools.date2hrot(date, fractional=True) for date in self.point_detection['Date']]
        harvey_lon = np.array([PCH_Tools.get_harvey_lon(date) for date in self.point_detection['Date']]).reshape(self.point_detection['Date'].size) * u.deg
        self.point_detection['H_StartLon'] = Longitude(harvey_lon + np.array(self.point_detection['StartLon']) * u.deg)
        self.point_detection['H_EndLon'] = Longitude(harvey_lon + np.array(self.point_detection['EndLon']) * u.deg)

    def write_table(self, write_dir=''):

        if self.begin_date == self.end_date:
            date_string = str(self.begin_date)
        else:
            date_string = str(self.begin_date)+'-'+str(self.end_date)

        if write_dir == '':
            write_dir = self.dir

        write_file = write_dir+'/'+self.detector+'_'+self.point_detection.meta['name']+date_string+'.csv'

        ascii.write(self.point_detection, write_file, format='ecsv')

