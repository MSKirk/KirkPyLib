import os, gc
import numpy as np
import warnings
from pathlib import Path
from skimage import exposure, morphology, measure

import PCH_Tools

import sunpy
from sunpy.coordinates.utils import GreatArc
from sunpy.coordinates.ephemeris import get_horizons_coord
from sunpy.map import Map

import astropy.units as u
from astropy.table import Table, join
from astropy.time import Time
from astropy.coordinates import Longitude
from astropy.io import ascii, fits
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning

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

    try:
        rsun = np.array([(inmap.rsun_arcseconds * u.arcsec).to('deg').value])
    except AttributeError:
        rsun = np.array([inmap.rsun_obs.to('deg').value])

    # Fix for missing Solar Radius information from EIT
    if rsun == 0:
        if inmap.detector == 'EIT':
            rsun = 0.27346886111111113
        else:
            rsun = np.nan

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

        mask_map.data[mask_map.data < 1] = 1

        mask_map.data[:] = mask_map.data[:] - mask_map.data.min()

        rsun_in_pix = rsun_pix(mask_map)

        # EUVI Wavelet adjustment
        # Give EIT some Love.
        if mask_map.detector in ['EUVI', 'EIT']:
            if mask_map.wavelength > 211*u.AA:
                   mask_map.mask = exposure.equalize_hist(mask_map.data, mask=np.logical_not(PCH_Tools.annulus_mask(mask_map.data.shape, (0,0), rsun_in_pix, center=mask_map.data.shape - np.flip(mask_map.wcs.wcs.crpix) -1)))
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
                #factor = 0.65
            if mask_map.wavelength == 304 * u.AA:
                factor = 0.15
            if mask_map.wavelength == 211 * u.AA:
                factor = 0.20
        if mask_map.detector == 'EIT':
            if mask_map.wavelength == 195 * u.AA:
                factor = 0.17
            if mask_map.wavelength == 171 * u.AA:
                factor = 0.30
            if mask_map.wavelength == 304 * u.AA:
                factor = 0.10
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
                #factor = 0.55
                factor = 0.78

        # Creating a kernel for the morphological transforms
        if mask_map.wavelength == 304 * u.AA:
            structelem = morphology.disk(np.round(np.average(rsun_in_pix * mask_map.meta['cdelt1']) * 0.007))
        else:
            structelem = morphology.disk(np.round(np.average(rsun_in_pix * mask_map.meta['cdelt1']) * 0.004))

        # First morphological pass...
        if mask_map.wavelength == 304 * u.AA:
            mask_map.mask = morphology.opening(morphology.closing(mask_map.mask, selem=structelem))
        else:
            mask_map.mask = morphology.closing(morphology.opening(mask_map.mask, selem=structelem))

        # Masking off limb structures and Second morphological pass...
        mask_map.mask = morphology.opening(mask_map.mask * PCH_Tools.annulus_mask(mask_map.data.shape, (0,0), rsun_in_pix, center=mask_map.data.shape - np.flip(mask_map.wcs.wcs.crpix)-1), selem=structelem)

        # Extracting holes...
        try:
            thresh = PCH_Tools.hist_percent(mask_map.mask[np.nonzero(mask_map.mask)], factor, number_of_bins=1000)
            mask_map.mask = np.where(mask_map.mask >= thresh, mask_map.mask, 0)
        except ValueError:
            mask_map.mask = np.ones_like(mask_map.mask)

        # Extracting annulus
        mask_map.mask[PCH_Tools.annulus_mask(mask_map.data.shape, rsun_in_pix*0.965, rsun_in_pix*0.995, center=mask_map.data.shape - np.flip(mask_map.wcs.wcs.crpix-1)) == False] = np.nan

        # Filter for hole size scaled to resolution of the image

        regions = measure.label(np.logical_not(mask_map.mask).astype(int), connectivity=1, background=0)

        if np.max(regions) > 0:
            for r_number in range(1,np.max(regions),1):
                if np.where(regions == r_number)[0].size < (0.4 * mask_map.mask.shape[0]):
                    regions[np.where(regions == r_number)] = 0
            regions[np.where(regions >= 1)] = 1
        mask_map.mask = np.logical_not(regions)

    # Explicity replacing the ephemeris of SOHO maps (can add other observatories if needed)
    if mask_map.observatory in ['SOHO']:
        new_coords = get_horizons_coord(mask_map.observatory.replace(' ', '-'), mask_map.date)
        mask_map.meta['HGLN_OBS'] = new_coords.lon.to('deg').value
        mask_map.meta['HGLT_OBS'] = new_coords.lat.to('deg').value
        mask_map.meta['DSUN_OBS'] = new_coords.radius.to('m').value

        mask_map.meta.pop('hec_x')
        mask_map.meta.pop('hec_y')
        mask_map.meta.pop('hec_z')


def pch_quality(masked_map, hole_start, hole_end, n_hole_pixels):
    # hole_start and hole_end are tuples of (lat, lon)
    # returns a fractional hole quality between 0 and 1
    # 1 is a very well defined hole, 0 is an undefined hole (not actually possible)

    arc_width = rsun_pix(masked_map)[0] * (0.995 - 0.965)
    degree_sep = GreatArc(hole_start, hole_end).inner_angle.to(u.deg)
    arc_length = 2 * np.pi * rsun_pix(masked_map)[0] * u.pix * (degree_sep/(360. * u.deg))
    quality_ratio = n_hole_pixels / (arc_length * arc_width)

    return quality_ratio


def pick_hole_extremes(hole_coordinates):
    # picks the points in the detected hole that are the furthest from each other

    hole_lat_max = np.where(hole_coordinates.heliographic_stonyhurst.lat == hole_coordinates.heliographic_stonyhurst.lat.max())[0]
    hole_lat_min = np.where(hole_coordinates.heliographic_stonyhurst.lat == hole_coordinates.heliographic_stonyhurst.lat.min())[0]
    hole_lon_max = np.where(hole_coordinates.heliographic_stonyhurst.lon == hole_coordinates.heliographic_stonyhurst.lon.max())[0]
    hole_lon_min = np.where(hole_coordinates.heliographic_stonyhurst.lon == hole_coordinates.heliographic_stonyhurst.lon.min())[0]

    test_coords_loc = [hole_lat_max[0], hole_lat_min[0], hole_lon_max[0], hole_lon_min[0]]
    max_sep = [hole_coordinates[hole_pix_loc].separation(hole_coordinates).max().value for hole_pix_loc in test_coords_loc]

    beginning_point = hole_coordinates[test_coords_loc[max_sep.index(max(max_sep))]]
    ending_point = hole_coordinates[np.where(beginning_point.separation(hole_coordinates) == beginning_point.separation(hole_coordinates).max())[0][0]]

    return beginning_point, ending_point


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

        if len(hole_xx) > 10:

            hole_coords = masked_map.pixel_to_world(hole_yy, hole_xx,0)

            # filtering points below 50 deg lat, i.e. not polar any more

            only_polar_holes = np.where(np.abs(hole_coords.heliographic_stonyhurst.lat) >= 50 * u.deg)

            if only_polar_holes[0].size > 0:
                hole_coords = hole_coords[only_polar_holes]
                hole_start, hole_end = pick_hole_extremes(hole_coords)
                pchq = pch_quality(masked_map, hole_start, hole_end, np.sum(holes == r_number) * u.pix)
                if pchq > 1:
                    pchq = 1.0
                edge_points.add_row((hole_start.heliographic_stonyhurst.lat, hole_start.heliographic_stonyhurst.lon,
                                    hole_end.heliographic_stonyhurst.lat, hole_end.heliographic_stonyhurst.lon,
                                    GreatArc(hole_start, hole_end).inner_angle.to(u.deg), pchq))

    # Adding in a Zero detection in case of no detection
    if not (edge_points['StartLat'] > 0).any():
        edge_points.add_row((90., 0, 90., 0, np.nan, 1.0))
    if not (edge_points['StartLat'] < 0).any():
        edge_points.add_row((-90., 0, -90., 0, np.nan, 1.0))

    # if only small holes are detected, add in a zero point which won't be filtered out
    if (edge_points['ArcLength'][np.where(edge_points['StartLat'] > 0)] < 3).any():
        edge_points.add_row((90., 0, 90., 0, np.nan, 1.0))
    if (edge_points['ArcLength'][np.where(edge_points['StartLat'] < 0)] < 3).any():
        edge_points.add_row((90., 0, 90., 0, np.nan, 1.0))

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

        if inmap.meta['MISSVALS'] > 0.01 * inmap.meta['TOTVALS']:
            good_image = False

        if inmap.meta['Quality'] not in [0, 2097152, 1073741824, 1073741828, 1075838976]:
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

    if not good_image:
        warnings.warn('Not Expected Level or Quality')

    return good_image


def file_integrity_check(infile):
    # Makes sure it can be fed to the Map
    warnings.simplefilter('ignore', AstropyWarning)

    file_path = Path(infile)

    if file_path.is_file():
        hdu1 = fits.open(file_path)

        head_loc = np.argmax([len(hdu.header) for hdu in hdu1])
            
        hdu1[head_loc].verify('fix')
        rewrite_needed = False

        # SWAP Stacked image fix
        if 'CTYPE1' not in hdu1[head_loc].header:
            hdu1[head_loc].header['CTYPE1'] = 'HPLN-TAN'
            rewrite_needed = True

        if 'CTYPE2' not in hdu1[head_loc].header:
            hdu1[head_loc].header['CTYPE2'] = 'HPLT-TAN'
            rewrite_needed = True

        if 'CUNIT1' not in hdu1[head_loc].header:
            hdu1[head_loc].header['CUNIT1'] = 'arcsec'
            rewrite_needed = True

        if 'CUNIT2' not in hdu1[head_loc].header:
            hdu1[head_loc].header['CUNIT2'] = 'arcsec'
            rewrite_needed = True

        if rewrite_needed:
            try:
                hdu1.writeto(file_path, overwrite=True)
            except (ValueError, fits.verify.VerifyError):
                print(f'Bad file: {file_path}')
                return False

        if 'NAXIS3' in hdu1[head_loc].header:
            warnings.warn('NAXIS greater than 2')
            return False

        #try:
        #    type(hdu1[head_loc].header['CRVAL1'])
        #except KeyError:
        #    warnings.warn('No CRVAL1 Specified')
        #    return False
        #if type(hdu1[head_loc].header['CRVAL1']) != float:
        #    warnings.warn('No CRVAL1 Specified')
        #    return False

        #try:
        #    type(hdu1[head_loc].header['CRVAL2'])
        #except KeyError:
        #    warnings.warn('No CRVAL2 Specified')
        #    return False
        #if type(hdu1[head_loc].header['CRVAL2']) != float:
        #    warnings.warn('No CRVAL2 Specified')
        #    return False

        try:
            type(hdu1[head_loc].header['CDELT1'])
        except KeyError:
            warnings.warn('No CDELT1 Detected')
            return False
        if type(hdu1[head_loc].header['CDELT1']) != float:
            warnings.warn('No CDELT1 Detected')
            return False

        try:
            type(hdu1[head_loc].header['CDELT2'])
        except KeyError:
            warnings.warn('No CDELT2 Detected')
            return False
        if type(hdu1[head_loc].header['CDELT2']) != float:
            warnings.warn('No CDELT2 Detected')
            return False
        else:
            return True
    else:
        warnings.warn(f'{file_path} is not a fits file')
        return False


class PCH_Detection:
    warnings.simplefilter('ignore', category=AstropyDeprecationWarning)

    def __init__(self, image_dir):

        # Quiet things a bit
        warnings.simplefilter('ignore', Warning)

        self.dir = os.path.abspath(image_dir)

        self.files = []

        for root, dirnames, filenames in os.walk(image_dir):
            for filename in filenames:
                if filename.endswith(('.fts', '.fits')) or filename.startswith('efz'):
                    self.files.append(os.path.join(root, filename))

        # # Check the CoM additions.
        self.point_detection = Table([[0], [0], [0], [0], [0], [0], [''], [Time('1900-01-04')]],
                                     names=('StartLat', 'StartLon', 'EndLat', 'EndLon', 'ArcLength', 'Quality', 'FileName', 'Date'),
                                     meta={'name': 'PCH_Detections'})

        for ii, image_file in enumerate(self.files):

            if file_integrity_check(image_file):
                solar_image = sunpy.map.Map(image_file)

                print(image_file)

                self.detector = solar_image.detector
                self.wavelength = solar_image.wavelength

                if image_integrity_check(solar_image):

                    # Resample AIA images to lower resolution for better processing times
                    if solar_image.dimensions[0] > 2049. * u.pixel:
                        dimensions = u.Quantity([2048, 2048], u.pixel)
                        solar_image = solar_image.resample(dimensions)

                    pch_mask(solar_image)
                    pts = pch_mark(solar_image)

                    print(pts)

                    if len(pts) > 0:
                        pts['FileName'] = [os.path.basename(image_file)] * len(pts)
                        pts['Date'] = [Time(solar_image.date)] * len(pts)

                        self.point_detection = join(pts, self.point_detection, join_type='outer')
                    # Variable Cleanup
                    del pts
                # Variable Cleanup
                del solar_image

        self.point_detection.remove_row(np.where(self.point_detection['Date'] == Time('1900-01-04'))[0][0])
        self.add_harvey_coordinates()
        self.point_detection.sort(['Harvey_Rotation'])

        self.begin_date = self.point_detection['Date'][0]
        self.end_date = self.point_detection['Date'][-1]

        # Adding in Area Calculation each point with one HR previous measurements
        area = []
        fit = []
        center = []
        for ii, h_rot in enumerate(self.point_detection['Harvey_Rotation']):
            if self.point_detection[ii]['StartLat'] > 0:
                northern = True
            else:
                northern = False
            ar, ft, cm = self.hole_area(h_rot, northern=northern)
            area = area + [ar]
            fit = fit + [ft]
            center = center + [cm]
        center = np.vstack([arr[0:2] for arr in center])

        self.point_detection['Area'] = np.asarray(area)[:, 1]
        self.point_detection['Area_min'] = np.asarray(area)[:, 0]
        self.point_detection['Area_max'] = np.asarray(area)[:, 2]

        self.point_detection['Fit'] = np.asarray(fit)[:, 1] * u.deg
        self.point_detection['Fit_min'] = np.asarray(fit)[:, 0] * u.deg
        self.point_detection['Fit_max'] = np.asarray(fit)[:, 2] * u.deg

        self.point_detection['Center_lat'] = center[:, 1] * u.deg
        self.point_detection['Center_lon'] = center[:, 0] * u.deg

    def hole_area(self, h_rotation_number, northern=True):
        # Returns the area as a fraction of the total solar surface area
        # Returns the location of the perimeter fit for the given h_rotation_number

        begin = np.min(np.where(self.point_detection['Harvey_Rotation'] > (h_rotation_number - 1)))
        end = np.max(np.where(self.point_detection['Harvey_Rotation'] == h_rotation_number))

        if northern:
            # A northern hole with Arclength Filter for eliminating small holes but not zeros
            index_measurements = np.where((self.point_detection[begin:end]['StartLat'] > 0) & np.logical_not(
                self.point_detection[begin:end]['ArcLength'] < 3.0))
        else:
            # A southern hole with Arclength Filter for eliminating small holes
            index_measurements = np.where((self.point_detection[begin:end]['StartLat'] < 0) & np.logical_not(
                self.point_detection[begin:end]['ArcLength'] < 3.0))

        index_measurements += begin

        # Filters for incomplete hole measurements: at least 10 points and half a harvey rotation needs to be defined
        if len(index_measurements[0]) < 10:
            return np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])

        elif self.point_detection['Harvey_Rotation'][index_measurements[0][-1]] - self.point_detection['Harvey_Rotation'][index_measurements[0][0]] < 0.5:
            return np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])

        else:
            lons = np.concatenate(np.vstack([self.point_detection[index_measurements]['H_StartLon'].data.data,
                                             self.point_detection[index_measurements]['H_EndLon'].data.data])) * u.deg
            lats = np.concatenate(np.vstack([self.point_detection[index_measurements]['StartLat'].data.data,
                                             self.point_detection[index_measurements]['EndLat'].data.data])) * u.deg
            errors = np.concatenate(np.vstack([np.asarray(1/self.point_detection[index_measurements]['Quality']),
                                               np.asarray(1/self.point_detection[index_measurements]['Quality'])]))

            perimeter_length = np.zeros(6) * u.rad
            fit_location = np.zeros(6) * u.rad
            hole_area = np.zeros(6)

            # Centroid offset for better fitting results
            if northern:
                offset_coords = np.transpose(np.asarray([lons, (90 * u.deg) - lats]))
            else:
                offset_coords = np.transpose(np.asarray([lons, (90 * u.deg) + lats]))

            offset_cm = PCH_Tools.center_of_mass(offset_coords, mass=1/errors) * u.deg
            offset_lons = np.mod(offset_coords[:, 0] * u.deg - offset_cm[0], 360* u.deg)
            offset_lats = offset_coords[:, 1] * u.deg - offset_cm[1]

            for ii, degrees in enumerate([4, 5, 6, 7, 8, 9]):
                try:
                    hole_fit = PCH_Tools.trigfit(np.deg2rad(offset_lons), np.deg2rad(offset_lats), degree=degrees,
                                                 sigma=errors)

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
            good_areas = hole_area[np.where((perimeter_length / (2*np.pi*u.rad)) -1 < 0.05)]
            good_fits = fit_location[np.where((perimeter_length / (2*np.pi*u.rad)) -1 < 0.05)]

            # A sphere is 4π steradians in surface area
            if good_areas.size > 0:
                percent_hole_area = (np.nanmin(good_areas) / (4 * np.pi), np.nanmean(good_areas) / (4 * np.pi), np.nanmax(good_areas) / (4 * np.pi))
                # in degrees
                hole_perimeter_location = (np.rad2deg(np.nanmin(good_fits)).value, np.rad2deg(np.nanmean(good_fits)).value, np.rad2deg(np.nanmax(good_fits)).value)
            else:
                percent_hole_area = (np.nan, np.nan, np.nan)
                hole_perimeter_location = np.array([np.nan, np.nan, np.nan])

            # From co-lat to lat

            if northern:
                if offset_cm[1] < 0:
                    offset_cm[1] = (90 * u.deg) + offset_cm[1]
                    offset_cm[0] = np.mod(offset_cm[0]-180*u.deg, 360* u.deg)
                else:
                    offset_cm[1] = (90 * u.deg) - offset_cm[1]
            else:
                if offset_cm[1] > 0:
                    offset_cm[1] = (-90 * u.deg) + offset_cm[1]
                else:
                    offset_cm[1] = (-90 * u.deg) - offset_cm[1]
                    offset_cm[0] = np.mod(offset_cm[0]-180*u.deg, 360* u.deg)

            # Tuples of shape (Min, Mean, Max)
            return np.asarray(percent_hole_area), np.asarray(hole_perimeter_location), np.asarray(offset_cm)

    def add_harvey_coordinates(self):
        # Modifies the point detection to add in harvey lon.

        self.point_detection['Harvey_Rotation'] = [PCH_Tools.date2hrot(date, fractional=True) for date in self.point_detection['Date']]
        self.point_detection['Harvey_Longitude'] = np.squeeze([PCH_Tools.get_harvey_lon(date) for date in self.point_detection['Date']])
        self.point_detection['H_StartLon'] = Longitude((np.squeeze(np.asarray(self.point_detection['Harvey_Longitude']))
                                                       + np.array(self.point_detection['StartLon'])) * u.deg)
        self.point_detection['H_EndLon'] = Longitude((np.squeeze(np.asarray(self.point_detection['Harvey_Longitude']))
                                                     + np.array(self.point_detection['EndLon'])) * u.deg)

    def write_table(self, write_dir='', format=''):

        if self.begin_date.datetime.year == self.end_date.datetime.year:
            date_string = str(self.begin_date.datetime.year)
        else:
            date_string = str(self.begin_date.datetime.year)+'-'+str(self.end_date.datetime.year)

        if write_dir == '':
            write_dir = self.dir

        write_file_name = self.detector+np.str(np.int(self.wavelength.value))+'_'+self.point_detection.meta['name']+date_string
        write_file = os.path.abspath(write_dir)+'/'+ write_file_name

        if format.lower() == 'votable':
            self.point_detection.write(write_file+'.vot', format='votable', overwrite=True)
        elif format.lower() == 'hdf':
            self.point_detection['Date'] = [datetime.jd for datetime in self.point_detection['Date']]
            self.point_detection['FileName'] = self.point_detection['FileName'].astype('S60')
            self.point_detection.write(write_file_name+'.hdf', path=os.path.abspath(write_dir), format='hdf5', overwrite=True)
        elif format.lower() == 'ascii':
            ascii.write(self.point_detection, write_file+'.csv', format='ecsv', overwrite=True)
        elif format.lower() == 'csv':
            ascii.write(self.point_detection, write_file + '.csv', format='csv', overwrite=True)
        elif format.lower() == 'fits':
            self.point_detection.write(write_file+'.fits', format='fits', overwrite=True)
        else:
            ascii.write(self.point_detection, write_file + '.csv', format='ecsv', overwrite=True)


def run_detection(image_file):
    pts = Table([[0], [0], [0], [0], [0], [0], [''], [Time('1900-01-04')]],
                            names=(
                            'StartLat', 'StartLon', 'EndLat', 'EndLon', 'ArcLength', 'Quality', 'FileName', 'Date'),
                            meta={'name': 'PCH_Detections'})

    if file_integrity_check(image_file):
        solar_image = sunpy.map.Map(image_file)

        print(image_file)

        if image_integrity_check(solar_image):

            # Resample AIA images to lower resolution for better processing times
            if solar_image.dimensions[0] > 2049. * u.pixel:
                dimensions = u.Quantity([2048, 2048], u.pixel)
                solar_image = solar_image.resample(dimensions)

            try:
                pch_mask(solar_image)
            except ValueError:
                solar_image.mask = np.zeros_like(solar_image.data[:])
            pts = pch_mark(solar_image)
            print(pts)

            if len(pts) > 0:
                pts['FileName'] = [os.path.basename(image_file)] * len(pts)
                pts['Date'] = [Time(solar_image.date)] * len(pts)

        # Variable Cleanup
        del solar_image

    return pts


def hole_area(point_detection, h_rotation_number, northern=True):
    # Returns the area as a fraction of the total solar surface area
    # Returns the location of the perimeter fit for the given h_rotation_number

    begin = np.min(np.where(point_detection['Harvey_Rotation'] > (h_rotation_number - 1)))
    end = np.max(np.where(point_detection['Harvey_Rotation'] == h_rotation_number))

    if northern:
        # A northern hole with Arclength Filter for eliminating small holes but not zeros
        index_measurements = np.where((point_detection[begin:end]['StartLat'] > 0) & np.logical_not(
            point_detection[begin:end]['ArcLength'] < 3.0))
    else:
        # A southern hole with Arclength Filter for eliminating small holes
        index_measurements = np.where((point_detection[begin:end]['StartLat'] < 0) & np.logical_not(
            point_detection[begin:end]['ArcLength'] < 3.0))

    index_measurements += begin

    # Filters for incomplete hole measurements: at least 10 points and half a harvey rotation needs to be defined
    if len(index_measurements[0]) < 10:
        return np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])

    elif point_detection['Harvey_Rotation'][index_measurements[0][-1]] - point_detection['Harvey_Rotation'][index_measurements[0][0]] < 0.5:
        return np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])

    else:
        lons = np.concatenate(np.vstack([point_detection[index_measurements]['H_StartLon'].data.data,
                                         point_detection[index_measurements]['H_EndLon'].data.data])) * u.deg
        lats = np.concatenate(np.vstack([point_detection[index_measurements]['StartLat'].data.data,
                                         point_detection[index_measurements]['EndLat'].data.data])) * u.deg
        errors = np.concatenate(np.vstack([np.asarray(1/point_detection[index_measurements]['Quality']),
                                           np.asarray(1/point_detection[index_measurements]['Quality'])]))

        perimeter_length = np.zeros(6) * u.rad
        fit_location = np.zeros(6) * u.rad
        hole_area = np.zeros(6)

        # Centroid offset for better fitting results
        if northern:
            offset_coords = np.transpose(np.asarray([lons, (90 * u.deg) - lats]))
        else:
            offset_coords = np.transpose(np.asarray([lons, (90 * u.deg) + lats]))

        offset_cm = PCH_Tools.center_of_mass(offset_coords, mass=1/errors) * u.deg
        offset_lons = np.mod(offset_coords[:, 0] * u.deg - offset_cm[0], 360* u.deg)
        offset_lats = offset_coords[:, 1] * u.deg - offset_cm[1]

        for ii, degrees in enumerate([4, 5, 6, 7, 8, 9]):
            try:
                hole_fit = PCH_Tools.trigfit(np.deg2rad(offset_lons), np.deg2rad(offset_lats), degree=degrees,
                                             sigma=errors)

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
        good_areas = hole_area[np.where((perimeter_length / (2*np.pi*u.rad)) -1 < 0.05)]
        good_fits = fit_location[np.where((perimeter_length / (2*np.pi*u.rad)) -1 < 0.05)]

        # A sphere is 4π steradians in surface area
        if good_areas.size > 0:
            percent_hole_area = (np.nanmin(good_areas) / (4 * np.pi), np.nanmean(good_areas) / (4 * np.pi), np.nanmax(good_areas) / (4 * np.pi))
            # in degrees
            hole_perimeter_location = (np.rad2deg(np.nanmin(good_fits)).value, np.rad2deg(np.nanmean(good_fits)).value, np.rad2deg(np.nanmax(good_fits)).value)
        else:
            percent_hole_area = (np.nan, np.nan, np.nan)
            hole_perimeter_location = np.array([np.nan, np.nan, np.nan])

        # From co-lat to lat

        if northern:
            if offset_cm[1] < 0:
                offset_cm[1] = (90 * u.deg) + offset_cm[1]
                offset_cm[0] = np.mod(offset_cm[0]-180*u.deg, 360* u.deg)
            else:
                offset_cm[1] = (90 * u.deg) - offset_cm[1]
        else:
            if offset_cm[1] > 0:
                offset_cm[1] = (-90 * u.deg) + offset_cm[1]
            else:
                offset_cm[1] = (-90 * u.deg) - offset_cm[1]
                offset_cm[0] = np.mod(offset_cm[0]-180*u.deg, 360* u.deg)

        # Tuples of shape (Min, Mean, Max)
        return np.asarray(percent_hole_area), np.asarray(hole_perimeter_location), np.asarray(offset_cm)


def hole_area_parallel(index):
    # Returns the area as a fraction of the total solar surface area
    # Returns the location of the perimeter fit for the given h_rotation_number

    warnings.simplefilter('ignore', category=AstropyDeprecationWarning)

    point_detection = Table(ascii.read('/Users/mskirk/data/PCH_Project/Temp.csv'))

    if point_detection[index]['StartLat'] > 0:
        northern = True
    else:
        northern = False

    h_rotation_number = point_detection['Harvey_Rotation'][index]

    begin = np.min(np.where(point_detection['Harvey_Rotation'] > (h_rotation_number - 1)))
    end = np.max(np.where(point_detection['Harvey_Rotation'] == h_rotation_number))

    if northern:
        # A northern hole
        index_measurements = np.where(point_detection[begin:end]['StartLat'] > 0)[0]
        null_fit = (90, 90, 90)
        null_center = (90, 0)
    else:
        # A southern hole
        index_measurements = np.where(point_detection[begin:end]['StartLat'] < 0)[0]
        null_fit = (-90, -90, -90)
        null_center = (-90, 0)

    index_measurements += begin

    # Filters for incomplete hole measurements: at least 10 points and a quarter harvey rotation needs to be defined
    if len(index_measurements) < 10:
        return {'Index': index, 'area': (np.nan, np.nan, np.nan), 'fit': (np.nan, np.nan, np.nan),
                'center': (np.nan, np.nan)}
    elif point_detection['Harvey_Rotation'][index_measurements[-1]] - point_detection['Harvey_Rotation'][index_measurements[0]] < 0.25:
        return {'Index': index, 'area': (np.nan, np.nan, np.nan), 'fit': (np.nan, np.nan, np.nan),
                'center': (np.nan, np.nan)}
    elif np.sum(point_detection[index_measurements]['ArcLength'] > 3.0) < 10:
        return {'Index': index, 'area': (0, 0, 0), 'fit': null_fit,
                'center': null_center}
    else:
        # small hole filter
        index_measurements = index_measurements[np.where(point_detection[index_measurements]['ArcLength'] > 3.0)]

        lons = np.concatenate(np.vstack([point_detection[index_measurements]['H_StartLon'].data.data,
                                         point_detection[index_measurements]['H_EndLon'].data.data])) * u.deg
        lats = np.concatenate(np.vstack([point_detection[index_measurements]['StartLat'].data.data,
                                         point_detection[index_measurements]['EndLat'].data.data])) * u.deg
        errors = np.concatenate(np.vstack([np.asarray(1/point_detection[index_measurements]['Quality']),
                                           np.asarray(1/point_detection[index_measurements]['Quality'])]))

        perimeter_length = np.zeros(6) * u.rad
        fit_location = np.zeros(6) * u.rad
        hole_area = np.zeros(6)

        # Centroid offset for better fitting results
        if northern:
            offset_coords = np.transpose(np.asarray([lons, (90 * u.deg) - lats]))
        else:
            offset_coords = np.transpose(np.asarray([lons, (90 * u.deg) + lats]))

        offset_cm = PCH_Tools.center_of_mass(offset_coords, mass=1/errors) * u.deg
        offset_lons = np.mod(offset_coords[:, 0] * u.deg - offset_cm[0], 360* u.deg)
        offset_lats = offset_coords[:, 1] * u.deg - offset_cm[1]

        for ii, degrees in enumerate([4, 5, 6, 7, 8, 9]):
            try:
                hole_fit = PCH_Tools.trigfit(np.deg2rad(offset_lons), np.deg2rad(offset_lats), degree=degrees,
                                             sigma=errors)

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
        good_areas = hole_area[np.where((perimeter_length / (2*np.pi*u.rad)) -1 < 0.05)]
        good_fits = fit_location[np.where((perimeter_length / (2*np.pi*u.rad)) -1 < 0.05)]

        # A sphere is 4π steradians in surface area
        if good_areas.size > 0:
            percent_hole_area = (np.nanmin(good_areas) / (4 * np.pi), np.nanmean(good_areas) / (4 * np.pi), np.nanmax(good_areas) / (4 * np.pi))
            # in degrees
            hole_perimeter_location = (np.rad2deg(np.nanmin(good_fits)).value, np.rad2deg(np.nanmean(good_fits)).value, np.rad2deg(np.nanmax(good_fits)).value)
        else:
            percent_hole_area = (0, 0, 0)
            hole_perimeter_location = null_fit

        # From co-lat to lat

        if northern:
            if offset_cm[1] < 0:
                offset_cm[1] = (90 * u.deg) + offset_cm[1]
                offset_cm[0] = np.mod(offset_cm[0]-180*u.deg, 360* u.deg)
            else:
                offset_cm[1] = (90 * u.deg) - offset_cm[1]
        else:
            if offset_cm[1] > 0:
                offset_cm[1] = (-90 * u.deg) + offset_cm[1]
            else:
                offset_cm[1] = (-90 * u.deg) - offset_cm[1]
                offset_cm[0] = np.mod(offset_cm[0]-180*u.deg, 360* u.deg)

        del point_detection, lats, lons, errors, offset_coords, offset_lats, offset_lons
        gc.collect()

        # Tuples of shape (Min, Mean, Max)
        return {'Index': index, 'area': percent_hole_area, 'fit': hole_perimeter_location,
                'center': offset_cm}
