import copy, os, glob, sys
from sunpy.time import parse_time
import astropy.units as u
sys.path.append(os.path.abspath('/Users/mskirk/py/aia/aia'))
from AIA_Response import AIAEffectiveArea as aia_area
from aiapy import calibrate
from astropy.io import fits
from sunpy.map import Map
import numpy as np
import cv2
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning


def aia_prepping_script(image_files, save_files, verbose=False, as_npz=False):
    warnings.simplefilter('ignore', category=AstropyDeprecationWarning)

    fits_files = glob.glob(os.path.join(os.path.abspath(image_files), '**/*.fits'), recursive=True)
    eff_area = aia_area()

    pointing_table = calibrate.util.get_pointing_table(parse_time(fits.getval(fits_files[0], 'DATE-OBS', 1)),
                                                       parse_time(fits.getval(fits_files[-1], 'DATE-OBS', 1)))
    bad_files = []

    for image in fits_files:
        if verbose:
            print(image)

        try:
            savepath = os.path.join(os.path.abspath(save_files), os.path.dirname(image.split(image_files)[1])[1:])
            os.makedirs(savepath, exist_ok=True)
            savename = os.path.join(savepath, os.path.basename(image).replace('lev1', 'lev15'))

            cal_map = Map(image)

            cal_map = calibrate.meta.fix_observer_location(cal_map)
            cal_map = calibrate.meta.update_pointing(cal_map, pointing_table=pointing_table)

            try:
                cal_map = calibrate.prep.normalize_exposure(cal_map)
                cal_map = aiaprep(cal_map)

                temp_data = cal_map.data / eff_area.effective_area_ratio(cal_map.fits_header['WAVELNTH'] * u.angstrom,
                                                            parse_time(cal_map.fits_header['DATE-OBS']).to_datetime())
                cal_map.data[:] = temp_data[:]
                if as_npz:
                    np.savez_compressed(savename.replace('.fits', '.npz'), cal_map.data)
                else:
                    try:
                        cal_map.save(savename, overwrite=True, hdu_type=fits.CompImageHDU)
                    except:
                        bad_files += [image]
                        print(f'FITS image write error... Skipping: {savename}')
            except ValueError:
                print(f'AIA Image has no integration time. Skipping: {image}')
                try:
                    os.remove(os.path.abspath(savename))
                    print('Cleaning up bad files...')
                    bad_files += [image]
                except OSError:
                    pass

        except(ValueError, OSError):
            print(f'{image} is not valid. Flagging for follow up.')
            bad_files += [image]
            os.rename(image, image.replace('aia.', 'CHECK.aia.'))

    return bad_files


def scale_rotate(image, angle=0, scale_factor=1, reference_pixel=None):
    """
    Perform scaled rotation with opencv. About 20 times faster than with Sunpy & scikit/skimage warp methods.
    The output is a padded image that holds the entire rescaled,rotated image, recentered around the reference pixel.
    Positive-angle rotation will go counterclockwise if the array is displayed with the origin on top (default),
    and clockwise with the origin at bottom.

    :param image: Numpy 2D array
    :param angle: rotation angle in degrees. Positive angle  will rotate counterclocwise if array origin on top-left
    :param scale_factor: ratio of the wavelength-dependent pixel scale over the target scale of 0.6 arcsec
    :param reference_pixel: tuple of (x, y) coordinate. Given as (x, y) = (col, row) and not (row, col).
    :return: padded scaled and rotated image
    """
    array_center = (np.array(image.shape)[::-1] - 1) / 2.0

    if reference_pixel is None:
        reference_pixel = array_center

    # convert angle to radian
    angler = angle * np.pi / 180
    # Get basic rotation matrix to calculate initial padding extent
    rmatrix = np.matrix([[np.cos(angler), -np.sin(angler)],
                         [np.sin(angler), np.cos(angler)]])

    extent = np.max(np.abs(np.vstack((image.shape * rmatrix,
                                      image.shape * rmatrix.T))), axis=0)

    # Calculate the needed padding or unpadding
    diff = np.asarray(np.ceil((extent - image.shape) / 2), dtype=int).ravel()
    diff2 = np.max(np.abs(reference_pixel - array_center)) + 1
    # Pad the image array
    pad_x = int(np.ceil(np.max((diff[1], 0)) + diff2))
    pad_y = int(np.ceil(np.max((diff[0], 0)) + diff2))

    padded_reference_pixel = reference_pixel + np.array([pad_x, pad_y])
    #padded_image = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=(0, 0))
    padded_image = aia_pad(image, pad_x, pad_y)
    padded_array_center = (np.array(padded_image.shape)[::-1] - 1) / 2.0

    # Get scaled rotation matrix accounting for padding
    rmatrix_cv = cv2.getRotationMatrix2D((padded_reference_pixel[0], padded_reference_pixel[1]), angle, scale_factor)
    # Adding extra shift to recenter:
    # move image so the reference pixel aligns with the center of the padded array
    shift = padded_array_center - padded_reference_pixel
    rmatrix_cv[0, 2] += shift[0]
    rmatrix_cv[1, 2] += shift[1]
    # Do the scaled rotation with opencv. ~20x faster than Sunpy's map.rotate()
    rotated_image = cv2.warpAffine(padded_image, rmatrix_cv, padded_image.shape, cv2.INTER_CUBIC)

    return rotated_image


def aiaprep(smap, cropsize=4096):
    # Adapted from AIA-reloaded to use sunpy maps

    header = smap.fits_header
    data = smap.data.astype(np.float64)

    # Target scale is 0.6 arcsec/px
    target_scale = 0.6
    scale_factor = header['CDELT1'] / target_scale
    # Center of rotation at reference pixel converted to a coordinate origin at 0
    reference_pixel = [header['CRPIX1'] - 1, header['CRPIX2'] - 1]
    
    # Rotation angle with openCV uses coordinate origin at top-left corner. 
    # For solar images in numpy we need to invert the angle.
    angle = -header['CROTA2']
    # Run scaled rotation. The output will be a rotated, rescaled, padded array.
    prepdata = scale_rotate(data, angle=angle, scale_factor=scale_factor, reference_pixel=reference_pixel)
    prepdata[prepdata < 0] = 0

    if cropsize is not None:
        center = ((np.array(prepdata.shape) - 1) / 2.0).astype(int)
        half_size = int(cropsize / 2)
        prepdata = prepdata[center[1] - half_size:center[1] + half_size, center[0] - half_size:center[0] + half_size]

    newmap = smap._new_instance(prepdata.astype(np.float32), copy.deepcopy(smap.meta))

    newmap.meta['r_sun'] = newmap.meta['rsun_obs'] / newmap.meta['cdelt1']
    newmap.meta['lvl_num'] = 1.5
    newmap.meta['bitpix'] = -32

    return newmap


# Alternate padding method. On AIA, it is ~6x faster than numpy.pad used in Sunpy's aiaprep
def aia_pad(image, pad_x, pad_y):
    newsize = [image.shape[0]+2*pad_y, image.shape[1]+2*pad_x]
    pimage = np.empty(newsize)
    pimage[0:pad_y,:] = 0
    pimage[:,0:pad_x]=0
    pimage[pad_y+image.shape[0]:, :] = 0
    pimage[:, pad_x+image.shape[1]:] = 0
    pimage[pad_y:image.shape[0]+pad_y, pad_x:image.shape[1]+pad_x] = image
    return pimage
