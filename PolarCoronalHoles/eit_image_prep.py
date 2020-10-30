from sunpy.coordinates.ephemeris import get_horizons_coord
from sunpy.time import parse_time
from sunpy.map import Map
import numpy as np
from astropy.io import fits
import cv2


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
    padded_image = image_pad(image, pad_x, pad_y)
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


# Alternate padding method. On AIA, it is ~6x faster than numpy.pad used in Sunpy's aiaprep
def image_pad(image, pad_x, pad_y):
    newsize = [image.shape[0]+2*pad_y, image.shape[1]+2*pad_x]
    pimage = np.empty(newsize)
    pimage[0:pad_y,:] = 0
    pimage[:,0:pad_x]=0
    pimage[pad_y+image.shape[0]:, :] = 0
    pimage[:, pad_x+image.shape[1]:] = 0
    pimage[pad_y:image.shape[0]+pad_y, pad_x:image.shape[1]+pad_x] = image
    return pimage


# Read an eit fits file and return a prepped array or sunpy Map
def eitprep(fitsfile, return_map=False):

    hdul = fits.open(fitsfile)
    hdul[0].verify('silentfix')
    header = hdul[0].header
    data = hdul[0].data.astype(np.float64)
    new_coords = get_horizons_coord(header['TELESCOP'], parse_time(header['DATE_OBS']))
    header['HGLN_OBS'] = new_coords.lon.to('deg').value
    header['HGLT_OBS'] = new_coords.lat.to('deg').value
    header['DSUN_OBS'] = new_coords.radius.to('m').value

    header.pop('hec_x')
    header.pop('hec_y')
    header.pop('hec_z')
    # Target scale is 2.63 arcsec/px
    target_scale = 2.63
    scale_factor = header['CDELT1'] / target_scale
    # Center of rotation at reference pixel converted to a coordinate origin at 0
    reference_pixel = [header['CRPIX1'] - 1, header['CRPIX2'] - 1]
    # Rotation angle with openCV uses coordinate origin at top-left corner. For solar images in numpy we need to invert the angle.
    angle = -header['SC_ROLL']
    # Run scaled rotation. The output will be a rotated, rescaled, padded array.
    prepdata = scale_rotate(data, angle=angle, scale_factor=scale_factor, reference_pixel=reference_pixel)
    prepdata[prepdata < 0] = 0
    prepdata[np.isnan(prepdata)] = 0

    if return_map:
        prepdata = Map(prepdata, header)

    return prepdata
