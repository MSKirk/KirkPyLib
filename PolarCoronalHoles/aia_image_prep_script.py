import sys, os, glob
from sunpy.time import parse_time
import astropy.units as u

sys.path.append(os.path.abspath('/Users/mskirk/py/AIA-reloaded'))
sys.path.append(os.path.abspath('/Users/mskirk/py/aia/aia'))
from calibration import AIAPrep
from AIA_Response import AIAEffectiveArea as aia_area


def aia_prepping_script(image_files, save_files, verbose=False):
    fits_files = glob.glob(os.path.join(os.path.abspath(image_files), '**/*.fits'), recursive=True)
    eff_area = aia_area()

    for image in fits_files:
        if verbose:
            print(image)

        savepath = os.path.join(os.path.abspath(save_files), os.path.dirname(image.split(image_files)[1])[1:])
        os.makedirs(savepath, exist_ok=True)

        im = AIAPrep(image)

        try:
            im.data /= eff_area.effective_area_ratio(im.header['WAVELNTH'] * u.angstrom, parse_time(im.header['DATE-OBS']))
            im.write_prepped(save_path=savepath)
        except ValueError:
            print('Invalid DATE-OBS... using T_OBS instead.')
            im.data /= eff_area.effective_area_ratio(im.header['WAVELNTH'] * u.angstrom, parse_time(im.header['T_OBS']))
            im.write_prepped(save_path=savepath)
        except:
            print('FITS image write error... Skipping file.')
