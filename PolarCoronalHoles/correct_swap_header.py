from astropy.io import fits
import glob, os
import numpy as np


def fix_swap_header():
    all_files = list(set([os.path.abspath(p) for p in glob.glob("/Volumes/CoronalHole/SWAP/*/*/*")]))

    for ii, file_path in enumerate(all_files):
        try:
            hdu1 = fits.open(file_path)

            head_loc = np.argmax([len(hdu.header) for hdu in hdu1])

            hdu1[head_loc].verify('fix')

            hdu1[head_loc].header['CTYPE1'] = 'HPLN-TAN'
            hdu1[head_loc].header['CTYPE2'] = 'HPLT-TAN'
            hdu1[head_loc].header['CUNIT1'] = 'arcsec'
            hdu1[head_loc].header['CUNIT2'] = 'arcsec'
            hdu1[head_loc].header['CRVAL1'] = 'arcsec'
            hdu1[head_loc].header['CRVAL2'] = 'arcsec'

            hdu1.writeto(file_path, overwrite=True)
        except OSError:
            print('file not readable... removing.')
            os.remove(file_path)
            pass

        print(int((ii/len(all_files))*100.), 'Percent Complete')
