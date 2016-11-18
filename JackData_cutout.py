##  Cutout for region in Ireland et al. 2015

opdir='/Volumes/DataDisk/AIA/211/'
svdir='/Users/mskirk/data/BM3D/'

import os
import fnmatch
import sunpy.map as map
import sunpy.instr.aia as aia
import astropy.units as u


fls=fnmatch.filter(os.listdir(opdir), 'BM3D*.fits')
itr=0

ji_xrange = [-539.16847142,-115.07568342] * u.arcsec

ji_yrange = [-116.93790609, 113.73012591] * u.arcsec


for file in fls:
    this_map=map.Map(opdir+file)
    # this_map=aia.aiaprep(this_map)
    submap = this_map.submap(ji_xrange, ji_yrange)
    submap.save(svdir+file, filetype='fits', clobber=True)
    itr = itr +1
    print itr



