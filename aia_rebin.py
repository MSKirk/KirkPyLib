## rebinning AIA data

opdir='/Users/mskirk/data/AIA/TempQuicklook/304/'

import os
import fnmatch
import sunpy.map as map
import sunpy.instr.aia as aia

fls=fnmatch.filter(os.listdir(opdir), '*.fits')
itr=0

for file in fls:
    aia_map=aia.aiaprep(map.Map(opdir+file))
    prep_map=aia_map.resample([1024,1024])
    prep_map.save(opdir+file, filetype='fits', clobber=True)
    itr = itr +1
    print itr



