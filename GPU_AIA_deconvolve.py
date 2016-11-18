from skimage import restoration as rest
from numbapro import vectorize, float32
from timeit import default_timer as timer
import os
import fnmatch
import sunpy.map as map
import astropy.units as u


@vectorize(['float32(float32, float32)'], target='cpu')
def decon_RL(image, psf):
    return rest.richardson_lucy(image, psf, iterations=20)



def main():

    opdir = '/home/mskirk/Data/AIA/335/'
    svdir = '/home/mskirk/Data/AIA/Decon_PSF/335/'
    
    psf_file = '/home/mskirk/Data/AIA/PSF/LMSAL/AIA_335_PSF.fits'
    psf = map.Map(psf_file)
    psf = psf.data.astype('float32')

    fls = fnmatch.filter(os.listdir(opdir), '*.fits')
    itr = 0

    for file in fls:
        start = timer()
        image_map = map.Map(opdir+file)
        image_map.data = decon_RL(image_map.data, psf)
        image_map.save(svdir+'PSF_'+file, filetype='fits', clobber=True)
        process_time = timer() - start
        
        itr = itr +1
        print itr
        print("PSF Deconvolution took %f seconds" % process_time)
        
        
if __name__ == '__main__':
    main()