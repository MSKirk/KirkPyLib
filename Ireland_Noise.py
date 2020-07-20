import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from bm3d import bm3d, BM3DProfile

from experiment_funcs import get_experiment_noise, get_psnr, get_cropped_psnr


bv = np.load('/Users/mskirk/data/Bradshaw/bv_simulation_high_fn_171.step1.npz', allow_pickle=True)       
imcube= bv['arr_0']

y = np.sqrt(imcube[:,:,-1])
y_cube = imcube/imcube.max()
y = y/y.max()
noise_type = 'g4w'
noise_var = y.min()/4.  # Noise variance
noise_cube_var = y_cube.min()/4
seed = 0  # seed for pseudorandom noise realization

noise, psd, kernel = get_experiment_noise(noise_type, noise_var, seed, y.shape)
noise_cube, psd_cube, kernel_cube = get_experiment_noise(noise_type, noise_cube_var, seed, y_cube.shape)

# Generate noisy image corrupted by additive spatially correlated noise
# with noise power spectrum PSD
z = (np.atleast_3d(y) + np.atleast_3d(noise))

# Call BM3D With the default settings.
y_est = bm3d(z, psd)

# Note: For white noise, you may instead of the PSD
# also pass a standard deviation
# y_est = bm3d(z, sqrt(noise_var));

psnr = get_psnr(y, y_est)
print("PSNR:", psnr)

# PSNR ignoring 16-pixel wide borders (as used in the paper), due to refiltering potentially leaving artifacts
# on the pixels near the boundary of the image when noise is not circulant
psnr_cropped = get_cropped_psnr(y, y_est, [16, 16])
print("PSNR cropped:", psnr_cropped)

# Ignore values outside range for display (or plt gives an error for multichannel input)
y_est = np.minimum(np.maximum(y_est, 0), 1)
z_rang = np.squeeze(np.minimum(np.maximum(z, 0), 1))
plt.title("y, z, y_est")
plt.imshow(np.concatenate((y, z_rang, y_est), axis=1), cmap='gray')
plt.show()