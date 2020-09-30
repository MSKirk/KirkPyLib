import numpy as np
import pywt
import matplotlib.pyplot as plt
from astropy.io import fits


def w2d(img, mode='bior5.5', level=1):
    # show wavelet decomp for fits images

    imArray = fits.getdata(img)
    # DataPrep conversions
    imArray[imArray < 0.1] = 0.1
    imArray = np.float32(np.sqrt(imArray))
    imArray /= imArray.max();

    # compute coefficients
    lw = pywt.swt2(imArray, mode, level=level)[0]

    plt.subplot(221)
    plt.imshow(lw[0], cmap='gray')
    plt.axis('off')
    plt.subplot(222)

    plt.imshow(np.abs(lw[1][0]), cmap='gray')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(np.abs(lw[1][1]), cmap='gray')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(np.abs(lw[1][2]), cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)


def w2decomp(img, mode='bior3.5', _compute=True):
    # Run all available wavelet decomp
    # return an image cube

    imArray = fits.getdata(img)
    # DataPrep conversions
    imArray[imArray < 0.1] = 0.1
    imArray = np.float32(np.sqrt(imArray))
    normal = imArray.max()
    imArray /= imArray.max();

    max_lvl = pywt.swt_max_level(len(imArray))

    out_imgs = np.zeros((imArray.shape[0], imArray.shape[1], max_lvl - 1))

    for level in range(1, max_lvl):
        if _compute:
            lw = pywt.swt2(imArray, mode, level=level)[0][0]
            out_imgs[:, :, level - 1] = lw

    return out_imgs


def imrecomb(imgs, weights):
    # recombining wavelet filtered image cube with weights

    recomb = np.zeros_like(imgs[:, :, 0])

    for index, w in enumerate(weights):
        recomb += (w * imgs[:, :, index])

    return recomb


def w_eit(img, mode='rbio5.5', weights=[1]):
    # Generate a wavelet filtered image: eit= w_eit('eit_l1_20080101_000010.fits')
    # Weights for EIT are a list or array of 9 values (can be negative)

    imgs = w2decomp(img, mode=mode)

    if len(weights) == 1:
        weights = np.ones_like(imgs[0, 0, :])

    wv_img = imrecomb(imgs, weights)

    plt.imshow(wv_img)

    return wv_img
