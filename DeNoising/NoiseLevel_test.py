import scipy.ndimage
import numpy as np
from numpy.linalg import matrix_rank
from scipy.stats import invgamma
from skimage.util import view_as_windows
from skimage import data


def convmtx2(H, m, n):
    # Specialized 2D convolution matrix generation
    # H — Input matrix
    # m — Rows in convolution matrix
    # n — Columns in convolution matrix

    s = np.shape(H)
    T = np.zeros([(m - s[0] + 1) * (n - s[1] + 1), m * n])

    k = 0
    for i in range((m - s[0] + 1)):
        for j in range((n - s[1] + 1)):
            for p in range(s[0]):
                index_a = (i - 1 + p - 1) * n + (j - 1)
                index_b = (i - 1 + p - 1) * n + (j - 1) + s[1] - 1

                if index_a == index_b:
                    T[k, index_a] = H[p, :]
                else:
                    T[k, index_a: index_b] = H[p, :]
            k = k + 1
    return T


def NoiseLevel_test(img = data.camera(), patchsize = 7, ecim=1, conf=1-1E-6, itr=3):

    try:
        third_dim_size = img.shape[2]
    except IndexError:
        img = np.expand_dims(img, 2)
        third_dim_size = img.shape[2]

    nlevel = np.ndarray(third_dim_size)
    th = np.ndarray(third_dim_size)
    num = np.ndarray(third_dim_size)

    kh = np.expand_dims(np.expand_dims(np.array([-0.5, 0, 0.5]), 1), 2)
    imgh = scipy.ndimage.correlate(img, kh, mode='nearest')
    imgh = imgh[:, 1: imgh.shape[1] - 1, :]
    imgh = imgh * imgh

    kv = np.expand_dims(np.matrix(kh).getH(), 2)
    imgv = scipy.ndimage.correlate(img, kv, mode='nearest')
    imgv = imgv[1: imgv.shape[0] - 1, :, :]
    imgv = imgv * imgv

    Dh = np.matrix(convmtx2(kh, patchsize, patchsize))
    Dv = np.matrix(convmtx2(kv, patchsize, patchsize))

    DD = Dh.getH() * Dh + Dv.getH() * Dv

    r = np.double(matrix_rank(DD))
    Dtr = np.trace(DD)

    tau0 = invgamma.cdf(conf, r / 2, scale=2 * Dtr / r)

    for cha in range(third_dim_size):
        X = view_as_windows(img[:, :, cha], (patchsize, patchsize))
        X = X.reshape(np.int(X.size / patchsize ** 2), patchsize ** 2, order='F')

        Xh = view_as_windows(imgh[:, :, cha], (patchsize, patchsize - 2))
        Xh = Xh.reshape(np.int(Xh.size / ((patchsize - 2) * patchsize)),
                        ((patchsize - 2) * patchsize), order='F')

        Xv = view_as_windows(imgv[:, :, cha], (patchsize - 2, patchsize))
        Xv = Xv.reshape(np.int(Xv.size / ((patchsize - 2) * patchsize)),
                        ((patchsize - 2) * patchsize), order='F')

        Xtr = np.expand_dims(np.sum(np.concatenate((Xh, Xv), axis=1), axis=1), 1)

        if decim > 0:
            XtrX = np.sort(np.concatenate((Xtr, X), axis=1), axis=1)
            p = np.floor(XtrX.shape[0] / (decim + 1))
            p = np.arange(0, p) * (decim + 1)
            Xtr = XtrX[p.astype('int'), 0]
            X = XtrX[p.astype('int'), 1:XtrX.shape[1]]

        # noise level estimation
        tau = np.inf

        if X.shape[1] > X.shape[0]:
            sig2 = 0
        else:
            cov = (np.asmatrix(X).getH() @ np.asmatrix(X)) / (X.shape[0] - 1)
            d = np.linalg.eig(cov)[0]
            sig2 = d[0]

        for i in range(1, itr):
            # weak texture selection
            tau = sig2 * tau0
            p = Xtr < tau
            Xtr = Xtr[p]
            X = X[p, :]

            # noise level estimation
            if X.shape[1] > X.shape[0]:
                break

            cov = (np.asmatrix(X).getH() @ np.asmatrix(X)) / (X.shape[0] - 1)
            d = np.linalg.eig(cov)[0]
            sig2 = d[0]

        nlevel[cha] = np.sqrt(sig2)
        th[cha] = tau
        num[cha] = X.shape[1]

    return nlevel, th, num


