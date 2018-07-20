import scipy.ndimage
import numpy as np
from numpy.linalg import matrix_rank
from scipy.stats import invgamma
from skimage.util import view_as_windows


class NoiseLevelEstimation:
    def __init__(self, img, patchsize=7, decim=0, conf=1-1E-6, itr=3):
        """
        NoiseLevel estimates noise level of input single noisy image.

        Input parameters:
            img: input single image array
            patchsize (optional): patch size (default: 7)
            decim (optional): decimation factor. If you put large number, the calculation will be accelerated. (default: 0)
            conf (optional): confidence interval to determin the threshold for the weak texture. In this algorithm, this value is usually set the value very close to one. (default: 0.99)
            itr (optional): number of iteration. (default: 3)

        Calculated parameters:
            nlevel: estimated noise levels.
            th: threshold to extract weak texture patches at the last iteration.
            num: number of extracted weak texture patches at the last iteration.
            mask: weak-texture mask. 0 and 1 represent non-weak-texture and weak-texture regions, respectively

        Example:
            estimate = NoiseLevelEstimation(image_array, patchsize=11, itr=20)

        Python Version: 20180718

        Translated from Noise Level Estimation Matlab code: noiselevel.m

        noiselevel.m Copyright (C) 2012-2015 Masayuki Tanaka

        Reference:
        Xinhao Liu, Masayuki Tanaka and Masatoshi Okutomi
        Noise Level Estimation Using Weak Textured Patches of a Single Noisy Image
        IEEE International Conference on Image Processing (ICIP), 2012.

        Xinhao Liu, Masayuki Tanaka and Masatoshi Okutomi
        Single-Image Noise Level Estimation for Blind Denoising Noisy Image
        IEEE Transactions on Image Processing, Vol.22, No.12, pp.5226-5237, December, 2013.

        """
        
        self.img = img
        self.patchsize = patchsize
        self.decim = decim
        self.conf = conf
        self.itr = itr

        self.nlevel, self.th, self.num = self.noiselevel()
        self.mask = self.weaktexturemask()

    def noiselevel(self):

        nlevel = np.ndarray(self.img.shape[2])
        th = np.ndarray(self.img.shape[2])
        num = np.ndarray(self.img.shape[2])

        kh = np.array([-1 / 2, 0, 1 / 2])
        imgh = scipy.ndimage.correlate(self.img, kh, mode='nearest').transpose()
        imgh = imgh[:, 1: imgh.shape[1] - 1, :]
        imgh = imgh * imgh

        kv = np.matrix(kh).getH()
        imgv = scipy.ndimage.correlate(self.img, kv, mode='nearest').transpose()
        imgv = imgv[1: imgv.shape[0] - 1, :, :]
        imgv = imgv * imgv

        Dh = np.matrix(self.convmtx2(kh, self.patchsize, self.patchsize))
        Dv = np.matrix(self.convmtx2(kv, self.patchsize, self.patchsize))

        DD = Dh.getH() * Dh + Dv.getH() * Dv

        r = np.double(matrix_rank(DD))
        Dtr = np.trace(DD)

        tau0 = invgamma.cdf(self.conf,r/2, scale=2*Dtr/r)

        for cha in range(self.img.shape[2]):
            X = view_as_windows(self.img[:,:,cha], (self.patchsize, self.patchsize))
            Xh = view_as_windows(imgh[:,:, cha], (self.patchsize, self.patchsize - 2))
            Xv = view_as_windows(imgv[:,:, cha], (self.patchsize - 2, self.patchsize))

            Xtr = np.sum(np.concatenate((Xh, Xv)), axis=0)

            if self.decim > 0:
                XtrX = np.concatenate((Xtr, X), axis=0)
                XtrX = np.asmatrix(XtrX).getH().sort(axis=1).getH()
                p = np.floor(XtrX.shape[1]/(self.decim+1))
                p = np.arange(0, p) * (self.decim+1)
                Xtr = XtrX[0,p]
                X = XtrX[1:XtrX.shape[0],p]

            # noise level estimation
            tau = np.inf

            if X.shape[1] > X.shape[0]:
                sig2 = 0
            else:
                cov = (X * np.asmatrix(X).getH())/(X.shape[1] -1)
                d = np.linalg.eig(cov)
                sig2 = d[0]

            for i in range(1,self.itr):
                # weak texture selection
                tau = sig2 * tau0
                p = Xtr < tau
                Xtr = Xtr[:, p]
                X = X[:, p]

                # noise level estimation
                if X.shape[1] < X.shape[0]:
                    break

                cov = (X * np.asmatrix(X).getH()) / (X.shape[1] - 1)
                d = np.linalg.eig(cov)
                sig2 = d[0]

            nlevel[cha] = np.sqrt(sig2)
            th[cha] = tau
            num[cha] = X.shape[1]

        return nlevel, th, num

    def convmtx2(self, H, m, n):
        s = np.shape(H)
        T = np.zeros([(m - s[0] + 1) * (n - s[1] + 1), m * n])

        k = 0
        for i in range((m - s[0] + 1)):
            for j in range(1, (m - s[1] + 1)):
                for p in range(s[0]):
                    T[k, (i - 1 + p - 1) * n + (j - 1) + 1:(i - 1 + p - 1) * n + (j - 1) + 1 + s[2] - 1] = H[p,:]
                k = k + 1
        
        return T

    def  weaktexturemask(self):

        kh = np.array([-1 / 2, 0, 1 / 2])
        imgh = scipy.ndimage.correlate(self.img, kh, mode='nearest').transpose()
        imgh = imgh[:, 1: imgh.shape[1] - 1, :]
        imgh = imgh * imgh

        kv = np.matrix(kh).getH()
        imgv = scipy.ndimage.correlate(self.img, kv, mode='nearest').transpose()
        imgv = imgv[1: imgv.shape[0] - 1, :, :]
        imgv = imgv * imgv

        s = self.img.shape
        msk = np.zeros_like(self.img)

        for cha in range(s[2]):
            m = np.zeros_like(view_as_windows(self.img[:,:,cha], (self.patchsize, self.patchsize)))
            Xh = view_as_windows(imgh[:,:, cha], (self.patchsize, self.patchsize - 2))
            Xv = view_as_windows(imgv[:,:, cha], (self.patchsize - 2, self.patchsize))

            Xtr = np.sum(np.concatenate((Xh, Xv)), axis=0)

            p = Xtr < self.th[cha]
            ind = 1

            for col in range(0,s[1]-self.patchsize+1):
                for row in range(0,s[0]-self.patchsize+1):
                    if p[ind] > 0:
                        msk[row: row + self.patchsize - 1, col: col + self.patchsize - 1, cha] = 1
                    ind = ind + 1

        return msk
