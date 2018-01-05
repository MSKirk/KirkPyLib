from sunpy import map
import numpy as np
from scipy import optimize

'''
A collection of tools used in PCH detection. This replicates the following IDL routines:
    trigfit.pro
    new_center.pro
    get_harlon.pro
    get_harrot.pro
    harrot2date.pro
    date2harrot.pro
    histpercent.pro
'''




def trigfit(self, theta, rho):
    tt = numpy.array(theta)
    yy = numpy.array(rho)
    # assume uniform spacing
    ff = numpy.fft.fftfreq(len(tt), (tt[1] - tt[0]))
    Fyy = abs(numpy.fft.fft(yy))

    # excluding the zero frequency "peak", which is related to offset
    guess_freq = abs(ff[numpy.argmax(Fyy[1:]) + 1])
    guess_amp = numpy.std(yy) * 2. ** 0.5
    guess_offset = numpy.mean(yy)
    guess = numpy.array([guess_offset, guess_amp, 2. * numpy.pi * guess_freq])

    def cos_series(t,A, degree=1):
        '''
        :param t: independent variables
        :param A: coefficients
        :param degree: degree of the function
        :return: function
        '''
        F = A[0]
        for n in range(1, degree+1):
            cx=np.cos(n*t + A[2*n])
            F += cx*A[2*n-1]
        return F

    popt, pcov = optimize.curve_fit(cos_series, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w / (2. * numpy.pi)
    fitfunc = lambda t: A * numpy.sin(w * t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1. / f, "fitfunc": fitfunc,
            "maxcov": numpy.max(pcov), "rawres": (guess, popt, pcov)}