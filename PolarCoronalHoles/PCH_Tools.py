import numpy as np
from scipy import optimize
from astropy.time import Time
import math
import astropy
import astropy.units as u
from astropy.coordinates import Longitude, Latitude

'''
A collection of tools used in PCH detection. This replicates the following IDL routines:
    trigfit.pro
    get_harlon.pro
    harrot2date.pro
    date2harrot.pro
    histpercent.pro
    com.pro
'''


def trigfit(theta, rho, sigma=None, degree=1):
    # A cosine series fit for data

    tt = np.array(theta)
    yy = np.array(rho)
    # assume uniform spacing
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))
    Fyy = abs(np.fft.fft(yy))

    # excluding the zero frequency "peak", which is related to offset
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])
    guess_amp = np.std(yy) * 2. ** 0.5
    guess_offset = np.mean(yy)

    # Guesses only first order cosine fit
    guess = np.append(np.array([guess_offset, guess_amp, 2. * np.pi * guess_freq]), np.zeros((degree-1)*2))

    if degree == 1:
        def cos_series(t, c, a1, p1):
            F = c
            F += a1*np.cos(1 * t + p1)
            return F

    if degree == 2:
        def cos_series(t, c, a1, p1, a2, p2):
            F = c
            F += a1 * np.cos(1 * t + p1)
            F += a2 * np.cos(2 * t + p2)
            return F

    if degree == 3:
        def cos_series(t, c, a1, p1, a2, p2, a3, p3):
            F = c
            F += a1 * np.cos(1 * t + p1)
            F += a2 * np.cos(2 * t + p2)
            F += a3 * np.cos(3 * t + p3)
            return F

    if degree == 4:
        def cos_series(t, c, a1, p1, a2, p2, a3, p3, a4, p4):
            F = c
            F += a1 * np.cos(1 * t + p1)
            F += a2 * np.cos(2 * t + p2)
            F += a3 * np.cos(3 * t + p3)
            F += a4 * np.cos(4 * t + p4)
            return F

    if degree == 5:
        def cos_series(t, c, a1, p1, a2, p2, a3, p3, a4, p4, a5, p5):
            F = c
            F += a1 * np.cos(1 * t + p1)
            F += a2 * np.cos(2 * t + p2)
            F += a3 * np.cos(3 * t + p3)
            F += a4 * np.cos(4 * t + p4)
            F += a5 * np.cos(5 * t + p5)
            return F

    if degree == 6:
        def cos_series(t, c, a1, p1, a2, p2, a3, p3, a4, p4, a5, p5, a6, p6):
            F = c
            F += a1 * np.cos(1 * t + p1)
            F += a2 * np.cos(2 * t + p2)
            F += a3 * np.cos(3 * t + p3)
            F += a4 * np.cos(4 * t + p4)
            F += a5 * np.cos(5 * t + p5)
            F += a6 * np.cos(6 * t + p6)
            return F

    if degree == 7:
        def cos_series(t, c, a1, p1, a2, p2, a3, p3, a4, p4, a5, p5, a6, p6, a7, p7):
            F = c
            F += a1 * np.cos(1 * t + p1)
            F += a2 * np.cos(2 * t + p2)
            F += a3 * np.cos(3 * t + p3)
            F += a4 * np.cos(4 * t + p4)
            F += a5 * np.cos(5 * t + p5)
            F += a6 * np.cos(6 * t + p6)
            F += a7 * np.cos(7 * t + p7)
            return F

    if degree == 8:
        def cos_series(t, c, a1, p1, a2, p2, a3, p3, a4, p4, a5, p5, a6, p6, a7, p7, a8, p8):
            F = c
            F += a1 * np.cos(1 * t + p1)
            F += a2 * np.cos(2 * t + p2)
            F += a3 * np.cos(3 * t + p3)
            F += a4 * np.cos(4 * t + p4)
            F += a5 * np.cos(5 * t + p5)
            F += a6 * np.cos(6 * t + p6)
            F += a7 * np.cos(7 * t + p7)
            F += a8 * np.cos(8 * t + p8)
            return F

    if degree == 9:
        def cos_series(t, c, a1, p1, a2, p2, a3, p3, a4, p4, a5, p5, a6, p6, a7, p7, a8, p8, a9, p9):
            F = c
            F += a1 * np.cos(1 * t + p1)
            F += a2 * np.cos(2 * t + p2)
            F += a3 * np.cos(3 * t + p3)
            F += a4 * np.cos(4 * t + p4)
            F += a5 * np.cos(5 * t + p5)
            F += a6 * np.cos(6 * t + p6)
            F += a7 * np.cos(7 * t + p7)
            F += a8 * np.cos(8 * t + p8)
            F += a9 * np.cos(9 * t + p9)
            return F

    if degree == 10:
        def cos_series(t, c, a1, p1, a2, p2, a3, p3, a4, p4, a5, p5, a6, p6, a7, p7, a8, p8, a9, p9, a10, p10):
            F = c
            F += a1 * np.cos(1 * t + p1)
            F += a2 * np.cos(2 * t + p2)
            F += a3 * np.cos(3 * t + p3)
            F += a4 * np.cos(4 * t + p4)
            F += a5 * np.cos(5 * t + p5)
            F += a6 * np.cos(6 * t + p6)
            F += a7 * np.cos(7 * t + p7)
            F += a8 * np.cos(8 * t + p8)
            F += a9 * np.cos(9 * t + p9)
            F += a10 * np.cos(10 * t + p10)
            return F

    popt, pcov = optimize.curve_fit(cos_series, tt, yy, p0=guess.tolist(), sigma=sigma)

    if degree == 1:
        fitfunc = lambda t: popt[0] + popt[1] * np.cos(1 * t + popt[2])

    if degree == 2:
        fitfunc = lambda t: popt[0] + popt[1] * np.cos(1 * t + popt[2]) + popt[3] * np.cos(2 * t + popt[4])

    if degree == 3:
        fitfunc = lambda t: popt[0] + popt[1] * np.cos(1 * t + popt[2]) + popt[3] * np.cos(2 * t + popt[4]) \
                            + popt[5] * np.cos(3 * t + popt[6])

    if degree == 4:
        fitfunc = lambda t: popt[0] + popt[1] * np.cos(1 * t + popt[2]) + popt[3] * np.cos(2 * t + popt[4]) \
                            + popt[5] * np.cos(3 * t + popt[6]) + popt[7] * np.cos(4 * t + popt[8])

    if degree == 5:
        fitfunc = lambda t: popt[0] + popt[1] * np.cos(1 * t + popt[2]) + popt[3] * np.cos(2 * t + popt[4]) \
                            + popt[5] * np.cos(3 * t + popt[6]) + popt[7] * np.cos(4 * t + popt[8]) \
                            + popt[9] * np.cos(5 * t + popt[10])

    if degree == 6:
        fitfunc = lambda t: popt[0] + popt[1] * np.cos(1 * t + popt[2]) + popt[3] * np.cos(2 * t + popt[4]) \
                            + popt[5] * np.cos(3 * t + popt[6]) + popt[7] * np.cos(4 * t + popt[8]) \
                            + popt[9] * np.cos(5 * t + popt[10]) + popt[11] * np.cos(6 * t + popt[12])

    if degree == 7:
        fitfunc = lambda t: popt[0] + popt[1] * np.cos(1 * t + popt[2]) + popt[3] * np.cos(2 * t + popt[4]) \
                            + popt[5] * np.cos(3 * t + popt[6]) + popt[7] * np.cos(4 * t + popt[8]) \
                            + popt[9] * np.cos(5 * t + popt[10]) + popt[11] * np.cos(6 * t + popt[12]) \
                            + popt[13] * np.cos(7 * t + popt[14])

    if degree == 8:
        fitfunc = lambda t: popt[0] + popt[1] * np.cos(1 * t + popt[2]) + popt[3] * np.cos(2 * t + popt[4]) \
                            + popt[5] * np.cos(3 * t + popt[6]) + popt[7] * np.cos(4 * t + popt[8]) \
                            + popt[9] * np.cos(5 * t + popt[10]) + popt[11] * np.cos(6 * t + popt[12]) \
                            + popt[13] * np.cos(7 * t + popt[14])+ popt[15] * np.cos(8 * t + popt[16])

    if degree == 9:
        fitfunc = lambda t: popt[0] + popt[1] * np.cos(1 * t + popt[2]) + popt[3] * np.cos(2 * t + popt[4]) \
                            + popt[5] * np.cos(3 * t + popt[6]) + popt[7] * np.cos(4 * t + popt[8]) \
                            + popt[9] * np.cos(5 * t + popt[10]) + popt[11] * np.cos(6 * t + popt[12]) \
                            + popt[13] * np.cos(7 * t + popt[14]) + popt[15] * np.cos(8 * t + popt[16]) \
                            + popt[17] * np.cos(9 * t + popt[18])

    if degree == 10:
        fitfunc = lambda t: popt[0] + popt[1] * np.cos(1 * t + popt[2]) + popt[3] * np.cos(2 * t + popt[4]) \
                            + popt[5] * np.cos(3 * t + popt[6]) + popt[7] * np.cos(4 * t + popt[8]) \
                            + popt[9] * np.cos(5 * t + popt[10]) + popt[11] * np.cos(6 * t + popt[12]) \
                            + popt[13] * np.cos(7 * t + popt[14]) + popt[15] * np.cos(8 * t + popt[16]) \
                            + popt[17] * np.cos(9 * t + popt[18]) + popt[19] * np.cos(10 * t + popt[20])

    return {'popt': popt, 'pcov': pcov, 'fitfunc': fitfunc}


def center_of_mass(coords, mass=1, distance=False):
    #
    # :param coords: an M-element x N-dimensional array of coordinates to find the center of mass. e.g. [[x1,y1,z1],[x2,y2,z2]]
    # :param mass: optional mass of each point
    # :param distance: boolean if true will scale the mass based upon distance from origin
    # :return: an N length array of the coordinate center of mass
    #

    if type(mass) == int:
        mass=np.ones_like(coords[:,0])

    if distance:
        mass *= np.sqrt(np.sum(coords**2,axis=1))

    mm = np.transpose(np.tile(mass, (coords.shape[1],1)))

    return np.sum(coords*mm,axis=0)/np.sum(mass)


def hrot2date(hrot):
    # Returns an astropy time object
    if hrot < 0:
        raise ValueError('Dubious date: The rotation number dates before 1900.')
    if hrot > 2000:
        raise ValueError('Dubious date: The rotation number dates after 2080.')

    jd = (((hrot - 1.) * 360.) / (360. / 33.)) + 2415023.5

    return Time(jd, format='jd', scale='utc')


def date2hrot(date, fractional=False):
    # 2415023.5 JD = Jan 4, 1900 => 1st Harvey Rotation

    if not isinstance(date, astropy.time.core.Time):
        raise ValueError('Input needs to be an astropy time object.')

    if fractional:
        return (((360. / 33.) * (date.jd - 2415023.5)) / 360.) + 1
    else:
        return np.int(np.floor((((360. / 33.) * (date.jd - 2415023.5)) / 360.)) + 1)


def get_harvey_lon(date, radians=False):
    # 2415023.5 JD = Jan 4, 1900 => 1st Harvey Rotation
    # 1 Harvey Rotation => 360 degrees in 33 days

    if not isinstance(date, astropy.time.core.Time):
        raise ValueError('Input needs to be an astropy time object.')

    if radians:
        return Longitude([math.radians(((360. / 33.) * (date.jd - 2415023.5)) - (np.floor(((360. / 33.) * (date.jd - 2415023.5)) / 360.) * 360.))] * u.rad)
    else:
        return Longitude([((360. / 33.) * (date.jd - 2415023.5)) - (np.floor(((360. / 33.) * (date.jd - 2415023.5)) / 360.) * 360.)] * u.deg)


def hist_percent(array, fraction, number_of_bins=200):
    # returns the value at which a fraction of a given array falls below

    histogram, bins = np.histogram(array, bins=number_of_bins)

    if np.max(np.abs(array)) == 0:
        return 0
    h = len(histogram)
    n = h
    i0 = 0.
    i = 0.
    a1 = 0.
    a = np.sum(histogram)

    while ((a1 / a) != fraction) and ((n - i0) > 1.):
        i = np.round((n - i0) / 2. + i0)
        n = np.round(n)
        i0 = np.round(i0)

        a1 = np.sum(histogram[0:int(i)])
        if (a1 / a) > fraction:
            n=i
        if (a1 / a) < fraction:
            n = i * 1.5
            i0 = i
            if n > h:
                n=h

    return (i*(bins[1]-bins[0])) + np.min(array)


def annulus_mask(array_shape, inner_radius, outer_radius, center=(0,0)):
    # returns a boolean array with an annulus mask for an ellipse
    # expects the inner_radius and outer_radius to be a tuple of (a,b)

    array_shape = np.array(array_shape)
    inner_radius = np.array(inner_radius).astype(float)
    outer_radius = np.array(outer_radius).astype(float)
    center = np.array(center).astype(float)

    yy = np.tile(np.arange(0, array_shape[1], 1), (array_shape[0], 1))-center[1]+1
    xx = np.rot90(np.tile(np.arange(0, array_shape[0], 1), (array_shape[1], 1))-center[0]+1)

    inner_radius[np.where(inner_radius <= 0)] = 0.0001
    outer_radius[np.where(outer_radius <= 0)] = 0.0001

    rr_in = (xx/inner_radius[0])**2 + (yy/inner_radius[1])**2 >= 1
    rr_out = (xx/outer_radius[0])**2 + (yy/outer_radius[1])**2 < 1

    return rr_in * rr_out


def coord_rec2spher(xx, yy, zz):
    # http://sce.uhcl.edu/helm/SpaceNavHandbook/Chapter5.pdf
    # SPACE NAVIGATION HANDBOOK  NAVPERS 92988 July 1, 1961

    rr = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    beta = Latitude(np.arctan(zz / np.sqrt(xx ** 2 + yy ** 2)))
    lamb = Longitude(np.arctan(yy / xx))

    # Dist from Sun, Latitude, Longitude
    return [rr, beta, lamb]


def coord_spher2rec(rr, beta, lamb):
    # http://sce.uhcl.edu/helm/SpaceNavHandbook/Chapter5.pdf

    xx = rr * np.cos(beta) * np.cos(lamb)
    yy = rr * np.cos(beta) * np.sin(lamb)
    zz = rr * np.sin(beta)

    return [xx, yy, zz]