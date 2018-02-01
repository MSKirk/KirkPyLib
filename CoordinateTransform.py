# Coordinate Translation between HEC_cartesian and HEC_spherical coordinates.
#
#
# http://sce.uhcl.edu/helm/SpaceNavHandbook/Chapter5.pdf
# SPACE NAVIGATION HANDBOOK  NAVPERS 92988 July 1, 1961

import numpy as np
from astropy.coordinates import Longitude, Latitude

def coord_rec2spher(xx, yy, zz):

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