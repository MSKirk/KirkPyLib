from PolarCoronalHoles import PCH_Tools, PCH_Detection
from sunpy import map
import sunpy.data.sample
import numpy as np

test_map = map.Map(sunpy.data.sample.AIA_171_IMAGE)


def assertEquals(var1, var2):
    if var1 == var2:
        return True
    else:
        return False


def test_rsun_pix():
    pixels = PCH_Detection.rsun_pix(test_map)

    if assertEquals(np.sum(pixels), 786.95568651283929):
        print('Pass')
    else:
        print('rsun_pix fails')


def test_pch_mask():
    PCH_Detection.pch_mask(test_map)

    if assertEquals(test_map.mask.dtype, 'bool'):
        print('Pass')
    else:
        print('pch_mask mask datatype invalid')

    if assertEquals(np.sum(test_map.mask), 1046296):
        print('Pass')
    else:
        print('pch_mask mask check sum fail')

