from PolarCoronalHoles import PCH_Tools, PCH_Detection
from sunpy import map
import sunpy.data.sample
import numpy as np
from skimage import measure
from sunpy.coordinates.utils import GreatArc
import matplotlib.pyplot as plt

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


def test_pick_hole_extremes():

    PCH_Detection.pch_mask(test_map)
    holes = measure.label(np.logical_not(test_map.mask).astype(int), connectivity=1, background=0)

    plot_map = map.Map(sunpy.data.sample.AIA_171_IMAGE)

    fig = plt.figure()
    ax = plt.subplot(projection=plot_map)
    plot_map.plot(axes=ax)

    for r_number in range(1, np.max(holes)+1, 1):
        hole_coords = test_map.pixel_to_world(np.where(holes == r_number)[1] * u.pixel,
                                                np.where(holes == r_number)[0] * u.pixel, 0)
        hole_start, hole_end = PCH_Detection.pick_hole_extremes(hole_coords)
        great_arc = GreatArc(hole_start, hole_end)
        ax.plot_coord(great_arc.coordinates(), color='c')
        print(hole_start, hole_end)

    plt.show()


def test_chole_area():
    PCH_Detection.pch_mask(test_map)

