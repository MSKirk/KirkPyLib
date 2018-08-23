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




if northern:
    offset_coords = np.transpose(np.asarray([lons, (90 * u.deg) - lats]))
else:
    offset_coords = np.transpose(np.asarray([lons, (90 * u.deg) + lats]))

offset = PCH_Tools.center_of_mass(offset_coords, mass=1 / errors)
offset_lons = offset_coords[:, 0] - offset[0]
offset_lats = offset_coords[:, 1] - offset[1]

hole_fit = PCH_Tools.trigfit(np.deg2rad(lons), np.deg2rad(lats), degree=6, sigma=errors)
co_hole_fit = PCH_Tools.trigfit(np.deg2rad(offset_coords[:,0])*u.rad, np.deg2rad(offset_coords[:,1])*u.rad, degree=6, sigma=errors)
offset_hole_fit = PCH_Tools.trigfit(np.deg2rad(offset_lons)*u.rad, np.deg2rad(offset_lats)*u.rad, degree=6, sigma=errors)


fit_yy =  hole_fit['fitfunc'](np.deg2rad(lons).value)
co_fit_yy = co_hole_fit['fitfunc'](np.deg2rad(offset_coords[:,0]))
offset_fit_yy = offset_hole_fit['fitfunc'](np.deg2rad(offset_lons))


plt.plot(np.deg2rad(lons).value, fit_yy - np.deg2rad(lats).value, '.')
plt.plot(np.deg2rad(offset_coords[:,0]), co_fit_yy - np.deg2rad(offset_coords[:,1]), '.')
plt.plot(np.deg2rad(offset_lons), offset_fit_yy - np.deg2rad(offset_lats), '.')

plt.plot(np.deg2rad(lons).value, fit_yy, '.')
plt.plot(np.deg2rad(offset_lons+np.deg2rad(offset[0])), offset_fit_yy - (np.pi *0.5)+ np.deg2rad(offset[1]), '.')
