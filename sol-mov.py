## sol-mov.py
## Script to generate multi-wavelength EUV animation of the sun

## Import libraries
import sunpy
from sunpy.net import vso
from sunpy.time import TimeRange
import sunpy.map
from astropy import units as u
import datetime
import os
import glob
import PIL

## Setup your directory location for data storage
datadir = '/Users/clowder/data/'

## Define search parameters
ndays = 4
tsample = 0.5*u.hour
curtime = datetime.datetime.utcnow()
starttime = curtime - datetime.timedelta(days=ndays+1)
endtime = curtime - datetime.timedelta(days=1)

## Initialize the VSO search
client = vso.VSOClient()

## Search for matching files
## Merge these two queries at some point...
qr = client.query(vso.attrs.Time(starttime, endtime), vso.attrs.Instrument('AIA'), vso.attrs.Resolution(0.25), vso.attrs.Sample(tsample), vso.attrs.Wave(193*u.AA,193*u.AA) | vso.attrs.Wave(304*u.AA,304*u.AA) | vso.attrs.Wave(171*u.AA,171*u.AA) | vso.attrs.Wave(211*u.AA,211*u.AA) | vso.attrs.Wave(131*u.AA,131*u.AA) | vso.attrs.Wave(335*u.AA,335*u.AA) | vso.attrs.Wave(94*u.AA,94*u.AA) | vso.attrs.Wave(1600*u.AA,1600*u.AA))
qr2 = client.query(vso.attrs.Time(starttime, endtime), vso.attrs.Instrument('HMI'), vso.attrs.Resolution(0.25), vso.attrs.Sample(tsample), vso.attrs.Physobs('los_magnetic_field'))

## Remove existing files
files = glob.glob(datadir+'download/*')
for f in files:
    os.remove(f)
files = glob.glob(datadir+'frames/*')
for f in files:
    os.remove(f)

## Download new files
res=client.get(qr, path=datadir+'download/{file}.fits').wait()
res2=client.get(qr2, path=datadir+'download/{file}.fits').wait()

## Grab a list of files
files = glob.glob(datadir+'download/*0193*')
mfiles = glob.glob(datadir+'download/hmi*')

## Loop through each frame
for f in arange(len(files)):

	## Load each filter into a separate sunpy map object
	m193 = sunpy.map.Map(files[f][:-14]+'0193.fits.fits')
	m304 = sunpy.map.Map(files[f][:-14]+'0304.fits.fits')
	m171 = sunpy.map.Map(files[f][:-14]+'0171.fits.fits')
	m211 = sunpy.map.Map(files[f][:-14]+'0211.fits.fits')
	m131 = sunpy.map.Map(files[f][:-14]+'0131.fits.fits')
	m335 = sunpy.map.Map(files[f][:-14]+'0335.fits.fits')
	m094 = sunpy.map.Map(files[f][:-14]+'0094.fits.fits')
	m1600 = sunpy.map.Map(files[f][:-14]+'1600.fits.fits')

	## Normalize to exposure time, and make some 'brightness' adjustments
	d193 = log((abs(m193.data) / float(m193.exposure_time/u.second) / 2000 * 10)+1.)
	d304 = log((abs(m304.data) / float(m304.exposure_time/u.second) / 100 * 10)+1.)
	d171 = log((abs(m171.data) / float(m171.exposure_time/u.second) / 2000 * 10)+1.)
	d211 = log((abs(m211.data) / float(m211.exposure_time/u.second) / 1500 * 10)+1.)
	d131 = log((abs(m131.data) / float(m131.exposure_time/u.second) / 100 * 10)+1.)
	d335 = log((abs(m335.data) / float(m335.exposure_time/u.second) / 50 * 10)+1.)
	d094 = log((abs(m094.data) / float(m094.exposure_time/u.second) / 50 * 10)+1.)
	d1600 = log((abs(m1600.data) / float(m1600.exposure_time/u.second) / 700 * 10)+1.)

	## Create a 4-channel image array from each filter colortable
	im193 = sunpy.cm.cm.sdoaia193(d193)
	im304 = sunpy.cm.cm.sdoaia304(d304)
	im171 = sunpy.cm.cm.sdoaia171(d171)
	im211 = sunpy.cm.cm.sdoaia211(d211)
	im131 = sunpy.cm.cm.sdoaia131(d131)
	im335 = sunpy.cm.cm.sdoaia335(d335)
	im094 = sunpy.cm.cm.sdoaia94(d094)
	im1600 = sunpy.cm.cm.sdoaia1600(d1600)

	## A quick function to define a gaussian mask
	def gen_filter(b,c):
		xarr = arange(1024)/1024.
		yarr = exp(-(xarr-b)**2 / (2*c**2))
		return yarr

	## Define the number of mask windows
	posarr = linspace(0.1,0.9,num=8)

	## For each filter, add a masked version into the output image arrays
	frmarr = array(im304)
	for d in arange(4): frmarr[:,:,d] = frmarr[:,:,d]*gen_filter(posarr[0],0.07)
	for d in arange(4): frmarr[:,:,d] = frmarr[:,:,d]+im1600[:,:,d]*gen_filter(posarr[1],0.07)
	for d in arange(4): frmarr[:,:,d] = frmarr[:,:,d]+im171[:,:,d]*gen_filter(posarr[2],0.07)
	for d in arange(4): frmarr[:,:,d] = frmarr[:,:,d]+im193[:,:,d]*gen_filter(posarr[3],0.07)
	for d in arange(4): frmarr[:,:,d] = frmarr[:,:,d]+im211[:,:,d]*gen_filter(posarr[4],0.07)
	for d in arange(4): frmarr[:,:,d] = frmarr[:,:,d]+im335[:,:,d]*gen_filter(posarr[5],0.07)
	for d in arange(4): frmarr[:,:,d] = frmarr[:,:,d]+im094[:,:,d]*gen_filter(posarr[6],0.07)
	for d in arange(4): frmarr[:,:,d] = frmarr[:,:,d]+im131[:,:,d]*gen_filter(posarr[7],0.07)
	frmarr[:,:,3] = zeros([1024, 1024], dtype='uint8')+1.

	## Normalize the image arrays, and alpha channel
	frmarr[:,:,0:3] = frmarr[:,:,0:3] / frmarr[:,:,0:3].max()
	frmarr = np.uint8(frmarr * 255)

	## Save this image to disk
	scipy.misc.imsave(datadir+'frames/frame'+'%05.f'%f+'.png', frmarr)

## Create an animation from this series
os.system("ffmpeg -r 15 -i /Users/clowder/data/frames/frame%05d.png -vcodec libx264 -pix_fmt yuv420p -crf 15 ./sol-mov.mov")