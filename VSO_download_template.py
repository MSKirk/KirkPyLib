## VSO Data Request Template

from sunpy.net import vso
import astropy.units as u

client=vso.VSOClient()

qr = client.query(vso.attrs.Time('2015/10/01T07:00:00', '2015/10/01T15:00:00'), vso.attrs.Instrument('aia'), vso.attrs.Wave(304* u.AA, 304* u.AA), vso.attrs.Provider('JSOC'))

len(qr) #number of records returned

res=client.get(qr, path='/Users/mskirk/data/AIA/20151001/{file}.fits')