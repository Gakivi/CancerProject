import readdatautil as ru
import plotutil as pu
import statsutil as su

def strlsttolst(s):
	s = s.replace("[", "").replace("]", "").strip()
	l = []

	for n in s.split(','):
		l.append((int(n)))

	return l

with open("tmp.txt") as f:
	content = f.readlines()

content = [x.strip() for x in content]

d = {}
for c in content:
	k = c.split(':')[0]
	v = c.split(':')[1]
	d[k] = strlsttolst(v)

idx = 0
for name in ru.getallnames():
	breath = [d['stand1'][idx], d['walk1'][idx], d['jog'][idx], d['run'][idx], 
				d['jump'][idx], d['walk2'][idx], d['stand2'][idx]]
	hr = ru.getdata(name, cols='hr')
	accels = su.getaccelsums(ru.getdata(name, cols='accels'))
	accles = su.zerobelowthreshold(accels, threshold=20, usenan=True)
	accels = (accels / float(su.np.nanmax(accels))) * max(hr)
	pu.plothrandbreath(hr, breath, accels)
	idx += 1
