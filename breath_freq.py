from readdatautil import getdata, getactivitydata, getallnames, \
	activities, getallidxnames, writedatatofile
import statsutil as su
import plotutil as pu
from plotutil import plot, plotfull, plotdataonsubplots

names=getallidxnames()
for name in names:
	print(name)
	data = getdata(name, cols='flow')
	freq = len(data)/(900) # samples per second
	breathspm = []
	breathspm_filtered = []

	winsize = 60
	half = int(winsize/2) # rounded down
	idx = half+1

	breathspm_filtered += [0]*int(half*freq)

	k = .4
	n = 2
	while idx <= 900-half+1:
		tstart = int((idx-half)*freq)
		tend = int((idx+half)*freq)
		subdata = su.digitalfilter(data[tstart:tend], k=k)
		localmin, _ = su.getlocalextremas(subdata,n)
		breathspm_filtered += [len(localmin) * (60/winsize)]*int(freq)

		idx += 1

	breathspm_filtered += [0]*int((half*freq))
	breathspm_filtered[0:int(2*60*freq)]=[0]*int(2*60*freq) # zero out first two minutes

	pu.plotfull(breathspm_filtered, name, savefig="figs/breath_freq/breathing_rate/{}.k_{}.n_{}.png"
		.format(name, k, n))
	writedatatofile(breathspm_filtered, getallnames()[name])


