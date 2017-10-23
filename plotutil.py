import matplotlib.pyplot as plt
import numpy as np
from math import ceil, sqrt, floor
from readdatautil import activity_time_ranges
from collections import OrderedDict

def saveorshow(savefig):
	if savefig:
		print("Saving: {}".format(savefig))
		plt.savefig(savefig)
		plt.clf()
	else:
		plt.show()

activity_time = [0, 2, 4, 6, 8, 10, 11, 13]
activities = ['off', 'stand1', 'walk1', 'jog', 'run', 'jump', 'walk2', 'stand2']

# this plot assumes full 15 minutes and plots with x axis labels
def plotfull(data, title="", yaxislabel="", addvertlines=True,savefig=None):
	if type(data) != list:
            lstdata = [data]
	else:
            lstdata = data

	xticks=[]
	for time in activity_time:
            x = floor((time/15)*len(lstdata[0]))
            xticks.append(x)
            if addvertlines:
                plt.axvline(x=x, linestyle='--', linewidth=1, color='gray')

	for d in lstdata:
            plt.plot(d)

	# plt.scatter(range(len(data)),data, s=1)
	plt.xticks(xticks, activities)

	plt.title(title)
	plt.ylabel(yaxislabel)

	saveorshow(savefig)

def plot(data, title="", ylim=None, savefig=None):
	plotdatacols([data], title, ylim, savefig)

# optional param: accels
def plothrandbreath(hr, breath, accels=None,savefig=None):
	fig, ax = plt.subplots()
	ax.plot(hr, label='heart rate')

	if accels is not None: # assume one col
		ax.scatter(range(len(accels)) ,accels, c='g', label='accelerations', s=1)

	for idx in range(len(breath)):
		tdiv = activity_time_ranges[idx+1]
		idxstart = int((tdiv[0]/15)*(hr.shape[0]))
		idxend = int((tdiv[1]/15)*(hr.shape[0]))
		mins = tdiv[1]-tdiv[0]

		ax.plot((idxstart, idxend), (breath[idx]/mins, breath[idx]/mins), c='r', label='breath rate')
		if idx+1 < len(breath):
			nextmins=activity_time_ranges[idx+2][1]-activity_time_ranges[idx+2][0]
			plt.plot((idxend, idxend), (breath[idx]/mins, breath[idx+1]/nextmins), c='r')

	xticks=[]
	for time in activity_time:
 		xticks.append(floor((time/15)*len(hr)))

	legend = ax.legend(loc='upper left', shadow=True)
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys(), loc='upper left', shadow=True)
 	
	plt.xticks(xticks, activities)

	saveorshow(savefig)

def plotdatacols(data, title="", ylim=None, savefig=None):
	if (type(data) == list):
		for idx in range(len(data)):
			print(len(data))
			plt.plot(data[idx], label=idx)
	else:
		numcols = data.shape[1]
		for idx in range(numcols):
			plt.plot(data[:,[idx]], label=str(idx))

	plt.title(title)
	plt.legend(loc='upper left')

	if ylim:
		plt.ylim(ylim[0], ylim[1])

	saveorshow(savefig)

def plothistogram(data, title="", savefig=None, nbins=10):
	plt.hist(data, bins=nbins)
	plt.title(title)

	saveorshow(savefig)

# assume data is a list
def plotflowwithbreathfreq(flowdata, maxfreq, minfreq, plotbits=0b111 ,savefig=None):
	if plotbits & 0b100: 
		plt.plot(flowdata)
		plt.scatter(range(len(flowdata)), flowdata, c='b', s=1)
	if plotbits & 0b010: plt.scatter(maxfreq, flowdata[maxfreq], c='r', s=8)
	if plotbits & 0b001: plt.scatter(minfreq, flowdata[minfreq], c='g', s=8)
	plt.title("Max breath: {} Min breath: {}".format(len(maxfreq), len(minfreq)))

	saveorshow(savefig)

# winsize in seconds
def plotbreath(flow, maxima, winsize=5, savefig=None):
	def countsinrange(tstart, tend): # returns the number of maxima in range
		return sum(tstart <= x < tend for x in maxima)

	freq = len(flow)/(15*60)

	tstart=0
	tend=winsize
	breathfreqs = []
	while tstart < (15*60) - winsize:
		breathfreqs += [0]*int(winsize/2)
		breathfreqs.append(countsinrange(int(tstart*freq), int(tend*freq)))
		breathfreqs += [0]*int(winsize/2)
		tstart += winsize
		tend += winsize

	breathfreqs = [float('nan') if x==0 else x for x in breathfreqs]
	plt.scatter(range(len(breathfreqs)), breathfreqs, s=2)
	saveorshow(savefig)


# given a dictionary (name->data), plot on subplots
def plotdataonsubplots(data, title="", savefig=None):
	x = int(sqrt(len(data)))
	y = ceil(len(data) / x)

	fig, axarr = plt.subplots(x, y)
	fig.suptitle(title, fontsize=16)

	for idx in range(len(data)):
		name = sorted(data.keys())[idx]
		d = data[name]
		shape = axarr.shape

		x = idx % shape[0]
		y = floor(idx/shape[0])
		if (type(d) == list):
			for col in d:
				axarr[x, y].plot(col)
		else:
			axarr[0, 0].plot(d)

		axarr[x, y].set_title(idx+1)

	fig.tight_layout()
	fig.subplots_adjust(top=0.88)

	saveorshow(savefig)

def plotbreathstepfunction(maxlocations, vals, datalen, title="",savefig=None):
	newarr = np.zeros(datalen)
	newarr[maxlocations] = vals

	prevx = -1
	prevy = -1
	for x in range(newarr.shape[0]):
		if newarr[x] == 0:
			continue
		elif prevx== -1:
			prevx = x
		else:
			plt.plot((prevx, x), (newarr[prevx], newarr[x]), 'r-')
			prevx = x

	plt.title(title)
	saveorshow(savefig)
