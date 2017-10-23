from readdatautil import *
from statsutil import *
from plotutil import *
import numpy

names = getallnames()
all_data = getalldata(cols=['co2', 'ox'], minutestart=3.5, breakcols=True)
#all_data = getnparraystartingat('alex', cols=['co2', 'ox'], minutestart=3.5)
#all_data = breakdata(all_data)

co2avg = []
oxavg = []
for name in names:
	co2avg.append(getaverages(all_data[name][0]))
	oxavg.append(getaverages(all_data[name][1]))

co2avg = sum(co2avg)/float(len(co2avg))
oxavg = sum(oxavg)/float(len(oxavg))

print(co2avg, oxavg)

allco2residuals = []
alloxresiduals = []

for name in names:
	all_data[name][0] = subtractfromcol(all_data[name][0], co2avg)
	all_data[name][1] = subtractfromcol(all_data[name][1], oxavg)
	allco2residuals.append(all_data[name][0])
	alloxresiduals.append(all_data[name][1])
	print("{0:.3f}".format(getcorrcoeff(all_data[name][0], all_data[name][1])[0, 1]))

plotdataonsubplots(all_data, savefig="figs/dc_co2_ox_graphs/allonone.png", title="All CO2/Ox residuals")

allco2residuals = flattenlistofarrays(allco2residuals)
alloxresiduals = flattenlistofarrays(alloxresiduals)

plothistogram(allco2residuals, title="CO2 Residuals", savefig="figs/histograms/co2resids.png", nbins=20)
plothistogram(alloxresiduals, title="Ox Residuals", savefig="figs/histograms/oxresids.png", nbins=20)
