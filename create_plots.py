import readdatautil as ru
import statsutil as su
import plotutil as pu
import numpy as np

''' this script is for creating all plots for report '''

rawflow = su.getdata(8, cols='flow')
rawflow2 = su.getdata(9, cols='flow')
rawx = su.getdata(8, cols='x')
rawy = su.getdata(8, cols='y')
rawz = su.getdata(8, cols='z')
rawhr = su.getdata(8, cols='hr')
rawco2 = su.getdata(8, cols='co2')
rawox = su.getdata(8, cols='ox')

pu.plotfull([rawflow, rawflow2], yaxislabel = "Liters/Min",title="Flow Rate", savefig="figs/for_report/flow.png")
pu.plotfull(rawx, yaxislabel = '$\mathit{g}$\'s',title="X Accel",savefig="figs/for_report/x.png")
pu.plotfull(rawy, yaxislabel = '$\mathit{g}$\'s',title="Y Accel", savefig="figs/for_report/y.png")
pu.plotfull(rawz, yaxislabel = '$\mathit{g}$\'s',title="Z Accel", savefig="figs/for_report/z.png")
pu.plotfull(rawox, yaxislabel = 'Oxygen %',title="Oxygen Levels", savefig="figs/for_report/ox.png")
pu.plotfull(rawco2, yaxislabel = '$CO_2$ %', title="$CO_2$", savefig="figs/for_report/co2.png")
pu.plotfull(rawhr, yaxislabel = 'Beats/Min', title="Heartrate", savefig="figs/for_report/hr.png")
