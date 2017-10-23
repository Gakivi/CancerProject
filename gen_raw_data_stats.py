import readdatautil as ru
import statsutil as su
import numpy as np

dfull = {}
dactive = {}

f = open('raw_data_stats.txt', 'w')

for header in ru.headerssimple:
    dfull[header] = np.empty(0,)
    dactive[header] = np.empty(0,)

for name in ru.getallidxnames():
    print(name)
    for feature in ru.headerssimple:
        rawdata = ru.getdata(name, cols=feature)
        rawactivedata = ru.getdata(name, cols=feature, minstart=2.1)
        
        dfull[feature] = np.append(dfull[feature], rawdata)
        dactive[feature] = np.append(dactive[feature], rawactivedata)
    

for header in ru.headerssimple:
    print(header, file=f)

    full = dfull[header]
    active = dactive[header]

    print("mean:", file=f)
    print(np.mean(full), file=f)
    print(np.mean(active), file=f)

    print("std. dev.:", file=f)
    print(np.std(full), file=f)
    print(np.std(active), file=f)

    print("median", file=f)
    print(np.median(full), file=f)
    print(np.median(active), file=f)

    print("variance", file=f)
    print(np.var(full), file=f)
    print(np.var(active), file=f)

    print("min, max", file=f)
    print(full.min(), full.max(), file=f)
    print(active.min(), active.max(), file=f)
