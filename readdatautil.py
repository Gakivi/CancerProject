import numpy as np
import requests
import re
from glob import glob
from numpy.lib import recfunctions as rfn
import os

''' 
This utility module should be very useful for extracting any data and in any order 
that I may need
'''

VERBOSE = True

headerssimple = ['co2', 'ox', 'x', 'y', 'z', 'flow', 'hr'] # order follows /d+.csv files
headersfull = ['CO2Percentage', 'oxygenPercentage', 'accelX', 'accelY', 'accelZ', 
			'flowData', 'hr']
activities = ['maskoff', 'stand1', 'walk1', 'jog', 'run', 'jump', 'walk2', 'stand2']

activity_time_ranges = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 11],
            [11, 13], [13, 15]]

base_path = os.path.dirname(os.path.abspath(__file__))
person_data_path = base_path + "/person_data"

# returns list of names
def getallnames():
        nameslist =  base_path + "/nameslist.txt"
        return [line.rstrip('\n') for line in open(nameslist)]

def getallidxnames():
        return range(len(getallnames()))

def getindexofname(name):
        _names = getallnames()
        if name in _names:
                return _names.index(name)
        else:
                return -1

# returns a dictionary with name : fulldata
# if breakcols is True, return list of cols
def getalldata(cols=headerssimple, minutestart=0, minuteend=15, breakcols=False):
        d = {}
        for name in getallnames():
                if VERBOSE: print("Getting data for {}".format(name))
                d[name]=getnparraystartingat(name, minutestart=minutestart, 
                        minuteend=minuteend, cols=cols)
                if breakcols:
                        d[name] = breakdata(d[name])
                        
        return d

# if names is a list, return a list of data
def getfullnparray(names, cols=headerssimple):
        if type(names) == list:
                lst = []
                for name in names:
                        lst.append(getdataforname(name, cols))
                return lst
        else:
                return getdataforname(names, cols)

# simpler name for function
def getdata(name, minstart = 0, minend=15, cols=headerssimple):
        if cols == 'accels': cols=['x', 'y', 'z']
        return getnparraystartingat(name, minstart, minend, cols)

# keeps the order of the columns
def getdataforname(name, cols=headerssimple):
        if type(name) == int:
                name = getallnames()[name]

        if type(cols) == str:
                cols = [cols]

        if not set(cols) <= set(headerssimple):
                print("Invalid columns passed in {}".format(cols))
                return None

        paths = glob('{}/person_data/{}_*/*csv'.format(base_path, name))
        pattern = re.compile("^.*\d+.csv$")
        path = [f for f in paths if pattern.match(f)][0]
        data = np.genfromtxt(path, delimiter=',', dtype=float)

        if cols != headerssimple:
                colindices = [headerssimple.index(col) for col in cols]
                newdata = np.empty([data.shape[0], len(cols)])
                for idx in colindices:
                        newdata[:,colindices.index(idx)] = data[:,idx]

                return newdata
        else:
                return data

# min to start at, can be float
def getnparraystartingat(name, minutestart, minuteend=15, cols=headerssimple):
        data = getfullnparray(name, cols)
        idxstart = int((minutestart/15)*(data.shape[0]))
        idxend = int((minuteend/15)*(data.shape[0]))
        return data[idxstart:idxend,]

# timetrim before and after, in seconds
def getactivitydata(name, activity, timetrim=5 ,cols=headerssimple):
        if activity not in activities:
                print("Invalid activity {}".format(activity))
                return None

        timetrimmins = float('%.4f' % (timetrim / 60))
        timediv = activity_time_ranges[activities.index(activity)]

        return getnparraystartingat(name, timediv[0]+timetrimmins, timediv[1]-timetrimmins, cols)

# this creates a new file and adds hr column to it
# do this for new trials
def generatefulldatawthhr(name):
        paths = glob('person_data/{}_*/*M.csv'.format(name))
        data = np.genfromtxt(paths[0], dtype=float, delimiter=',', usecols=(0, 5, 1, 2, 3, 4))

        hrpaths = glob('person_data/{}_*/hr.csv'.format(name))
        heartrates = np.genfromtxt(hrpaths[0], dtype=int, delimiter=',')  # 900 HRs for every second
        a = np.zeros(len(data))
        data_freq = len(data) / 900
        idx = 0
        while idx < len(data):
                a[idx] = hr_arr[int(idx/data_freq)]
                idx += 1

        heartrates = a

        data = rfn.append_fields(data, names='hr', data=heartrates)

        np.savetxt(hrpaths[0].replace("hr.csv", "{}.csv".format(getallnames().index(name))), 
                                data, fmt="%.3f, %.3f, %i, %i, %i, %i, %i")

# given a numpy array with multiple columns, return those columns as such: [c1, c2, ...]
def breakdata(data):
        return [data[:,col] for col in range(data.shape[1])]

def updatenameslist():
        url = 'http://maskapp.herokuapp.com/'
        filename = "nameslist.txt"

        response = requests.get(url)

        match = re.findall(b"data-id='([a-zA-Z0-9_]+)'><", response.content)

        print("Found {} names, writing to {}".format(len(match), filename))

        f = open(filename, "w+")
        for m in match:
                f.write(m.decode("utf-8") + "\n")

        f.close()

# for now assume the data is one col, such as flow rate
def writedatatofile(data, name, filename="breath.csv"):
        if type(name) == int:
                name = getallnames()[name]

        filename = glob('person_data/{}_*/hr.csv'.format(name))[0].replace('hr.csv', filename)
        f = open(filename, 'w')
        for d in data:
                f.write("{}\n".format(d))

        f.close()

# this returns an array of 900
def gethrdata(name):
        if type(name) == int:
                name = getallnames()[name]

        hrsfile = glob(person_data_path + '/{}_*/hr.csv'.format(name))[0]

        lst = [int(line.rstrip('\n')) for line in open(hrsfile)]
        return np.array(lst).reshape(len(lst), 1) # reshape to give us (900,1) shape
