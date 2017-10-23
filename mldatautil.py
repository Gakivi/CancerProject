import numpy as np

basefolder = 'working_data/'
trainfolder = basefolder + 'train/'
testfolder = basefolder + 'test/'

params = ['co2', 'ox', 'x', 'y', 'z', 'flow', 'hr'] # order follows /d+.csv files
params_mag = ['co2', 'ox', 'accel_mag', 'flow', 'hr'] # order follows /d+.csv files

def getfilename(folder, param, scaling=1):
    filename = folder + param
    key = folder + param
    if scaling == 2:
        filename += '_fs'
    elif scaling == 3:
        filename += '_std'
    
    return filename + '.csv'

def getcombinebool(b):
    if b in ['T', 't', 'True', 'true', True]:
        combine = True
    else:
        combine = False
    return combine

def printresults(alg, test_acc, train_acc):
    test_acc = float(test_acc)
    train_acc = float(train_acc)
    print("{} test: {:.3f} train: {:.3f}".format(alg, test_acc, train_acc))

def printparamsused(params_to_use, combineacts):
    combineacts = getcombinebool(combineacts)

    params_to_use = str(params_to_use)
    if len(params_to_use) == 5:
        base_params = params_mag 
    elif len(params_to_use) == 7:
        base_params = params
    else:
        print("Wrong number of params")
        quit()

    s = ""
    for i in range(len(params_to_use)):
        param = base_params[i]
        n = int(params_to_use[i])
        if n == 0:
            s += param + ":no, "
        elif n == 1:
            s += param + ":raw, "
        elif n == 2:
            s += param + ":fs, "
        elif n == 3:
            s += param + ":std, "

    if combineacts:
        s += " labels: 5"
    else:
        s += " labels: 7"

    print(s)

# params -> base3 representation - 0 -> don't use, 1 -> use raw, 2 -> use fs
# orders follows params array
def gettraintestdata(params_to_use, combineacts = False):
    combineacts = getcombinebool(combineacts)

    params_to_use = str(params_to_use)   
    if len(params_to_use) == 5:
        base_params = params_mag 
    elif len(params_to_use) == 7:
        base_params = params
    else:
        print("Wrong number of params")
        quit()

    xTrain = None
    xTest = None
    yTrain = np.genfromtxt(getfilename(trainfolder, 'label'), dtype='int')
    yTest = np.genfromtxt(getfilename(testfolder, 'label'), dtype='int')

    if combineacts:
        yTrain[yTrain == 5] = 1
        yTrain[yTrain == 6] = 0
        yTest[yTest == 5] = 1
        yTest[yTest == 6] = 0

    for i in range(len(params_to_use)):
        param = base_params[i] 
        n = int(params_to_use[i]) 
        if n == 0:
            continue
        elif n in [1,2,3]:
            filename_train = getfilename(trainfolder, param, scaling=n)
            filename_test = getfilename(testfolder, param, scaling=n)
        else:
            print("Invalid param: {}".format(str(n)))
            quit()
            
        data_train = np.genfromtxt(filename_train, delimiter=',', dtype='float32')
        data_test = np.genfromtxt(filename_test, delimiter=',', dtype='float32')
        
        if len(data_train.shape) == 1:
            data_train = np.reshape(data_train, (data_train.shape[0],1))

        if len(data_test.shape) == 1:
            data_test = np.reshape(data_test, (data_test.shape[0],1))

        xTrain = np.concatenate((xTrain, data_train), axis=1) if xTrain is not None else data_train
        xTest = np.concatenate((xTest, data_test), axis=1) if xTest is not None else data_test

    return xTrain, yTrain, xTest, yTest

