from sklearn import svm
import mldatautil as ml
import sys

if len(sys.argv) != 3:
    print('usage: sript.py feature_flags combine_activities')
    quit()

feature_flags = sys.argv[1]
combine = sys.argv[2]

# get the data into numpy arrays
train_x, train_y, test_x, test_y = ml.gettraintestdata(feature_flags, combine)

clf = svm.SVC()
clf.fit(train_x, train_y)

correct = 0
tot = 0
for idx in range(test_y.shape[0]):
    test_y_label = test_y[idx]
    reshaped = test_x[idx].reshape(1, -1)
    tot +=1
    if test_y_label == clf.predict(reshaped)[0]:
        correct += 1

test_acc = correct/tot

correct = 0
tot = 0
for idx in range(train_y.shape[0]):
    train_y_label = train_y[idx]
    reshaped = train_x[idx].reshape(1, -1)
    tot +=1
    if train_y_label == clf.predict(reshaped)[0]:
        correct += 1

train_acc = correct/tot

ml.printresults("svm", test_acc, train_acc)