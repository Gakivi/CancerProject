from sklearn.ensemble import RandomForestClassifier
import mldatautil as ml
import sys

if len(sys.argv) != 3:
    print('usage: sript.py feature_flags combine_activities')
    quit()

feature_flags = sys.argv[1]
combine = sys.argv[2]

# get the data into numpy arrays
train_x, train_y, test_x, test_y = ml.gettraintestdata(feature_flags, combine)

clf = RandomForestClassifier(n_jobs=2)
clf.fit(train_x, train_y)

pred = clf.predict(test_x)
correct = 0
tot = 0
for idx in range(test_y.shape[0]):
    if pred[idx] == test_y[idx]:
        correct += 1
    tot += 1
test_acc = correct/tot


pred = clf.predict(train_x)
correct = 0
tot = 0
for idx in range(train_y.shape[0]):
    if pred[idx] == train_y[idx]:
        correct += 1
    tot += 1
train_acc = correct/tot
ml.printresults("random forest", test_acc, train_acc)

