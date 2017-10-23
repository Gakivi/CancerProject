from sklearn.naive_bayes import GaussianNB
import mldatautil as ml
import sys

if len(sys.argv) != 3:
    print('usage: sript.py feature_flags combine_activities')
    quit()

feature_flags = sys.argv[1]
combine = sys.argv[2]

# get the data into numpy arrays
train_x, train_y, test_x, test_y = ml.gettraintestdata(feature_flags, combine)

model = GaussianNB()
model.fit(train_x, train_y)

predicted = model.predict(test_x)
correct = 0
tot = 0
for idx in range(test_y.shape[0]):
    if test_y[idx] == predicted[idx]:
        correct += 1

    tot += 1

test_acc = correct/tot

predicted = model.predict(train_x)
correct = 0
tot = 0
for idx in range(train_x.shape[0]):
    if train_y[idx] == predicted[idx]:
        correct += 1

    tot += 1

train_acc = correct/tot
ml.printresults("naive bayes", test_acc, train_acc)

