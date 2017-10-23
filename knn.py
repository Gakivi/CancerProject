from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import mldatautil as ml
import sys

if len(sys.argv) != 3:
    print('usage: sript.py feature_flags combine_activities')
    quit()

feature_flags = sys.argv[1]
combine = sys.argv[2]

# get the data into numpy arrays
train_x, train_y, test_x, test_y = ml.gettraintestdata(feature_flags, combine)

accuracies = []
accuracies_train = []
for n_neighbors in [1,5,10]:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    knn.fit(train_x, train_y)

    pred = knn.predict(test_x)
    correct = 0
    tot = 0
    for idx in range(pred.shape[0]):
        if pred[idx] == test_y[idx]:
            correct += 1

        tot += 1 
    accuracies.append("{0:.2f}".format(correct/tot))
    
    pred = knn.predict(train_x)
    correct = 0
    tot = 0
    for idx in range(pred.shape[0]):
        if pred[idx] == train_y[idx]:
            correct += 1

        tot += 1 
    accuracies_train.append("{0:.2f}".format(correct/tot))

print("knn test: " + ",".join(accuracies) + " train: " + ",".join(accuracies_train))
