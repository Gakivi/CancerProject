import tensorflow as tf
from numpy import genfromtxt
import numpy as np
import sys
import os
import mldatautil as ml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if len(sys.argv) != 3:
    print('usage: sript.py feature_flags combine_activities')
    quit()

feature_flags = sys.argv[1]
combine = sys.argv[2]

# get the data into numpy arrays
train_x, train_y, test_x, test_y = ml.gettraintestdata(feature_flags, combine)

learning_rate = .001
training_epochs = 80
batch_size = 100
display_step = 20

#Network Parameters
n_hidden_1 = train_x.shape[1]
n_hidden_2 = train_x.shape[1]
n_hidden_3 = train_x.shape[1]
n_input = train_x.shape[1]
n_classes = np.unique(train_y).shape[0]

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# create onehot test data
test_y_onehot = np.zeros((test_y.shape[0], n_classes), dtype="int")
test_y_onehot[np.arange(test_y.shape[0]), test_y] = 1

# create onehot test data
train_y_onehot = np.zeros((train_y.shape[0], n_classes), dtype="int")
train_y_onehot[np.arange(train_y.shape[0]), train_y] = 1

def multilayer_perceptron(x, weights, biases):
    # hidden with RELU act
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # hidden with RELU act
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # layer_3 = tf.nn.sigmoid(layer_3)
    # output layer with lin act
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_input, n_hidden_2])),
    # 'h3': tf.Variable(tf.random_normal([n_input, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    # 'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = multilayer_perceptron(x, weights, biases)

# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# # Init variables
init = tf.global_variables_initializer()

def getBatchedData(batchId, batchSize, data_x, data_y):
    batch_x = data_x[batchId*batchSize : batchId*batchSize + batchSize]
    batch_y = data_y[batchId*batchSize : batchId*batchSize + batchSize]
    # batch_y needs to be one-hot
    batch_y_onehot = np.zeros((batchSize, n_classes), dtype="int")
    batch_y_onehot[np.arange(batchSize), batch_y] = 1

    return batch_x, batch_y_onehot

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(train_x.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = getBatchedData(i, batch_size, train_x, train_y)
            # Run optimization op (backdrop_ and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute avg loss
            avg_cost += c / total_batch
        # display logs per epoch
        if epoch % display_step == 0:
            # print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        
            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Epoch: {} Accuracy:".format(epoch), accuracy.eval({x: test_x, y: test_y_onehot}))
 
    test_acc = str(accuracy.eval({x: test_x, y: test_y_onehot}))
    train_acc = str(accuracy.eval({x: train_x, y: train_y_onehot}))
    ml.printresults("multilayer", test_acc, train_acc)

