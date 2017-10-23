import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os
import math
import sys

if len(sys.argv) != 3:
    print('usage: sript.py feature_flags combine_activities')
    quit()

feature_flags = sys.argv[1]
combine = sys.argv[2]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("running multilayer")
# get the data into numpy arrays

train_x, train_y, test_x, test_y = ml.gettraintestdata(feature_flags, combine)
def next_batch(num):
    indices = random.sample(range(0, train_x.shape[0]), num)
    targets = np.array(train_y[indices,]).reshape(-1)
    one_hot_targets = np.eye(labels)[targets]
    return train_x[indices,], one_hot_targets

labels = np.unique(train_y).shape[0]
params = train_x.shape[1]

# print("running softmax on {} labels, with {} parameters".format(labels, params))

x = tf.placeholder(tf.float32, [None, params])

W = tf.Variable(tf.zeros([params, labels]))
b = tf.Variable(tf.zeros([labels]))

y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, labels])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

for i in range(5):
    tf.global_variables_initializer().run()

    for _ in range(300):
        batch_xs, batch_ys = next_batch(1000)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    one_hot_test_y = np.eye(labels)[test_y]
    one_hot_train_y = np.eye(labels)[train_y]
    
    yes1 = 0 # top 1
    yes2 = 0 # top 2
    yes3 = 0 # top 3
    tot = 0
    for idx in range(test_x.shape[0]):
        a = test_x[idx]
        w = W.eval()
        bb = b.eval()
        yy = np.dot(a,w) + bb
       
        yy_exp = [math.pow(math.e, yy_) for yy_ in yy]
        sum_yy_exp = sum(yy_exp)
        softmax = [i / sum_yy_exp for i in yy_exp]
        sorted_sm_indices = [b[0] for b in sorted(enumerate(softmax),key=lambda i:i[1], reverse=True)]

        if test_y[idx] in sorted_sm_indices[0:1]: yes1 += 1
        if test_y[idx] in sorted_sm_indices[0:2]: yes2 += 1
        if test_y[idx] in sorted_sm_indices[0:3]: yes3 += 1
         
        tot += 1
    
    print("top1: " + str(yes1 / tot))
    print("top2: " + str(yes2 / tot))
    print("top3: " + str(yes3 / tot))
    print(sess.run(accuracy, feed_dict={x: test_x, y_: one_hot_test_y}))

