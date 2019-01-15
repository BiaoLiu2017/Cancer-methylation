#-*- coding:utf-8 -*-
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import os
import sys

tf.reset_default_graph()
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape = (n_x, None), name = "X")
    Y = tf.placeholder(tf.float32, shape = (n_y, None), name = "Y")
    return X, Y

#initialize parameters
def initialize_parameters():
    W1 = tf.get_variable("W1", [100, 13], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [100, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [100, 100], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [100, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1, 100], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [1, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    return parameters

#forward propagation
def forward_propagation(X, parameters, is_training):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.matmul(W1, X)
    B1 = tf.layers.batch_normalization(Z1, axis = 0, training = is_training) 
    A1 = tf.nn.relu(B1) 
    Z2 = tf.matmul(W2, A1)          
    B2 = tf.layers.batch_normalization(Z2, axis = 0, training = is_training) 
    A2 = tf.nn.relu(B2)                               
    Z3 = tf.add(tf.matmul(W3, A2 ), b3)                 

    return Z3    


X, Y = create_placeholders(13, 1)
is_training = tf.placeholder(tf.bool)
parameters = initialize_parameters()
Z3 = forward_propagation(X, parameters, is_training)

correct_prediction = tf.equal(tf.to_float(tf.sigmoid(Z3) >= 0.5), Y)
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
config=tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=config)

#load data
X_test_org = np.loadtxt(sys.argv[1])
#Y_test_org = np.loadtxt("label_163.list")
X_test = X_test_org.transpose()
#Y_test = Y_test_org.reshape(1,169)

saver1 = tf.train.Saver()
saver1.restore(sess, '../models/model_0.5-3970')

f1 = open(sys.argv[2], 'a')
f2 = open(sys.argv[3], 'a')
#f3 = open("result_accuracy_promoter.txt", 'a')
    
Z_test = sess.run(Z3, feed_dict = {X : X_test, is_training: False})
A_test = tf.to_float(tf.sigmoid(Z_test)).eval(session=sess)
Y3_test = tf.to_float(A_test >= 0.5 ).eval(session=sess)
for i in range(X_test_org.shape[0]):
    f1.write("%f"%(A_test[0,i])+'\n')
    f2.write("%d"%(Y3_test[0,i])+'\n')

#test_acc = str(accuracy.eval({X: X_test, Y: Y_test, is_training: False}, session=sess))
#f3.write('Test accuracy: ' + test_acc + '\n')

f1.close()
f2.close()
#f3.close()

sess.close()

