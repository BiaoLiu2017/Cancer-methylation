#-*- coding:utf-8 -*-
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
tf.reset_default_graph()
def create_placeholders(n_x):
    X = tf.placeholder(tf.float32, shape = (n_x, None), name = "X")
    return X
def initialize_parameters():
    #tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [100, 12], initializer = tf.contrib.layers.xavier_initializer())
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

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Z1 = tf.add(tf.matmul(W1, X), b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                              # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2 ), b3)                                              # Z3 = np.dot(W3,Z2) + b3
    return Z3          #[1,8656]/[1,1100]

X = create_placeholders(12)
print(X.shape)
parameters = initialize_parameters()
print(parameters['W1'])
print(parameters['W2'])
print(parameters['W3'])
Z3 = forward_propagation(X, parameters)

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)

X_train_org = np.loadtxt("final_top12_marker_450K_test_3055_scaler.txt", delimiter="\t")
Y_train_org = np.loadtxt("TCGA_GEO_test_3055.label", delimiter="\t")
X_train = X_train_org.transpose()
Y_train = Y_train_org.reshape(1,3055)

saver = tf.train.Saver()
#saver = tf.train.import_meta_graph('logs_and_models/model_0.5_formeta-0.meta')
saver.restore(sess, 'logs_and_models/train_450K/model_0.5-1000')

#print(sess.run(Z3, feed_dict = {X : X_train[:,0:1]}))

f1 = open("result_sigmoid.txt_3", 'a')
f2 = open("result_prediction.txt_3", 'a')
for i in range(20):
    Z = sess.run(Z3, feed_dict = {X : X_train[:,i:i+1]})
    A3 = tf.to_float(tf.sigmoid(Z)).eval(session=sess)
    Y3 = tf.to_float(A3 >= 0.5 ).eval(session=sess)
    #f1.write(A3)
    #f1.write('\n')
    #f2.write(Y3))
    #f2.write('\n')
    f1.write("%f"%(A3[0,0])+'\n')
    f2.write("%d"%(Y3[0,0])+'\n')


f1.close()
f2.close()

sess.close()

