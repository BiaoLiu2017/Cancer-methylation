#-*- coding:utf-8 -*-
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

#load data
X_train_org = np.loadtxt("final_top12_marker_450K_train_8620_scaler.txt", delimiter="\t")
Y_train_org = np.loadtxt("TCGA_GEO_train_8620.label", delimiter="\t")
X_test_org = np.loadtxt("final_top12_marker_450K_validation_2158_scaler.txt", delimiter="\t")
Y_test_org = np.loadtxt("TCGA_GEO_validation_2158.label", delimiter="\t")
X_train = X_train_org.transpose()
Y_train = Y_train_org.reshape(1,8620)
X_test = X_test_org.transpose()
Y_test = Y_test_org.reshape(1,2158)

#create placeholders
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    Arguments:
    n_x -- scalar, length of every sample
    n_y -- scalar, number of classes,1
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """
    X = tf.placeholder(tf.float32, shape = (n_x, None), name = "X")
    Y = tf.placeholder(tf.float32, shape = (n_y, None), name = "Y")
    return X, Y

#initialize parameters
def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [100, 22547]
                        b1 : [100, 1]
                        W2 : [100, 100]
                        b2 : [100, 1]
                        W3 : [1, 100]
                        b3 : [1, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
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

#forward propagation
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> Sigmoid
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    # Retrieve the parameters from the dictionary "parameters" 
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

#compute_cost
def compute_cost(Z3, Y):
    """
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (2, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    Returns:
    cost - Tensor of the cost function
    """
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    return cost

#random_mini_batches
def random_mini_batches(X, Y, mini_batch_size = 32):
    """
    Creates a list of random minibatches from (X, Y)
    Arguments:
    X -- input data, of shape (input size, number of examples)[4,4]
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)[1,4]
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    #np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))  #Randomly permute a sequence  eg:[1,3,2,0]
    shuffled_X = X[:, permutation]   #sorted according to 2ed dim 
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))  

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

#model
def model(X_train, Y_train, X_test, Y_test, learning_rate_base = 0.01,
          num_epochs = 2000, minibatch_size = 16, regularaztion_rate = 0.0001, model_save_path_name = "./model", print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    Arguments:
    X_train -- training set, of shape (input size = 4, number of training examples = 4)
    Y_train -- test set, of shape (output size = 1, number of training examples = 4)
    X_test -- training set, of shape (input size = 4, number of training examples = 4)
    Y_test -- test set, of shape (output size = 1, number of test examples = 4)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 10 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    #tf.set_random_seed(1)                             # to keep consistent results
    #seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)[4,4]
    (n_x_t, m_t) = Y_train.shape
    n_y = Y_train.shape[0]                            # n_y : output size[1]
    costs = []                                        # To keep track of the cost
    costs_test = []
    acc_train = []
    acc_test = []
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)


    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, parameters['W1']) 
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, parameters['W2'])
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, parameters['W3']) 
    regularizer = tf.contrib.layers.l2_regularizer(scale = regularaztion_rate) 
    reg_term = tf.contrib.layers.apply_regularization(regularizer) 
    cost = (compute_cost(Z3, Y) + reg_term)
    #cost = compute_cost(Z3, Y)
    
    #Get learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step,
                                           decay_steps, decay_rate, staircase=True)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver(max_to_keep = 5)

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
            # Run the initialization
            sess.run(init)
        
            # Do the training loop
            for epoch in range(num_epochs):
                epoch_cost = 0.
                epoch_cost_test = 0.
                global_step = epoch
                # Defines a cost related to an epoch
                num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
                minibatch_size_t = int(m_t / num_minibatches)
                #seed = seed + 1
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
                minibatches_t = random_mini_batches(X_test, Y_test, minibatch_size_t)

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                
                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                    #_ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    _  = sess.run(optimizer, feed_dict={X: minibatch_X, Y: minibatch_Y})
                    
                epoch_cost = sess.run(cost, feed_dict={X: X_train, Y: Y_train})
                epoch_cost_test = sess.run(cost, feed_dict={X: X_test, Y: Y_test})
                    

                #parameters = sess.run(parameters)
                if epoch % 100 == 0:
                    correct_prediction = tf.equal(tf.to_float(tf.sigmoid(Z3) >= 0.5 ), Y)
                    accuracy_train_test = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    acc_train.append(accuracy_train_test.eval({X: X_train, Y: Y_train}))
                    acc_test.append(accuracy_train_test.eval({X: X_test, Y: Y_test}))

                # Print the cost every epoch
                if print_cost == True and epoch % 100 == 0:
                    f.write("<Train>Cost after epoch %i: %f\n" % (epoch, epoch_cost))
                    f.write("<Test>Cost after epoch %i: %f\n" % (epoch, epoch_cost_test))
                    #print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 100 == 0:
                    costs.append(epoch_cost)
                    costs_test.append(epoch_cost_test)
                #save model/weights
                if epoch % 1000 == 0:
                    saver.save(sess, model_save_path_name, global_step = epoch, write_meta_graph = True)
                
            np.save('costs_epoch_train_450K_0.5', costs)
            np.save('costs_epoch_test_450K_0.5', costs_test)
            np.save("accuracy_train_450K_0.5", acc_train)
            np.save("accuracy_test_450K_0.5", acc_test)
            '''
            # plot the cost
            fig = plt.figure(figsize=(9,6))
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('epoch')
            plt.title("Learning rate =" + str(learning_rate))
            #plt.show()
            fig.savefig("train_cost_epoch.png")
            '''

            # lets save the parameters in a variable
            parameters = sess.run(parameters)
            #print ("Parameters have been trained!")

            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.to_float(tf.sigmoid(Z3) >= 0.5), Y)

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            #print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            #print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
            f.write("\nTrain Accuracy:")
            f.write(str(accuracy.eval({X: X_train, Y: Y_train})))
            f.write("\n")
            f.write("\nTest Accuracy:")
            f.write(str(accuracy.eval({X: X_test, Y: Y_test})))
            f.write("\n")
        
            return parameters
        

if __name__ == '__main__':
    with tf.device('/gpu:1'):

        #data_path = "/cloud/liubiao/methylation_project/promoter_cancer/tfrecords/data.tfrecords"

        #learning_rate = 0.00002
        learning_rate_base = 0.00002
        decay_steps = 2000
        decay_rate = 0.85
        regularaztion_rate = 0.001
        #moving_average_decay = 0.99
        num_epochs = 5000
        minibatch_size = 64

        model_save_path = "/cloud/liubiao/methylation_project/TCGA_GEO_data/final_marker_train/logs_and_models/train_450K_new/"
        model_name = "model_0.5"
        model_save_path_name = model_save_path + model_name

        out_file = model_save_path + 'train_output_0.5.txt'
        f = open(out_file, 'a')

        
        parameters = model(X_train, Y_train, X_test, Y_test, learning_rate_base, num_epochs, minibatch_size, regularaztion_rate, model_save_path_name)

        f.close()




