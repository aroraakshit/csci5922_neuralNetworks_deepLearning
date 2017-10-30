#Assignment 2 - Question 1      Akshit Arora
#Implementing perceptron learning rule. Report the training and test set performance in terms of % classified correctly!

from numpy import *
from math import *
import matplotlib.pyplot as plt
from sklearn import metrics

LR = 1.0 #starting learning rate.
time_decay_LR_flag = True #flag to enable/disable time decay

def time_decay_LR(i): #time decay learning rate
    if time_decay_LR_flag == True:
        return (LR/i)
    else:
        return (LR)

def predict(W,X):
    var = dot(X,W[1:]) + W[0]
    if var > 0.0:
        return 1.0
    else:
        return 0.0

def error(points,W):
    check = 0.0
    for i in range (len(points)):
        X = points[i, 2:7]
        y = points[i, 7]
        check += abs(- y + predict(W,X))
    return (check / len(points) * 100)

def step_gradient(W_current, points):
    new_W = zeros(6)
    for i in range(len(points)):
        b = W_current[0]
        W = W_current[1:]
        X = points[i, 2:7]
        desired = points[i, 7]
        y = predict(W_current,X)
        W_current[0] = W_current[0] + (desired - y) * time_decay_LR(i+1)
        W_current[1:] = W_current[1:] + (desired - y) * X * time_decay_LR(i+1)
    new_W = W_current
    return new_W

def perceptron_runner(points, points_testing, starting_W, num_iterations, ERROR_test,ERROR_training):
    W = starting_W
    for i in range(num_iterations):
        W = step_gradient(W, array(points))
        ERROR_test[i] = error(points_testing,W)
        ERROR_training[i] = error(points,W)
    return W, ERROR_training, ERROR_test

def run():
    #Step 1 - collect our data in the form of ndarray
    print "Collecting input"
    points_training = genfromtxt('train_data.txt',delimiter=",", skip_header = 1)
    points_testing = genfromtxt('test_data.txt',delimiter=",", skip_header = 1)

    print "Normalizing input data"
    #Doing input normalization for training set
    for i in range(2,7):
        m = mean(points_training[:,i])
        sd = std(points_training[:,i])
        for j in range(len(points_training)):
            points_training[j,i] = (points_training[j,i] - m) / sd

    #Doing input normalization for test set
    for i in range(2,7):
        m = mean(points_testing[:,i])
        sd = std(points_testing[:,i])
        for j in range(len(points_testing)):
            points_testing[j,i] = (points_testing[j,i] - m) / sd

    #Step 2 - define our hyperparameters
    num_iterations = 100
    initial_W = zeros(6) #w[0] is bias
    ERROR_training = arange(num_iterations, dtype=float)
    ERROR_test = arange(num_iterations, dtype=float)

    #Step 3 - train our model
    print 'Training the Perceptron'
    W,ERROR_training, ERROR_test = perceptron_runner(points_training, points_testing, initial_W, num_iterations, ERROR_training, ERROR_test)

    print "Perceptron has been trained! weights: " +str(W[1:]) + " and bias: " + str(W[0])

    #Step 4 - Results
    print "Results:"
    print "Final training set accuracy: " + str(100-error(points_training,W))
    print "Final test set accuracy: " + str(100-error(points_testing,W))
    plt.title('Performance plot for Perceptron (train: green, test: red)')
    plt.xlabel('Total number of Epochs')
    plt.ylabel('Performance (% examples classified correctly)')
    plt.plot(100-ERROR_training,'g')
    plt.plot(100-ERROR_test,'r')
    plt.show()

if __name__ == '__main__':
    run()
