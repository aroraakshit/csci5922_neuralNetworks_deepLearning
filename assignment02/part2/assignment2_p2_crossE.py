#Assignment 2 - Question 2     Akshit Arora
#Implementing backpropagation (in mini-batch setting)

import numpy as np
import matplotlib.pyplot as plt
#import time #used while debugging

H = 5 #number of hidden units
N = 100 #size of mini batch
NUMITR = 20 #number of Epochs
LR = 0.1 #learning rate

ERROR_training = np.zeros(NUMITR,dtype=np.float)
ERROR_test = np.zeros(NUMITR,dtype = np.float)

def sigmoid(x,deriv=False):
    if(deriv==True):
        temp = np.multiply(sigmoid(x),sigmoid(1-x))
        return temp
    temp = 1/(1+np.exp(-x))
    return temp

def error(y,t): #returns number of misclassified examples. Used for calculating accuracy.
    return abs(t-y)

def cross_entropy(y,t):
    return -np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))

def current_value(X,W_1,b_1,W_2,b_2):
    Z_1 = np.matmul(W_1,np.transpose(np.asmatrix(X))) + b_1
    A_1 = sigmoid(Z_1)
    Z_2 = np.matmul(W_2, A_1) + b_2
    A_2 = sigmoid(Z_2)
    A_2[A_2 >= 0.5] = 1.0
    A_2[A_2 < 0.5] = 0.0
    return A_2

def train(W_1,b_1,W_2,b_2,points_training,points_testing):
    for j in xrange(NUMITR):
        num_batches = len(points_training) / N
        for i in xrange(num_batches):
            dW_2 = 0.0
            db_2 = 0.0
            dW_1 = 0.0
            db_1 = 0.0
            for k in xrange(i*N,(i+1)*N):
                X = points_training[k, 2:7]
                Y = points_training[k, 7]
                #feed forward
                Z_1 = np.matmul(W_1,np.transpose(np.asmatrix(X))) + b_1
                A_1 = sigmoid(Z_1)
                Z_2 = np.matmul(W_2, A_1) + b_2
                A_2 = sigmoid(Z_2)
                dZ_2 = A_2 - Y
                #backpropagation
                #accumulating gradients
                dW_2 += dZ_2 * np.transpose(A_1)
                db_2 += dZ_2
                dZ_1 = np.multiply(np.transpose(W_2) * dZ_2, sigmoid(Z_1),dtype=float)
                dW_1 += dZ_1 * np.transpose(X)
                db_1 += dZ_1
            #updating weights
            W_1 = W_1 - LR * 1.0/N * dW_1
            b_1 = b_1 - LR * 1.0/N * db_1
            W_2 = W_2 - LR * 1.0/N * dW_2
            b_2 = b_2 - LR * 1.0/N * db_2
            #print np.sum(error(current_value(points_training[:,2:7],W_1,b_1,W_2,b_2),points_training[:,7])) / len(points_training) * 100
        ERROR_training[j] = np.sum(error(current_value(points_training[:,2:7],W_1,b_1,W_2,b_2),points_training[:,7])) / len(points_training) * 100
        ERROR_test[j] = np.sum(error(current_value(points_testing[:,2:7],W_1,b_1,W_2,b_2),points_testing[:,7])) / len(points_testing) * 100
    return [W_1,b_1,W_2,b_2]

def run():
    #Step 1 - collect our data in the form of ndarray
    print "Collecting input"
    points_training = np.genfromtxt('train_data.txt',delimiter=",", skip_header = 1)
    points_testing = np.genfromtxt('test_data.txt',delimiter=",", skip_header = 1)

    #Doing input normalization for training set
    print "Normalizing input data"
    for i in range(2,7):
        m = np.mean(points_training[:,i])
        sd = np.std(points_training[:,i])
        for j in range(len(points_training)):
            points_training[j,i] = (points_training[j,i] - m) / sd

    #Doing input normalization for test set
    for i in range(2,7):
        m = np.mean(points_testing[:,i])
        sd = np.std(points_testing[:,i])
        for j in range(len(points_testing)):
            points_testing[j,i] = (points_testing[j,i] - m) / sd

    #initializing weights and bias
    print "Initializing weights and bias"
    n = [5,H,1] #[number of input features, number of hidden units, number of output units]
    W_1 = np.random.randn(n[1],n[0]) * 0.01
    b_1 = np.zeros((n[1],1))
    W_2 = np.random.randn(n[2],n[1]) * 0.01
    b_2 = np.zeros((n[2],1))

    #training the neuron
    print "Training the neuron"
    [W_1,b_1,W_2,b_2] = train(W_1,b_1,W_2,b_2,points_training,points_testing)

    #maximum accuracy achieved
    print "Accuracy on training set: " + str(100 - min(ERROR_training))
    print "Accuracy on test set: " + str(100 - min(ERROR_test))

    #visualizing results
    plt.title("Performance on training (green) and test (red) set")
    plt.xlabel('Number of Epochs')
    plt.ylabel('Performance (% examples classified correctly)')
    plt.plot(100-ERROR_training,'g')
    plt.plot(100-ERROR_test,'r')
    plt.show()

if __name__ == '__main__':
    run()
