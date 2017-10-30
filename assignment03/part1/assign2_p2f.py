#Assignment 2 - Question 2     Akshit Arora
#Implementing backpropagation (in mini-batch setting). Corrected for obtaining plot like assignment 3's part 1.

import numpy as np
import matplotlib.pyplot as plt
#import time #used while debugging

H = 5 #number of hidden units
N = 100 #size of mini batch
NUMITR = 50 #number of Epochs
LR = 0.001 #learning rate

ERROR_training = np.zeros(5,dtype=np.float)
ERROR_test = np.zeros(5,dtype = np.float)
ACC = np.ndarray(shape=(NUMITR,5), dtype=float, order='F')

def sigmoid(x,deriv=False):
    if(deriv==True):
        temp = np.multiply(sigmoid(x),sigmoid(1-x))
        return temp
    temp = 1/(1+np.exp(-x))
    return temp

def error(y,t): #returns number of misclassified examples. Used for calculating accuracy.
    return abs(t-y)

def current_value(X,W_1,b_1,W_2,b_2): #returns the current output of given weights. Used for calculating accuracy.
    Z_1 = np.matmul(W_1,np.transpose(np.asmatrix(X))) + b_1
    A_1 = sigmoid(Z_1)
    Z_2 = sigmoid(np.matmul(W_2, A_1) + b_2)
    print Z_2
    A_2 = Z_2
    #A_2 = sigmoid(Z_2)
    A_2[A_2 >= 0.5] = 1.0   #thresholding applied.
    A_2[A_2 < 0.0] = 0.0
    print A_2
    return A_2

def train(W_1,b_1,W_2,b_2,points_training,points_testing,hl):
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
                dZ_2 = abs(A_2 - Y)
                #print dZ_2
                #backpropagation
                #accumulating gradients
                dW_2 += dZ_2 * np.transpose(A_1)
                db_2 += dZ_2
                dZ_1 = np.multiply(np.transpose(W_2) * dZ_2, sigmoid(Z_1,deriv=True),dtype=float)
                dW_1 += dZ_1 * np.transpose(X)
                db_1 += dZ_1
            #updating weights
            W_1 = W_1 - LR * 1.0/N * dW_1
            b_1 = b_1 - LR * 1.0/N * db_1
            W_2 = W_2 - LR * 1.0/N * dW_2
            b_2 = b_2 - LR * 1.0/N * db_2
        ACC[j][hl] = 100 - np.sum(error(current_value(points_testing[:,2:7],W_1,b_1,W_2,b_2),points_testing[:,7])) / len(points_testing) * 100
        print "accuracy for epoch: " + str(j) + " is: " + str(ACC[j][hl])
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

    i = -1.0
    for H in [1,2,5,10,20]:
        i = i + 1.0
        #initializing weights and bias
        print "Initializing weights and bias hidden unit:" + str(H)
        n = [5,H,1] #[number of input features, number of hidden units, number of output units]
        W_1 = np.random.randn(n[1],n[0]) * 0.01
        b_1 = np.zeros((n[1],1))
        W_2 = np.random.randn(n[2],n[1]) * 0.01
        b_2 = np.zeros((n[2],1))

        #training the neuron
        print "Training the neuron hidden unit:" + str(H)
        [W_1,b_1,W_2,b_2] = train(W_1,b_1,W_2,b_2,points_training,points_testing,i)

        # if i<=3:
        #     j = i-1
        # elif i==5:
        #     j = 2
        # elif i==10:
        #     j = 3
        # else:
        #     i = 4
        #ERROR_training[j] = np.sum(error(current_value(points_training[:,2:7],W_1,b_1,W_2,b_2),points_training[:,7])) / len(points_training) * 100
        #j = 0

    #visualizing results
    plt.title("Percentage Performance on test set")
    plt.xlabel('Number of Epochs')
    plt.ylabel('Performance (% examples classified correctly)')
    plt.plot(ACC)
    plt.legend(['1','2','5','10','20'],loc='best',fancybox=True, framealpha=0.5)
    plt.show()

if __name__ == '__main__':
    run()
