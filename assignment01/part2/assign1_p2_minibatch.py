#Assignment 1 - Question 2      Akshit Arora
#Implementing mini-batch gradient descent for linear regression using numpy.
#Given the sweeps (10,000), determine the setting (learningRate, batch/minibatch/online) with least error.

#Additional notes:
#Used for online / incremental updating of weights and bias. We look at one example at a time.
#it is again, not a vectorized implementation => only one CPU used at a time.

from numpy import *
import matplotlib.pyplot as plt

NUMITR = 10000      #number of iterations
B_SIZE = 5          #batch size
NUMEXAMPLES = 100   #number of rows in given data
ERROR = ndarray(shape=(NUMITR,NUMEXAMPLES / B_SIZE), dtype=float)
W1 = ndarray(shape=(NUMITR,NUMEXAMPLES / B_SIZE), dtype=float)
W2 = ndarray(shape=(NUMITR,NUMEXAMPLES / B_SIZE), dtype=float)
B = ndarray(shape=(NUMITR,NUMEXAMPLES / B_SIZE), dtype=float)

def error(w1, w2, b, points):
    totalError = 0
    for i in range(0, len(points)):
        x1 = points[i, 0]
        x2 = points[i, 1]
        y = points[i, 2]
        totalError += (y - (w1 * x1 + w2 * x2 + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, w1_current, w2_current, points, learningRate):
    b_gradient = 0
    w1_gradient = 0
    w2_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x1 = points[i, 0]
        x2 = points[i, 1]
        y = points[i, 2]
        b_gradient += -(2/N) * (y - ((w1_current * x1) + (w2_current * x2) + b_current))
        w1_gradient += -(2/N) * x1 * (y - ((w1_current * x1) + (w2_current * x2) + b_current))
        w2_gradient += -(2/N) * x2 * (y - ((w1_current * x1) + (w2_current * x2) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_w1 = w1_current - (learningRate * w1_gradient)
    new_w2 = w2_current - (learningRate * w2_gradient)
    return [new_b, new_w1, new_w2]

def mini_batch_gradient_descent_runner(points, starting_w1, starting_w2, starting_b, learning_rate, num_iterations):
    b = starting_b
    w1 = starting_w1
    w2 = starting_w2
    for l in range(num_iterations):
        random.shuffle(points)
        for i in range(len(points)/B_SIZE):
            ERROR[l,i] = error(w1,w2,b,points[i*B_SIZE:(i+1)*B_SIZE])
            [b, w1, w2] = step_gradient(b, w1, w2, points[i*B_SIZE:(i+1)*B_SIZE], learning_rate)
            W1[l,i] = w1
            W2[l,i] = w2
            B[l,i] = b
    #w1 = mean(W1[num_iterations/B_SIZE - 1], axis=0)
    #w2 = mean(W2[num_iterations/B_SIZE - 1], axis=0)
    #b = mean(B[num_iterations/B_SIZE - 1], axis=0)
    return [w1, w2, b]

def run():

    #Step 1 - collect our data in the form of ndarray
    points = genfromtxt('assign1_data.txt')
    points = delete(points, (0), axis=0)    #the first row is actually blank; so removing it

    #Step 2 - define our hyperparameters
    learning_rate = 0.1
    #y = w1 * x1 + w2 * x2 + b
    initial_w1 = 0
    initial_w2 = 0
    initial_b = 0
    num_iterations = NUMITR

    #Step 3 - train our model
    print 'starting mini-batch gradient descent at w1 = {0}, w2 = {1}, b = {2}, batch size = {3}, error w.r.t. first batch= {4}'.format(initial_w1,initial_w2, initial_b, B_SIZE, error(initial_w1, initial_w2, initial_b, points[0:10]))
    print "Running..."
    [w1, w2, b] = mini_batch_gradient_descent_runner(points, initial_w1, initial_w2, initial_b, learning_rate, num_iterations)
    print "After {0} iterations, w1 = {1}, w2 = {2}, b = {3}, error for the whole dataset (calc for comparison) = {4}".format(num_iterations, w1, w2, b, error(w1,w2,b,points))
    plt.plot(ERROR.flatten()) #working!
    print "Total iterations (number of times gradient_step was executed):" + str(len(ERROR.flatten()) * B_SIZE)
    print "Total epochs:" + str(NUMITR)
    #ax.scatter(W1,W2,ERROR) #working!
    #not able to get error surface properly though.
    plt.show()

if __name__ == '__main__':
    run()
