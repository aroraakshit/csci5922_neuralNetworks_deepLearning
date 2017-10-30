#Assignment 1 - Question 2      Akshit Arora
#Implementing batch gradient descent for linear regression using numpy.
#Given the error, determine the setting (learningRate, batch/minibatch/online) with minimal sweeps.

#Additional notes:
#Will try to plot using matplotlib. Was able to get error vs number of iterations plot but not able to get the error surface and the gradient descent path.
#not a vectorized implementation!

from numpy import *
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

NUMITR = 100000         #maximum number of iterations possible
ERR_ALLOWED = 0.039409   #Error threshold.
ERROR = arange(NUMITR, dtype=float)
W1 = arange(NUMITR, dtype=float)
W2 = arange(NUMITR, dtype=float)

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

def gradient_descent_runner(points, starting_w1, starting_w2, starting_b, learning_rate, num_iterations):
    b = starting_b
    w1 = starting_w1
    w2 = starting_w2
    for i in range(num_iterations):
        [b, w1, w2] = step_gradient(b, w1, w2, array(points), learning_rate)
        ERROR[i] = error(w1,w2,b,points)
        W1[i] = w1
        W2[i] = w2
        if ( error(w1, w2, b, points) <= ERR_ALLOWED):
            print "number of epochs recorded!: " + str(i)
            return [w1, w2, b]
    print "MAXIMUM NUMBER OF ITERATIONS (" + str(NUMITR) + ") EXCEEDED!"
    return [w1, w2, b]

def run():

    #Step 1 - collect our data in the form of ndarray
    points = genfromtxt('assign1_data.txt')
    points = delete(points, (0), axis=0)

    #Step 2 - define our hyperparameters
    learning_rate = 0.1
    #y = w1 * x1 + w2 * x2 + b
    initial_w1 = 0
    initial_w2 = 0
    initial_b = 0
    num_iterations = NUMITR

    #Step 3 - train our model
    print 'starting batch gradient descent at w1 = {0}, w2 = {1}, b = {2}, error = {3}'.format(initial_w1,initial_w2, initial_b,error(initial_w1, initial_w2, initial_b, points))
    print "Running..."
    [w1, w2, b] = gradient_descent_runner(points, initial_w1, initial_w2, initial_b, learning_rate, num_iterations)
    print "After {0} iterations w1 = {1}, w2 = {2}, b = {3}, error w.r.t. whole data set (for comparison) = {4}".format(num_iterations, w1, w2, b, error(w1, w2, b, points))
    plt.plot(ERROR) #working!
    print "Total iterations (number of times gradient_step was executed):" + str(len(ERROR) * len(points))
    #print "Total epochs:" + str(NUMITR)
    #ax.scatter(W1,W2,ERROR) #working!
    #not able to get error surface properly though.
    plt.show()

if __name__ == '__main__':
    run()
