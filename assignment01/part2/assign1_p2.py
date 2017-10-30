#initially I thought I should use linear_model.SGDClassifier of sklearn. but implementing all natively now.

#Stochastic Gradient Descent implementation. Used for online / incremental updating of weights and bias. We look at one example at a time.
#it is again, not a vectorized implementation => only one CPU used at a time.

from numpy import *
import matplotlib.pyplot as plt

NUMITR = 1000
NUMEXAMPLES = 100
ERROR = ndarray(shape=(NUMITR,NUMEXAMPLES), dtype=float)
W1 = ndarray(shape=(NUMITR,NUMEXAMPLES), dtype=float)
W2 = ndarray(shape=(NUMITR,NUMEXAMPLES), dtype=float)
B = ndarray(shape=(NUMITR,NUMEXAMPLES), dtype=float)

def error(w1, w2, b, points):
    Error = 0
    x1 = points[0]
    x2 = points[1]
    y = points[2]
    Error += (y - (w1 * x1 + w2 * x2 + b)) ** 2
    return Error

def step_gradient(b_current, w1_current, w2_current, points, learningRate):
    b_gradient = 0
    w1_gradient = 0
    w2_gradient = 0
    x1 = points[0]
    x2 = points[1]
    y = points[2]
    b_gradient += - (y - ((w1_current * x1) + (w2_current * x2) + b_current))
    w1_gradient += - x1 * (y - ((w1_current * x1) + (w2_current * x2) + b_current))
    w2_gradient += - x2 * (y - ((w1_current * x1) + (w2_current * x2) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_w1 = w1_current - (learningRate * w1_gradient)
    new_w2 = w2_current - (learningRate * w2_gradient)
    return [new_b, new_w1, new_w2]

def sgd_runner(points, starting_w1, starting_w2, starting_b, learning_rate, num_iterations):
    b = starting_b
    w1 = starting_w1
    w2 = starting_w2
    for l in range(num_iterations):
        random.shuffle(points) #important for stochastic gradient Descent
        for i in range(len(points)):
            ERROR[l,i] = error(w1,w2,b,points[i])
            [b, w1, w2] = step_gradient(b, w1, w2, points[i], learning_rate)
            W1[l,i] = w1
            W2[l,i] = w2
            B[l,i] = b
    w1 = mean(W1[num_iterations - 1], axis=0)
    w2 = mean(W2[num_iterations - 1], axis=0)
    b = mean(B[num_iterations - 1], axis=0)
    return [w1, w2, b]

def run():

    #Step 1 - collect our data in the form of ndarray
    points = genfromtxt('assign1_data.txt')
    points = delete(points, (0), axis=0)
    random.shuffle(points) #important for stochastic gradient Descent

    #Step 2 - define our hyperparameters
    learning_rate = 0.0001
    #y = w1 * x1 + w2 * x2 + b
    initial_w1 = 0
    initial_w2 = 0
    initial_b = 0
    num_iterations = NUMITR

    #Step 3 - train our model
    print 'starting stochastic gradient descent at w1 = {0}, w2 = {1}, b = {2}, error w.r.t. first example = {3}'.format(initial_w1,initial_w2, initial_b,error(initial_w1, initial_w2, initial_b, points[0]))
    print "Running..."
    [w1, w2, b] = sgd_runner(points, initial_w1, initial_w2, initial_b, learning_rate, num_iterations)
    print "After {0} iterations w1 = {1}, w2 = {2}, b = {3}, error w.r.t the last example = {4}".format(num_iterations, w1, w2, b, error(w1, w2, b, points[99]))
    plt.plot(ERROR.flatten()) #working!
    #ax.scatter(W1,W2,ERROR) #working!
    #not able to get error surface properly though.
    plt.show()

if __name__ == '__main__':
    run()
