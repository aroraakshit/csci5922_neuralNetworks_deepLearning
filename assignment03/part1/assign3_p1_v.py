#Assignment 3 Part 1c)
#variant of previous code to try out different architectures with two hidden layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import math
points_training = np.genfromtxt('train_data.txt',delimiter=",", skip_header = 1).astype(np.float32)
points_testing = np.genfromtxt('test_data.txt',delimiter=",", skip_header = 1).astype(np.float32)
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

hm_epochs = 250
N = 100
i = -1.0
ACC = np.ndarray(shape=(hm_epochs,4*4), dtype=float, order='F')
for H in [2,5,10,20]:
    for H2 in [2,5,10,20]:
        i = i+1
        print "number of hidden units: " + str(H) + "," + str(H2)
        n = [5,H,H2,1]
        sess = tf.InteractiveSession()

        x = tf.placeholder(tf.float32, shape=[None,n[0]], name="x")
        y = tf.placeholder(tf.float32, shape=[None, n[3]], name="y")

        hidden_1_layer = {'weights':tf.Variable(tf.random_normal([n[0],n[1]])),'biases':tf.Variable(tf.random_normal([n[1]]))}
        hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n[1],n[2]])),'biases':tf.Variable(tf.random_normal([n[2]]))}
        output_layer = {'weights':tf.Variable(tf.random_normal([n[2],n[3]])),'biases':tf.Variable(tf.random_normal([n[3]]))}

        def neural_network_model(data):
            l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
            l1 = tf.nn.relu(l1)
            l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
            l2 = tf.nn.relu(l2)
            output = tf.matmul(l2,output_layer['weights']) + output_layer['biases']
            return output

        def calc_accuracy(testing_data_x,testing_data_y,W_1,b_1,W_2,b_2,W_3,b_3):
            l1 = np.matmul(testing_data_x, W_1) + b_1
            l1 = l1 * (l1 > 0) #relu imlementation in numpy
            l2 = np.matmul(l1, W_2) + b_2
            l2 = l2 * (l2 > 0) #relu imlementation in numpy
            output = np.matmul(l2,W_3)+b_3
            output = np.transpose(output)
            output[output < 0.5] = 0.0
            output[output >= 0.5] = 1.0
            accuracy = 100 - (np.sum(np.abs(output - testing_data_y)) / len(points_testing) * 100)
            return accuracy

        def train_neural_network():
            prediction = neural_network_model(x)
            cost = tf.reduce_mean( tf.losses.mean_squared_error(predictions=prediction, labels=y) )
            optimizer = tf.train.AdamOptimizer().minimize(cost)
            W_1 = 0.0
            W_2 = 0.0
            W_3 = 0.0
            b_3 = 0.0
            b_1 = 0.0
            b_2 = 0.0
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(hm_epochs):
                    epoch_loss = 0
                    for _ in range(int(len(points_training) / N)):
                        epoch_x = points_training[_*N:(_+1)*N,2:7]
                        epoch_y = points_training[_*N:(_+1)*N,7].reshape(100,1)
                        _, c, W_1, b_1, W_2, b_2,W_3,b_3 = sess.run([optimizer, cost, hidden_1_layer['weights'], hidden_1_layer['biases'],hidden_2_layer['weights'], hidden_2_layer['biases'], output_layer['weights'], output_layer['biases']], feed_dict={x: epoch_x, y: epoch_y})
                        epoch_loss += c
                    ACC[epoch][i] = calc_accuracy(points_testing[:,2:7], points_testing[:,7],W_1,b_1,W_2,b_2,W_3,b_3)
                    print('Accuracy after epoch!: ',ACC[epoch][i])
                    print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            print('FINAL Accuracy: ',calc_accuracy(points_testing[:,2:7], points_testing[:,7],W_1,b_1,W_2,b_2,W_3,b_3))

        train_neural_network()

#visualizing results
plt.title("Percentage Performance on test set")
plt.xlabel('Number of Epochs')
plt.ylabel('Performance (% examples classified correctly)')
plt.plot(ACC[:,0:8])
plt.legend(['5-2-2-1','5-2-5-1','5-2-10-1','5-2-20-1','5-5-2-1','5-5-5-1','5-5-10-1','5-5-20-1'],loc='best',fancybox=True, framealpha=0.5)
plt.show()

plt2.title("Percentage Performance on test set")
plt2.xlabel('Number of Epochs')
plt2.ylabel('Performance (% examples classified correctly)')
plt2.plot(ACC[:,8:16])
plt2.legend(['5-10-2-1','5-10-5-1','5-10-10-1','5-10-20-1','5-20-2-1','5-20-5-1','5-20-10-1','5-20-20-1'],loc='best',fancybox=True, framealpha=0.5)
plt2.show()
A = ACC
for i in range(16):
    print "for architecture: " + str(i) + " the maximum accuracy is: " + str(np.amax(A[:,i])) + " at epoch: " + str(np.argmax(A[:,i])) + "\n"
#writer = tf.summary.FileWriter("/tmp/a3p1/1") #for the purpose of tensorboard
#writer.add_graph(sess.graph) #for visualizing graph in tensorboard
