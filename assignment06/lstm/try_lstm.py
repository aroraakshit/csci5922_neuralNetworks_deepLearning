from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random

print (tf.VERSION)

# ====================
#  TOY DATA GENERATOR
# ====================

class ToySequenceData(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])

    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, n_samples=1000, max_seq_len=2):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(max_seq_len)
            xor = lambda x: 1 if (x % 2 == 1) else 0
            sequences = np.random.choice([0, 1], size=[max_seq_len], replace=True)
            counts = np.count_nonzero(sequences == 1, axis=0)
            y =xor(counts)
            s = np.expand_dims(sequences, axis=2)
            self.data.append(s)
            if(y == 0):
                self.labels.append([0., 1.])
            else:
                self.labels.append([1., 0.])
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen

		

number_of_models = 8
replications = 10
training_steps = 1500
acc_test = np.zeros([number_of_models,replications,training_steps])
mean_acc_test = np.zeros([number_of_models,training_steps])
acc_training = np.zeros([number_of_models,replications,training_steps])
mean_acc_training = np.zeros([number_of_models,training_steps])
for n_model,s_l,n_h in [[0,2,5],[1,10,5],[2,25,5],[3,50,5],[4,2,25],[5,10,25],[6,25,25],[7,50,25]]:
    for i in range(replications):
        print("\nModel: "+str(n_model)+" : seq len = "+str(s_l)+", num_hidden = "+str(n_h)+", replication: "+str(i)+"\n")
        # ==========
        #   MODEL
        # ==========

        # Parameters
        learning_rate = 1.0
        batch_size = 128
        display_step = 100
        tf.reset_default_graph()
        # Network Parameters
        seq_max_len = s_l # Sequence max length
        n_hidden = n_h # hidden layer num of features
        n_classes = 2 # number of output classes

        trainset = ToySequenceData(n_samples=10000, max_seq_len=seq_max_len)
        testset = ToySequenceData(n_samples=10000, max_seq_len=seq_max_len)

        # tf Graph input
        x = tf.placeholder("float", [None, seq_max_len, 1])
        y = tf.placeholder("float", [None, n_classes])
        # A placeholder for indicating each sequence length
        seqlen = tf.placeholder(tf.int32, [None])

        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        
        def dynamicRNN(x, seqlen, weights, biases):

            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, n_steps, n_input)
            # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

            # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            x = tf.unstack(x, seq_max_len, 1)

            # Define a lstm cell with tensorflow
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

            # Get lstm cell output, providing 'sequence_length' will perform dynamic
            # calculation.
            outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32,sequence_length=seqlen)

            # When performing dynamic calculation, we must retrieve the last
            # dynamically computed output, i.e., if a sequence length is 10, we need
            # to retrieve the 10th output.
            # However TensorFlow doesn't support advanced indexing yet, so we build
            # a custom op that for each sample in batch size, get its length and
            # get the corresponding relevant output.

            # 'outputs' is a list of output at every timestep, we pack them in a Tensor
            # and change back dimension to [batch_size, n_step, n_input]
            outputs = tf.stack(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2])

            # Hack to build the indexing and retrieve the right output.
            batch_size = tf.shape(outputs)[0]
            # Start indices for each sample
            index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
            # Indexing
            outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

            # Linear activation, using outputs computed above
            return tf.matmul(outputs, weights['out']) + biases['out']
        
        pred = dynamicRNN(x, seqlen, weights, biases)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        
        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            for step in range(1, training_steps+1):
                batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,seqlen: batch_seqlen})
                test_data = testset.data
                test_label = testset.labels
                test_seqlen = testset.seqlen
                acc_test[n_model][i][step-1] = sess.run(accuracy, feed_dict={x: test_data, y: test_label,seqlen: test_seqlen})
                acc_training[n_model][i][step-1] = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,seqlen: batch_seqlen})
                if step % display_step == 0 or step == 1:
                    # Calculate batch accuracy & loss
                    acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y,seqlen: batch_seqlen})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))

            print("Optimization Finished!")

            # Calculate accuracy
            test_data = testset.data
            test_label = testset.labels
            test_seqlen = testset.seqlen
            print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={x: test_data, y: test_label,seqlen: test_seqlen}))
				
    for i in range(training_steps):
        for j in range(10):
            mean_acc_test[n_model][i] += acc_test[n_model][j][i]
            mean_acc_training[n_model][i] += acc_training[n_model][j][i]
        mean_acc_test[n_model][i] /= 10
        mean_acc_training[n_model][i] /= 10
		
np.save(file="mean_acc_test.npy",arr=mean_acc_test)
np.save(file="mean_acc_training.npy", arr = mean_acc_training)
np.save(file="acc_test.npy",arr=acc_test)
np.save(file="acc_training.npy",arr=acc_training)