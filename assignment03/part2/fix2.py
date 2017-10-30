
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
from random import shuffle
import sklearn.utils as skutils


# In[2]:

n = [36,15,24] #[input units, hl1, output units]
num_splits = 20 #number of splits
hm_epochs = 1500 #number of epochs for training per split!
ACC = np.ndarray(shape=(hm_epochs,num_splits), dtype=float, order='F') #just for collecting accuracy for one split
test_accuracy = np.ndarray(shape=(num_splits), dtype=float, order='F') #declare array for test set accuracy storage for all splits.


# In[3]:

def split_data(dataset, train_size):
    X = dataset["X"]
    y = dataset["y"]
    X_shuff, y_shuff = skutils.shuffle(X, y)

    num_people = dataset["names"].shape[0]
    num_relatioons = dataset["relationships"].shape[0]

    return {
        "training" : {
            "X"    :   X_shuff[:train_size, :],
            "y"    :   y_shuff[:train_size]
        },
        "test" : {
            "X"     :   X_shuff[train_size:, :],
            "y"     :   y_shuff[train_size:]
        }
    }


# In[4]:

def read_family_tree_data():

    # Get names and relationships
    names = []
    relationships = []
    raw_file_data = []
    with open("relations.txt", "r") as infile:
        for line in infile:
            raw_file_data.append(line)
            split_line = line.split()
            names.append(split_line[0])
            relationships.append(split_line[1])
            names.append(split_line[2])

    # Make names/relationships lists unique
    names = np.unique(names)
    relationships = np.unique(relationships)

    # One-hot encoding
    X, y = [], []
    for relationship in raw_file_data:
        split_relation = relationship.split()

        x_name = split_relation[0]
        x_relation = split_relation[1]
        y_name = split_relation[2]

        X_one_hot_names = np.zeros(len(names))
        X_one_hot_names[np.where(names == x_name)] = 1.0

        X_one_hot_relations = np.zeros(len(relationships))
        X_one_hot_relations[np.where(relationships == x_relation)] = 1.0

        X_data = np.r_[X_one_hot_names, X_one_hot_relations]

        Y_one_hot_names = np.zeros(len(names))
        Y_one_hot_names[np.where(names == y_name)] = 1.0

        X.append(X_data)
        y.append(Y_one_hot_names)

    return {"X" : np.array(X), "y" : np.array(y), "names" : names, "relationships" : relationships}


# In[5]:

def calc_accuracy(output, testing_data_y):
    temp=0.0
    for i in range(len(output)):
        ax = output[i]
        for j in range(len(output[i])):
            if(j==np.argmax(output[i])):
                ax[j] == 1.0
            else:
                ax[j] = 0.0
        if (np.array_equal(ax,testing_data_y[i])):
            temp+=1.0
    accuracy = temp / len(output) * 100.0
    return accuracy


# In[6]:

def neural_network_model(data):
    with tf.variable_scope("input"):
        global W1
        W1 = tf.get_variable("weights_input", shape=[n[0], n[1]], initializer=tf.contrib.layers.xavier_initializer())

        global b1
        b1 = tf.get_variable("bias_input", shape=[n[1]], initializer=tf.contrib.layers.xavier_initializer())

        global z1
        z1 = tf.nn.sigmoid(tf.matmul(data, W1) + b1, name="activation_input")

    global a1
    a1 = tf.nn.sigmoid(z1)

    with tf.variable_scope("output"):
        W2 = tf.get_variable("weights_output", shape=[n[1], n[2]], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("bias_output", shape=[n[2]], initializer=tf.contrib.layers.xavier_initializer())
        z2 = tf.nn.sigmoid(tf.matmul(a1, W2) + b2, name="activation_out")

    output = tf.nn.sigmoid(z2)

    return output


# In[11]:

def train_neural_network(training_data, test_data, split):
    train_x = training_data["X"]
    train_y = training_data["y"]
    test_x = test_data["X"]
    test_y = test_data["y"]

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.square(prediction - train_y))
    diff = prediction - train_y
    optimizer = tf.train.AdamOptimizer()
    grad_step = optimizer.compute_gradients(cost)
    optimizer = optimizer.minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            _, epoch_loss = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})
            ACC[epoch][split] = calc_accuracy(prediction.eval(feed_dict={x: test_x}), test_y)
            print('Epoch', epoch, 'completed out of', hm_epochs,'loss:', epoch_loss,'accuracy:', ACC[epoch][split])
    return ACC[hm_epochs-1][split]


# In[12]:

tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, n[0]], name="x")
y = tf.placeholder(tf.float32, shape=[None, 24], name="y")

if __name__ == "__main__":

    dataset = read_family_tree_data() #list

    #loop of 20 iterations
    for split in range(num_splits):

        data = split_data(dataset, train_size=89)
        training_data = data["training"]
        test_data = data["test"]

        #train the network
        train_neural_network(training_data, test_data, split)

    #print test set accuracy mean and standard deviation
    print test_accuracy


# In[ ]:




# In[ ]:
