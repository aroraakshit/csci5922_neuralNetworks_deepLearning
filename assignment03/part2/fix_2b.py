import tensorflow as tf
import numpy as np
from random import shuffle
import sklearn.utils as skutils
import matplotlib.pyplot as plt

hm_epochs = 8000 #number of epochs for training per split!
ACC_tr = np.ndarray(shape=(hm_epochs), dtype=float, order='F') #just for collecting training accuracy
#aliases of weights of the layers outside the layer
w1_1 = np.zeros(shape=[24,6]) #person 1-> 6units
w1_2 = np.zeros(shape=[12,6]) #relationship-> 6units
w2 = np.zeros(shape=[12,12]) #12->12units fully connected!
w3 = np.zeros(shape=[12,6]) #12 units to 6units
w4 = np.zeros(shape=[6,24]) #6 units to output!
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

    return {"X" : np.array(X), "y" : np.array(y), "names" : names, "relationships" : relationships},names,relationships


def neural_network_model(data):
    #with tf.variable_scope("input"):
    with tf.variable_scope("input"):
        #for person1
        global W1_1
        W1_1 = tf.get_variable("weights_input_1", shape=[24,6], initializer=tf.contrib.layers.xavier_initializer())
        global b1_1
        b1_1 = tf.get_variable("bias_input_1", shape=[6], initializer=tf.contrib.layers.xavier_initializer())
        global z1_1
        z1_1 = tf.nn.sigmoid(tf.matmul(data[:,0:24], W1_1) + b1_1, name="activation_input_p1")
        #for relationships
        global W1_2
        W1_2 = tf.get_variable("weights_input_2", shape=[12, 6], initializer=tf.contrib.layers.xavier_initializer())
        global b1_2
        b1_2 = tf.get_variable("bias_input_2", shape=[6], initializer=tf.contrib.layers.xavier_initializer())
        global z1_2
        z1_2 = tf.nn.sigmoid(tf.matmul(data[:,24:36], W1_2) + b1_2, name="activation_input_rel")

    global a1
    #don't know how to combine output of two separate layers and feed it into the next layer. No append function for tensorflow exists!
    a1 = tf.concat([z1_1,z1_2],1)

    with tf.variable_scope("intermediate", reuse=None):
        global W2
        W2 = tf.get_variable("weights_1", shape=[12, 12], initializer=tf.contrib.layers.xavier_initializer())
        global b2
        b2 = tf.get_variable("bias_1", shape=[12], initializer=tf.contrib.layers.xavier_initializer())
        global z2
        z2 = tf.nn.sigmoid(tf.matmul(a1, W2) + b2, name="six_6To12")

        global a2
        a2 = z2

        global W3
        W3 = tf.get_variable("weights_2", shape=[12, 6], initializer=tf.contrib.layers.xavier_initializer())
        global b3
        b3 = tf.get_variable("bias_2", shape=[6], initializer=tf.contrib.layers.xavier_initializer())
        global z3
        z3 = tf.nn.sigmoid(tf.matmul(a2, W3) + b3, name="twelveTo6")

        global a3
        a3=z3

    with tf.variable_scope("output"):
        global W4
        W4 = tf.get_variable("weights_output", shape=[6, 24], initializer=tf.contrib.layers.xavier_initializer())
        global b4
        b4 = tf.get_variable("bias_output", shape=[24], initializer=tf.contrib.layers.xavier_initializer())
        global z4
        z4 = tf.nn.sigmoid(tf.matmul(a3, W4) + b4, name="activation_out")

    output = z4

    return output


# In[11]:

def train_neural_network(training_data):
    train_x = training_data["X"]
    train_y = training_data["y"]

    prediction = neural_network_model(x)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,1),tf.argmax(train_y,1)),tf.float32))
    cost = tf.reduce_mean(tf.square(prediction - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    optimizer = optimizer.minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            _, epoch_loss,ACC_tr[epoch],w1,w2 = sess.run([optimizer, cost, acc, W1_1, W1_2], feed_dict={x: train_x, y: train_y})
            print('Epoch', epoch, 'completed out of', hm_epochs,'loss:', epoch_loss, 'train accuracy',ACC_tr[epoch])
    return w1,w2

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

if __name__ == "__main__":
    dataset,names,relationships = read_family_tree_data()
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None, 36], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 24], name="y")
    data = split_data(dataset, train_size=104)
    training_data = data["training"]
    w1_1,w1_2 = train_neural_network(training_data)
    print names
    print relationships
    hinton(w1_1)
    plt.show()
    hinton(w1_2)
    plt.show()
