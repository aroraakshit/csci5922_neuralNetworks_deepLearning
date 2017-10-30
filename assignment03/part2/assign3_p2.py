import tensorflow as tf
import numpy as np
from random import shuffle

n = [36,10,24] #[input units, hl1, output units]
x = tf.placeholder(tf.float32, shape=[None,n[0]], name="x")
y = tf.placeholder(tf.float32, shape=[None, n[2]], name="y")
num_splits = 20 #number of splits
hm_epochs = 1500 #number of epochs for training per split!
ACC = np.ndarray(shape=(hm_epochs,num_splits), dtype=float, order='F') #just for collecting accuracy for one split
test_accuracy = np.ndarray(shape=(num_splits), dtype=float, order='F') #declare array for test set accuracy storage for all splits.
hidden_1_layer = {'weights':tf.Variable(tf.random_normal([n[0],n[1]])),'biases':tf.Variable(tf.random_normal([n[1]]))}
output_layer = {'weights':tf.Variable(tf.random_normal([n[1],n[2]])),'biases':tf.Variable(tf.random_normal([n[2]]))}
sess = tf.InteractiveSession()

def familytree():
    def bitvec(ix,nbit):
        out = []
        for i in range(nbit):
            out.append((i==ix)+0.0)
        return np.array(out)

    names = [ "Christopher", "Andrew", "Arthur", "James", "Charles", "Colin", "Penelope", "Christine", "Victoria", "Jennifer", "Margaret", "Charlotte", "Roberto", "Pierro", "Emilio", "Marco", "Tomaso", "Alfonso", "Maria", "Francesca", "Lucia", "Angela", "Gina", "Sophia"]
    relations = [ "husband", "wife", "son", "daughter", "father", "mother", "brother", "sister", "nephew", "niece", "uncle", "aunt"]

    dataset = []
    with open('relations.txt','r') as f:
        for line in f:
            sline = line.split();
            p1 = names.index(sline[0])
            r = relations.index(sline[1])
            p2 = names.index(sline[2])
            d = [ np.concatenate((bitvec(p1,len(names)),bitvec(r,len(relations)))),
                  bitvec(p2,len(names)) ]
            dataset.append(d)
    return dataset

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.sigmoid(l1)
    output = tf.matmul(l1,output_layer['weights']) + output_layer['biases']
    return output

def calc_accuracy():
    l1 = np.matmul(testing_data_x, W_1) + b_1
    l1[l1 > 10.0] = 1.0
    l1 = 1.0 / (1.0 + np.exp(-1.0 * l1))
    output = np.matmul(l1,W_2)+b_2
    output[output < 0.5] = 0.0
    output[output >= 0.5] = 1.0
    temp = 0.0
    for i in range(len(output)):
        if (np.array_equal(output[i],testing_data_y[i])):
            temp+=1.0
    accuracy = temp / len(testing_data_x) * 100.0
    return accuracy

def train_neural_network(train_x,train_y,test_x,test_y,split):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, epsilon=1e-4).minimize(cost)
    W_1 = 0.0
    W_2 = 0.0
    b_1 = 0.0
    b_2 = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_x = train_x
            epoch_y = train_y
            _, epoch_loss, W_1, b_1, W_2, b_2 = sess.run([optimizer, cost, hidden_1_layer['weights'], hidden_1_layer['biases'], output_layer['weights'], output_layer['biases']], feed_dict={x: epoch_x, y: epoch_y})
            ACC[epoch][split] = calc_accuracy(test_x, test_y,W_1,b_1,W_2,b_2)
            print('Accuracy after epoch!: ',ACC[epoch][split])
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
    print('FINAL Accuracy: ',calc_accuracy(test_x, test_y,W_1,b_1,W_2,b_2))
    return ACC[epoch][split]

if __name__ == "__main__":
    data = familytree() #list
    #loop of 20 iterations
    for split in range(num_splits):
        #shuffle data
        shuffle(data)
        #split into 89 and 15 (training and test set)
        testing_data = data[89:]
        training_data = data[:89]
        #convert list of lists to ndarray
        testing_data = np.array([np.array(xi) for xi in testing_data])
        training_data = np.array([np.array(yi) for yi in training_data])
        #correcting shapes
        test_x = np.ndarray(shape=(15,36), dtype=float, order='F')
        for i in range(len(testing_data)):
            for j in range(len(testing_data[i][0])):
                test_x[i][j]=testing_data[i][0][j]
        test_y = np.ndarray(shape=(15,24), dtype=float, order='F')
        for i in range(len(testing_data)):
            for j in range(len(testing_data[i][1])):
                test_y[i][j]=testing_data[i][1][j]
        train_x = np.ndarray(shape=(89,36), dtype=float, order='F')
        for i in range(len(training_data)):
            for j in range(len(training_data[i][0])):
                train_x[i][j]=training_data[i][0][j]
        train_y = np.ndarray(shape=(89,24), dtype=float, order='F')
        for i in range(len(testing_data)):
            for j in range(len(training_data[i][1])):
                train_y[i][j]=training_data[i][1][j]
        #train the network
        test_accuracy[split] = train_neural_network(train_x,train_y,test_x,test_y,split)
        #store test set accuracy in 1-D array.
    #loop end
    #print test set accuracy mean and standard deviation
    print test_accuracy
