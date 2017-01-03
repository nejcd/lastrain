import tensorflow as tf
import numpy as np
import datetime
import random
from tensorflow.python.ops import rnn, rnn_cell



path = '/media/nejc/Prostor/AI/data/test_arranged_class_labels/class_5-6_balanced_MMM/'

filename_train = 'train_k01.las'
filename_test = 'train_k02.las'

featureset_train = np.load(path + filename_train + '.npy')
featureset_test = np.load(path + filename_test + '.npy')

train_x = list(featureset_train[:,0] / 255)
train_y = list(featureset_train[:,1])
test_x = list(featureset_test[:,0] / 255)
test_y = list(featureset_test[:,1])

hm_epochs = 3
n_classes = 3
batch_size = 128
chunk_size = 32
n_chunks = 32
rnn_size = 128
img_depth = 3

x = tf.placeholder('float', [None, img_depth, n_chunks, chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, img_depth, 0,2])
    x = tf.reshape(x, [-1, img_depth, chunk_size])
    x = tf.split(0, n_chunks, x)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            time_start = datetime.datetime.now()

            i = 0
            while i < len(train_x):
                start = i 
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end]).reshape((batch_size, img_depth, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c

                i += batch_size

            time_epoch = datetime.datetime.now() - time_start
            print('Epoch', epoch, 'completed out of',hm_epochs, 'loss:',epoch_loss )
            print('On epoch in {0} . Time to graduation: {1}'.format(time_epoch, (hm_epochs-epoch-1)*time_epoch))

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        i = 0
        n = 0 
        acc = 0
        while i < len(test_x):
            start = i
            end = i + batch_size_eval
            batch_x_test = np.array(test_x[start:end])
            batch_y_test = np.array(test_y[start:end])
            acc = acc + accuracy.eval({x:batch_x_test.reshape((-1, img_depth, n_chunks, chunk_size)), y:batch_y_test})
            n += 1
            i += batch_size_eval

        print("Accuracy: ", acc/n)

train_neural_network(x)

