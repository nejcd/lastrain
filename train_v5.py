import tensorflow as tf
import numpy as np
import datetime
import random

path = '/media/nejc/Prostor/AI/data/test_arranged_class_labels/class_5-6_balanced_MMM_kelag_part/'

filename_train = 'kel_class_mgi_000084.las.npy_merged'
filename_test = 'test'

featureset_train = np.load(path + filename_train + '.npy')
featureset_test = np.load(path + filename_test + '.npy')

train_x = list(featureset_train[:,0] / 255)
train_y = list(featureset_train[:,1])
test_x = list(featureset_test[:,0] / 255)
test_y = list(featureset_test[:,1])

#train_x = train_x.reshape(len(train_x), 40, 40, 3)
#test_x = test_x.reshape(len(test_x), 40, 40, 3)

n_classes = len(train_y[1])
batch_size = 512
batch_size_eval = 1024
hm_epochs = 10
img_size = 32
img_depth = 3

x = tf.placeholder(tf.float32,
                shape = (None, img_size, img_size, img_depth))

y = tf.placeholder(tf.float32, shape = (None, n_classes))

#x = tf.cast(train_x, tf.float32)
#y = tf.cast(train_y, tf.float32)

keep_rate = 0.9
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,img_depth,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_conv3':tf.Variable(tf.random_normal([5,5,64,64])),
               'W_fc1':tf.Variable(tf.random_normal([img_size // 4 * img_size // 4 * 64, 1024])),
               'W_fc2':tf.Variable(tf.random_normal([1024, 1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_conv3':tf.Variable(tf.random_normal([64])),
               'b_fc1':tf.Variable(tf.random_normal([1024])),
               'b_fc2':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #x = tf.reshape(x, shape=[-1, 40, 40, 3])

    conv1 = tf.nn.relu(tf.nn.sigmoid(conv2d(x, weights['W_conv1']) + biases['b_conv1']))
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(tf.nn.sigmoid(conv2d(conv1, weights['W_conv2']) + biases['b_conv2']))
    conv2 = maxpool2d(conv2)

    conv3 = tf.nn.relu(tf.nn.sigmoid(conv2d(conv2, weights['W_conv3']) + biases['b_conv3']))
    conv3 = maxpool2d(conv3)

    fc1 = tf.reshape(conv3,[-1, img_size // 4 * img_size // 4 * 64])
    fc1 = tf.nn.relu(tf.matmul(fc1, weights['W_fc1']) + biases['b_fc1'])
    fc1 = tf.nn.dropout(fc1, keep_rate)

    
    fc2 = tf.nn.relu(tf.matmul(fc1, weights['W_fc2']) + biases['b_fc2'])
    fc2 = tf.nn.dropout(fc2, keep_rate)


    output = tf.matmul(fc2, weights['out']) + biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            time_start = datetime.datetime.now()

            i = 0
            while i < len(train_x):
                start = i 
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

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
            acc = acc + accuracy.eval({x:batch_x_test, y:batch_y_test})
            n += 1
            i += batch_size_eval

        print("Accuracy: ", acc/n)

        saver.save(sess, 'my-model_v1')

train_neural_network(x)