import tensorflow as tf
import numpy as np

path = '/media/nejc/Prostor/Dropbox/dev/Data/'

filename_train = '29'
filename_test = '01'

featureset_train = np.load(path + filename_train + '.npy')
featureset_test = np.load(path + filename_test + '.npy')

train_x = list(featureset_train[:,0])
train_y = list(featureset_train[:,1])
test_x = list(featureset_test[:,0])
test_y = list(featureset_test[:,1])

#train_x = train_x.reshape(len(train_x), 40, 40, 3)
#test_x = test_x.reshape(len(test_x), 40, 40, 3)

n_classes = 2
batch_size = 32

x = tf.placeholder(tf.float32,
                shape = (None, 40, 40, 3))

y = tf.placeholder(tf.float32, shape = (None, n_classes))

#x = tf.cast(train_x, tf.float32)
#y = tf.cast(train_y, tf.float32)

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,3,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([40 // 4 * 40 // 4 * 64, 512])),
               'out':tf.Variable(tf.random_normal([512, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([512])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #x = tf.reshape(x, shape=[-1, 40, 40, 3])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 40 // 4 * 40 // 4 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 2
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i 
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c

                i += batch_size

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)