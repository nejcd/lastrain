import tensorflow as tf
import numpy as np
import datetime
import laspy, laspy.file
import las2featureMMM as las2feature


path = '/media/nejc/Prostor/AI/data/test_arranged_class_labels/all_classes/'

filename_train = 'train_k03.las'
filename_test = 'train_k02.las'

featureset_train = np.load(path + filename_train + '.npy')
featureset_test = np.load(path + filename_test + '.npy')

train_x = list(featureset_train[:,0] / 255)
train_y = list(featureset_train[:,1])
test_x = list(featureset_test[:,0] / 255)
test_y = list(featureset_test[:,1])

n_classes = len(train_y[1])
batch_size = 512
batch_size_eval = 1024
hm_epochs = 2
img_size = 32
img_depth = 3

x = tf.placeholder(tf.float32,
                shape = (None, img_size, img_size, img_depth))

y = tf.placeholder(tf.float32, shape = (None, n_classes))

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,img_depth,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([img_size // 4 * img_size // 4 * 64, 512])),
               'out':tf.Variable(tf.random_normal([512, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([512])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    conv1 = tf.nn.relu(tf.nn.sigmoid(conv2d(x, weights['W_conv1']) + biases['b_conv1']))
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, img_size // 4 * img_size // 4 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output

def predict(las, batch_size=1024):
    pointsin = np.vstack((las.x, las.y, las.z)).transpose()
    extend = las2feature.get_extend(las)
    grid = las2feature.create_main_grid(pointsin, extend)
    
    labels = []
    prediction = convolutional_neural_network(x)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            restorer = tf.train.import_meta_graph('my-model_v1')
            restorer.restore(sess, tf.train.latest_checkpoint('./'))
        except Exception as e:
            print(str(e))

        for point in pointsin:
            feature = np.array(las2feature.create_feature([point], grid, extend))
            labels.append(int(sess.run(tf.argmax(prediction.eval(feed_dict={x:feature}), 1))))
        
    return prediction

path = '/media/nejc/Prostor/AI/data/'
filename = '01'
las = laspy.file.File(path + filename + '.las', mode='rw')

labels = predict(las)

print labels

las.Classification = labels
las.close()


