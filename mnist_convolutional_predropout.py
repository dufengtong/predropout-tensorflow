import tensorflow as tf
import tensorflowvisu
import math
import pickle
from generate_noise_mnist import read_h5_data


from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import pickle
from tensorflow.python import debug as tf_debug
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
# mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
noise_var = 64
iteration = 10000
code_num = 100
mnist = read_h5_data('data/noisy_mnist_sigma_%d.hdf5'%noise_var, reshape=False)
# mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer
batch_size = 100
# load mask vectors with dimension [class_num, N]
with open('feature_vectors/fv_10_%d_%d_50.pickle'%(code_num, N), 'rb') as f:
    masks = pickle.load(f)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10], name='label')
# mask of last fc layer
Mask = tf.constant(masks.T, dtype=tf.float32, name='mask')
# variable learning rate
lr = tf.placeholder(tf.float32)


W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1, name='Y1')
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2, name='Y2')
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3, name='Y3')

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M], name='YY')

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4, name='Y4')
# masked layer for training
W5_masked = tf.multiply(W5, Mask, name='masked_W5')
Ylogits = tf.matmul(Y4, W5_masked, name='Ylogits_masked') + B5
# no mask layer for testing
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*batch_size

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
# allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1])], 0)
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)
datavis = tensorflowvisu.MnistDataVis()

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(batch_size)

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, w, b= sess.run([accuracy, cross_entropy, I, allweights, allbiases], {X: batch_X, Y_: batch_Y})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        datavis.append_training_curves_data(i, a, c)
        datavis.update_image1(im)
        datavis.append_data_histograms(i, w, b)

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], {X: mnist.test.images, Y_: mnist.test.labels})
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate})



movie_name = 'movies/mnist-noise-%d-pd-iter-%d-fv-%d-%d.mp4' % (noise_var, iteration, N, code_num)
datavis.animate(training_step, iteration+1, train_data_update_freq=10, test_data_update_freq=100, save_movie=movie_name)


# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

text_file = open("max_accuracy.txt", "a")
text_file.write("Video:%s; max accuracy:%s\n" % (movie_name, str(datavis.get_max_test_accuracy())))
text_file.close()

# layers 4 8 12 200, best 0.989 after 10000 iterations
# layers 4 8 12 200, best 0.9892 after 10000 iterations
# layers 4 8 12 200, concept encoding number 20, noise 64, max accuracy: 0.9535
# layers 4 8 12 200, concept encoding number 40, noise 0, max accuracy: 0.9895
# layers 4 8 12 200, concept encoding number 100, noise 0, max accuracy: 0.9907
# layers 4 8 12 200, concept encoding number 100, noise 64, max accuracy: 0.9565
# layers 4 8 12 200, concept encoding number 150, noise 0, max accuracy: 0.9905
# layers 4 8 12 200, concept encoding number 150, noise 64, max accuracy: 0.9637
# layers 4 8 12 40, concept encoding number 10, random fv max overlap 5, noise 0, max accuracy: 0.9863
# layers 4 8 12 40, concept encoding number 20, random fv max overlap 10, noise 0, max accuracy: 0.988
# layers 4 8 12 40, concept encoding number 20, random fv max overlap 10, noise 64, max accuracy: 0.9549
# layers 4 8 12 200, concept encoding number 100, random fv max overlap 120, noise 0, max accuracy: 0.9909
# layers 4 8 12 200, concept encoding number 100, random fv max overlap 120, noise 64, max accuracy: 0.9678
# layers 4 8 12 64, concept encoding number 10, share 4, max overlap 4, noise 0, max accuracy: 0.9886

# layers 4 8 12 200, concept encoding number 100, random fv max overlap 50, noise 64, max accuracy: 0.9764
# layers 4 8 12 200, concept encoding number 100, random fv max overlap 50, noise 0, max accuracy: 0.9911
