from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cbook as cbook
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from PIL import Image
import tensorflow as tf
from caffe_classes import class_names

train_x = zeros((1, 227, 227, 3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

################################################################################
# Read Image, and change to BGR

im = np.random.rand(227, 227, 3) * 255 - 128

# im1 = (imread("poodle.png")[:, :, :3]).astype(float32)
# im1 = im1 - mean(im1)
# im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
#
# im2 = (imread("poodle.png")[:, :, :3]).astype(float32)
# im2[:, :, 0], im2[:, :, 2] = im2[:, :, 2], im2[:, :, 0]

################################################################################


net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def gaussian_kernel(size: int, mean: float, std: float):

    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


x = tf.Variable(im, dtype=tf.float32)
y = tf.placeholder(tf.float32, (1, 1000))
do_gauss = tf.placeholder(tf.bool)
x_temp = tf.reshape(x, (1, 227, 227, 3))

def modify_gradient(x):
    def grad(dy):
        return dy
    return tf.identity(x), grad

x_g = modify_gradient(x_temp)

# conv1
# conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
conv1W = tf.constant(net_data["conv1"][0])
conv1b = tf.constant(net_data["conv1"][1])
conv1_in = conv(x_temp, conv1W, conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

# lrn1
# lrn(2, 2e-05, 0.75, name='norm1')
lrn1 = tf.nn.local_response_normalization(conv1,
                                          depth_radius=2,
                                          alpha=2e-05,
                                          beta=0.75,
                                          bias=1.0)

# maxpool1
# max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# conv2
# conv(5, 5, 256, 1, 1, group=2, name='conv2')
conv2W = tf.constant(net_data["conv2"][0])
conv2b = tf.constant(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding="SAME", group=2)
conv2 = tf.nn.relu(conv2_in)

# lrn2
# lrn(2, 2e-05, 0.75, name='norm2')
lrn2 = tf.nn.local_response_normalization(conv2,
                                          depth_radius=2,
                                          alpha=2e-05,
                                          beta=0.75,
                                          bias=1.0)

# maxpool2
# max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# conv3
# conv(3, 3, 384, 1, 1, name='conv3')
conv3W = tf.constant(net_data["conv3"][0])
conv3b = tf.constant(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=1)
conv3 = tf.nn.relu(conv3_in)

# conv4
# conv(3, 3, 384, 1, 1, group=2, name='conv4')
conv4W = tf.constant(net_data["conv4"][0])
conv4b = tf.constant(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=2)
conv4 = tf.nn.relu(conv4_in)

# conv5
# conv(3, 3, 256, 1, 1, group=2, name='conv5')
conv5W = tf.constant(net_data["conv5"][0])
conv5b = tf.constant(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding="SAME", group=2)
conv5 = tf.nn.relu(conv5_in)

# maxpool5
# max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

# fc6
# fc(4096, name='fc6')
fc6W = tf.constant(net_data["fc6"][0])
fc6b = tf.constant(net_data["fc6"][1])
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

# fc7
# fc(4096, name='fc7')
fc7W = tf.constant(net_data["fc7"][0])
fc7b = tf.constant(net_data["fc7"][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

# fc8
# fc(1000, relu=False, name='fc8')
fc8W = tf.constant(net_data["fc8"][0])
fc8b = tf.constant(net_data["fc8"][1])
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

# prob
prob = tf.nn.softmax(fc8)

loss_0 = fc8 * y

x_tr = tf.transpose(x, [2, 0, 1])
fft = tf.spectral.rfft2d((x_tr + 128) / 255.) / (227 * 227)
fft2 = tf.abs(fft) * tf.abs(fft)

fft_std = np.fromfunction(lambda c, x, y:
                          1./((114 - np.abs(x - 114))**2 + (y)**2), (3, 227, 114))
fft_std[:, 0, 0] = 1
fft_std = tf.constant(fft_std, dtype=tf.float32)

fft_mask = np.fromfunction(lambda c, x, y: (x + y > 5), (3, 227, 114))

fft_loss = tf.reduce_sum(tf.abs(tf.abs(fft) - tf.sqrt(fft_std)))

loss = - tf.reduce_sum(loss_0) + fft_loss

decay_op = tf.assign(x, (1 - 0.0001) * x)

gauss_kernel = gaussian_kernel(1, 0., 1.)
# Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
gauss_kernel = tf.tile(gauss_kernel, [1, 1, 3, 1])

# Convolve.
gauss = tf.nn.depthwise_conv2d(x_temp, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
gauss_op = tf.assign(x, tf.where(do_gauss, tf.reshape(gauss, [227, 227, 3]), x))
clip_op = tf.assign(x, tf.clip_by_value(x, -128., 127.))

with tf.control_dependencies([decay_op, gauss_op, clip_op]):
    train_op = tf.train.GradientDescentOptimizer(learning_rate=1000).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

label = 123
gt = np.zeros([1000])
gt[label] = 1.

for i in range(1201):
    if i % 100 == 0:
        flag = True
    else:
        flag = False
    op, im, output, im_loss, im_fft_loss, im_fft = sess.run([train_op, x, fc8, loss, fft_loss, fft], feed_dict={y: [gt], do_gauss: flag})
    if i % 100 == 0:
        print('step %d, output %g, fft_loss %g, loss %g' % (i, output[0][label], im_fft_loss, im_loss))

plt.imshow((im + 128.) / 255.)
plt.show()
