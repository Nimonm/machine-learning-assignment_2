from pylab import *
from scipy.misc import imread
from scipy.misc import imsave
import tensorflow as tf
# from scipy.ndimage.filters import gaussian_filter


class Alex(object):
    def __init__(self, flag=True, layer=None, target=None, img_real=None):
        train_x = zeros((1, 227, 227, 3)).astype(float32)
        train_y = zeros((1, 1000))
        xdim = train_x.shape[1:]
        ydim = train_y.shape[1]

        self.img_real = img_real
        self.layer = layer

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
                input_groups = tf.split(input, group, 3)
                kernel_groups = tf.split(kernel, group, 3)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat(output_groups, 3)
            return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

        if (flag):
            self.x=tf.Variable(tf.zeros([1,227,227,3]))
            self.img=tf.placeholder(tf.float32, (1,) + xdim)

        else:
            self.x = tf.placeholder(tf.float32, (None,) + xdim)


        # conv1
        # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        conv1W = tf.Variable(net_data["conv1"][0])
        conv1b = tf.Variable(net_data["conv1"][1])
        self.conv1_in = conv(self.x, conv1W, conv1b, 11, 11, 96, 4, 4, padding="SAME", group=1)
        self.conv1 = tf.nn.relu(self.conv1_in)
        # x[?,227,227,3]
        # conv1[57,57,96]

        # lrn1
        # lrn(2, 2e-05, 0.75, name='norm1')
        self.lrn1 = tf.nn.local_response_normalization(self.conv1,
                                                       depth_radius=2,
                                                       alpha=2e-05,
                                                       beta=0.75,
                                                       bias=1.0)
        # lrn1.shape[?,57,57,96]

        # maxpool1
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        padding = 'VALID'
        self.maxpool1 = tf.nn.max_pool(self.lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=padding)
        # maxpool1.shape [?,28,28,96]

        # conv2
        # conv(5, 5, 256, 1, 1, group=2, name='conv2')
        group = 2
        conv2W = tf.Variable(net_data["conv2"][0])
        conv2b = tf.Variable(net_data["conv2"][1])
        self.conv2_in = conv(self.maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding="SAME", group=group)
        self.conv2 = tf.nn.relu(self.conv2_in)
        # conv2.shape [?,28,28,256]

        # lrn2
        # lrn(2, 2e-05, 0.75, name='norm2')
        self.lrn2 = tf.nn.local_response_normalization(self.conv2,
                                                       depth_radius=2,
                                                       alpha=2e-05,
                                                       beta=0.75,
                                                       bias=1.0)
        # lrn2.shape [?,28,28,256]

        # maxpool2
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
        padding = 'VALID'
        self.maxpool2 = tf.nn.max_pool(self.lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=padding)
        # maxpool2.shape [?,13,13,256]

        # conv3
        # conv(3, 3, 384, 1, 1, name='conv3')
        group = 1
        conv3W = tf.Variable(net_data["conv3"][0])
        conv3b = tf.Variable(net_data["conv3"][1])
        self.conv3_in = conv(self.maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding="SAME", group=group)
        self.conv3 = tf.nn.relu(self.conv3_in)
        # conv3.shape [?,13,13,384]

        # conv4
        # conv(3, 3, 384, 1, 1, group=2, name='conv4')
        group = 2
        conv4W = tf.Variable(net_data["conv4"][0])
        conv4b = tf.Variable(net_data["conv4"][1])
        self.conv4_in = conv(self.conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding="SAME", group=group)
        self.conv4 = tf.nn.relu(self.conv4_in)
        # conv4.shape [?,13,13,384]

        # conv5
        # conv(3, 3, 256, 1, 1, group=2, name='conv5')
        group = 2
        conv5W = tf.Variable(net_data["conv5"][0])
        conv5b = tf.Variable(net_data["conv5"][1])
        self.conv5_in = conv(self.conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding="SAME", group=group)
        self.conv5 = tf.nn.relu(self.conv5_in)
        # conv5.shape [?,13,13,256]

        # maxpool5
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        padding = 'VALID'
        self.maxpool5 = tf.nn.max_pool(self.conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=padding)
        # print(maxpool5.shape)

        # fc6
        # fc(4096, name='fc6')
        fc6W = tf.Variable(net_data["fc6"][0])
        fc6b = tf.Variable(net_data["fc6"][1])
        self.fc6 = tf.nn.relu_layer(tf.reshape(self.maxpool5, [-1, int(prod(self.maxpool5.get_shape()[1:]))]), fc6W,
                                    fc6b)

        # fc7
        # fc(4096, name='fc7')
        fc7W = tf.Variable(net_data["fc7"][0])
        fc7b = tf.Variable(net_data["fc7"][1])
        self.fc7 = tf.nn.relu_layer(self.fc6, fc7W, fc7b)

        # fc8
        # fc(1000, relu=False, name='fc8')
        fc8W = tf.Variable(net_data["fc8"][0])
        fc8b = tf.Variable(net_data["fc8"][1])
        self.fc8 = tf.nn.xw_plus_b(self.fc7, fc8W, fc8b)

        # prob
        # soft-max(name='prob'))
        self.prob = tf.nn.softmax(self.fc8)

        if flag:
            # visualization

            self.sq_diff = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.fc8, target)),
                                           reduction_indices=1))
            self.regularizer = 0.0001*tf.nn.l2_loss(self.x)
            self.loss = self.sq_diff+self.regularizer
            # self.opt = tf.train.AdagradOptimizer(1).minimize(self.loss,var_list=[self.x])
            self.opt = tf.train.AdamOptimizer(1).minimize(self.loss, var_list=[self.x])


def visualize():

    model = Alex(flag=False)
    # the specific layer
    model.layer = model.fc8
    # the input image
    im = (imread("laska.png")[:, :, :3]).astype(float32)
    im = im - mean(im)
    im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
    im_list = []
    im_list.append(im)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        target = sess.run(model.layer, feed_dict={model.x: im_list})

    # training
    model = Alex(flag=True, layer=None, target=target)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1001):
            loss = sess.run([model.x, model.sq_diff, model.regularizer, model.loss, model.opt])

            if i % 200 == 0:
                image = loss[0][0]
                # for channel in range(3):
                #     cimg = gaussian_filter(image[:, :, channel], 1)
                #     image[:, :, channel] = cimg
                imsave("gen/" + str(i) + "_laska.png", image)


if __name__ == '__main__':
    visualize()