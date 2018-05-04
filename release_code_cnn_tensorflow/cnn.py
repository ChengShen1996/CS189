
import numpy as np
import tensorflow as tf
#import yolo.config_card as cfg

import IPython

slim = tf.contrib.slim


class CNN(object):

    def __init__(self,classes,image_size):
        '''
        Initializes the size of the network
        '''

        self.classes = classes
        self.num_class = len(self.classes)
        self.image_size = image_size

        self.output_size = self.num_class
        self.batch_size = 40

        self.images = tf.placeholder(tf.float32, [None, self.image_size,self.image_size,3], name='images')


        self.logits = self.build_network(self.images, num_outputs=self.output_size)

        self.labels = tf.placeholder(tf.float32, [None, self.num_class])

        self.loss_layer(self.logits, self.labels)
        self.total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,
                      images,
                      num_outputs,
                      scope='yolo'):

        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):

                '''
                Fill in network architecutre here
                Network should start out with the images function
                Then it should return net
                '''
                if True:
                    # A convolutional layer with 5 filters of size of 15 by 15
                    net = slim.conv2d(images, 5, [5,5], scope = 'conv_0', activation_fn = tf.nn.relu)
                    self.response_map = net
                    # A max pooling operation with filter size of 3 by 3
                    net = slim.max_pool2d(net, [2, 2], scope = 'pool')
                    # A fully connected layer with output size 512
                    net = slim.flatten(net, scope = 'flat')
                else:
                    print("The fully connected version")
                    net = slim.flatten(images,scope = 'flat')
                    net = slim.fully_connected(net, 1, scope = 'fc_1')

                net = slim.fully_connected(net, 512, scope = 'fc_2', activation_fn = tf.nn.relu)
                # A fully connected layer wih output size 25
                net = slim.fully_connected(net, num_outputs, scope = 'fc_3', activation_fn = tf.nn.relu)

        return net



    def get_acc(self,y_,y_out):

        '''
        compute accurracy given two tensorflows arrays
        y_ (the true label) and y_out (the predict label)
        '''

        cp = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))

        ac = tf.reduce_mean(tf.cast(cp, tf.float32))

        return ac

    def loss_layer(self, predicts, classes, scope='loss_layer'):
        '''
        The loss layer of the network, which is written for you.
        You need to fill in get_accuracy to report the performance
        '''
        with tf.variable_scope(scope):

            self.class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = classes,logits = predicts))

            self.accuracy = self.get_acc(classes,predicts)
