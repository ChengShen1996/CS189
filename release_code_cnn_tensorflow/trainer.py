import tensorflow as tf
import datetime
import os
import sys
import argparse

slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net, data, max_iter):

     
        self.net = net
        self.data = data
       
        #Number of iterations to train for
        self.max_iter = max_iter
        #Every 200 iterations please record the trest and train loss
        self.summary_iter = 200
        


        '''
        Tensorflow is told to use a gradient descent optimizer 
        In the function optimize you will iteratively apply this on batches of data
        '''
        self.train_step = tf.train.MomentumOptimizer(.003, .9)
        self.train = self.train_step.minimize(self.net.class_loss)
        

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    


    def optimize(self):

        #Append the current train and test accuracy every 200 iterations
        self.train_accuracy = []
        self.test_accuracy = []
        print("optimizing")
        '''
        Performs the training of the network. 
        Implement SGD using the data manager to compute the batches
        Make sure to record the training and test accuracy through out the process
        '''
        for step in range(1,self.max_iter+1):
            images, labels = self.data.get_train_batch()
            print("Progress: {0}%".format(step/self.max_iter*100),end="\r")
            feed_dict = {self.net.images: images, self.net.labels: labels}

            if step%self.summary_iter == 0:

                accuracy = self.sess.run(self.net.accuracy,feed_dict = feed_dict)

                self.train_accuracy.append(accuracy)


                images_t, labels_t = self.data.get_validation_batch()

                feed_dict_test = {self.net.images: images_t, self.net.labels: labels_t}

                test_acc = self.sess.run(self.net.accuracy, feed_dict = feed_dict_test)

                self.test_accuracy.append(test_acc)

                print("Current test accuracy is: {0} at step: {1} ".format(test_acc,step))

                if test_acc > 0.9:
                    return 

            else:
                self.sess.run([self.train],feed_dict = feed_dict)
            


    def save_cfg(self):
        with open(os.path.join(self.output_dir,'config.txt'),'w') as f:
            self.cfg_dict = self.cfg__dict__
            for key in sorted(self.cfg_dict.keys()):
                if key[0].isupper():
                    self.cfg_str = '{}: {}\n'.format(key, self.cfg_dict[key])
                    f.write(self.cfg_str)