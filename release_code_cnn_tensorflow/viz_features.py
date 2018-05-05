from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import cv2
import IPython
import numpy as np



class Viz_Feat(object):


    def __init__(self,val_data,train_data, class_labels,sess):

        self.val_data = val_data
        self.train_data = train_data
        self.CLASS_LABELS = class_labels
        self.sess = sess





    def vizualize_features(self,net):

        images = [0,10,100]
        '''
        Compute the response map for the index images
        '''
        for idx in images:

            datum = self.val_data[idx]
            batch_eval = np.zeros([1,datum['features'].shape[0],datum['features'].shape[1],datum['features'].shape[2]])
            batch_eval[0,:,:,:] = datum['features']

            batch_label = np.zeros([1,len(self.CLASS_LABELS)])
            batch_label[0,:] = datum['label']

            response_map = self.sess.run(net.response_map,feed_dict = {net.images: batch_eval, net.labels: batch_label})

            img_name = 'image_lambda_{0}/image_'.format(net.weights_regularizer) + str(idx) + '.png'
            cv2.imwrite(img_name,datum['c_img'])
            for i in range(5):
                img = self.revert_image(response_map[0,:,:,i])

                img_name = 'image_lambda_{0}/image_'.format(net.weights_regularizer) + str(idx) + '_filter_' + str(i) + '.png'

                cv2.imwrite(img_name,img)
                # cv2.waitKey(300)
        print("Images saved")


    def revert_image(self,img):
        '''
        Used to revert images back to a form that can be easily visualized
        '''

        img = (img+1.0)/2.0*255.0

        img = np.array(img,dtype=int)

        blank_img = np.zeros([img.shape[0],img.shape[1],3])

        blank_img[:,:,0] = img
        blank_img[:,:,1] = img
        blank_img[:,:,2] = img

        img = blank_img.astype("uint8")

        return img

        




