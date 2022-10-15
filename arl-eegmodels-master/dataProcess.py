'''
2018.12.03
@lsy

Database: SEED
Function:

'''

from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function 
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils as np_utils
import numpy as np 
import scipy.io as scio 
import os 

class DataGenerate:
    def __init__(self, data, label, subject, testSub):
        self.data = data
        self.label = label
        self.subject = subject
        self.testSub = testSub
        self.divideTrainTest()
        self.dataPreprocess()
        self.shuffleData()


    def divideTrainTest(self):       
        '''
        Divide data into train data and test data.
        '''
        idx = [i for i in range(self.label.shape[0]) if self.subject[i] != self.testSub]      
        self.train_data = self.data[idx]
        self.train_label = self.label[idx]

        idx = []
        idx = [i for i in range(self.label.shape[0]) if self.subject[i] == self.testSub]
        self.test_data = self.data[idx]
        self.test_label = self.label[idx]

    def dataPreprocess(self):
       # self.complenetTrainData()
        chans, samples, kernels = 128, 125, 1
        self.train_data = self.train_data.reshape(self.train_data.shape[0], chans, samples, kernels)

        self.test_data = self.test_data.reshape(self.test_data.shape[0], chans, samples, kernels)
    # def complenetTrainData(self):
    #     m = self.train_label.shape[0]
    #
    #     for i in range(m):
    #         self.train_data = np.concatenate((self.train_data, self.train_data[i:i+1, :, :]), axis=0)
    #         self.train_label = np.concatenate((self.train_label, self.train_label[i:i+1]), axis=0)

    def shuffleData(self):
        idx = np.random.permutation(len(self.train_label))
        self.train_data = self.train_data[idx, :, :, :]
        self.train_label = self.train_label[idx]

        self.train_label = np_utils.to_categorical(self.train_label, num_classes=2)
        seed = 1
        self.train_X, self.X_validate, self.train_Y, self.Y_validate = train_test_split(self.train_data,self.train_label, test_size=0.3, random_state=seed)
        #return train_X, X_validate, train_Y, Y_validate, test_X, test_Y





