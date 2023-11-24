import numpy as np
import random
import os
import cv2
# from keras.utils.data_utils import Sequence
from scipy import misc
import common
import scipy.io
import h5py
from PIL import Image
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

class ISICImageDataset(tf.keras.utils.Sequence):
    def __init__(self, args):
        self.fids, self.labels, self.class_names, self.skip_class = common.load_dataset_ISIC(args.dataset_root, args.is_train, args.num_classes, args.skip_class)
        self.num_classes = args.num_classes
        self.dataset_root = args.dataset_root
        self.counter = 0
        self.epoch_counter = 0
        self.is_train = args.is_train
        self.load_weights = args.load_weights
        self.init_epoch = args.init_epoch
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.augment = args.augment
        self.sizeW = args.sizeW
        self.sizeH = args.sizeH
        dTypes = ['Validation', 'Training', 'Testing']
        print ('Loading '+dTypes[self.is_train]+' Dataset...')

        self.batch_count = len(self.fids)//self.batch_size # Total number of batches in each epoch
        self.indices = np.arange(len(self.fids)) # Assigining ID # to each data for shuffling at each epoch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.batch_count
    
    def __getitem__(self, idx):
        i = np.sort(self.indices[idx * self.batch_size:(idx + 1) * self.batch_size])
        
        Xs = np.zeros((self.batch_size,self.sizeH,self.sizeW, 3))
        Ys = np.zeros((self.batch_size,self.num_classes))
        for thisI, j in enumerate(i):
            thisImg = Image.open(self.fids[j]+'.jpg').convert("RGB")

            if self.augment and self.is_train ==1:
                thisImg = thisImg.crop((np.shape(thisImg)[1]//100*random.randint(1, 10), np.shape(thisImg)[0]//100*random.randint(1, 10), np.shape(thisImg)[1]-np.shape(thisImg)[1]//100*random.randint(1, 10),np.shape(thisImg)[0]-np.shape(thisImg)[0]//100*random.randint(1, 10))) # Center crop taking out up to 10% of the image dimension
                thisImg = np.asarray(thisImg.resize((224,224)))/255
                thisImg = thisImg[::random.choice([-1, None])]
                thisImg = thisImg[:,::random.choice([-1, None])]
            else:
                thisImg = np.asarray(thisImg.resize((224,224)))/255

            Xs[thisI,] = thisImg
            Ys[thisI,:] = np.asarray(self.labels[self.fids[j][self.fids[j].rfind('/')+1:]])
        
        self.counter += 1
        return Xs, Ys

    def on_epoch_end(self):
        """Method called at the end of every epoch.
            """
        self.epoch_counter +=1
                
        # suffle data along the first dimension
        np.random.shuffle(self.indices)
        return
