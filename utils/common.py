""" A bunch of general utilities shared by train/embed/eval """

from argparse import ArgumentTypeError
import os

import numpy as np
import tensorflow as tf
import random
from scipy import misc
import csv

# Commandline argument parsing
def check_directory(arg, access=os.W_OK, access_str="writeable"):
    """ Check for directory-type argument validity.

    Checks whether the given `arg` commandline argument is either a readable
    existing directory, or a createable/writeable directory.

    Args:
        arg (string): The commandline argument to check.
        access (constant): What access rights to the directory are requested.
        access_str (string): Used for the error message.

    Returns:
        The string passed din `arg` if the checks succeed.

    Raises:
        ArgumentTypeError if the checks fail.
    """
    path_head = arg
    while path_head:
        if os.path.exists(path_head):
            if os.access(path_head, access):
                # Seems legit, but it still doesn't guarantee a valid path.
                # We'll just go with it for now though.
                return arg
            else:
                raise ArgumentTypeError(
                    'The provided string `{0}` is not a valid {1} path '
                    'since {2} is an existing folder without {1} access.'
                    ''.format(arg, access_str, path_head))
        path_head, _ = os.path.split(path_head)

    # No part of the provided string exists and can be written on.
    raise ArgumentTypeError('The provided string `{}` is not a valid {}'
                            ' path.'.format(arg, access_str))


def number_greater_x(arg, type_, x):
    try:
        value = type_(arg)
    except ValueError:
        raise ArgumentTypeError('The argument "{}" is not an {}.'.format(
            arg, type_.__name__))

    if value > x:
        return value
    else:
        raise ArgumentTypeError('Found {} where an {} greater than {} was '
            'required'.format(arg, type_.__name__, x))


def positive_int(arg):
    return number_greater_x(arg, int, 0)


def nonnegative_int(arg):
    return number_greater_x(arg, int, -1)


def positive_float(arg):
    return number_greater_x(arg, float, 0)


def float_or_string(arg):
    """Tries to convert the string to float, otherwise returns the string."""
    try:
        return float(arg)
    except (ValueError, TypeError):
        return arg


# Dataset handling
def load_dataset_ISIC(root, is_train, num_classes, skip_class, fail_on_missing=True):
    """ Loads a dataset, returning fids or path to files
        
        Args:
        image_root (string): The path to which the image files list of labels. Used for verification purposes.
        is_train (int): Indicator variable for test (0), training (1), and validation (2) split
        num_classes (int): Number of classes to consider in classification problem
        skip_class (string): New class labels unseen during training
        
        Returns:
        (fids) a list numpy string arrays of path to each data
        (labels) corresponding label of fids.
        i.e. FIDs, i.e. the filenames.
        (class_names) name of in-distribution classes used for training.
        (skip_class) name of out-of-distribution classes unseen during trianing.
        
        Raises:
        IOError if any one file is missing and `fail_on_missing` is True.
        """

    data_root = root+"ISIC_2019_Training_Input/"
    label_root = root+"ISIC_2019_Training_GroundTruth.csv"
    class_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    if num_classes == 9:
        class_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    if num_classes == 7:
        class_names.remove(class_names, skip_class)
    if num_classes == 6:
        class_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'SCC']
        skip_class = ['DF', 'VASC']

    print('class_names:'+str(class_names))
    print('skip_class:'+str(skip_class))
    dirlist = sorted(os.listdir(data_root))
    if is_train == 1:
        dirlist = np.delete(dirlist, range(0,len(dirlist), 10)) # training data
    elif is_train == 2:
        dirlist = dirlist[::10] # test data
    elif is_train ==0:
    	class_names = skip_class
    elif is_train ==-1:
        class_names = skip_class[0]
    elif is_train ==-2:
        class_names = skip_class[1]
    
    # labels
    with open(label_root ) as csv_file:
	    csv_reader = csv.DictReader(csv_file)
	    labels_all = {row['image']:[int(float(row[thisclass])) for thisclass in class_names] for row in csv_reader}
	
    # file name of image input data (fids)
    labels = {}
    fids = {}
    fcount = 0
    for files in dirlist:
        if ".jpg" in files:
            if np.sum(labels_all[files[:-4]])>0:
                fids[fcount] =  data_root+str(files[:-4])
                labels[files[:-4]] =labels_all[files[:-4]]
                fcount = fcount + 1
    
    return fids, labels, class_names, skip_class 

