import os
import torch
import numpy as np
import scipy.misc as smp
import scipy.ndimage
from random import randint

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_one_hot(labels, num_classes, device):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
        labels : torch.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
        num_classes : int
        Number of classes
        device: string
        Device to place the new tensor on. Should be same as input
    Returns
    -------
        target : torch.Tensor on given device
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    labels=labels.unsqueeze(1)
    #print("Labels here is",labels.shape,labels.type())
    one_hot = torch.FloatTensor(labels.size(0), num_classes, labels.size(2), labels.size(3)).zero_()
    one_hot = one_hot.to(device)
    target = one_hot.scatter_(1, labels.data, 1) 
    return target

def generate_target(y_test):
    my_test_original = y_test
    my_test = np.argmax(y_test[0,:,:,:], axis = 1)
    preds = smp.toimage(my_test)
    #plt.imshow(preds, cmap='jet')
    #plt.show()
    y_target = y_test

    target_class = 13

    dilated_image = scipy.ndimage.binary_dilation(y_target[0, target_class, :, :], iterations=6).astype(y_test.dtype)

    for i in range(256):
        for j in range(256):
            y_target[0, target_class, i, j] = dilated_image[i,j]

    for i in range(256):
        for j in range(256):
            potato = np.count_nonzero(y_target[0,:,i,j])
            if (potato > 1):
                x = np.where(y_target[0, : ,i, j] > 0)
                k = x[0]
                if k[0] == target_class:
                    y_target[0,k[1],i,j] = 0.
                else:
                    y_target[0, k[0], i, j] = 0.

    my_target = np.argmax(y_target[0,:,:,:], axis = 1)
    preds = smp.toimage(my_target)
    return y_target

def generate_target_swap(y_test):
    #my_test_original = y_test
    #my_test = np.argmax(y_test[0,:,:,:], axis = -1)
    #preds = smp.toimage(my_test)
    #plt.imshow(preds, cmap='jet')
    #plt.show()

    y_target = y_test

    y_target_arg = np.argmax(y_test, axis = 1)

    y_target_arg_no_back = np.where(y_target_arg>0)

    y_target_arg = y_target_arg[y_target_arg_no_back]

    classes  = np.unique(y_target_arg)

    #print(classes)

    if len(classes) > 3:

        first_class = 0

        second_class = 0

        third_class = 0

        while first_class == second_class == third_class:
            first_class = classes[randint(0, len(classes)-1)]
            f_ind = np.where(y_target_arg==first_class)
            #print(np.shape(f_ind))

            second_class = classes[randint(0, len(classes)-1)]
            s_ind = np.where(y_target_arg == second_class)

            third_class = classes[randint(0, len(classes) - 1)]
            t_ind = np.where(y_target_arg == third_class)

            summ = np.shape(f_ind)[1] + np.shape(s_ind)[1] + np.shape(t_ind)[1]

            if summ < 1000:
                first_class = 0

                second_class = 0

                third_class = 0

        for i in range(256):
            for j in range(256):
                temp = y_target[0,second_class, i,j]
                y_target[0,second_class, i,j] = y_target[0,first_class,i,j]
                y_target[0, first_class,i, j] = temp

        '''
        print('New target')
        my_target = np.argmax(y_target[0,:,:,:], axis = -1)
        my_test = np.argmax(y_test[0, :, :, :], axis=-1)
        print('potato')
        print(np.shape(my_target))
        print(np.shape(my_test))
        #my_test = np.reshape(my_test, (256, 256))
        together = np.concatenate((my_test, my_target), axis = 1)
        preds = smp.toimage(together)
        plt.imshow(preds, cmap='jet')
        plt.show()
        '''
    else:
        y_target = y_test
        print('Not enough classes to swap!')
    return y_target
