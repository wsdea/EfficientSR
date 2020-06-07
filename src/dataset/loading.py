# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def image_list(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]

def im_show(array, name='img', folder='', save=True):
    if array.dtype in [np.float32,np.float64]:
        array = np.uint8(np.round(np.clip(255*array,0,255)))
    elif array.dtype != 'uint8':
        raise Exception('Accepting uint8 or np.float32/64 only')
        
    plt.figure()
    if len(array.shape)==3:
        if array.shape[-1] == 1:
            plt.imshow(array.reshape(array.shape[:-1]))
        else:
            plt.imshow(array[:,:,::-1])
    else:
        plt.imshow(array)
    plt.title(name)
    plt.show()
    
    if save:
        cv2.imwrite(os.path.join(folder,'{}.png'.format(name)),array)

