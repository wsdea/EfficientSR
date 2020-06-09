import os
import cv2
import numpy as np
import random

import torch
from skimage.util.shape import view_as_windows

from ..preprocessing.utils import uint8_to_float32

class RandomCropDataset(torch.utils.data.Dataset):
    def __init__(self, batch_size, R, lr_crop_size, data_folder='data'):

        self.DIV2K = {}
        self.DIV2K['base_folder'] = "{}/DIV2K".format(data_folder)
        self.DIV2K['HR_folder'] = os.path.join(self.DIV2K['base_folder'], "DIV2K_train_HR")
        self.DIV2K['LR_folder'] = os.path.join(self.DIV2K['base_folder'], "DIV2K_train_LR_bicubic/X4")
        self.DIV2K['im_list'] = os.listdir(self.DIV2K['HR_folder'])

        self.Flickr2K = {}
        self.Flickr2K['base_folder'] = "{}/Flickr2K".format(data_folder)
        self.Flickr2K['HR_folder'] = os.path.join(self.Flickr2K['base_folder'], "Flickr2K_HR")
        self.Flickr2K['LR_folder'] = os.path.join(self.Flickr2K['base_folder'], "Flickr2K_LR_bicubic/X4")
        self.Flickr2K['im_list'] = os.listdir(self.Flickr2K['HR_folder'])

        self.R = R
        self.lr_crop_size = lr_crop_size
        self.hr_crop_size = self.R * self.lr_crop_size
        self.batch_size = batch_size #n_crops per image

    def __len__(self):
        return len(self.DIV2K['im_list']) + len(self.Flickr2K['im_list'])

    def __getitem__(self, index):
        if index < len(self.DIV2K['im_list']):
            #loading from DIV2K
            im       = cv2.imread(os.path.join(self.DIV2K['HR_folder'],
                                               self.DIV2K['im_list'][index]))
            im_small = cv2.imread(os.path.join(self.DIV2K['LR_folder'],
                                               self.DIV2K['im_list'][index].replace('.png', 'x4.png')))

        else:
            #loading from Flickr2K
            index = index - len(self.DIV2K['im_list'])
            im       = cv2.imread(os.path.join(self.Flickr2K['HR_folder'],
                                               self.Flickr2K['im_list'][index]))
            im_small = cv2.imread(os.path.join(self.Flickr2K['LR_folder'],
                                               self.Flickr2K['im_list'][index].replace('.png', 'x4.png')))


        im, im_small = self.random_flip_and_rotation(im, im_small)

        im       = uint8_to_float32(im)
        im_small = uint8_to_float32(im_small)

        im       = np.ascontiguousarray(im)
        im_small = np.ascontiguousarray(im_small)

        small_crops, big_crops = self.select_random_crops(im_small, im, self.batch_size)

        # channel first
        return np.moveaxis(small_crops, 3, 1), np.moveaxis(big_crops, 3, 1)

    def random_flip_and_rotation(self, a1, a2):
        if random.random() > 0.5:
            a1 = np.fliplr(a1)
            a2 = np.fliplr(a2)

        rotation = int(random.random() * 4) #random number between 0 and 3
        return np.rot90(a1, rotation), np.rot90(a2, rotation)

    def select_random_crops(self, im_small, im_big, n_crops):
        w_big   = view_as_windows(im_big,
                                (self.hr_crop_size, self.hr_crop_size, 3),
                                step=self.R)[..., 0, :, :, :]
        w_small = view_as_windows(im_small,
                                (self.lr_crop_size, self.lr_crop_size, 3))[..., 0, :, :, :]

        if w_small.shape[:2] != w_big.shape[:2]:
            raise Exception('Not equal shapes for views, big {} vs. small {}'.format(w_big.shape,
                                                                                     w_small.shape))

        # Index and get our specific windows
        indexes_X = np.random.randint(0, w_small.shape[1], n_crops)
        indexes_Y = np.random.randint(0, w_small.shape[0], n_crops)

        big_crops   = w_big  [indexes_Y, indexes_X]
        small_crops = w_small[indexes_Y, indexes_X]

        return small_crops, big_crops


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder="data"):
        self.im_list = [
                      '0808.png',
                      '0809.png',
                      '0816.png',
                      '0824.png',
                      '0829.png',
                      '0842.png',
                      '0856.png',
                      '0868.png',
                      '0884.png',
                      '0893.png',
                      ]

        self.n = len(self.im_list)

        self.base_folder = "{}/DIV2K".format(data_folder)
        self.HR_folder = os.path.join(self.base_folder, "DIV2K_val_HR")
        self.LR_folder = os.path.join(self.base_folder, "DIV2K_valid_LR_bicubic/X4")

        self.hr_shape = None

        print('Loading the validation tensor')
        for i, im_name in enumerate(self.im_list):
            im       = cv2.imread(os.path.join(self.HR_folder,
                                               self.im_list[i]))
            im_small = cv2.imread(os.path.join(self.LR_folder,
                                               self.im_list[i].replace('.png', 'x4.png')))

            if self.hr_shape is None:
                self.hr_shape = im.shape
                self.lr_shape = im_small.shape

                self.hr_array = np.empty((self.n, ) + self.hr_shape, dtype=np.float32)
                self.lr_array = np.empty((self.n, ) + self.lr_shape, dtype=np.float32)


            self.lr_array[i] = uint8_to_float32(im_small)
            self.hr_array[i] = uint8_to_float32(im)

        self.lr_array = torch.Tensor(np.moveaxis(self.lr_array, 3, 1))
        self.hr_array = torch.Tensor(np.moveaxis(self.hr_array, 3, 1))

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.lr_array, self.hr_array

