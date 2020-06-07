# -*- coding: utf-8 -*-
import numpy as np
import torch
import cv2
import os
from .gpuMatlabResize import imresize_np

def Bicubic(im, R=None, output_shape=None):
    if not R is None:
        return imresize_np(im, scalar_scale=float(R))
    elif not output_shape is None:
        return imresize_np(im, output_shape=output_shape)
    else:
        raise Exception('R or shape required')

def float32_to_uint8(array, cuda=False):
    if cuda:
        with torch.no_grad():
            array = torch.Tensor(array).to('cuda')
            array = (array * 255.).clamp(0., 255.).round().type(torch.uint8)
            return array.cpu().numpy()
    else:
        print('slow (cpu)')
        return np.uint8(np.round(np.clip(255 * array, 0, 255)))

def uint8_to_float32(array, cuda=False):
    if cuda:
        print('slow (cuda)')
        with torch.no_grad():
            array = torch.Tensor(array).to('cuda')
            return (array.type(torch.float32) / 255.).cpu().numpy()
    else:
        return array.astype(np.float32)/255.

def downsampling(im,R):
    if int(R) != R:
        raise Exception('R should be integer (got {})'.format(R))

    if im.shape[0] % R != 0 or im.shape[1] % R != 0:
        raise Exception('Image shape should be multiple of {}'.format(R))
        
    if im.dtype not in [np.float16, np.float32, np.float64] or np.max(im)>1:
        raise Exception('Image should be float values between 0 and 1')
    
    return imresize_np(im, scalar_scale=1./float(R)).astype(np.float32)
    
def extract_crops(im,crop_size,stride=None):
    stride = stride or crop_size
    h,w,c = im.shape
    L = [im[y:y+crop_size,x:x+crop_size] for y in range(0,h,stride) for x in range(0,w,stride)]
    return [x for x in L if x.shape == (crop_size,crop_size,3)]
    
def generate_folder(in_folder, out_folder, scale):
    for im_name in os.listdir(in_folder):
        print(im_name)
        im = cv2.imread(os.path.join(in_folder, im_name))
        small = imresize_np(uint8_to_float32(im), scale)
        np.save(os.path.join(out_folder, im_name.replace('.png', '')), small)

            