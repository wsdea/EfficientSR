import os
import time
import cv2
import shutil

from tqdm import tqdm
import torch
from ..preprocessing.utils import float32_to_uint8

def save_submission(subs):
    sub_folder = os.path.join('submissions', str(int(time.time())))
    os.mkdir(sub_folder) #creating archive folder
    for image_full_name, array in tqdm(subs.items()):
        im_name = os.path.basename(image_full_name)
        if not im_name.endswith('.png'):
            raise Exception('Base image is not .png')
        path = os.path.join(sub_folder, im_name) 
        if array.dtype != 'uint8':
            print('Warning : Converting to uint8')
            array = float32_to_uint8(array, cuda=True)
        cv2.imwrite(path, array)
    
    #creating zip and deleting the old folder
    print("zipping")
    shutil.make_archive(sub_folder, 'zip', sub_folder)
    shutil.rmtree(sub_folder) 
    print('Saved as {}.zip !'.format(sub_folder))
