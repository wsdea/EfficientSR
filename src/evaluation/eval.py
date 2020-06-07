import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ..preprocessing.utils import uint8_to_float32, float32_to_uint8

#def PSNR(im1, im2, scale):
#    if im1.dtype != im2.dtype:
#        raise Exception('Images of different dtype, {} vs {}'.format(im1.dtype, 
#                                                                     im2.dtype))
#    
#    if im1.dtype == 'uint8':
#        im1 = uint8_to_float32(im1)
#        im2 = uint8_to_float32(im2)
#        
#    if scale > 0:
#        border = 6 + scale + 1
#        im1 = im1[border: -border, border: -border]
#        im2 = im2[border: -border, border: -border]
#        
#    MSE = np.mean((im1 - im2) ** 2)
#    if MSE == 0:
#        return 100
#    else:
#        return -10 * np.log(MSE) / np.log(10)
    
    
def official_PSNR(im1, im2, border=0):
    # im1 and im2 have range [0, 255]
    
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
        
    if len(im1.shape) > 3:
        raise Exception('Only image at a time')
        
    if im1.dtype != im2.dtype:
        raise Exception('Images of different dtype, {} vs {}'.format(im1.dtype, 
                                                                     im2.dtype))
    
    if im1.dtype == np.float32:
        print('Converting to uint8')
        im1 = float32_to_uint8(im1, cuda=True)
        im2 = float32_to_uint8(im2, cuda=True)
        
    elif im1.dtype != np.uint8:
        raise Exception('Only uint8 and float32 are supported')
        
    if border > 0:
        h, w = im1.shape[:2]
        im1 = im1[border:h-border, border:w-border]
        im2 = im2[border:h-border, border:w-border]

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    mse = np.mean((im1 - im2)**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def fast_PSNR(t1, t2):
    """expecting tensors of shape with len = 3 or 4 with values between 0 and 1,
    calculated PSNR should be equal to official_PSNR with a minimum of 3 digits after comma
    X.XXX00000
    """
    if isinstance(t1, np.ndarray):
        t1 = torch.Tensor(t1)
        t2 = torch.Tensor(t2)
    
    if not t1.shape == t2.shape:
        raise ValueError('Input tensors must have the same dimensions.')
        
    if t1.dtype != torch.float or t2.dtype != torch.float:
        raise Exception('Tensors should be float')
    
    with torch.no_grad():
        return_scalar = False
        if len(t1.shape) < 4:
            t1 = t1.unsqueeze(0)
            t2 = t2.unsqueeze(0)
            return_scalar = True
        
        t1 = t1.to('cuda')
        t2 = t2.to('cuda')
            
        mse = ((t1 - t2)**2).mean([1, 2, 3]) #list of len = n_images
        psnr_list = (-10 * torch.log10(mse)).cpu().numpy()
        if return_scalar:
            return psnr_list[0]
        else:
            return psnr_list.tolist()
    
def sub_PSNR(sub, ground_truth_folder, verbose=False):
    gt_images = os.listdir(ground_truth_folder)
    
    if len(gt_images) > len(sub):
        print("Warning submission doesn't contain all images")
    
    all_psnrs = []
    for im_name in tqdm(sub):
        #loading ground truth
        gt_im   = cv2.imread(os.path.join(ground_truth_folder, im_name))
        gt_im   = uint8_to_float32(gt_im)
        
        score = fast_PSNR(sub[im_name], gt_im)
        all_psnrs.append(score)
        if verbose:
            print('{} -> {:.4f}'.format(im_name, score))
    
    return np.mean(all_psnrs)
