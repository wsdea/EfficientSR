# -*- coding: utf-8 -*-
import os
import numpy as np
import random

import torch

from src.training.Torch_utils  import DefaultTrainer
from src.evaluation.submission import save_submission
from src.evaluation.eval       import sub_PSNR
from src.models.models         import small_ESRGAN, baseline_model, FasterMSRResNet, MyModel_debug, small_baseline_model
#from src.models.SRResNet       import MSRResNet
from src.dataset.loading       import image_list, im_show

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
random.seed(42)

class Trainer(DefaultTrainer):
    def __init__(self, model_fun, **kwargs):
        #model
        self.model = model_fun()

        super().__init__(**kwargs)

 ############################################################################################ Datasets

if __name__ == "__main__":
    data_folder = "data"
    data_folder = "D:/ML/SR dataset"

    LR_val_folder = data_folder + "/DIV2K/DIV2K_valid_LR_bicubic/X4"
    HR_val_folder = data_folder + "/DIV2K/DIV2K_val_HR"


    baseline_ckpt = os.path.join('Challenge files',
                                 'MSRResNet',
                                 'MSRResNetx4_model',
                                 'MSRResNetx4.pth')

#    model_fun = baseline_model
#    model_fun = FasterMSRResNet
#    model_fun = small_baseline_model
    model_fun = MyModel_debug

    t = Trainer(model_fun, data_folder=data_folder)

#    t.model.load_weights(baseline_ckpt)


    RETRAIN = 1
    if RETRAIN:
        n_iterations = 1000000
        t.train(n_iterations)

    else:
        sub = t.generate_sub(image_list(LR_val_folder), batch_size=1)
        for im_name in sub:
            im_show(sub[im_name])
            break

        print(sub_PSNR(sub, HR_val_folder))



        if input('Do you want to save the test submission ? Y/N').lower()[0] == 'y':
            save_submission(sub)























