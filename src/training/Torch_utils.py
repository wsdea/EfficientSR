import os
import cv2
import numpy as np
import time
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import l1_loss

from ..dataset.torch_dataset import RandomCropDataset, ValDataset
from ..preprocessing.utils   import uint8_to_float32
from ..evaluation.eval       import fast_PSNR

class ModelCheckpoint:
    def __init__(self, logdir, model):
        self.min_loss = None
        self.max_psnr = None
        self.logdir = logdir
        self.model = model

    def update(self, step, loss, psnr):
        self.model.save_weights(self.logdir, "last_model.pt")

        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model (loss)")
            self.model.save_weights(self.logdir, "best_val_loss_{:03d}.pt".format(step))
            self.min_loss = loss

        if (self.max_psnr is None) or (psnr > self.max_psnr):
            print("Saving a better model (psnr)")
            self.model.save_weights(self.logdir, "best_val_psnr_{:03d}.pt".format(step))
            self.max_psnr = psnr

class CenteredL1Loss(torch.nn.Module):
    def __init__(self, margin):
        super(CenteredL1Loss, self).__init__()
        self.m = margin

    def forward(self, true, preds):
        return l1_loss(preds[:, :, self.m:-self.m, self.m:-self.m],
                       true [:, :, self.m:-self.m, self.m:-self.m])



class DefaultTrainer:
    def __init__(self,
                 device=None,
                 lr=2e-4,
                 batch_size=16,
                 iterations_step=1000,
                 drop_lr_iterations=50000,
                 data_folder="data"):
        self.tqdm = None
        self.R = 4
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.drop_lr_iterations = drop_lr_iterations

        # dataset
        self.set_crop_size(48)
        self.val_dataset   = ValDataset(self.data_folder)

        self.set_loading_threads(1)

        # device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        #training params
#        self.f_loss = torch.nn.L1Loss()
        self.f_loss = CenteredL1Loss(margin=self.R * 5)

        self.iterations_counter = 0
        self.iterations_step = iterations_step

        #model
        self.model.name += "_{}".format(self.model.trainable_params())
        self.model = self.model.to(self.device)
        print('Trainable parameters : {}'.format(self.model.trainable_params()))

        # optimizer
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = self.optimizer,
#                                                                    factor    = 0.5,
#                                                                    patience  = 10,
#                                                                    threshold_mode = 'abs',
#                                                                    min_lr    = 1e-7,
#                                                                    eps       = 1e-5,
#                                                                    verbose   = True,
#                                                                    mode      = 'min')
        self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, lambda x : 0.5)

        self.logdir = None

    def set_crop_size(self, size):
        self.crop_size = size
        self.train_dataset = RandomCropDataset(self.batch_size,
                                               self.R,
                                               self.crop_size,
                                               data_folder=self.data_folder)

    def set_loading_threads(self, threads):
        self.loading_threads = threads

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=1,
                                                        shuffle=True,
                                                        num_workers=self.loading_threads)


    def setup_logs(self):
        self.logdir = os.path.join('logs', "{}-{}_{}".format(int(time.time()),
                                                                            self.model.name,
                                                                            self.lr))

        if not os.path.isdir(self.logdir):
            os.mkdir(self.logdir)
        else:
            raise Exception('Log path already exists!')

        self.model_checkpoint = ModelCheckpoint(self.logdir, self.model)
        self.tb = SummaryWriter(log_dir = self.logdir)


    def train_one_step(self):
        loss_sum = 0
        N = 0
        PSNR_list = []
        old_iteration_counter = self.iterations_counter

        self.model.train()

        self.tqdm = tqdm(range(self.iterations_step),
                         initial=old_iteration_counter)
        for (inputs, targets), _ in zip(self.train_loader,
                                        self.tqdm):
            inputs  = inputs[0]  #torch.Size([n_patches, crop_size, crop_size, 3])
            targets = targets[0]

            inputs  = inputs.to(self.device)
            batch_size = len(inputs)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            #calculating metrics
            targets = targets.to(self.device)

            # Calculate loss
            loss = self.f_loss(outputs, targets)
            loss_sum += batch_size * loss.item()
            N += batch_size

            #PSNR
            PSNR_list += fast_PSNR(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.iterations_counter += 1

            if self.iterations_counter % self.drop_lr_iterations == 0:
                print('Reducing lr')
                self.scheduler.step()

        return loss_sum / N,  np.mean(PSNR_list)


    def network_pass(self, inputs, targets=None, wanted_batch_size=None, verbose=False):
        test = targets is None
        wanted_batch_size = wanted_batch_size or self.batch_size

        if test:
            all_outputs = []
        else:
            loss_sum = 0
            N = 0
            PSNR_list = []

        self.model.eval()

        inputs = inputs.to(self.device)
        n_patches = len(inputs)

        batch_indexes = range((n_patches - 1)//wanted_batch_size + 1)
        if verbose:
            batch_indexes = tqdm(batch_indexes)

        for batch_id in batch_indexes:
            batch_in =  inputs[batch_id * wanted_batch_size: (batch_id + 1) * wanted_batch_size]

            with torch.no_grad():
                outputs = self.model(batch_in)

            if test:
                all_outputs.append(outputs.cpu().numpy())

            else:
                #we have the ground truth
                targets = targets.to(self.device)
                batch_ta = targets[batch_id * wanted_batch_size: (batch_id + 1) * wanted_batch_size]

                #the last batch is going to be smaller
                batch_size = len(batch_in)

                # Calculate loss
                loss = self.f_loss(outputs, batch_ta)
                loss_sum += batch_size * loss.item()
                N += batch_size

                #PSNR
                PSNR_list += fast_PSNR(outputs, batch_ta)

        if not test:
            return loss_sum / N,  np.mean(PSNR_list)
        else:
            return np.concatenate(all_outputs)


    def validate(self):
        inputs, targets = self.val_dataset[0]
        return self.network_pass(torch.Tensor(inputs),
                                 torch.Tensor(targets),
                                 wanted_batch_size=1)

    def _train_and_validate(self, n_iterations):
        if self.logdir is None:
            self.setup_logs()

        n_steps = (n_iterations - 1)//self.iterations_step + 1
        for step in range(n_steps):
            print("Step {}/{}".format(step, n_steps))

            #training
            train_loss, train_psnr = self.train_one_step()

            #validating
            val_loss, val_psnr = self.validate()

            #tensorboard
            self.tb.add_scalar('lr', self.scheduler.optimizer.param_groups[0]['lr'], self.iterations_counter)
            self.tb.add_scalar('loss/train', train_loss, self.iterations_counter)
            self.tb.add_scalar('PSNR/train', train_psnr, self.iterations_counter)
            self.tb.add_scalar('loss/val', val_loss, self.iterations_counter)
            self.tb.add_scalar('PSNR/val', val_psnr, self.iterations_counter)

            print('Epoch {} Loss : {:.5f}/{:.5f}, PSNR : {:.5f}/{:.5f}'.format(self.iterations_counter,
                                                                               train_loss,
                                                                               val_loss,
                                                                               train_psnr,
                                                                               val_psnr))


            self.model_checkpoint.update(self.iterations_counter, val_loss, val_psnr)

        print('# Iterations : {}'.format(self.iterations_counter))


    def generate_sub(self, img_list, batch_size):
        arrays_by_shape = {} #key : str(shape), value : list of numpy arrays
        names_by_shape  = {} #key : str(shape), value : list of image names
        all_images      = {} #key image name, value : numpy array of the resulting image
        for im_name in img_list:
            im = cv2.imread(im_name)
            im = uint8_to_float32(im)

            key = str(im.shape)
            im_name = os.path.basename(im_name) #we don't need the folder name
            im_name = im_name.replace('x4.png', '.png')

            arrays_by_shape[key] = arrays_by_shape.get(key, []) + [im]
            names_by_shape [key] = names_by_shape .get(key, []) + [im_name]
            print("Loaded {} with shape {}".format(im_name, key))

        for shape in arrays_by_shape:
            print('Generating outputs for shape :', shape)
            inputs = np.stack(arrays_by_shape[shape])
            inputs = torch.Tensor(np.moveaxis(inputs, 3, 1))

            outputs = self.network_pass(inputs, wanted_batch_size=batch_size, verbose=False)
            outputs = np.moveaxis(outputs, 1, 3)
            for im_name, im_out in zip(names_by_shape[shape], outputs):
                print(im_name, '->', im_out.shape)
                all_images[im_name] = im_out

        return all_images



    def train(self, n_iterations):
        try:
            self._train_and_validate(n_iterations)

        except KeyboardInterrupt:
            print('Training stopped by user')

        except BrokenPipeError:
            print('Broken pipe Error -> switching to monothread')
            self.set_loading_threads(0)
            return self.train(n_iterations)

        finally:
            self.tqdm.close()


