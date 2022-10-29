# Weather4cast 2022 Starter Kit
#
# Copyright (C) 2022
# Institute of Advanced Research in Artificial Intelligence (IARAI)

# This file is part of the Weather4cast 2022 Starter Kit.
# 
# The Weather4cast 2022 Starter Kit is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# 
# The Weather4cast 2022 Starter Kit is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran


import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.evaluate import *        
from torch.autograd import Variable
import numpy as np
import copy
# from utils.data_utils import mixup_data, mixup_criterion
from torch.nn.functional import normalize
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#models
from models.baseline_UNET3D_bottleneck import UNetBottle as Base_UNET3D # 3_3_2 model selection

VERBOSE = False
# VERBOSE = True
# 
class UNetBottle_Lightning(pl.LightningModule):
    def __init__(self, UNet_params: dict, params: dict,
                 **kwargs):
        super(UNetBottle_Lightning, self).__init__()

        self.in_channels = params['in_channels']
        self.start_filts = params['init_filter_size']
        self.transfer = params['transfer']
        self.categorize = params['categorize']
        self.model = Base_UNET3D(in_channels=self.in_channels,start_filts =  self.start_filts, transfer = self.transfer, categorize=self.categorize)
        
        if params['freeze']: 
            self.freeze(params['freeze'])

        self.save_hyperparameters()
        self.params = params
        #self.example_input_array = np.zeros((44,252,252))
   
        self.main_metric = 'BCE with logits' #mse [log(y+1)-yhay]'

        self.val_batch = 0
        
        self.prec = 7
        
        self.mixup = params['mixup']
        self.alpha = params['alpha']
        self.cutmix_prob = params['cutmix_p']
        self.masking = False
        self.mask_p = 0.9

        pos_weight = torch.tensor(params['pos_weight']);
        if VERBOSE: print("Positive weight:",pos_weight);

        self.loss = params['loss']
        self.bs = params['batch_size']
        # kwargs = {"weight": 0.2, "gamma": 2.0, "reduction": 'mean'}
        self.loss_fn = {
            'smoothL1': nn.SmoothL1Loss(), 'L1': nn.L1Loss(), 'mse': F.mse_loss,
            'BCELoss': nn.BCELoss(), 
            'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(pos_weight=pos_weight), 'CrossEntropy': nn.CrossEntropyLoss(), 'DiceBCE': DiceBCELoss(), 'DiceLoss': DiceLoss(), 'FocalLoss': FocalLoss(weight=torch.FloatTensor(params['loss_weight'])),
            }[self.loss]
        
        # self.valid_loss_fn = DiceLoss()
        self.relu = nn.ReLU() # None
        
        t = f"============== n_workers: {params['n_workers']} | batch_size: {params['batch_size']} \n"+\
            f"============== loss: {self.loss} | weight: {pos_weight} (if using BCEwLL)"
        print(t)
        
        # self.valid_log = []
        # self.valid_cf = []
        # for i in range(1,self.start_filts+1):
        #     self.valid_log.append(open(f'./logs/valid_{i}h.log',"w"))
        #     self.valid_cf.append(open(f'./logs/valid_cf_{i}h.log',"w"))
        #     self.valid_log[i-1].write('recall\tprecision\tf1\tcsi\tacc')
        #     self.valid_log[i-1].flush()
        #     self.valid_cf[i-1].write('tn\tf\nfp\ntp')
        #     self.valid_cf[i-1].flush()
        # self.test_log = []
        # self.test_cf = []
        # for i in range(1,self.start_filts+1):
        #     self.test_log.append(open(f'./logs/test_{i}h.log',"w"))
        #     self.test_cf.append(open(f'./logs/test_cf_{i}h.log',"w"))
        #     self.test_log[i-1].write('recall\tprecision\tf1\tcsi\tacc')
        #     self.test_log[i-1].flush()
        #     self.test_cf[i-1].write('tn\tf\nfp\ntp')
        #     self.test_cf[i-1].flush()
         
    def freeze(self, option):
        if option == "upconv":
           ## if self.transfer is true, freeze only upconv
          for m in self.model.down_convs:
            for pn, p in m.named_parameters():
                p.requires_grad = False
        elif option == '~film_final':
          """ freeze the model without film_layer and final_layer"""
          for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('film_scale') or pn.endswith('film_bias') or ('reduce_channels' in fpn):
                    p.requires_grad = True
                    print(fpn)
                else:
                    p.requires_grad = False
      
    
    def on_fit_start(self):
        """ create a placeholder to save the results of the metric per variable """
        metric_placeholder = {self.main_metric: -1}
        self.logger.log_hyperparams(self.hparams, metric_placeholder)
        
    def forward(self, x):
        x, reg = self.model(x)
        #if self.loss =='BCELoss':
        #x = self.relu(x)
        if self.categorize:
          assert x.shape[1] == 2, x.shape
        return x, reg

    def retrieve_only_valid_pixels(self, x, m):
        """ we asume 1s in mask are invalid pixels """
        ##print(f"x: {x.shape} | mask: {m.shape}")
        return x[~m]

    def get_target_mask(self, metadata):
        mask = metadata['target']['mask']
        #print("mask---->", mask.shape)
        return mask
    
    
    def _compute_valid_loss(self, y_hat, y, agg=True, mask=None):
        if mask is not None:
            y_hat = self.retrieve_only_valid_pixels(y_hat, mask)
            y = self.retrieve_only_valid_pixels(y, mask)
        # print("================================================================================")
        # print(y_hat.shape, y_hat.min(), y_hat.max())
        # print(y.shape, y.min(), y.max())
        if agg:
            loss = self.valid_loss_fn(y_hat, y)
        else:
            loss = self.valid_loss_fn(y_hat, y, reduction='none')
        return loss

    def _compute_loss_categorize(self, y_hat, y):
        # print("================================================================================")
        # print(y_hat.shape, y_hat.min(), y_hat.max())
        # print(y.shape, y.min(), y.max())
        # print(y_hat.get_device(), y.get_device())
        y = torch.squeeze(y, dim=1) 
        # print(y.max(), y.min(), y_hat.max(), y_hat.min())
        # criterion = nn.CrossEntropyLoss(reduction='none')

        loss = self.loss_fn(y_hat, y)
        
        # if VERBOSE: print("loss shape: ", loss.shape)
        return loss
    
    def _compute_loss(self, y_hat, y, agg=True, mask=None, reg=None, r_idx=None, mode='train'):
        if mask is not None:

            y_hat = self.retrieve_only_valid_pixels(y_hat, mask)
            y = self.retrieve_only_valid_pixels(y, mask)
        # print("================================================================================")
        # print(y_hat.shape, y_hat.min(), y_hat.max())
        # print(y.shape, y.min(), y.max())
        if agg:
            loss = self.loss_fn(y_hat, y, reg, r_idx, mode=mode, categorize=self.categorize)
        else:
            loss = self.loss_fn(y_hat, y, reduction='none', mode=mode, categorize=self.categorize)
        return loss
    
    def mixup_data(self, x, y, r_idx, metadata, alpha=1.0):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        r_idx_a, r_idx_b = r_idx, r_idx[index]
        
        metadata_a = metadata
        metadata_b = copy.deepcopy(metadata)
        metadata_b['target']['mask'] = metadata['target']['mask'][index]
        
        # metadata['target']['mask'] = metadata['target']['mask'] | metadata['target']['mask'][index]
        return mixed_x, y_a, y_b, r_idx_a, r_idx_b, metadata_a, metadata_b, lam
    
    def cutmix_data(self,x, y, metadata, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1            
            
        W = x.size()[2]
        H = x.size()[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        y[:, :, bbx1:bbx2, bby1:bby2] = y[index, :, bbx1:bbx2, bby1:bby2]
        metadata['target']['mask'][:, :, bbx1:bbx2, bby1:bby2] = metadata['target']['mask'][index, :, bbx1:bbx2, bby1:bby2]
        
        return x, y, metadata
    
    def _mixup_criterion(self, y_hat, y_a, y_b, reg, r_idx_a, r_idx_b, lam, agg=True, mask_a=None, mask_b=None):
        if mask_a is not None:
            y_hat_a = self.retrieve_only_valid_pixels(y_hat, mask_a)
            y_a = self.retrieve_only_valid_pixels(y_a, mask_a)
        else:
            y_hat_a = y_hat

        if mask_b is not None:
            y_hat_b = self.retrieve_only_valid_pixels(y_hat, mask_b)
            y_b = self.retrieve_only_valid_pixels(y_b, mask_b)
        else:
            y_hat_b = y_hat
        
        if agg:
            loss = lam * self.loss_fn(y_hat_a, y_a, reg, r_idx_a) + (1 - lam) * self.loss_fn(y_hat_b, y_b, reg, r_idx_b)
        else:
            loss = lam * self.loss_fn(y_hat_a, y_a, reg, r_idx_a, reduction='none') + (1 - lam) * self.loss_fn(y_hat_b, y_b, reg, r_idx_b, reduction='none')
            
        return loss

    def training_step(self, batch, batch_idx, phase='train'):
        x, y, metadata, r_idx  = batch
        y_a, y_b, r_idx_a, r_idx_b, lam = None, None, None, None, None
        
        r = np.random.rand(1)
            
        if self.mixup == 'mixup':
            x, y_a, y_b, r_idx_a, r_idx_b, metadata_a, metadata_b, lam = self.mixup_data(x, y, r_idx, metadata, self.alpha)
            x, y_a, y_b, r_idx_a, r_idx_b = map(Variable, (x, y_a, y_b, r_idx_a, r_idx_b))
        elif self.mixup == 'cutmix' and r < self.cutmix_prob:
            x, y, metadata = self.cutmix_data(x, y, metadata, self.alpha)
        
        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')
        y_hat, reg = self.forward(x)
        if self.categorize:
          assert y_hat.shape[1] == 2, y_hat.shape
        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')

        
        if self.masking:
            mask_a = self.get_target_mask(metadata_a)
            mask_b = self.get_target_mask(metadata_b)
            tmp = torch.rand(mask_a.shape, device=mask_a.device) > self.mask_p
            tmp[y == 1] = 0
            mask_a = mask_a + tmp
            del tmp
            
            tmp = torch.rand(mask_b.shape, device=mask_b.device) > self.mask_p
            tmp[y == 1] = 0
            mask_b = mask_b + tmp
            del tmp
            
        
        if self.mixup == 'mixup':
            loss = self._mixup_criterion(y_hat, y_a, y_b, reg, r_idx_a, r_idx_b, lam, mask_a=mask_a, mask_b=mask_b)
        elif self.categorize:
              loss = self._compute_loss_categorize(y_hat, y)
              # loss = self._compute_loss(y_hat, y, mask=None, reg=reg, r_idx=r_idx, mode='valid')
        else: loss = self._compute_loss(y_hat, y, mask=mask) #cutmix
        
        if not self.transfer:
            loss += kor_reg(self.model)
            
        self.log(f'{phase}_loss', loss,batch_size=self.bs)
        return loss
                
    def validation_step(self, batch, batch_idx, phase='val'):
        #data_start = timer()
        x, y, metadata, r_idx  = batch
        #data_end = timer()
        
        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')
        y_hat, reg = self.forward(x)
        if self.categorize:
          assert y_hat.shape[1] == 2, y_hat.shape
        mask = self.get_target_mask(metadata)
        if VERBOSE:
            print(torch.unique(y[:,0,:,:,:], return_counts=True)) #0~128
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')
        if self.categorize:
          loss = self._compute_loss_categorize(y_hat, y)
        else:
          loss = self._compute_loss(y_hat, y, mask=mask, reg=reg, r_idx=r_idx, mode='valid')
#         loss = self._compute_valid_loss(y_hat, y, mask=mask)

        # todo: add the same plot as in `test_step`
        if self.loss=="BCEWithLogitsLoss":
            print("applying thresholds to y_hat logits")
            # set the logits threshold equivalent to sigmoid(x)>=0.5
            idx_gt0 = y_hat>=0
            y_hat[idx_gt0] = 1
            y_hat[~idx_gt0] = 0
        elif self.loss=='DiceBCE' or self.loss == 'DiceLoss':
            threshold = 0
            idx_gt0 = y_hat>=threshold
            y_hat[idx_gt0] = 1
            y_hat[~idx_gt0] = 0    
        elif self.loss == 'CrossEntropy':
            self.model.cate_layer.weight.div_(torch.norm(self.model.cate_layer.weight, p=5, dim=-1, keepdim=True))
            y_hat, reg = self.model(x) # B, 128, 32, 252, 252
            # print("apply softmax")
            y_hat = torch.nn.functional.softmax(y_hat, dim=1)
            # print(y_hat[0,:,0,0,0])

            s = y_hat.shape
            flatten = y_hat.view(s[0], s[1], -1)
            val, ind = flatten.max(1)
            # print(val[0,:], ind[0,:])

            # ind = torch.argmax(y_hat, dim=1)
            # if VERBOSE: print(ind.shape, torch.unique(ind, return_counts=True))
            y_hat = ind.view(s[0], 1, 32, 252, 252)
            # if VERBOSE: print(y_hat.shape)
            # print(torch.unique(y_hat, return_counts=True))
            threshold = 0 
            idx_gt0 = y_hat>threshold
            y_hat[idx_gt0] = 1
            y_hat[~idx_gt0] = 0   
        elif self.loss == 'FocalLoss':
            y_hat, reg = self.model(x) # B, 2, 32, 252, 252
            # print("apply softmax")
            y_hat = torch.nn.functional.softmax(y_hat, dim=1)
            # print(y_hat[0,:,0,0,0])

            s = y_hat.shape
            flatten = y_hat.view(s[0], s[1], -1)
            val, ind = flatten.max(1)
            # print(val[0,:], ind[0,:])

            # ind = torch.argmax(y_hat, dim=1)
            if VERBOSE: print(ind.shape, torch.unique(ind, return_counts=True))
            y_hat = ind.view(s[0], 1, 32, 252, 252)
 
        recall, precision, F1, acc, csi = recall_precision_f1_acc(y, y_hat)
              # if self.params['logging']:
              #     logs = write_temporal_recall_precision_f1_acc(y.squeeze(), y_hat.squeeze(),32)
        
        # for i in range(self.start_filts):
        #     self.valid_log[i].write(logs['log'][i])
        #     self.valid_cf[i].write(logs['cf'][i])
        #     self.valid_log[i].flush()
        #     self.valid_cf[i].flush()
            
        iou = iou_class(y_hat, y)

        #LOGGING
        self.log(f'{phase}_loss', loss,batch_size=self.bs)
        values = {'val_acc': acc, 'val_recall': recall, 'val_precision': precision, 'val_F1': F1, 'val_iou': iou, 'val_CSI': csi}
        self.log_dict(values, batch_size=self.bs)
    
#         return csi
        return loss

    def validation_epoch_end(self, outputs, phase='val'):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log(f'{phase}_loss_epoch', avg_loss, prog_bar=True,batch_size=self.bs)
        self.log(self.main_metric, avg_loss,batch_size=self.bs)


    def test_step(self, batch, batch_idx, phase='test'):
        x, y, metadata = batch
        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')
        y_hat = self.forward(x)
        mask = self.get_target_mask(metadata)
        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')
        loss = self._compute_loss(y_hat, y, mask=mask)
        ## todo: add the same plot as in `test_step`
        if self.loss=="BCEWithLogitsLoss":
            print("applying thresholds to y_hat logits")
            # set the logits threshold equivalent to sigmoid(x)>=0.5
            idx_gt0 = y_hat>=0
            y_hat[idx_gt0] = 1
            y_hat[~idx_gt0] = 0
        elif self.loss=='DiceBCE' or self.loss == 'DiceLoss':
            idx_gt0 = y_hat>=0
            y_hat[idx_gt0] = 1
            y_hat[~idx_gt0] = 0     
        elif self.loss == 'CrossEntropy':
            self.model.cate_layer.weight.div_(torch.norm(self.model.cate_layer.weight, p=5, dim=-1, keepdim=True))
            y_hat, reg = self.model(x) # B, 128, 32, 252, 252
            # print("apply softmax")
            y_hat = torch.nn.functional.softmax(y_hat, dim=1)
            # print(y_hat[0,:,0,0,0])

            s = y_hat.shape
            flatten = y_hat.view(s[0], s[1], -1)
            val, ind = flatten.max(1)
            # print(val[0,:], ind[0,:])

            # ind = torch.argmax(y_hat, dim=1)
            # if VERBOSE: print(ind.shape, torch.unique(ind, return_counts=True))
            y_hat = ind.view(s[0], 1, 32, 252, 252)
            # if VERBOSE: print(y_hat.shape)
            # print(torch.unique(y_hat, return_counts=True))
            threshold = 0 
            idx_gt0 = y_hat>threshold
            y_hat[idx_gt0] = 1
            y_hat[~idx_gt0] = 0   
        elif self.loss == 'FocalLoss':
            y_hat, reg = self.model(x) # B, 2, 32, 252, 252
            # print("apply softmax")
            y_hat = torch.nn.functional.softmax(y_hat, dim=1)
            # print(y_hat[0,:,0,0,0])

            s = y_hat.shape
            flatten = y_hat.view(s[0], s[1], -1)
            val, ind = flatten.max(1)
            # print(val[0,:], ind[0,:])

            # ind = torch.argmax(y_hat, dim=1)
            if VERBOSE: print(ind.shape, torch.unique(ind, return_counts=True))
            y_hat = ind.view(s[0], 1, 32, 252, 252)     
        
        recall, precision, F1, acc, csi = recall_precision_f1_acc(y, y_hat)
        if self.params['logging']:
            logs=write_temporal_recall_precision_f1_acc(y.squeeze(), y_hat.squeeze(),32, test=True)
        # for i in range(self.start_filts):
        #     self.test_log[i].write(logs['log'][i])
        #     self.test_cf[i].write(logs['cf'][i])
            
        #     self.test_log[i].flush()
        #     self.test_cf[i].flush()
            
        iou = iou_class(y_hat, y)

        #LOGGING
        self.log(f'{phase}_loss', loss,batch_size=self.bs)
        values = {'test_acc': acc, 'test_recall': recall, 'test_precision': precision, 'test_F1': F1, 'test_iou': iou, 'test_CSI': csi}
        self.log_dict(values, batch_size=self.bs)
        
        return 0, y_hat

    def predict_step(self, batch, batch_idx, phase='predict'):
        x, y, metadata, r_idx = batch
        y_hat, reg = self.model(x) # B, 128, 32, 252, 252
        mask = self.get_target_mask(metadata)
        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')
        if self.loss=="BCEWithLogitsLoss":
            print("applying thresholds to y_hat logits")
            # set the logits threshold equivalent to sigmoid(x)>=0.5
            idx_gt0 = y_hat>=0
            y_hat[idx_gt0] = 1
            y_hat[~idx_gt0] = 0
        elif self.loss=='DiceBCE' or self.loss == 'DiceLoss':
            idx_gt0 = y_hat>= 0
            y_hat[idx_gt0] = 1
            y_hat[~idx_gt0] = 0
        elif self.loss == 'CrossEntropy':
            self.model.cate_layer.weight.div_(torch.norm(self.model.cate_layer.weight, p=5, dim=-1, keepdim=True))
            y_hat, reg = self.model(x) # B, 128, 32, 252, 252
            # print("apply softmax")
            y_hat = torch.nn.functional.softmax(y_hat, dim=1)
            # print(y_hat[0,:,0,0,0])

            s = y_hat.shape
            flatten = y_hat.view(s[0], s[1], -1)
            val, ind = flatten.max(1)
            # print(val[0,:], ind[0,:])

            # ind = torch.argmax(y_hat, dim=1)
            # if VERBOSE: print(ind.shape, torch.unique(ind, return_counts=True))
            y_hat = ind.view(s[0], 1, 32, 252, 252)
            # if VERBOSE: print(y_hat.shape)
            # print(torch.unique(y_hat, return_counts=True))
            threshold = 0 
            idx_gt0 = y_hat>threshold
            y_hat[idx_gt0] = 1
            y_hat[~idx_gt0] = 0   
            
            if VERBOSE: print(y_hat.shape, torch.unique(y_hat, return_counts=True))
        elif self.loss == 'FocalLoss':
            y_hat, reg = self.model(x) # B, 2, 32, 252, 252
            # print("apply softmax")
            y_hat = torch.nn.functional.softmax(y_hat, dim=1)
            # print(y_hat[0,:,0,0,0])

            s = y_hat.shape
            flatten = y_hat.view(s[0], s[1], -1)
            val, ind = flatten.max(1)
            # print(val[0,:], ind[0,:])

            # ind = torch.argmax(y_hat, dim=1)
            if VERBOSE: print(ind.shape, torch.unique(ind, return_counts=True))
            y_hat = ind.view(s[0], 1, 32, 252, 252) 
        y_hat = y_hat.float()
        return y_hat

    def configure_optimizers(self):
        if VERBOSE: print("Learning rate:",self.params["lr"], "| Weight decay:",self.params["weight_decay"])
        
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose3d, torch.nn.ConvTranspose2d)
        blacklist_weight_modules = (torch.nn.Linear, torch.nn.GroupNorm, torch.nn.BatchNorm3d, torch.nn.BatchNorm2d, )
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('film_scale') or pn.endswith('film_bias'):
                    no_decay.add(fpn)


        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        
        decay_params = []
        no_decay_params = [param_dict[pn] for pn in sorted(list(no_decay))]
        
        for pn in sorted(list(decay)):
            if (param_dict[pn].shape == 5) and (param_dict[pn].shape[2] == 1):
                no_decay_params.append(param_dict[pn])
            else:
                decay_params.append(param_dict[pn])

        # create the pytorch optimizer object
        optim_groups = [
            {"params": decay_params, "weight_decay": float(self.params["weight_decay"])},
            {"params": no_decay_params, "weight_decay": 0},
        ]
            
        optimizer = torch.optim.AdamW(optim_groups,
                                     lr=float(self.params["lr"]))
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
        return [optimizer], [scheduler]

    def seq_metrics(self, y_true, y_pred):
        text = ''
        cm = confusion_matrix(y_true, y_pred).ravel()
        if len(cm)==4:
            tn, fp, fn, tp = cm
            recall, precision, F1 = 0, 0, 0

            if (tp + fn) > 0:
                recall = tp / (tp + fn)
            r = f'r: {recall:.2f}'
            
            if (tp + fp) > 0:
                precision = tp / (tp + fp)
            p = f'p: {precision:.2f}'

            if (precision + recall) > 0:
                F1 = 2 * (precision * recall) / (precision + recall)
            f = f'F1: {F1:.2f}'

            acc = (tn + tp) / (tn+fp+fn+tp)
            text = f"{r} | {p} | {f} | acc: {acc} "

        return text

def main():
    print("running")
if __name__ == 'main':
    main()

#PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, targets, inputs, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)    
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        idx_binary = targets > 0
        targets[idx_binary] = 1.
        targets[~idx_binary] = 0.
        
        intersection = (inputs * targets).sum()      
#         print('intersection', intersection)
#         print('inputs', inputs.sum())
#         print('targets', targets.sum())                     
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
#         print("dice", dice)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, reg, r_idx, smooth=1, mode='train', categorize=False):
        r_idx_label = nn.functional.one_hot(r_idx, num_classes=3)

        if categorize:
          assert inputs.shape[1] == 2, inputs.shape
          # inputs_, ind = torch.topk(inputs, 1, dim=-1)
          inputs, ind = torch.max(inputs, dim=-1)
          # assert inputs.shape == targets.shape, (inputs.shape, targets.shape)
        #comment out if your model contains a sigmoid or equivalent activation layer
        else: inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        # inputs = inputs.flatten()
        targets = targets.view(-1)
        # targets = targets.flatten()
        assert inputs.shape == targets.shape, (inputs.shape, targets.shape)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        reg_loss = F.binary_cross_entropy(reg.to(torch.float32), r_idx_label.to(torch.float32), reduction='mean')
        BCE =  F.binary_cross_entropy(inputs, targets, reduction='mean') # + 0.1 * F.binary_cross_entropy(inputs, 1-targets, reduction='mean')

        if mode == 'valid':
            return dice_loss
        else:
            Dice_BCE =  0.9 * BCE + 1.1 * dice_loss # + 0.5* reg_loss
        
            return Dice_BCE 
            
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

def label_to_one_hot_label(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    ignore_index=255,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.

    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([
                [[0, 1], 
                [2, 0]]
            ]) #(1, 2, 2)
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]]) #(1, 3, 2, 2)

    """
    shape = labels.shape #(B, 32, 252, 252)
    # one hot : (B, C=ignore_index+1, H, W)
    one_hot = torch.zeros((shape[0], ignore_index+1) + shape[1:], device=device, dtype=dtype)
    
    # labels : (B, H, W) -> (B, 32, 252, 252)
    # labels.unsqueeze(1) : (B, C=1, H, W) -> (B, 1, 32, 252, 252)
    # one_hot : (B, C=ignore_index+1, H, W)
    print(labels.unsqueeze(1).shape)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
    
    # ret : (B, C=num_classes, H, W) -> (B, 2, 32, 252, 252)
    ret = torch.split(one_hot, [num_classes, ignore_index+1-num_classes], dim=1)[0]

    
    return ret


# https://github.com/zhezh/focalloss/blob/master/focalloss.py
def focal_loss(input, target, alpha, gamma, reduction, eps, ignore_index):
    
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Return:
        the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    # input : (B, C, H, W) -> (B, bin, T, H, W)
    n = input.size(0) # B
    
    # out_sie : (B, H, W) -> (B, T, H, W)
    out_size = input.size()[0:1] + input.size()[3:]
    
    # input : (B, C, H, W) -> (B, bin, T, H, W)
    # target : (B, H, W) -> (B, T, H, W)
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")
    
    if isinstance(alpha, float):
        pass
    elif isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha)
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
    elif isinstance(alpha, torch.Tensor):
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)       
        

    # compute softmax over the classes axis
    # input_soft : (B, C, H, W) -> (B, C, T, H, W)
    input_soft = F.softmax(input, dim=1) + eps
    
    # create the labels one hot tensor
    # target_one_hot : (B, C, H, W) -> (B, C, T, H, W)
    if VERBOSE: print(torch.unique(target, return_counts=True))
    target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device, dtype=input.dtype, ignore_index=ignore_index)
    print("target_one_hot shape ", target_one_hot.shape)

    # compute the actual focal loss
    weight = torch.pow(1.0 - input_soft, gamma)
    
    # alpha, weight, input_soft : (B, C, H, W)
    # focal : (B, C, H, W)
    focal = -alpha * weight * torch.log(input_soft)
    
    # loss_tmp : (B, H, W)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        # loss : (B, H, W)
        loss = loss_tmp
    elif reduction == 'mean':
        # loss : scalar
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        # loss : scalar
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class _FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math:

        FL(p_t) = -alpha_t(1 - p_t)^{gamma}, log(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha, gamma = 2.0, reduction = 'mean', eps = 1e-8, ignore_index=30):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps, self.ignore_index)

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        # self.weight = torch.FloatTensor([0.2, 0.8]) #weight parameter will act as the alpha parameter to balance class weights
        self.weight = weight

    def forward(self, input, target):
        # print(input.dtype, target.dtype, self.weight.dtype)
        ce_loss = F.cross_entropy(input, target.long(),reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def kor_reg(mdl, lamb=1.0):
    # Consider the below facotrs.
    # factor1: which kind layer (e.g., original(stem), pointwise, depthwise, fc_layer)
    # factor2: power of regularization (i.e., lambda). Maybe, we should differ from each class of layer's lambda.
    # How do we handle these?
    # 'lamb_list' is a list of hyperparmeters for each class of layer. [origin_conv, pointwise, depthwise, fully coneected layer]
    # 'lamb_list' length is 4.
    # 'opt' : position of pointwise convolution. - expansion stage (exp), reduction stage (rec), both.
    
    l2_reg = None

    for module in mdl.modules():
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
            if isinstance(module, nn.Conv3d):
                if module.weight.shape[2] == 1:
                    # pointwise conv
                    W = module.weight
            elif isinstance(module, nn.Linear):
                # fully connected layer 
                W = module.weight
            else:
                continue
                
            cols = W[0].numel()
            w1 = W.view(-1, cols) # W.shape[1] * W.shape[0]
            wt = torch.transpose(w1, 0, 1)
            if W.shape[0]< W.shape[1]:
                m = torch.matmul(wt, w1)
                ident = Variable(torch.eye(cols, cols)).cuda()
            else:
                m = torch.matmul(w1, wt)
                ident = Variable(torch.eye(m.shape[0], m.shape[0])).cuda()
   
            
            w_tmp = m-ident
            height = w_tmp.size(0)
            u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
            v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
            u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
            sigma = torch.dot(u, torch.matmul(w_tmp, v))**2
            
            if l2_reg is None:
                l2_reg = lamb * sigma
                num = 1
            else:
                l2_reg += lamb * sigma
                num += 1
        else:
            continue
    

    return l2_reg/num