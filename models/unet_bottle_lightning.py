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

#models
from models.baseline_UNET3D_bottleneck import UNetBottle as Base_UNET3D # 3_3_2 model selection

VERBOSE = False
#VERBOSE = True

class UNetBottle_Lightning(pl.LightningModule):
    def __init__(self, UNet_params: dict, params: dict,
                 **kwargs):
        super(UNetBottle_Lightning, self).__init__()

        self.in_channels = params['in_channels']
        self.start_filts = params['init_filter_size']
        self.transfer = params['transfer']
        self.model = Base_UNET3D(in_channels=self.in_channels,start_filts =  self.start_filts, transfer = self.transfer)
        
        
        if params['freeze'] == True: 
            self.freeze()

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
        self.teacher = None
        self.distill = params['distill']
        
        pos_weight = torch.tensor(params['pos_weight']);
        if VERBOSE: print("Positive weight:",pos_weight);

        self.loss = params['loss']
        self.bs = params['batch_size']
        self.loss_fn = {
            'smoothL1': nn.SmoothL1Loss(), 'L1': nn.L1Loss(), 'mse': F.mse_loss,
            'BCELoss': nn.BCELoss(), 
            'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(pos_weight=pos_weight), 'CrossEntropy': nn.CrossEntropyLoss(), 'DiceBCE': DiceBCELoss(), 'DiceLoss': DiceLoss()
            }[self.loss]
        
        self.valid_loss_fn = DiceLoss()
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
    
    def distillation(self):
        print("distillation")
        self.teacher = copy.deepcopy(self.model)
        for _, m in self.teacher.named_modules():
            for _, p in m.named_parameters():
                p.requires_grad = False
    
        
    def freeze(self):
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
        
    def forward(self, x, r_idx=None, r_idx_b=None, lam=None, phase='val'):
        if phase == 'train':
            x = self.model(x, r_idx, r_idx_b, lam)
        else:
            x = self.model(x, r_idx)

            
        return x

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
    
    
    def _compute_loss(self, y_hat, y, agg=True, mask=None, mode='train'):
        if mask is not None:
            y_hat = self.retrieve_only_valid_pixels(y_hat, mask)
            y = self.retrieve_only_valid_pixels(y, mask)
        # print("================================================================================")
        # print(y_hat.shape, y_hat.min(), y_hat.max())
        # print(y.shape, y.min(), y.max())
        if agg:
            loss = self.loss_fn(y_hat, y, mode=mode)
        else:
            loss = self.loss_fn(y_hat, y, reduction='none', mode=mode)
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
        
        if self.distill:
            y_a, y_b = self.teacher(x)[0], self.teacher(x[index])[0]
        else:
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
    
    def _mixup_criterion(self, y_hat, y_a, y_b, lam, agg=True, mask_a=None, mask_b=None):
        if mask_a is not None:
            # y_hat_a = self.retrieve_only_valid_pixels(y_hat, mask_a)
            # y_a = self.retrieve_only_valid_pixels(y_a, mask_a)
            y_hat_a = y_hat
        else:
            y_hat_a = y_hat

        if mask_b is not None:
            # y_hat_b = self.retrieve_only_valid_pixels(y_hat, mask_b)
            # y_b = self.retrieve_only_valid_pixels(y_b, mask_b)
            y_hat_b = y_hat
        else:
            y_hat_b = y_hat
        
        if agg:
            loss = lam * self.loss_fn(y_hat_a, y_a, distill = self.distill) + (1 - lam) * self.loss_fn(y_hat_b, y_b, distill = self.distill)
        else:
            loss = lam * self.loss_fn(y_hat_a, y_a, reduction='none', distill = self.distill) + (1 - lam) * self.loss_fn(y_hat_b, y_b, reduction='none', distill = self.distill)
            
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
        y_hat = self.forward(x, r_idx_a, r_idx_b, lam, phase=phase)
        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')
        mask_a = self.get_target_mask(metadata_a)
        mask_b = self.get_target_mask(metadata_b)
        
        if self.masking:
            tmp = torch.rand(mask_a.shape, device=mask_a.device) < self.mask_p
            tmp[y == 1] = 0
            mask_a = mask_a + tmp
            del tmp
            
            tmp = torch.rand(mask_b.shape, device=mask_b.device) < self.mask_p
            tmp[y == 1] = 0
            mask_b = mask_b + tmp
            del tmp
            
        
        if self.mixup == 'mixup':
            loss = self._mixup_criterion(y_hat, y_a, y_b, lam, mask_a=mask_a, mask_b=mask_b)
        else: # cutmix also uses this loss function
            loss = self._compute_loss(y_hat, y, mask=mask)
        
        if not self.transfer:
            loss += 0.1 * kor_reg(self.model, device = mask_a.device)
            
        self.log(f'{phase}_loss', loss,batch_size=self.bs)
        return loss
                
    def validation_step(self, batch, batch_idx, phase='val'):
        #data_start = timer()
        x, y, metadata, r_idx  = batch
        #data_end = timer()
        
        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')
        y_hat = self.forward(x, r_idx)
        mask = self.get_target_mask(metadata)
        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')

        loss = self._compute_loss(y_hat, y, mask=mask, mode='valid')

        # todo: add the same plot as in `test_step`

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
            

#         y shape: torch.Size([16, 1, 32, 252, 252]), y_hat shape: torch.Size([16, 1, 32, 252, 252])

        recall, precision, F1, acc, csi = recall_precision_f1_acc(y, y_hat)
        if self.params['logging']:
            logs = write_temporal_recall_precision_f1_acc(y.squeeze(), y_hat.squeeze(),32)

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
        x, y, metadata, r_idx = batch
        if VERBOSE:
            print('x', x.shape, 'y', y.shape, '----------------- batch')
        y_hat = self.forward(x, r_idx)
        mask = self.get_target_mask(metadata)
        if VERBOSE:
            print('y_hat', y_hat.shape, 'y', y.shape, '----------------- model')
        loss = self._compute_loss(y_hat, y, mask=mask, mode='valid')
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
        y_hat = self.model(x, r_idx)
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
            idx_gt0 = y_hat>= 0 # -0.4, -0.62
            print(F.sigmoid(y_hat))
            y_hat[idx_gt0] = 1
            y_hat[~idx_gt0] = 0
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
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
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
        self.bcelogits = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(6.114572))

    def forward(self, inputs, targets, smooth=1, mode='train', distill=False):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        tau = 1
        
        if distill:
            scale = tau **2 if tau >1. else tau
            BCE = - scale * torch.mean(F.sigmoid(targets) * F.logsigmoid(inputs/tau) + (1-F.sigmoid(targets)) * torch.log(1-F.sigmoid(inputs/tau)))
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)  
        if distill:
            targets = F.sigmoid(targets)
        
        # if not distill:
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

        if not distill:
            BCE =  F.binary_cross_entropy(inputs, targets, reduction='mean') # + 0.1 * F.binary_cross_entropy(inputs, 1-targets, reduction='mean')

        if mode == 'valid':
            return dice_loss
        else:
            loss = F.binary_cross_entropy(inputs, targets, reduction='mean') + dice_loss
            # loss = self.bcelogits(inputs, targets) + dice_loss
        
            return loss 

        
def kor_reg(mdl, device, lamb=1.0):
    # Consider the below facotrs.
    # factor1: which kind layer (e.g., original(stem), pointwise, depthwise, fc_layer)
    # factor2: power of regularization (i.e., lambda). Maybe, we should differ from each class of layer's lambda.
    # How do we handle these?
    # 'lamb_list' is a list of hyperparmeters for each class of layer. [origin_conv, pointwise, depthwise, fully coneected layer]
    # 'lamb_list' length is 4.
    # 'opt' : position of pointwise convolution. - expansion stage (exp), reduction stage (rec), both.
    
    l2_reg = None

    for name, module in mdl.named_modules():
        if 'condition' in name:
            continue
            
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
            if isinstance(module, nn.Conv3d):
                if module.weight.shape[3] == 1:
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
                ident = Variable(torch.eye(cols, cols)).to(m.device)
            else:
                m = torch.matmul(w1, wt)
                ident = Variable(torch.eye(m.shape[0], m.shape[0])).to(m.device)
   
            
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