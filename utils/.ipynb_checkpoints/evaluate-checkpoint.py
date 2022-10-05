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


from sklearn.metrics import confusion_matrix
import numpy as np
import torch as t
import wandb

def get_confusion_matrix(y_true, y_pred): 
    """get confusion matrix from y_true and y_pred

    Args:
        y_true (numpy array): ground truth 
        y_pred (numpy array): prediction 

    Returns:
        confusion matrix
    """    
    return confusion_matrix(y_true, y_pred).ravel()

def recall_precision_f1_acc(y, y_hat):
    """ returns metrics for recall, precision, f1, accuracy

    Args:
        y (numpy array): ground truth 
        y_hat (numpy array): prediction 

    Returns:
        recall(float): recall/TPR 
        precision(float): precision/PPV
        F1(float): f1-score
        acc(float): accuracy
        csi(float): critical success index
    """  
      
    # pytorch to numpy
    y, y_hat = [o.cpu() for o in [y, y_hat]]
    y, y_hat = [np.asarray(o) for o in [y, y_hat]]

    cm = get_confusion_matrix(y.ravel(), y_hat.ravel())
    if len(cm)==4:
        tn, fp, fn, tp = cm
        recall, precision, F1 = 0, 0, 0

        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        
        if (precision + recall) > 0:
            F1 = 2 * (precision * recall) / (precision + recall)
        
        if (tp + fn + fp) > 0: 
            csi = tp / (tp + fn + fp)

        acc = (tn + tp) / (tn+fp+fn+tp)

    return recall, precision, F1, acc, csi

def write_temporal_recall_precision_f1_acc(y, y_hat, time_dim, test=False):
    y, y_hat = [o.cpu() for o in [y, y_hat]]
    y, y_hat = [np.asarray(o) for o in [y, y_hat]]
    
    logs = {'log':[], 'cf':[]}
    
#     [16, 32, 252, 252]
    for i in range(time_dim):
        
      if int(i)%4==3:
        cm = get_confusion_matrix(y[:, i, :, :].ravel(), y_hat[:, i, :, :].ravel())
        if len(cm)==4:
            tn, fp, fn, tp = cm
            recall, precision, F1 = 0, 0, 0

            if (tp + fn) > 0:
                recall = tp / (tp + fn)

            if (tp + fp) > 0:
                precision = tp / (tp + fp)

            if (precision + recall) > 0:
                F1 = 2 * (precision * recall) / (precision + recall)

            if (tp + fn + fp) > 0: 
                csi = tp / (tp + fn + fp)

            acc = (tn + tp) / (tn+fp+fn+tp)

            # logs['log'].append(f'{recall}\t{precision}\t{F1}\t{csi}\t{acc}\n')
            # logs['cf'].append(f'{tn}\t{fn}\n{fp}\t{tp}\n')
            if test:
              wandb.log({"recall_t": recall, "precision_t": precision, "F1_t": F1, "csi_t": csi, "acc_t": acc, "tn_t": tn, "fn_t": fn, "fp_t": fp, "tp_t": tp})
            else:
              wandb.log({"recall": recall, "precision": precision, "F1": F1, "csi": csi, "acc": acc, "tn": tn, "fn": fn, "fp": fp, "tp": tp})
    return logs
SMOOTH = 1e-6
def iou_class(y_pred: t.Tensor, y_true: t.Tensor):
    #y_true, y_pred = [o.cpu() for o in [y_true, y_pred]]
    #y_true, y_pred = [np.asarray(o) for o in [y_true, y_pred]]
    y_pred = y_pred.int()
    y_true = y_true.int()
    # Outputs: BATCH X H X W

    intersection = (y_pred & y_true).float().sum(
        (0, 1, 2, 3, 4))  # Will be zero if Truth=0 or Prediction=0
    union = (y_pred | y_true).float().sum(
        (0, 1, 2, 3, 4))  # Will be zero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH
                                     )  # We smooth our devision to avoid 0/0
    iou = iou.cpu()
    return iou
