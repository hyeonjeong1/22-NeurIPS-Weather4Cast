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


import argparse
from config import seed_everything
seed_everything(42)
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import datetime
import os
import torch 
import wandb
from tqdm import tqdm
from models.unet_lightning import UNet_Lightning as UNetModel
from models.unet_bottle_lightning import UNetBottle_Lightning as UNetBModel
from utils.data_utils import load_config
from utils.data_utils import get_cuda_memory_usage
from utils.data_utils import tensor_to_submission_file
from utils.w4c_dataloader import RainData
import torchvision.transforms

class DataModule(pl.LightningDataModule):
    """ Class to handle training/validation splits in a single object
    """
    def __init__(self, params, training_params, mode):
        super().__init__()
        self.params = params     
        self.training_params = training_params
        if mode in ['train']:
            self.train_ds = RainData('training', **self.params)
            self.val_ds = RainData('validation', **self.params)
        if mode in ['val']:
            self.val_ds = RainData('validation', **self.params)    
        if mode in ['predict', 'generate']:    
            self.test_ds = RainData('test', **self.params)

    def __load_dataloader(self, dataset, shuffle=True, pin=True):
        dl = DataLoader(dataset, 
                        batch_size=self.training_params['batch_size'],
                        num_workers=self.training_params['n_workers'],
                        shuffle=shuffle, pin_memory=pin, prefetch_factor=2,
                        persistent_workers=True)
        return dl
    
    def train_dataloader(self):
        return self.__load_dataloader(self.train_ds, shuffle=True, pin=True)
    
    def val_dataloader(self):
        return self.__load_dataloader(self.val_ds, shuffle=False, pin=True)

    def test_dataloader(self):
        return self.__load_dataloader(self.test_ds, shuffle=False, pin=True)


def load_model(Model, params, checkpoint_path=''):
    """ loads a model from a checkpoint or from scratch if checkpoint_path='' """
    p = {**params['experiment'], **params['dataset'], **params['train']} 
    if checkpoint_path == '':
        print('-> Modelling from scratch!  (no checkpoint loaded)')
        model = Model(params['model'], p)            
    else:
        print(f'-> Loading model checkpoint: {checkpoint_path}')
        model = Model.load_from_checkpoint(checkpoint_path, UNet_params=params['model'], params=p, strict=False)
    return model

def get_trainer(gpus,params):
    """ get the trainer, modify here its options:
        - save_top_k
     """
    max_epochs=params['train']['max_epochs'];
    print("Trainig for",max_epochs,"epochs");
    checkpoint_callback = ModelCheckpoint(monitor='val_loss_epoch', save_top_k=3, save_last=True,
                                          filename='{epoch:02d}-{val_loss_epoch:.6f}')
    
    paralel_training = None
    ddppplugin = None   
    strategy = None
    if gpus[0] == -1:
        gpus = None
    elif len(gpus) > 1:
#         paralel_training = 'ddp'
        paralel_training = 'gpu'
        strategy = 'ddp'
#         ddppplugin = DDPPlugin(find_unused_parameters=True)
    print(f"====== process started on the following GPUs: {gpus} | accelerator: {paralel_training} ======")
    date_time = datetime.datetime.now().strftime("%m%d-%H:%M")
    version = params['experiment']['name']
    version = version + '_' + date_time

    #SET LOGGER 
    if params['experiment']['logging']: 
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=params['experiment']['experiment_folder'],name=params['experiment']['sub_folder'], version=version, log_graph=True)
    else: 
        tb_logger = False

    if params['train']['early_stopping']: 
        early_stop_callback = EarlyStopping(monitor="val_loss",
                                            patience=params['train']['patience'],
                                            mode="min")
        callback_funcs = [checkpoint_callback, early_stop_callback]
    else: 
        callback_funcs = [checkpoint_callback]
   
    trainer = pl.Trainer(gpus=gpus, max_epochs=max_epochs,
                         gradient_clip_val=params['model']['gradient_clip_val'],
                         gradient_clip_algorithm=params['model']['gradient_clip_algorithm'],
                         accelerator='gpu',
                         callbacks=callback_funcs,logger=tb_logger,
                         profiler='simple',precision=params['experiment']['precision'],
                         plugins=ddppplugin,
                         strategy=strategy
                        )
#     accelerator="gpu", devices=8, strategy="ddp"
    return trainer

def do_predict(trainer, model, predict_params, test_data):
    scores = trainer.predict(model, dataloaders=test_data)
    scores = torch.concat(scores)   
    tensor_to_submission_file(scores,predict_params)

def do_test(trainer, model, test_data):
    scores = trainer.test(model, dataloaders=test_data)

def generate_pseudo_labels(model, test_data):
  ## test_data is DataLoader
  transforms = torch.nn.Sequential(
      torchvision.transforms.CenterCrop((224, 224)),
      # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
  )
  print(type(test_data))
  for i, batch in enumerate(tqdm(test_data)):
    print(batch[0].shape)
    res = model(transforms(batch[0]))
    print(res.shape)

def train(params, gpus, mode, checkpoint_path, model=UNetBModel): 
# def train(params, gpus, mode, checkpoint_path, model=UNetModel):
    """ main training/evaluation method
    """
    # wandb
    if params['logging']:
        wandb.init(project=params['model']['model_name'], name=params['experiment']['name'],  entity="w4c")
    
    # ------------
    # model & data
    # ------------
    get_cuda_memory_usage(gpus)
    data = DataModule(params['dataset'], params['train'], mode)
    model = load_model(model, params, checkpoint_path)
    if params['logging']:
        wandb.watch(model)
    # ------------
    # Add your models here
    # ------------
    
    # ------------
    # trainer
    # ------------
    trainer = get_trainer(gpus, params)
    get_cuda_memory_usage(gpus)
    # ------------
    # train & final validation
    # ------------
    if mode == 'train':
        print("------------------")
        print("--- TRAIN MODE ---")
        print("------------------")
        trainer.fit(model, data)
    
    
    if mode == "val":
    # ------------
    # VALIDATE
    # ------------
        print("---------------------")
        print("--- VALIDATE MODE ---")
        print("---------------------")
        do_test(trainer, model, data.val_dataloader()) 


    if mode == 'predict':
    # ------------
    # PREDICT
    # ------------
        print("--------------------")
        print("--- PREDICT MODE ---")
        print("--------------------")
        if len(params["dataset"]["regions"]) > 1 or params["predict"]["region_to_predict"] != str(params["dataset"]["regions"][0]):
            print("EXITING... \"regions\" and \"regions to predict\" must indicate the same region name in your config file.")
        else:
            do_predict(trainer, model, params["predict"], data.test_dataloader())
      
    if mode == 'generate':
    # ------------
    # Generate
    # ------------
        print("--------------------")
        print("--- Generate MODE ---")
        print("--------------------") 
        aux_params=dict(
            pooling='max',             # one of 'avg', 'max'
            # dropout=0.2,               # dropout ratio, default is None
            activation='sigmoid',      # activation function, default is None
            classes=1,                 # define number of output labels
        )
        model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_depth=5,
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            decoder_attention_type="scse",
            in_channels=11,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
            aux_params=aux_params,
        )

        generate_pseudo_labels(model, data.test_dataloader())
    
    get_cuda_memory_usage(gpus)

def update_params_based_on_args(options):
    config_p = os.path.join('models/configurations',options.config_path)
    params = load_config(config_p)
    
    if options.name != '':
        print(params['experiment']['name'])
        params['experiment']['name'] = options.name

    params['logging'] = options.logging
    return params
    
def set_parser():
    """ set custom parser """
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--config_path", type=str, required=False, default='./configurations/config_basline.yaml',
                        help="path to config-yaml")
    parser.add_argument("-g", "--gpus", type=int, nargs='+', required=False, default=1, 
                        help="specify gpu(s): 1 or 1 5 or 0 1 2 (-1 for no gpu)")
    parser.add_argument("-m", "--mode", type=str, required=False, default='train', 
                        help="choose mode: train (default) / val / predict")
    parser.add_argument("-c", "--checkpoint", type=str, required=False, default='', 
                        help="init a model from a checkpoint path. '' as default (random weights)")
    parser.add_argument("-n", "--name", type=str, required=False, default='', 
                         help="Set the name of the experiment")
    parser.add_argument("-l", "--logging", action='store_true',
                        help="wandb logging true or not")
    parser.add_argument("--freeze", type=str, required=False, default='', help='freeze [upconv] or all except film&final layer [~film_final]')

    return parser

def main():
    parser = set_parser()
    options = parser.parse_args()
    print(torch.get_num_threads())
    torch.set_num_threads(64)
    params = update_params_based_on_args(options)
    params['freeze'] = options.freeze
    print("freeze: ", params['freeze'])
    
    train(params, options.gpus, options.mode, options.checkpoint)

if __name__ == "__main__":
    main()
    """ examples of usage:

    1) train from scratch on one GPU
    python train.py --gpus 1 3 4 --mode train --config_path config_baseline.yaml --name baseline_train

    2) train from scratch on four GPUs
    python train.py --gpus 0 1 2 3 --mode train --config_path config_baseline.yaml --name baseline_train
    
    3) fine tune a model from a checkpoint on one GPU
    python train.py --gpus 1 --mode train  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_tune
    
    4) evaluate a trained model from a checkpoint on two GPUs
    python train.py --gpus 0 1 --mode val  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_validate
    python train.py --gpus 6 7 --mode val  --config_path config_baseline.yaml  --checkpoint "lightning_logs/baseline/baseline_train_0830-05:22/checkpoints/epoch=27-val_loss_epoch=0.789238.ckpt" --name baseline_validate

    5) generate predictions (plese note that this mode works only for one GPU)
    python train.py --gpus 1 --mode predict  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt"

    """
    
    """
    
    input shape: torch.Size([16, 11, 4, 252, 252]), label shape: torch.Size([16, 1, 32, 252, 252])
    BATCH X CHANNEL X TIME X IMG SIZE
    
    """