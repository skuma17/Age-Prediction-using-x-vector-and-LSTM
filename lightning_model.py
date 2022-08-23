import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

import wavencoder
import torchmetrics
from pytorch_lightning import LightningModule
import pytorch_lightning as pl

from torchmetrics import Accuracy, ConfusionMatrix
from torchmetrics import F1Score, AUROC
from torchmetrics import MeanSquaredError  as MSE
from torchmetrics import MeanAbsoluteError as MAE



from models import Wav2VecLSTM_Base
from models import SpeechBrainLSTM


import pandas as pd
import wavencoder

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class LightningModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super(LightningModel, self).__init__()
        # HPARAMS
        self.save_hyperparameters()
        self.model = SpeechBrainLSTM(HPARAMS['model_hidden_size'])
        #self.model = Wav2VecLSTM_Base(HPARAMS['model_hidden_size'])

        self.regression_criterion = MSE()
        self.mae_criterion = MAE()
        self.rmse_criterion = RMSELoss()
        self.accuracy = Accuracy()

        self.alpha = HPARAMS['model_alpha']
        self.beta = HPARAMS['model_beta']
        self.gamma = HPARAMS['model_gamma']

        self.lr = HPARAMS['training_lr']
        
        self.csv_path = HPARAMS['speaker_csv_path']
        self.df = pd.read_csv(self.csv_path, sep=' ')
        
        self.a_mean = self.df['Age'].mean()
        self.a_std = self.df['Age'].std()
        
        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
  
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y_a, _ = batch
        y_hat_a = self(x)

        y_a = y_a.view(-1).float()
        y_hat_a = y_hat_a.view(-1).float()

        age_loss = self.regression_criterion(y_hat_a, y_a)
        loss = self.beta * age_loss

        age_mae =self.mae_criterion(y_hat_a*self.a_std+self.a_mean, y_a*self.a_std+self.a_mean)
        age_rmse =self.rmse_criterion(y_hat_a*self.a_std+self.a_mean, y_a*self.a_std+self.a_mean)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        return {'loss':loss, 
                'train_age_mae':age_mae.item(),
                'train_age_rmse':age_rmse.item(),
                 }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
                
        age_mae = torch.tensor([x['train_age_mae'] for x in outputs]).sum()/n_batch
        age_rmse = torch.tensor([x['train_age_rmse'] for x in outputs]).sum()/n_batch
        
        self.log('epoch_loss' , loss, prog_bar=True, sync_dist=True)
        self.log('Age_mae',age_mae.item(), prog_bar=True, sync_dist=True)
        self.log('Age_rmse',age_rmse.item(), prog_bar=True, sync_dist=True)
        
    def validation_step(self, batch, batch_idx):
        x, y_a, _ = batch       
        y_hat_a = self(x)

        y_a = y_a.view(-1).float()
        y_hat_a = y_hat_a.view(-1).float()

        age_loss = self.regression_criterion(y_hat_a, y_a)
        loss = self.beta * age_loss
        
        age_mae =self.mae_criterion(y_hat_a*self.a_std+self.a_mean, y_a*self.a_std+self.a_mean)
        age_rmse =self.rmse_criterion(y_hat_a*self.a_std+self.a_mean, y_a*self.a_std+self.a_mean)
        
        return {'val_loss':loss, 
                'val_age_mae':age_mae.item(),
                'val_age_rmse':age_rmse.item(),
        }


    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        
        age_mae = torch.tensor([x['val_age_mae'] for x in outputs]).sum()/n_batch
        age_rmse = torch.tensor([x['val_age_rmse'] for x in outputs]).sum()/n_batch
        
        self.log('v_loss' , val_loss, prog_bar=True, sync_dist=True)
        self.log('v_a_mae',age_mae.item(), prog_bar=True, sync_dist=True)
        self.log('v_a_rmse',age_rmse.item(), prog_bar=True, sync_dist=True)
                
    def test_step(self, batch, batch_idx):       
        x, y_a, y_g  = batch
        y_hat_a = self(x)

        y_a = y_a.view(-1).float()
        y_hat_a = y_hat_a.view(-1).float()

        idx = y_g.view(-1).long()
        female_idx = torch.nonzero(idx).view(-1)
        male_idx = torch.nonzero(1-idx).view(-1)
        
        male_age_mae = self.mae_criterion(y_hat_a[male_idx]*self.a_std+self.a_mean, y_a[male_idx]*self.a_std+self.a_mean)      
        male_age_mae = torch.nan_to_num(male_age_mae)
        female_age_mae = self.mae_criterion(y_hat_a[female_idx]*self.a_std+self.a_mean, y_a[female_idx]*self.a_std+self.a_mean)
        female_age_mae = torch.nan_to_num(female_age_mae)
        
        male_age_rmse = self.rmse_criterion(y_hat_a[male_idx]*self.a_std+self.a_mean, y_a[male_idx]*self.a_std+self.a_mean)
        male_age_rmse = torch.nan_to_num(male_age_rmse)
        female_age_rmse = self.rmse_criterion(y_hat_a[female_idx]*self.a_std+self.a_mean, y_a[female_idx]*self.a_std+self.a_mean)
        female_age_rmse = torch.nan_to_num(female_age_rmse)
        
        return {
                'male_age_mae':male_age_mae.item(),
                'female_age_mae':female_age_mae.item(),
                'male_age_rmse':male_age_rmse.item(),
                'female_age_rmse':female_age_rmse.item(),
         }


    def test_epoch_end(self, outputs):
        n_batch = len(outputs)

        male_age_mae = torch.tensor([x['male_age_mae'] for x in outputs]).mean()
        female_age_mae = torch.tensor([x['female_age_mae'] for x in outputs]).mean()

        male_age_rmse = torch.tensor([x['male_age_rmse'] for x in outputs]).mean()
        female_age_rmse = torch.tensor([x['female_age_rmse'] for x in outputs]).mean()
        
        pbar = {'male_age_mae':male_age_mae.item(),
                'female_age_mae': female_age_mae.item(),
                'male_age_rmse':male_age_rmse.item(),                
                'female_age_rmse': female_age_rmse.item(),
                } 

        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)
        

