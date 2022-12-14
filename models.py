import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy

import wavencoder

import speechbrain
from speechbrain.lobes.features import MFCC, Fbank
from speechbrain.lobes.models.Xvector import Xvector
from speechbrain.lobes.models.Xvector import Classifier
from speechbrain.lobes.models.Xvector import Discriminator

class SpeechBrainLSTM(nn.Module):
    def __init__(self, lstm_h, lstm_inp=512):
        super().__init__()

        self.feature_maker = Fbank()
        self.compute_xvect = Xvector(device='gpu') 

        self.lstm = nn.LSTM(lstm_inp, lstm_h, batch_first=True)
 
        self.age_regressor = nn.Sequential(
            nn.Linear(lstm_h, lstm_h),
            nn.ReLU(),
            nn.Linear(lstm_h, 1))

    def forward(self, x):

        x = x.squeeze(1)
        feats = self.feature_maker(x)        
        xvector = self.compute_xvect(feats.float())         
        output, (hidden, _) = self.lstm(xvector)
        attn_output = output.squeeze(1)              
        age = self.age_regressor(attn_output)
        return age

  

class Wav2VecLSTM_Base(nn.Module):
    def __init__(self, lstm_h, lstm_inp=512):
        super().__init__()
        self.encoder = wavencoder.models.Wav2Vec(pretrained=True)

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder.feature_extractor.conv_layers[5:].parameters():
            param.requires_grad = True

               
        self.lstm = nn.LSTM(lstm_inp, lstm_h, batch_first=True)
        self.attention = wavencoder.layers.SoftAttention(lstm_h, lstm_h) 
 
        self.age_regressor = nn.Linear(lstm_h, 1)

    def forward(self, x):
        x = self.encoder(x)

        output, (hidden, _) = self.lstm(x.transpose(1,2))
        attn_output = self.attention(output)
        
        age = self.age_regressor(attn_output)
        return age
     
    
