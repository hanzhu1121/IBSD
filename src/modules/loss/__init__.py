import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
import pdb

class WKD_Loss(nn.modules.loss._Loss):
    def __init__(self):
        super(WKD_Loss, self).__init__()
        self.loss = nn.L1Loss()
        self.dwt = DWTForward(J=4, mode='zero', wave='haar').cuda()	

    def forward(self, sr_out, sr_t):
        sd_loss = 0
        
        _, s_h = self.dwt(sr_out)
        _, t_h = self.dwt(sr_t)

        for index in range(len(s_h)):
            sd_loss += self.loss(t_h[index], s_h[index])

        return sd_loss
