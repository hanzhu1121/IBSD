import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from modules.model import common

def make_model(args, parent=False):
    return IBNet(args)


class IBNet(nn.Module):
    def __init__(self, args, nf=64, unf=64):
        super(IBNet, self).__init__()
        m_ibs = []
        for _ in range(1, 4):
            m_ibs.append(common.get_mean_var(nf, nf))
        self.ibs = nn.Sequential(*m_ibs)

    def forward(self, fea_stu):
        upper_bound = 0
        for idx in range(1, 4):
            f_s, f_r = fea_stu[idx], fea_stu[idx-1]

            mu, var = self.ibs[idx-1](f_s)

        # complete club
            var = var * 0.1 + 1e-6
            positive = - torch.abs(mu - f_r) / var

            prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
            y_samples_1 = f_r.unsqueeze(0)    # shape [1,nsample,dim]

            # log of conditional probability of negative sample pairs
            negative = - (torch.abs(y_samples_1 - prediction_1)).mean(dim=1) / var

            upper_bound += ((positive.sum(dim = -1) - negative.sum(dim = -1)).mean())

        return upper_bound

    def learning_loss(self, fea_stu):
        loss = 0
        for idx in range(1, 4):
            f_s, f_r = fea_stu[idx], fea_stu[idx-1]

            mu, var = self.ibs[idx-1](f_s)
            var = var * 0.1 + 1e-6
            numerator = torch.abs(mu - f_r)
            loss += (mu.shape[1] * np.log(2*math.pi)/2 + torch.log(2*var) + numerator / var).sum(dim=1).mean()

        return loss
