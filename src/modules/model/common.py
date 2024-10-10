import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class get_dynamic_kernel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(get_dynamic_kernel, self).__init__()
        m_kernel_mu = []
        m_kernel_mu.append(nn.Conv2d(in_channel, in_channel, 1, padding=0, bias=True))
        m_kernel_mu.append(nn.AdaptiveAvgPool2d(1))
        self.k_mu = nn.Sequential(*m_kernel_mu)

        m_kernel_var = []
        m_kernel_var.append(nn.Conv2d(in_channel, in_channel, 1, padding=0, bias=True))
        m_kernel_var.append(nn.AdaptiveAvgPool2d(1))
        self.k_var = nn.Sequential(*m_kernel_var)

    def forward(self, x):
        kernel_mu = self.k_mu(x).unsqueeze(1)
        kernel_var = self.k_var(x).unsqueeze(1)
        return kernel_mu, kernel_var

class get_mean_var(nn.Module):
    '''calculate mean and var'''
    def __init__(self, in_channel, out_channel):
        super(get_mean_var, self).__init__()
        self.mean = self.get_adaptation_layer(in_channel, out_channel, False)
        self.var = self.get_adaptation_layer(in_channel, out_channel, False)
        self.softplus = nn.Softplus()

        self.kernel = get_dynamic_kernel(in_channel, in_channel)

    def get_adaptation_layer(self, in_channels, out_channels, adap_avg_pool):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=1, stride=1, padding=0 )
        )
        if adap_avg_pool:
            layer.add_module(str(len(layer)+1), nn.AdaptiveAvgPool2d(1))

        return layer

    def forward(self, x):
        mean = self.mean(x)
        var = self.var(x)
        
        k_mu, k_var = self.kernel(x)
        sp_mean = []
        sp_var = []
        for i in range(k_mu.shape[0]):
            sp_mean.append(F.conv2d(x[i,:,:,:].unsqueeze(0), k_mu[i,:,:,:,:], stride=1, padding=0))
            sp_var.append(F.conv2d(x[i,:,:,:].unsqueeze(0), k_var[i,:,:,:,:], stride=1, padding=0))
        sp_mean = torch.cat(sp_mean, 0)
        sp_var = torch.cat(sp_var, 0)

        mean = mean + sp_mean
        var = self.softplus(var + sp_var)

        return mean, var
