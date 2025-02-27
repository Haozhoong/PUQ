import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from .model_utils import *

#region Convolution Block
def conv_block(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size//2),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(),
    )
#endregion

#region DM Block (CNN-based Mapping)
class cnn_mapping(nn.Module):
    def __init__(self, input_ch, param_ch, ch=64, n_layers=5):
        super().__init__()
        layers = []
        layers += conv_block(in_channels=input_ch, out_channels=ch, kernel_size=1)
        for _ in range(n_layers - 1):
            layers += conv_block(in_channels=ch, out_channels=ch, kernel_size=1)
        layers += nn.Sequential(nn.Conv2d(in_channels=ch, out_channels=param_ch, kernel_size=1, padding=0, bias=False))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        output = self.net(x)
        return output
#endregion

#region DR Block (CNN-based Reconstruction)
class cnn_denoiser(nn.Module):
    def __init__(self, input_ch, ch, n_layers=5):
        super().__init__()
        layers = []
        layers += conv_block(in_channels=input_ch, out_channels=ch, kernel_size=3)
        for _ in range(n_layers - 1):
            layers += conv_block(in_channels=ch, out_channels=ch, kernel_size=3)
        layers += nn.Sequential(nn.Conv2d(in_channels=ch, out_channels=input_ch, kernel_size=3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        idt = x
        output = - self.net(x) + idt
        return output

class cnn_denoiser_PDplus1(nn.Module):
    def __init__(self, ch, n_layers=5):
        super().__init__()
        self.net_PD = cnn_denoiser(input_ch=2, ch=ch, n_layers=n_layers)
        self.net_P1 = cnn_denoiser(input_ch=1, ch=ch, n_layers=n_layers)
    def forward(self, x):
        # x: (bsz, 3, Ny, Nx)
        assert x.size(1) == 3
        x_PD = x[:, 0:2, :, :]
        x_P1 = x[:, 2:3, :, :]
        z_PD = self.net_PD(x_PD)
        z_P1 = self.net_P1(x_P1)
        z = torch.cat([z_PD, z_P1], dim=1)
        output = nn.functional.sigmoid(z)
        return output
#endregion

class Dopamine_Net(nn.Module):
    def __init__(self, Nw, ch, Nconv, Niter, lambda_k=0.01, mu_k=0.01):
        super().__init__()
        self.Nw = Nw
        self.ch = ch
        self.Nconv = Nconv
        self.Niter = Niter
        # networks
        self.DM = cnn_mapping(input_ch=Nw*2, param_ch=3, ch=ch, n_layers=Nconv)
        self.DR = nn.ModuleList([
            cnn_denoiser_PDplus1(ch=ch, n_layers=Nconv)
            for k in range(Niter)
        ])
        # coefficients
        self.lambda_k = nn.Parameter(torch.Tensor([lambda_k]), requires_grad=True)
        self.mu_k = nn.Parameter(torch.Tensor([mu_k]), requires_grad=True)
    def forward(self, imgW_init, signal_model, smaps, masks, kdatas):
        '''
            imgW_init:  complex, (bsz, 1, Nw, Ny, Nx)
            signal_model:  implemented with method:
                CalcGradientTerm(self, x, smap, mask, kdata)
                CalcWeightedImage(self, x)
            smaps:   (bsz, Nc, 1, Ny, Nx)
            masks:   (bsz, 1, Nw, Ny, Nx)
            kdatas:  (bsz, Nc, Nw, Ny, Nx)
        '''
        bsz, _, Nw, Ny, Nx = imgW_init.size()
        imgW_init = torch.reshape(imgW_init, (bsz, Nw, Ny, Nx))
        # ------ CNN mapping
        x0 = self.DM(c2r(imgW_init, axis=1))  # (bsz, 3, Ny, Nx)
        x0 = nn.functional.sigmoid(x0)  # range [0, 1]
        xk = x0.clone()
        Nparam = xk.size(1)
        # ------ iterative CNN denoising of quantitative maps 'xk'
        for k in range(self.Niter):
            # denoised map
            DRxk = self.DR[k](xk)
            # gradient term respect to the quantitative parameters
            xtemp = torch.reshape(xk, (bsz, 1, Nparam, Ny, Nx))  # (bsz, 1, Nparam, Ny, Nx)
            grad_term = signal_model.CalcGradientTerm(x=xtemp, smap=smaps, mask=masks, kdata=kdatas)  # (bsz, 1, Nparam, Ny, Nx)
            grad_term = torch.reshape(grad_term, (bsz, Nparam, Ny, Nx))
            # update new map
            xk = (1.0 - 2.0 * self.lambda_k * self.mu_k) * xk \
               + 2.0 * self.lambda_k * self.mu_k * DRxk \
               - 2.0 * self.mu_k * grad_term
        # ------ output
        Qmap0 = torch.reshape(x0, (bsz, Nparam, Ny, Nx))
        Qmap = torch.reshape(xk, (bsz, Nparam, Ny, Nx))
        return Qmap0, Qmap