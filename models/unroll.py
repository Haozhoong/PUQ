import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from models.model_utils import c2r, r2c

#CNN denoiser ======================
def conv_block_ds(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False, padding_mode= 'circular'),
        nn.Conv2d(in_channels, out_channels, 1, bias=False, padding_mode= 'circular'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fcb=True):
        super().__init__()
        self.layers = conv_block(in_channels, out_channels)
        if in_channels != out_channels:
            self.resample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.resample = nn.Identity()

    def forward(self, input):
        shortcut = self.resample(input)
        return self.layers(input) + shortcut


class cnn_denoiser(nn.Module):
    def __init__(self, n_channels, n_layers, n_hiddens, dropout_rate,
                 res, outchannels=None, shortcut=True, relu=False):
        super().__init__()
        if outchannels is None:
            outchannels = n_channels
        layers = []
        layers += conv_block(n_channels, n_hiddens)

        for _ in range(n_layers-2):
            if res:
                layers += nn.Sequential(ResBlock(n_hiddens, n_hiddens))
            else:
                layers += conv_block(n_hiddens, n_hiddens)
            if dropout_rate > 0:
                layers += nn.Sequential(nn.Dropout2d(dropout_rate))

        layers += nn.Sequential(
            nn.Conv2d(n_hiddens, outchannels, 3, padding=1),
            nn.BatchNorm2d(outchannels)
        )
        if relu:
            layers += nn.Sequential(nn.ReLU())

        self.shortcut = shortcut
        self.nw = nn.Sequential(*layers)

    def forward(self, x):
        idt = x # (n_channels, nrow, ncol)
        dw = self.nw(x)
        if self.shortcut:
            dw = dw + idt # (n_channels, nrow, ncol)
        return dw

class project(nn.Module):
    """
    performs DC step
    """
    def __init__(self):
        super(project, self).__init__()

    def forward(self, im, measure, csm, mask): #step for batch image
        """
        :im: complex image (B x channel x nrow x nrol)
        measure # complex (B x channel x ncoil x nrow x ncol)
        csm # complex (B x 1 x ncoil x nrow x ncol)
        mask # complex (B x channel x ncoil x  nrow x ncol)
        """
        
        im_coil = csm * r2c(im, axis=1).unsqueeze(2)
        k_iter = torch.fft.fft2(im_coil, norm='ortho')
        k_iter = k_iter * (1 - mask) + measure * mask
        im_u_coil = torch.fft.ifft2(k_iter, norm='ortho')
        im_u = torch.sum(im_u_coil * csm.conj(), axis=2)
        
        return c2r(im_u, axis=1)


#CG algorithm ======================
class myAtA(nn.Module):
    """
    performs DC step
    """
    def __init__(self, csm, mask, lam):
        super(myAtA, self).__init__()
        self.csm = csm # complex (B x ncoil x nrow x ncol)
        self.mask = mask # complex (B x nrow x ncol)
        self.lam = lam 

    def forward(self, im): #step for batch image
        """
        :im: complex image (B x nrow x nrol)
        """
        im = im.unsqueeze(2)
        im_coil = self.csm * im # split coil images (B x phase x ncoil x nrow x ncol)
        k_full = torch.fft.fft2(im_coil, norm='ortho') # convert into k-space 
        k_u = k_full * self.mask # undersampling
        im_u_coil = torch.fft.ifft2(k_u, norm='ortho') # convert into image domain
        im_u = torch.sum(im_u_coil * self.csm.conj(), axis=2) # coil combine (B x phase x nrow x ncol)
        return im_u.squeeze(1) + self.lam * im.squeeze(2)

#model =======================    
class MoDL(nn.Module):
    def __init__(self, n_channels, n_layers, n_hiddens,
                 k_iters, dropout_rate, res=False,
                 varhead=False, wshare=False,
                 s_min=-10, s_max=10, var_clip=False):
        """
        :n_layers: number of layers
        :k_iters: number of iterations
        """
        super().__init__()
        self.k_iters = k_iters
        self.dw = nn.ModuleList([])
        if not wshare:
            for i in range(k_iters-1):
                self.dw.append(cnn_denoiser(n_channels, n_layers, n_hiddens, dropout_rate, res))

        self.dw.append(cnn_denoiser(n_channels, n_layers, n_hiddens, dropout_rate, res))
        if varhead:
            self.dvar = cnn_denoiser(n_channels, n_layers, n_hiddens,
                                    dropout_rate, res, n_channels//2, False)

        self.dc = project()
        self.varhead = varhead
        self.wshare = wshare

    def forward(self, x0, measure, csm, mask):
        """
        :x0: zero-filled reconstruction (B, 2, nrow, ncol) - float32
        :csm: coil sensitivity map (B, ncoil, nrow, ncol) - complex64
        :mask: sampling mask (B, nrow, ncol) - int8
        """
        x_k = x0.clone()

        for k in range(self.k_iters):
            #dw
            if self.wshare:
                dw_k = self.dw[0]
            else:
                dw_k = self.dw[k]
        
            if self.varhead and k == self.k_iters - 1:
                var = self.dvar(x_k)
            z_k = dw_k(x_k) 
            
            #dc
            x_k = self.dc(z_k, measure, csm, mask) # (2, nrow, ncol)

        if self.varhead:
            return x_k, var
        else:
            return x_k
