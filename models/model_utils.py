import torch
import torch.fft
import torch.nn as nn
import numpy as np

def c2r(complex_img, axis=1):
    """
    :input shape: 1 x row x col (complex64)
    :output shape: 2 x row x col (float32)
    """
    if isinstance(complex_img, np.ndarray):
        real_img = np.concatenate((complex_img.real, complex_img.imag), axis=axis)
    elif isinstance(complex_img, torch.Tensor):
        real_img = torch.cat((complex_img.real, complex_img.imag), dim=axis)
    else:
        raise NotImplementedError
    return real_img

def r2c(images, axis=1):
    """
    :input shape: 2c x row x col (float32)
    :output shape: 1c x row x col (complex64)
    """
    C = int(images.shape[axis]/2)
    images = torch.complex(torch.index_select(images, axis, torch.tensor(range(C), device=images.device)),
                           torch.index_select(images, axis, torch.tensor(range(C, images.shape[axis]), device=images.device)))
    return images

def unroll_recon_forward(model, images, measurement,
                         sensmap, mask, sampling_times,
                         args, delta=2, grad=False):
    
    if args.model_name in ['unroll']:
        images_zf = MRI_AHop(measurement, sensmap, mask=None).squeeze(2)
        images_zf = c2r(images_zf, axis=1)
        
    elif args.model_name in ['crnn']:
        images_zf = MRI_AHop(measurement, smap=None, mask=None)
        images_zf = c2r(images_zf, axis=2)
        images_zf = images_zf.permute(1, 0, 2, 3, 4)
        measurement = measurement.permute(1, 0, 2, 3, 4)
        mask = mask.permute(1, 0, 2, 3, 4)
        
    elif args.model_name in ['unet']:
        images_zf = torch.fft.ifft2(measurement, dim=(-2,-1), norm='ortho')
        images_zf = images_zf.view(images_zf.shape[0], images_zf.shape[1]*images_zf.shape[2],
                                   images_zf.shape[3], images_zf.shape[4])
        images_zf = c2r(images_zf, axis=1)
    
    out_samplings = torch.zeros(sampling_times, images.shape[0], images.shape[1],
                                images.shape[3], images.shape[4], device=images.device)
    if args.varhead:
        apara_samplings = torch.zeros(sampling_times, images.shape[0], images.shape[1],
                                      images.shape[3], images.shape[4], device=images.device)
    with torch.set_grad_enabled(grad):
        for i in range(sampling_times):
            if args.model_name in ['unroll']:
                if args.varhead:
                    pred, apara = model(images_zf, measurement, sensmap, mask)
                    apara_samplings[i] = apara
                else:
                    pred = model(images_zf, measurement, sensmap, mask)
                    apara = None
                pred = r2c(pred, axis=1).real
                
            elif args.model_name in ['unet']:
                if args.varhead:
                    pred, apara = model(images_zf)
                    apara_samplings[i] = apara
                else:
                    pred = model(images_zf)
                    apara = None
            elif args.model_name in ['crnn']:
                pred = model(images_zf, measurement, mask, test=False)
                pred = r2c(pred.transpose(0, 1), axis=2)
                pred = torch.sum(torch.conj(sensmap) * pred, dim=2, keepdim=False).real
                apara = None
            else:
                print('Unknown model ', args.model_name)
                raise(ValueError)
            
            out_samplings[i] = pred

        if sampling_times>1:
            if args.std:
                e_var = torch.std(out_samplings, dim=0)
            else:
                e_var = torch.var(out_samplings, dim=0)
            recon_mean = torch.mean(out_samplings, dim=0)
            if args.varhead:
                if delta == 2:
                    a_var = torch.mean(torch.exp(apara_samplings), dim=0)
                elif delta == 1:
                    a_var = torch.mean(torch.sqrt(torch.exp(apara_samplings)), dim=0)
                else:
                    raise(ValueError)
            else:
                a_var = None
        else:
            e_var = None
            recon_mean = pred
            if args.varhead:
                if delta == 2:
                    a_var = torch.exp(apara)
                elif delta == 1:
                    a_var = torch.sqrt(torch.exp(apara))
                else:
                    raise(ValueError)
            else:
                a_var = None

    recon_result = {'recon_mean':recon_mean, 'evar':e_var, 'avar':a_var, 'apara': apara}
    return recon_result

def divide_zero(input, other, eps=1e-9):
    # input, other same shape
    result_shape = [max(i, o) for i,o in zip(input.shape, other.shape)]
    input = input.broadcast_to(result_shape)
    other = other.broadcast_to(result_shape)
    result = torch.zeros_like(other, dtype=torch.result_type(input, other))
    nozero_mask = torch.abs(other) > eps
    result[nozero_mask] = torch.div(input[nozero_mask],
                                    other[nozero_mask])
    return result

#region MRI encoding operator
def MRI_Aop(img, smap, mask=None):
    kspc = torch.fft.fft2(smap * img, dim=(-2,-1), norm='ortho')
    if mask is not None:
        kspc = mask * kspc
    return kspc

def MRI_AHop(kspc, smap=None, mask=None, dim=2):
    if mask is not None:
        kspc = mask * kspc
    if smap is not None:
        img = torch.sum(torch.conj(smap) * torch.fft.ifft2(kspc, dim=(-2,-1), norm='ortho'), dim=dim, keepdim=True)
    else:
        img = torch.fft.ifft2(kspc, dim=(-2,-1), norm='ortho')
    
    return img


#region Signal Model (non-linear MR signal encoding model respect to quantitative parameters)
def SignalEquation_T2decay(PD, T2, TE,
                           clamp=False, clampexp=0,
                           Nwdim=2):
    '''
        PD: (bsz, 1, 1, Ny, Nx)
        T2: (bsz, 1, 1, Ny, Nx)
        TE: tensor (Nw,)
        output: (bsz, 1, Nw, Ny, Nx)
    '''
    if T2.dim() == 5:
        if Nwdim == 2:
            TE = TE.view(1,1,-1,1,1)
        else:
            TE = TE.view(1,-1,1,1,1)
    else:
        TE = TE.view(1,-1,1,1)
        
    exp_term = - divide_zero(TE, T2)  # -TE/T2
    if clamp:
        exp_term = clampexp - nn.functional.relu(clampexp - exp_term)
    output = PD * torch.exp(exp_term)
    return output

class SignalModel_T2decay(nn.Module):
    '''
        Encoding system:
            y = A(x) = M * F * S * f(x)
            where: f(x) = PD * exp(-t / (T2 * T2MAX))
                   PD ~ [0, 1]
                   T2 ~ [0, 1]
        Calculate the Gradient Term:
            L(PD, T2) = ||A(x) - y||_2^2
                      = ||M * F * S * f(x) - y||_2^2
            dL / dPD = (df / dPD) * SH * FH * MH * (M * F * S * f(x) - y)
            dL / dT2 = (df / dT2) * SH * FH * MH * (M * F * S * f(x) - y)
        Derivative:
            df / dPD = exp(-t / (T2 * T2MAX))
            df / dT2 = PD * exp(-t / (T2 * T2MAX)) * (- t / T2MAX) * (- 1 / T2^2)
                     = PD * exp(-t / (T2 * T2MAX)) * (t / T2^2) * (1 / T2MAX)
                     = PD * exp(-t / (T2 * T2MAX)) * (t / (T2 * T2MAX)^2) * T2MAX
        Let:
            T2true = T2 * T2MAX
            df / dPD = exp(-t / T2true)
            df / dT2 = PD * exp(-t / T2true) * (t / T2true^2) * T2MAX
    '''
    def __init__(self, TE, T2MAX):
        # TE: tensor (Nw,)
        super().__init__()
        self.TE = TE
        self.T2MAX = T2MAX
    def CalcGradientTerm(self, x, smap, mask, kdata):
        '''
            x:         (bsz, 1, 3, Ny, Nx)
            smap:      (bsz, Nc, 1, Ny, Nx)
            mask:      (bsz, 1, Nw, Ny, Nx)
            kdata:     (bsz, Nc, Nw, Ny, Nx)
            GradTerm:  (bsz, 1, 3, Ny, Nx)
        '''
        PD = x[:, :, 0:2, :, :]
        T2 = x[:, :, 2:3, :, :]
        # map
        PDcplx = r2c(PD, axis=2)  # (bsz, 1, 1, Ny, Nx)
        T2true = T2 * self.T2MAX  # (bsz, 1, 1, Ny, Nx)
        T2true = nn.functional.relu(T2true) # clamp to positive value
        # forward
        sig = SignalEquation_T2decay(PD=PDcplx, T2=T2true, TE=self.TE)  # (bsz, 1, Nw, Ny, Nx)
        Ax = MRI_Aop(img=sig, smap=smap, mask=mask)  # (bsz, Nc, Nw, Ny, Nx)
        # adjoint
        temp = Ax - kdata
        temp = MRI_AHop(kspc=temp, smap=smap, mask=mask, dim=1)  # (bsz, 1, Nw, Ny, Nx)
        # ------ calculate derivatives
        # df / dPD = exp(-t / T2true)
        df_dPD = SignalEquation_T2decay(PD=torch.ones_like(PDcplx), T2=T2true, TE=self.TE)  # (bsz, 1, Nw, Ny, Nx)
        # df / dT2 = PD * exp(-t / T2true) * (t / T2true^2) * T2MAX
        df_dT2 = SignalEquation_T2decay(PD=PDcplx, T2=T2true, TE=self.TE) \
                 * divide_zero(self.TE.view(1,1,-1,1,1), T2true * T2true) * self.T2MAX  # (bsz, 1, Nw, Ny, Nx)
        # ------ calculate gradient (conjugate or not ?)
        Grad_PD = torch.sum(torch.conj(df_dPD) * temp, dim=2, keepdim=True)  # (bsz, 1, 1, Ny, Nx)
        Grad_T2 = torch.sum(torch.conj(df_dT2) * temp, dim=2, keepdim=True)  # (bsz, 1, 1, Ny, Nx)
        # gradient
        Grad_PD = c2r(Grad_PD, axis=2)  # (bsz, 1, 2, Ny, Nx)
        Grad_T2 = torch.abs(Grad_T2)  # (bsz, 1, 1, Ny, Nx)
        GradTerm = torch.cat([Grad_PD, Grad_T2], dim=2)  # (bsz, 1, 3, Ny, Nx)
        return GradTerm
    
    def CalcWeightedImage(self, x):
        '''
            x:         (bsz, 1, 3, Ny, Nx)
        '''
        PD = x[:, :, 0:2, :, :]
        T2 = x[:, :, 2:3, :, :]
        # map
        PDcplx = r2c(PD, axis=2)  # (bsz, 1, 1, Ny, Nx)
        T2true = T2 * self.T2MAX  # (bsz, 1, 1, Ny, Nx)
        # weighted-image
        imgW = SignalEquation_T2decay(PD=PDcplx, T2=T2true, TE=self.TE)  # (bsz, 1, Nw, Ny, Nx)
        return imgW
    
# endregion

# IR TI signal model
# f(x) = A * (1 - 2 * exp(-TI / (T1 * T2MAX)))
def SignalEquation_T1decay(A, T1, TI,
                           clamp=False, clampexp=0,
                           Nwdim=2):
    '''
        PD: (bsz, 1, 1, Ny, Nx)
        T2: (bsz, 1, 1, Ny, Nx)
        TE: tensor (Nw,)
        output: (bsz, 1, Nw, Ny, Nx)
    '''
    if T1.dim() == 5:
        if Nwdim == 2:
            TI = TI.view(1,1,-1,1,1)
        else:
            TI = TI.view(1,-1,1,1,1)
    else:
        TI = TI.view(1,-1,1,1)
        
    exp_term = -  divide_zero(TI, T1)
    if clamp:
        exp_term = clampexp - nn.functional.relu(clampexp - exp_term)
    output = A * (1 - 2 * torch.exp(exp_term))
    return output

class SignalModel_T1decay(nn.Module):
    '''
        Encoding system:
            y = H(x) = M * F * S * f(x)
            where: f(x) = A * (1 - 2 * exp(-TI / (T1 * T1MAX)))
                   A ~ [-1, 1] (c)
                   T1 ~ [0, 1]
        Calculate the Gradient Term:
            L(A, T1) = ||A(x) - y||_2^2
                      = ||M * F * S * f(x) - y||_2^2
            dL / dA = (df / dA) * SH * FH * MH * (M * F * S * f(x) - y)
            dL / dT1 = (df / dT1) * SH * FH * MH * (M * F * S * f(x) - y)
        Derivative:
            df / dA = 1 - 2 * exp(-TI / T1 * T1MAX)
            df / dT1 = -2 * A * exp(-TI / T1 * T1MAX) * (-TI / T1MAX) * (-1 / T1^2)
                     = -2 * A * exp(-TI / T1 * T1MAX) * (TI / T1^2) * (1 / T1MAX)
                     = -2 * A * exp(-TI / T1 * T1MAX) * (TI / (T1 * T1MAX)^2) * T1MAX
        Let:
            T1true = T1 * T1MAX
            df / dA = 1 - 2 * exp(-TI / T1true)
            df / dT1 = -2 * A * exp(-TI / T1true) * (TI / T1true^2) * T1MAX
    '''
    def __init__(self, TI, T1MAX):
        # TE: tensor (Nw,)
        super().__init__()
        self.TI = TI
        self.T1MAX = T1MAX
        
    def CalcGradientTerm(self, x, smap, mask, kdata):
        '''
            x:         (bsz, 1, 3, Ny, Nx)
            smap:      (bsz, Nc, 1, Ny, Nx)
            mask:      (bsz, 1, Nw, Ny, Nx)
            kdata:     (bsz, Nc, Nw, Ny, Nx)
            GradTerm:  (bsz, 1, 2, Ny, Nx)
        '''
        A = x[:, :, 0:2, :, :]
        T1 = x[:, :, 2:3, :, :]
        # map
        Acplx = r2c(A, axis=2)
        T1true = T1 * self.T1MAX
        T1true = nn.functional.relu(T1true)
        # forward
        sig = SignalEquation_T1decay(A=Acplx, T1=T1true, TI=self.TI)
        Ax = MRI_Aop(img=sig, smap=smap, mask=mask)
        # adjoint
        temp = Ax - kdata
        temp = MRI_AHop(kspc=temp, smap=smap, mask=mask, dim=1)
        # ------ calculate derivatives
        # df / dA = 1 - 2 * exp(-TI / T1true)
        df_dA = SignalEquation_T1decay(A=torch.ones_like(Acplx), T1=T1true, TI=self.TI)
        # df / dT1 = - 2 * A * exp(-TI / T1true) * (TI / T1true^2) * T1MAX
        df_dT1 = - 2 * Acplx * torch.exp(-divide_zero(self.TI.view(1,1,-1,1,1), T1true)) *\
                    divide_zero(self.TI.view(1,1,-1,1,1), T1true * T1)
        # ------ calculate gradient (conjugate or not ?)
        Grad_A = torch.sum(torch.conj(df_dA) * temp, dim=2, keepdim=True)
        Grad_T1 = torch.sum(torch.conj(df_dT1) * temp, dim=2, keepdim=True)
        # gradient
        Grad_A = c2r(Grad_A, axis=2)
        Grad_T1 = torch.abs(Grad_T1)
        GradTerm = torch.cat([Grad_A, Grad_T1], dim=2)
        
        return GradTerm
    
    def CalcWeightedImage(self, x):
        '''
            x:         (bsz, 1, 3, Ny, Nx)
        '''
        PD = x[:, :, 0:2, :, :]
        T1 = x[:, :, 2:3, :, :]
        # map
        PDcplx = r2c(PD, axis=2)  # (bsz, 1, 1, Ny, Nx)
        T1true = T1 * self.T1MAX  # (bsz, 1, 1, Ny, Nx)
        # weighted-image
        imgW = SignalEquation_T1decay(PDcplx, T1=T1true, TI=self.TI)
        return imgW