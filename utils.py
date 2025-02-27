from math import e
import numpy as np
import torch
import sys
import random
import yaml
import torch.nn.functional as F
import os
from types import SimpleNamespace
from torch.nn import Module
from torch import Tensor
from zmq import has

from models.unet import Unet
from models.unroll import MoDL
from models.dopamine import Dopamine_Net
from models.fcn import FCN
from models.crnn import CRNN_MRI

def get_args(config_path):
    with open(config_path, 'r') as f:
        args_all = yaml.safe_load(f)
        
    args_data = args_all['data']
    args_model = args_all['model']
    args_training = args_all['train']
    args_sl = args_all['sl']
    args = SimpleNamespace(**args_data, **args_model, **args_training, **args_sl)
    

    # default network setting
    if not hasattr(args, 'conv'):
        args.conv = False
    if not hasattr(args, 'norm'):
        args.norm = None
    if not hasattr(args, 'sigmoid'):
        args.sigmoid = True
    if not hasattr(args, 'shortcut'):
        args.shortcut = True
    if not hasattr(args, 'cat'):
        args.cat = True
        
    # default uncertainty setting for recon
    if not hasattr(args, 'varhead'):
        args.varhead = False
        
    if not hasattr(args, 'var_clip'):
        args.var_clip = False
        args.s_min = -10
        args.s_max = 10
        
    if not hasattr(args, 'delta'):
        args.delta = 1
        
    return args, args_model
    
def get_model(args):
    if args.model_name == 'MANTIS':
        model = Unet(in_ch=len(args.TEs)*args.n_coils*2, out_ch=2,
                     dropout_rate=args.dropout, sigmoid=True)
        
    elif args.model_name == 'Dopamine':
        model = Dopamine_Net(Nw=len(args.TEs), ch=args.hidden,
                             Nconv=args.Nconv, Niter=args.Niter,
                             lambda_k=args.lambda_k, mu_k=args.mu_k)
        
    elif args.model_name == 'crnn':
        model = CRNN_MRI(n_ch=len(args.TEs)*2, nf=args.hidden, nc=args.nc, nd=args.nd)
        
    elif args.model_name == 'unroll':
        model = MoDL(n_channels=len(args.TEs)*2, n_layers=args.n_layers, n_hiddens=args.n_hiddens,
                    k_iters=args.k_iters, dropout_rate=args.dropout,
                    res=args.res, varhead=args.varhead,)
        
    elif args.model_name == 'convnet':
        in_ch = len(args.TEs)
        model = ConvNet(in_ch=in_ch, out_ch=2, hidden=args.hidden,
                        dropout_rate=args.dropout, guidetype=args.guidetype,
                        guide_layers=(args.guide_layers if hasattr(args, 'guide_layers') else None),
                        input_attention=args.input_attention,
                        conv=args.conv, sigmoid=args.sigmoid, norm=args.norm,
                        shortcut=args.shortcut, cat=args.cat)
        
    elif args.model_name == 'unet':
        if args.state == 'fitting':
            model = Unet(in_ch=len(args.TEs), out_ch=2,
                        dropout_rate=args.dropout, sigmoid=args.sigmoid,)
        else:
            model = Unet(in_ch=len(args.TEs)*args.n_coils*2, out_ch=len(args.TEs),
                        dropout_rate=args.dropout, sigmoid=args.sigmoid,)
            
    elif args.model_name == 'fcn':
        if args.pipe_type in ['gt', 'model']:
            in_ch = len(args.TEs)
        elif args.pipe_type == 'model_uq':
            in_ch = len(args.TEs)*2
        else:
            print('Unknown pipe type!')
            raise(ValueError)
        model = FCN(in_ch=in_ch, out_ch=1,
                    hidden_ch=args.hidden_ch, num_layers=args.num_layers,
                    sigmoid=(args.sigmoid if hasattr(args, 'sigmoid') else False),)
    else:
        print('Unknown model!')
        raise(ValueError)

    return model

def get_optimizer(model, args):
    if args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.LR, alpha=0.99,
                                        eps=1e-08, weight_decay=args.weight_decay,
                                        momentum=args.momentum, centered=False)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, betas=(0.9, 0.999),
                                        eps=1e-08, weight_decay=args.weight_decay,
                                        amsgrad=False)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.LR, momentum=args.momentum,
                                    dampening=0, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, betas=(0.9, 0.999),
                                         eps=1e-08, weight_decay=args.weight_decay,
                                         amsgrad=False)
    else:
        raise(ValueError)
    return optimizer


def get_savepath(args, args_model):
    if args.model_name == 'fcn':
        if len(args.save_sub_dir) == 0:
            save_sub_dir = args.model_name
            for key in args_model:
                if key != 'model_name':
                    save_sub_dir = save_sub_dir + '__' + key + '_' + str(args_model[key])
        else:
            save_sub_dir = args.save_sub_dir
            
        if len(args.save_label) == 0:
            save_label = os.path.basename(os.path.dirname(args.data_dir)) + '_' +\
                os.path.basename(args.data_dir) + '__'
            save_label += '__' + str(args.optimizer) +\
                '__' + 'LR' + '_' + str(args.LR) + \
                '__' + 'wd' + '_' + str(args.weight_decay) +\
                ('_' + 'clip' + str(args.grad_max_norm) if args.grad_max_norm > 0 else '')+\
                '__' + 'epoch' + '_' + str(args.epoches)
                
        else:
            save_label = args.save_label
            
        return save_sub_dir, save_label

    # model save_setting
    if len(args.save_sub_dir) == 0:
        save_sub_dir = args.model_name
        for key in args_model:
            if key != 'model_name':
                save_sub_dir = save_sub_dir + '__' + key + '_' + str(args_model[key])
    else:
        save_sub_dir = args.save_sub_dir

    if len(args.save_label) == 0:
        save_label = str(args.data_type) + '_' + str(args.mapping) + \
            '__' + str(args.mask_type) + '_' + str(args.sampling_factor) + \
            '__' + str(args.optimizer) +\
            '__' + 'LR' + '_' + str(args.LR) + \
            '__' + 'wd' + '_' + str(args.weight_decay) +\
            ('__' + 'schedule' + '_' + str(args.scheduler) if hasattr(args, 'scheduler') else '') +\
            ('_' + 'clip' + str(args.grad_max_norm) if args.grad_max_norm > 0 else '')+\
            ('__' + 'N' + '_' + str(args.sampling_times) if hasattr(args, 'sampling_times') else '')+\
            ('__' + 'lm' + '_' + str(args.loss_mask) if hasattr(args, 'loss_mask') and args.loss_mask else '')+\
            ('__' + 'T2max' + '_' + str(args.T2max) if hasattr(args, 'T2_th') and args.T2_th else '')+\
            ('__' + 'beta' + '_' + str(args.beta) if hasattr(args, 'beta') and args.beta != 0 else '')+\
            ('__' + 'varclip' + '_' + str(args.s_min) + '_' + str(args.s_max) if hasattr(args, 'var_clip') and args.var_clip else '')
            
        save_label += '__' + 'loss' + '_' + str(args.loss)
        save_label += '__s' + str(args.seed)

    else:
        save_label = args.save_label
            
    return save_sub_dir, save_label

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def activate_drop(model):
    for m in model.modules():
      if m.__class__.__name__.startswith('Dropout'):
        m.train()

def save_ckpt(path, epoch, model, best_score, optimizer, scheduler=None):
    """ save current model
    """
    if scheduler is not None:
        torch.save({
            "cur_epoch": epoch,
            "model_state": model.state_dict(),
            "best_score": best_score,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, path)
    else:
        torch.save({
            "cur_epoch": epoch,
            "model_state": model.state_dict(),
            "best_score": best_score,
            "optimizer": optimizer.state_dict()
        }, path)
        
    print("Model saved as %s" % path)

def load_ckpt(save_path, model, optimizer, scheduler):
    print('Loading: ', save_path + '_last_.pth')
    checkpoint = torch.load(save_path + '_last_.pth')
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['cur_epoch'] + 1
    best_score = checkpoint['best_score']
    
    return start_epoch, best_score
        
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def var_loss(output, target, loss_type='L2', nll=False):
    if nll:
        exp_var = torch.exp(-output['apara'])
        if loss_type == 'L2':
            loss1 = torch.mul(exp_var, (output['recon_mean'] - target) ** 2)
        elif loss_type == 'L1':
            loss1 = torch.mul(exp_var, torch.abs(output['recon_mean'] - target))
        else:
            raise(ValueError)
        loss2 = output['apara']

        loss_sum = 0.5 * loss1 + 0.5 * loss2
        loss = loss_sum.mean()

    else:
        if loss_type == 'L2':
            loss = F.mse_loss(output['recon_mean'], target)
        elif loss_type == 'L1':
            loss = F.l1_loss(output['recon_mean'], target)
        else:
            raise(ValueError)
        
    return loss

class SSIMLoss(Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5) -> None:
        """Computes the structural similarity (SSIM) index map between two images.
        Args:
            kernel_size (int): Height and width of the gaussian kernel.
            sigma (float): Gaussian standard deviation in the x and y direction.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.in_ch = 2
        self.gaussian_kernel = self._create_gaussian_kernel(self.kernel_size, self.sigma)
    def forward(self, x: Tensor, y: Tensor, as_loss: bool = True) -> Tensor:
        if not self.gaussian_kernel.is_cuda:
            self.gaussian_kernel = self.gaussian_kernel.to(x.device)
        ssim_map = self._ssim(x, y)
        if as_loss:
            return 1 - ssim_map.mean()
        else:
            return ssim_map
    def _ssim(self, x: Tensor, y: Tensor) -> Tensor:
        # Compute means
        ux = F.conv2d(x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.in_ch)
        uy = F.conv2d(y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.in_ch)
        # Compute variances
        uxx = F.conv2d(x * x, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.in_ch)
        uyy = F.conv2d(y * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.in_ch)
        uxy = F.conv2d(x * y, self.gaussian_kernel, padding=self.kernel_size // 2, groups=self.in_ch)
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
        denominator = (ux ** 2 + uy ** 2 + c1) * (vx + vy + c2)
        return numerator / (denominator + 1e-12)
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> Tensor:
        start = (1 - kernel_size) / 2
        end = (1 + kernel_size) / 2
        kernel_1d = torch.arange(start, end, step=1, dtype=torch.float)
        kernel_1d = torch.exp(-torch.pow(kernel_1d / sigma, 2) / 2)
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)
        kernel_2d = torch.matmul(kernel_1d.t(), kernel_1d)
        kernel_2d = kernel_2d.expand(self.in_ch, 1, kernel_size, kernel_size).contiguous()
        return kernel_2d