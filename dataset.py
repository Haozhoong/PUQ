import numpy as np
import torch
import os
from subsample_strict import create_mask_for_mask_type
from torch.utils.data import Dataset, DataLoader
from models.model_utils import MRI_Aop

class MyDataset(Dataset):
    def __init__(self,
                 directory, center_fractions=0.08,
                 accelerations=4, mapping='T2',
                 q_th=None, q_max=1):

        fh = os.listdir(directory)
        fh.sort()
        fn = [os.path.join(directory, k) for k in fh]

        self.mask_func = create_mask_for_mask_type(mask_type_str="random", center_fractions=[center_fractions],
                                                    accelerations=[accelerations])
        self.fn = fn
        self.accelerations = accelerations
        self.q_th = q_th
        self.q_max = q_max
        
        if mapping == 'T2':
            self.qmap = 't2'
        elif mapping == 'T1':
            self.qmap = 't1'
        else:
            raise ValueError('Unknown mapping type')

    def __getitem__(self, index):
        dict = np.load(self.fn[index], allow_pickle=True).item()
        name = self.fn[index]
        pdmap = torch.from_numpy(dict['pd']).float().unsqueeze(0)
        qmap = torch.from_numpy(dict[self.qmap]).float().unsqueeze(0)
        if self.q_th is not None:
            qmap[qmap>self.q_max] = self.q_max
            qmap[qmap<0] = 0
        if 'roi' in dict.keys():
            roi = torch.from_numpy(dict['roi']).float().unsqueeze(0)
        else:
            roi = torch.ones_like(qmap)
        if np.iscomplex(dict['image']).any():
            images = torch.from_numpy(dict['image']).cfloat()
        else:
            images = torch.from_numpy(np.abs(dict['image'])).float()
        sensmap = torch.from_numpy(dict['sens']).cfloat()
        
        rng = np.random.default_rng(dict['seed'])
        sampling_seeds = rng.choice(0xffffff, images.shape[0] * sensmap.shape[0])
        mask = np.zeros((images.shape[0], images.shape[-2], images.shape[-1]))
        for i in range(mask.shape[0]):
            ss = sampling_seeds[i]
            mask_gen = self.mask_func(shape=(images.shape[-1], images.shape[-1]), seed=ss)
            mask_gen = mask_gen.repeat(images.shape[-2], 0)
            mask[i] = mask_gen

        mask = torch.from_numpy(mask).float()
        pdmap = pdmap.unsqueeze(0)
        qmap = qmap.unsqueeze(0)
        images = images.unsqueeze(1)
        sensmap = sensmap.unsqueeze(0)
        mask = mask.unsqueeze(1)
        
        mask = torch.fft.fftshift(mask, dim=(-2, -1))
        measurement = MRI_Aop(images, sensmap, mask=mask)
        
        # shape (without the batch dim)
        # qmap, pd_map: [1, 1, h, w]
        # roi: [1, h, w]
        # images: [phase, 1, h, w]
        # sensmap: [1, coil, h, w]
        # mask: [phase, 1, h, w]
        # measurement: [phase, coil, h, w]
        return {'qmap':qmap, 'pdmap':pdmap, 'images':images,
                'measurement':measurement, 'sensmap':sensmap,
                'mask':mask, 'roi':roi, 'name':name}

    def __len__(self):
        return len(self.fn)