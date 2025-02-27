import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import os
from models.model_utils import *
from dataset import MyDataset
from utils import *
from train_twostep import unroll_recon_forward

# prepare data recon
parser = argparse.ArgumentParser('Training command parser', add_help=False)
parser.add_argument('-c', '--config_path', default='/data/smart/shz/project/uq/config/exp_t2_cov/beta0.25_1e-2_10.yaml',
                    type=str, help='config path')
#/data/smart/shz/project/uq/config/exp_t2_cov/baseline_t2_10.yaml
parser.add_argument('-t', '--pipe_type', default='model', type=str, help='pipeline type')
parser.add_argument('-d', '--data_type', default='t2', type=str, help='data type')
parser.add_argument('-ut', '--uq_type', default='evar', type=str, help='uq type')
parser.add_argument('-s', '--s_time', default=1, type=int, help='sampling time')

cmd_par =  parser.parse_args()

sample_times = cmd_par.s_time
uq_type = cmd_par.uq_type
pipe_type = cmd_par.pipe_type
config_path = cmd_par.config_path
data_type = cmd_par.data_type

normalized_path = os.path.normpath(config_path)
parent_path, last_part = os.path.split(normalized_path)
grandparent_path, parent_part = os.path.split(parent_path)
save_config_path = parent_part + '_slash_' + last_part

args, args_model = get_args(config_path)

save_sub_dir, save_label = get_savepath(args, args_model)
random_seed = args.seed
set_random_seed(random_seed, True)
root_dir = args.root_dir
train_dataset = MyDataset(directory=os.path.join(root_dir, 'train'), q_th=args.q_th, q_max=args.q_max,
                            center_fractions=args.center_fractions, accelerations=args.sampling_factor,
                            mapping=args.mapping)   

val_dataset = MyDataset(directory=os.path.join(root_dir, 'val'), q_th=args.q_th, q_max=args.q_max,
                        center_fractions=args.center_fractions, accelerations=args.sampling_factor,
                        mapping=args.mapping)
g = torch.Generator()
g.manual_seed(random_seed)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=1,
                                            num_workers=16,
                                            shuffle=False)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                        batch_size=1,
                                        num_workers=16,
                                        shuffle=False)
# three pipelines
# 1. use data point from the ground truth
# 2. use data point from the model without uq
# 3. use data point from the model with uq
export_path = '/data/smart/shz/dataset/brainmapping_pixels_' + data_type
export_sub = f'pixels_{pipe_type}_down_{args.sampling_factor}'
if sample_times > 1:
    export_sub += f'_s{sample_times}'
if pipe_type in ['model', 'model_uq']:
    export_sub += '_' + save_config_path
export_folder = os.path.join(export_path, export_sub)
if not os.path.exists(export_folder):
    os.makedirs(export_folder)
    
if pipe_type in ['model', 'model_uq']:
    model = get_model(args)
    model = model.to(args.device)
    save_path = os.path.join(args.save_dir, save_sub_dir, save_label)
    model.load_state_dict(torch.load(save_path + '_best_.pth')["model_state"])
    model.eval()
    if sample_times > 1:
        activate_drop(model)
else:
    model = None

def save_pixel_data(data_loader, export_base_folder,
                    state='train', pipe_type='gt',
                    model_recon=None, args=None):
    if not os.path.exists(export_base_folder):
        os.makedirs(export_base_folder)
    pixels_count = 0
    
    all_images = []
    all_maps = []
    for batch_idx, datas in enumerate(data_loader):
        images = datas['images']
        qmap = datas['qmap']
        pdmap = datas['pdmap']
        roi = datas['roi']
        name = datas['name'][0]
        if pipe_type == 'gt':
            images = images.squeeze()
        elif pipe_type in ['model', 'model_uq']:
            measurement = datas['measurement']
            sensmap = datas['sensmap']
            mask = datas['mask']
            output_recon = unroll_recon_forward(model_recon, images.to(device=args.device),
                                                measurement.to(device=args.device),
                                                sensmap.to(device=args.device),
                                                mask.to(device=args.device), sample_times,
                                                args, delta=1, grad=False)

            if pipe_type == 'model':
                images = output_recon['recon_mean'].cpu().squeeze()
            else:
                mean = output_recon['recon_mean']
                if uq_type == 'avar':
                    uq = output_recon['avar']
                elif uq_type == 'evar':
                    uq = output_recon['evar']
                else:
                    raise ValueError('Unknown uq type')

                images = torch.cat([mean, uq], dim=1).cpu().squeeze()
                
        qmap = qmap.squeeze()
        pdmap = pdmap.squeeze()
        roi = roi.squeeze().bool()
        #print(images.size(), qmap.size(), pdmap.size())
        images = images[:, roi]
        qmap = qmap[roi.expand_as(qmap)]
        pdmap = pdmap[roi.expand_as(pdmap)]
        #print(images.size(), qmap.size(), pdmap.size())
        images = images.view(images.size(0), -1)
        qmap = qmap.view(-1)
        pdmap = pdmap.view(-1)
        data_map = torch.stack([qmap, pdmap], dim=0)
        pixels_count += data_map.size(-1)
        
        all_images.append(images)
        all_maps.append(data_map)
        print(f'{pixels_count} pixels saved in', name)
            
    # save all images and maps
    all_images = torch.cat(all_images, dim=-1)
    all_maps = torch.cat(all_maps, dim=-1)
    torch.save(all_images, os.path.join(export_base_folder, f'{state}_images.pt'))
    torch.save(all_maps, os.path.join(export_base_folder, f'{state}_maps.pt'))
            
    '''
    for i in range(images.size(-1)):
        data = images[:, i]
        data_map = torch.stack([qmap[i], pdmap[i]], dim=0)
        pixels_count += 1
        data_name = os.path.join(export_base_folder, os.path.basename(name)[:-4] + f'_{i:06d}.npy')
        print(f'data saved in', data_name)
        #np.save(data_name, {'data': data.numpy(), 'map': data_map.numpy()})
        break
    '''

save_pixel_data(train_loader, export_folder, 'train', pipe_type, model, args)
save_pixel_data(val_loader, export_folder, 'val', pipe_type, model, args)
print(f'All pixels saved in {export_folder}')