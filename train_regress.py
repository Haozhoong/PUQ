import torch
import numpy as np
import os
import time
import json
import argparse
import shutil
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
from utils import get_args, get_model, get_optimizer,\
    seed_worker, set_random_seed, load_ckpt, save_ckpt, get_savepath

class MyDataset(Dataset):
    def __init__(self, directory,
                 state='train', image_th=1e-3,
                 map_th=2, var_norm=1, gt=False):
        # load
        image_dir = os.path.join(directory, f'{state}_images.pt')
        images = torch.load(image_dir)
        map_dir = os.path.join(directory, f'{state}_maps.pt')
        maps = torch.load(map_dir)
        
        # real
        if gt:
            images = images.real

        # remove the images too small
        non_image_idx = abs(images[0, :]) < image_th
        images = images[:, ~non_image_idx]
        maps = maps[0, ~non_image_idx]
        # clip the qmap
        maps[maps > map_th] = map_th
        maps[maps < 0] = 0
        
        if var_norm != 1:
            phases = images.shape[0] // 2
            images[phases:, :] = images[phases:, :] * var_norm
        
        self.images = images
        self.maps = maps

    def __getitem__(self, index):
        datamap = self.maps[index].unsqueeze(0)
        sig = self.images[:, index]
        
        # normalize
        norm_factor = sig[0]
        sig = sig / norm_factor

        return {'datamap':datamap, 'sig':sig}
    
    def __len__(self):
        return self.images.shape[1]

def train(model, train_loader, optimizer, epoch, args):
    running_loss = {'sum':0.}
    for batch_idx, datas in enumerate(train_loader):
        sig = datas['sig'].to(args.device)
        datamap = datas['datamap'].to(args.device)
        if args.sigmoid:
            datamap = datamap / args.q_max
        
        if args.model_name == 'fcn':
            map_pred = model(sig)
            loss = F.mse_loss(map_pred, datamap)
            running_loss['sum'] += loss.item()
        else:
            raise ValueError('Unknown model name')
        
        optimizer.zero_grad()
        loss.backward()
        if args.grad_max_norm > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_max_norm, norm_type=2)
        optimizer.step()
        
        if batch_idx % args.vis_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: sum {:.12f}'.format(
                epoch, batch_idx * sig.shape[0], len(train_loader) * sig.shape[0],
                100. * batch_idx / len(train_loader), loss.data))
            
    for key in running_loss.keys():
        running_loss[key] = running_loss[key] / len(train_loader)
    print(' Train set: Average loss: ', running_loss)
    
    return running_loss

def val(model, val_loader, epoch, args):
    running_loss = {'sum':0.}
    criterion_output = {'nmse':0.}
    
    norm_counts = 0
    for batch_idx, datas in enumerate(val_loader):
        sig = datas['sig'].to(args.device)
        datamap = datas['datamap'].to(args.device)
        if args.sigmoid:
            datamap = datamap / args.q_max
        
        with torch.no_grad():
            if args.model_name == 'fcn':
                map_pred = model(sig)
                loss = F.mse_loss(map_pred, datamap)
                running_loss['sum'] += loss.item()
            else:
                raise ValueError('Unknown model name')
            
            if args.sigmoid:
                datamap = datamap * args.q_max
                map_pred = map_pred * args.q_max
            
            mse = np.mean((map_pred.cpu().numpy() - datamap.cpu().numpy()) ** 2)
            norms = np.mean(datamap.cpu().numpy() ** 2)
            
            if norms != 0:
                criterion_output['nmse'] +=  mse / norms
                norm_counts += 1 
        
    for key in running_loss.keys():
        running_loss[key] = running_loss[key] / len(val_loader)
    for key in criterion_output.keys():
        criterion_output[key] = criterion_output[key] / norm_counts
    print('Test set: Average loss: ', running_loss, ' ', criterion_output)
    
    return running_loss, criterion_output
            

def main():
    parser = argparse.ArgumentParser('Training command parser', add_help=False)
    parser.add_argument('-c', '--config_path',
                        default='/data/smart/shz/project/uq/config/exp_fcn/t2_ai_10_c115.yaml',
                        type=str, help='config path')
    cmd_par =  parser.parse_args()
    config_path = cmd_par.config_path
    args, args_model = get_args(config_path)
    save_sub_dir, save_label = get_savepath(args, args_model)
    save_path = os.path.join(args.save_dir, save_sub_dir, save_label)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    shutil.copyfile(config_path, save_path + '.yaml')
    
    bset_score_criterion = 'nmse'
    
    random_seed = args.seed
    set_random_seed(random_seed, True)

    img_th = 1e-3
    #img_th = 1e-1 if args.data_type == 'phantom' else 1e-3
    train_dataset = MyDataset(args.data_dir, 'train', map_th=args.q_max, var_norm=1, image_th=img_th, gt=args.pipe_type=='gt')
    val_dataset = MyDataset(args.data_dir, 'val', map_th=args.q_max, var_norm=1, image_th=img_th, gt=args.pipe_type=='gt')
    g = torch.Generator()
    g.manual_seed(random_seed)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=16,
                                               worker_init_fn=seed_worker,
                                               generator=g,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=16,
                                             worker_init_fn=seed_worker,
                                             generator=g,
                                             shuffle=False)
    
    # get model
    model = get_model(args)
    model = model.to(args.device)

    # loss function
    optimizer = get_optimizer(model, args)
    scheduler = None
    
    start_epoch = 0
    best_score = 10000000000000.
    if args.resume_training:
        start_epoch, best_score = load_ckpt(save_path, model, optimizer, scheduler)
        
    for epoch in range(start_epoch, args.epoches):
        epoch_start = time.time()
        model.train()
        loss_train = train(model, train_loader, optimizer, epoch, args)
        model.eval()
        loss_val, criterion = val(model, val_loader, epoch, args)
        loss_val_save = {l+"_val": loss_val[l] for l in loss_val.keys()}
        
        save_dict = {'epoch':epoch, **criterion, **loss_train, **loss_val_save}
        f = open(save_path + '_metric.txt', 'a')
        f.write(json.dumps(save_dict))
        f.write('\n')
        f.close()
        if criterion[bset_score_criterion] < best_score:  # save best model
            best_score = criterion[bset_score_criterion]
            save_ckpt(save_path + '_best_.pth', epoch, model, best_score, optimizer, scheduler)
        print(f"time cost: {time.time()-epoch_start} s")
        save_ckpt(save_path + '_last_.pth', epoch, model, best_score, optimizer, scheduler)
        print(f"time cost: {time.time()-epoch_start} s")

    torch.save(model.state_dict(), save_path + '_alllast_.pth')
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Script interrupted by user. Exiting...")