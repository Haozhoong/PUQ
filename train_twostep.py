import argparse
from numpy import add
import torch
import torch.fft
import os
import json
import shutil
import time
from torch.nn import functional as F
from utils import *
from dataset import MyDataset
from models.model_utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def train(model, model_recon, state, train_loader, optimizer, scheduler, epoch, args, args_recon=None):
    running_loss = {'sum':0.}
    for batch_idx, datas in enumerate(train_loader):
        qmap = datas['qmap'].to(device=args.device)
        pdmap = datas['pdmap'].to(device=args.device)
        images = datas['images'].to(device=args.device)
        measurement = datas['measurement'].to(device=args.device)
        sensmap = datas['sensmap'].to(device=args.device)
        mask = datas['mask'].to(device=args.device)
        roi = datas['roi'].to(device=args.device)
        map_target = torch.cat([pdmap, qmap / args.q_max], dim=1).squeeze(2)
        
        if args.mapping == 'T1':
            images_target = images.real.squeeze(2)
        elif args.mapping == 'T2':
            images_target = torch.abs(images).squeeze(2)
        else:
            print('Unknown mapping type')
            raise(ValueError)
        
        if state == 'recon':
            output = unroll_recon_forward(model, images, measurement,
                                            sensmap, mask, 1, args, grad=True)
            loss = var_loss(output, images_target, args.loss, args.varhead)

        if state == 'fitting':
            output_recon = unroll_recon_forward(model_recon, images, measurement,
                                                sensmap, mask, args.sampling_times,
                                                args_recon, delta=args.delta, grad=False)
            map_out = model(output_recon['recon_mean'])
            
            if hasattr(args, 'loss_mask') and args.loss_mask:
                loss = F.mse_loss(map_out*roi, map_target*roi)
            else:
                loss = F.mse_loss(map_out, map_target)
        
        running_loss['sum'] += loss.item()
            
        optimizer.zero_grad()
        loss.backward()
        if args.grad_max_norm > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_max_norm, norm_type=2)
        optimizer.step()
        
        if batch_idx % args.vis_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: sum {:.12f}'.format(
                state, epoch, batch_idx * qmap.shape[0], len(train_loader),
                100. * batch_idx / len(train_loader), loss.data))
            
    if scheduler is not None:
        scheduler.step()
    for key in running_loss.keys():
        running_loss[key] = running_loss[key] / len(train_loader)
    print(state, ' Train set: Average loss: ', running_loss)
    
    return running_loss

def val(model, model_recon, state, val_loader, args, args_recon=None):
    running_loss = {'sum': 0.}
    if state == 'recon':
        criterion_output = {"nmse": 0., "psnr": 0., "ssim":0.}
    elif state == 'fitting':
        criterion_output = {"nmse_q": 0., "nmse_pd": 0.}
        
    for batch_idx, datas in enumerate(val_loader):
        with torch.no_grad():
            qmap = datas['qmap'].to(device=args.device)
            pdmap = datas['pdmap'].to(device=args.device)
            images = datas['images'].to(device=args.device)
            measurement = datas['measurement'].to(device=args.device)
            sensmap = datas['sensmap'].to(device=args.device)
            mask = datas['mask'].to(device=args.device)
            roi = datas['roi'].to(device=args.device)
            map_target = torch.cat([pdmap, qmap / args.q_max], dim=1).squeeze(2)
            
            if args.mapping == 'T1':
                images_target = images.real.squeeze(2)
            elif args.mapping == 'T2':
                images_target = torch.abs(images).squeeze(2)
            else:
                print('Unknown mapping type')
                raise(ValueError)
            
            if state == 'recon':
                output = unroll_recon_forward(model, images, measurement,
                                                sensmap, mask, 1, args, grad=False)
                loss = var_loss(output, images_target, args.loss, args.varhead)
            
            if state == 'fitting':
                output_recon = unroll_recon_forward(model_recon, images, measurement,
                                                    sensmap, mask, args.sampling_times,
                                                    args_recon, delta=args.delta, grad=False)
                map_out = model(output_recon['recon_mean'])

                if hasattr(args, 'loss_mask') and args.loss_mask:
                    map_out = map_out * roi
                    map_target = map_target * roi 

                loss = F.mse_loss(map_out, map_target)
                
            running_loss['sum'] += loss.item()

        if state == 'recon':
            full_i = images_target[0, 0].cpu().numpy()
            out_i = output['recon_mean'][0, 0].cpu().numpy()
            mse = np.mean((out_i - full_i) ** 2)
            norms = np.mean(full_i ** 2)
            criterion_output["nmse"] += mse / norms
            criterion_output["psnr"] += peak_signal_noise_ratio(out_i, full_i, data_range=1)
            criterion_output["ssim"] += structural_similarity(out_i, full_i, win_size=11, channel_axis=0, data_range=1)
            
        elif state == 'fitting':
            qmap_target = map_target[0, 1].cpu().numpy() * args.q_max
            pdmap_target = map_target[0, 0].cpu().numpy()
            qmap_output = map_out[0, 1].cpu().numpy() * args.q_max
            pdmap_output = map_out[0, 0].cpu().numpy()
            mse = np.mean((qmap_output - qmap_target) ** 2)
            norms = np.mean(qmap_target ** 2)
            criterion_output["nmse_q"] += mse / norms
            mse = np.mean((pdmap_output - pdmap_target) ** 2)
            norms = np.mean(pdmap_target ** 2)
            criterion_output["nmse_pd"] += mse / norms
            
    for key in running_loss.keys():
        running_loss[key] = running_loss[key] / len(val_loader)
    for key in criterion_output.keys():
        criterion_output[key] = criterion_output[key] / len(val_loader)
        
    print(state, 'Test set: Average loss: ', running_loss, ' ', criterion_output)
    
    return criterion_output, running_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training command parser', add_help=False)
    parser.add_argument('-c', '--config_path', default='./config/MANTIS.yaml',
                        type=str, help='config path')
    cmd_par =  parser.parse_args()
    config_path = cmd_par.config_path
    args, args_model = get_args(config_path)
    
    if args.state == "fitting":
        args_recon, args_model_recon = get_args(args.recon_model_path)
        save_sub_dir, save_label = get_savepath(args, args_model)
        save_sub_dir_recon, save_label_recon = get_savepath(args_recon, args_model_recon)
        save_path_recon = os.path.join(args.save_dir, save_sub_dir_recon, save_label_recon)
        save_path = os.path.join(args.save_dir, save_sub_dir + '_recon_' + save_sub_dir_recon,
                                 save_label + '_recon_' + save_label_recon)
        
        model_recon = get_model(args_recon).to(args.device)
        checkpoint = torch.load(save_path_recon + '_best_.pth')
        model_recon.load_state_dict(checkpoint['model_state'])
        model_recon.eval()
        activate_drop(model_recon)
        
        bset_score_criterion = "nmse_q"
    else:
        model_recon = None
        args_recon = None
        bset_score_criterion = "nmse"
        save_sub_dir, save_label = get_savepath(args, args_model)
        save_path = os.path.join(args.save_dir, save_sub_dir, save_label)
        
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    print('Export path: ', save_path)
    shutil.copyfile(config_path, save_path + '.yaml')
    
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
                                               batch_size=args.batch_size,
                                               num_workers=16,
                                               worker_init_fn=seed_worker,
                                               generator=g,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=1,
                                             num_workers=16,
                                             worker_init_fn=seed_worker,
                                             generator=g,
                                             shuffle=False)

    model = get_model(args).to(args.device)
    optimizer = get_optimizer(model, args)
    scheduler = None
    
    start_epoch = 0
    best_score = 10000000000000.
    if args.resume_training:
        start_epoch, best_score = load_ckpt(save_path, model, optimizer, scheduler)
        
    for epoch in range(start_epoch, args.epoches):
        epoch_start = time.time()
        model.train()
        loss_train = train(model, model_recon, args.state, train_loader, optimizer, scheduler, epoch, args, args_recon)
        model.eval()
        criterion, loss_val = val(model, model_recon, args.state, val_loader, args, args_recon)
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