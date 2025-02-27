import argparse
from turtle import pd
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

def train(model, train_loader, optimizer, scheduler, epoch, args):
    # setting loss recorder
    if args.model_name in ['Dopamine']:
        running_loss = {'x0': 0., 'out':0., 'sum':0.}
        ssim_loss_f = SSIMLoss()
    elif args.model_name in ['MANTIS']:
        running_loss = {'map': 0., 'measure':0., 'sum':0.}
        
    for batch_idx, datas in enumerate(train_loader):
        qmap = datas['qmap'].to(device=args.device)
        pdmap = datas['pdmap'].to(device=args.device)
        images = datas['images'].to(device=args.device)
        measurement = datas['measurement'].to(device=args.device)
        sensmap = datas['sensmap'].to(device=args.device)
        mask = datas['mask'].to(device=args.device)
        roi = datas['roi'].to(device=args.device)
        TE_t = torch.tensor(args.TEs, device=qmap.device)
        map_target = torch.cat([pdmap, qmap / args.q_max], dim=1).squeeze(2)

        if args.model_name in ['Dopamine']:
            images_zf = torch.fft.ifft2(measurement, dim=(-2,-1), norm='ortho') #(B, phase, coils, H, W)
            images_zf_combine = torch.sum(images_zf * sensmap.conj(), dim=2) #(B, phase, H, W)
            images_zf_combine = images_zf_combine.unsqueeze(1)
            sgmodel = SignalModel_T2decay(TE_t, args.q_max).to('cuda')
            
            sensmap = torch.permute(sensmap, (0,2,1,3,4))
            mask = torch.permute(mask, (0,2,1,3,4))
            measurement = torch.permute(measurement, (0,2,1,3,4))
            map_x0, map_out = model(imgW_init=images_zf_combine, signal_model=sgmodel,
                                smaps=sensmap, masks=mask, kdatas=measurement)
            map_out = torch.cat([map_out[:,:1], map_out[:,2:]], dim=1)
            image_x0 = sgmodel.CalcWeightedImage(x=map_x0.unsqueeze(1)).squeeze(1)
            # compute loss
            x0_loss = F.l1_loss(c2r(image_x0, axis=1), c2r(images_zf_combine.squeeze(1), axis=1))
            if hasattr(args, 'loss_mask') and args.loss_mask:
                out_loss = F.l1_loss(map_out * roi, map_target * roi)
                ssim_loss = ssim_loss_f(map_out * roi, map_target * roi)
            else:
                out_loss = F.l1_loss(map_out, map_target)
                ssim_loss = ssim_loss_f(map_out, map_target)
            loss = x0_loss + out_loss + 0.5 * ssim_loss
            
            running_loss['sum'] += loss.item()
            running_loss['x0'] += x0_loss.item()
            running_loss['out'] += out_loss.item()
            
            if batch_idx % args.vis_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: sum {:.6f} / x0 {:.6f} / out {:.6f} / ssim {:.6f}'.format(
                    epoch, batch_idx , len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, x0_loss.data, out_loss.data, ssim_loss.data))
            
        elif args.model_name in ['MANTIS']:
            images_zf = torch.fft.ifft2(measurement, dim=(-2,-1), norm='ortho')
            images_zf = images_zf.view([images_zf.shape[0], -1, images_zf.shape[-2], images_zf.shape[-1]])
            images_zf = c2r(images_zf, axis=1)

            map_out = model.forward(images_zf)
            image_x0 = SignalEquation_T2decay(PD=map_out[:,:1].unsqueeze(2),
                                              T2=map_out[:,1:].unsqueeze(2) * args.q_max,
                                              TE=TE_t, Nwdim=1)
            kspace_out = MRI_Aop(image_x0, sensmap, mask=mask)

            if hasattr(args, 'loss_mask') and args.loss_mask:
                map_loss = F.mse_loss(map_out * roi, map_target * roi)
            else:
                map_loss = F.mse_loss(map_out, map_target)
            measure_loss = F.mse_loss(torch.view_as_real(measurement), torch.view_as_real(kspace_out))
            loss = map_loss + measure_loss * args.lambda_k
            
            running_loss['sum'] += loss.item()
            running_loss['measure'] += measure_loss.item()
            running_loss['map'] += map_loss.item()
            
            if batch_idx % args.vis_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: sum {:.6f} / map {:.6f} / measure {:.6f}'.format(
                    epoch, batch_idx , len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, map_loss.data, measure_loss.data))
        else:
            raise(ValueError)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_max_norm > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_max_norm, norm_type=2)
        optimizer.step()

    for key in running_loss.keys():
        running_loss[key] = running_loss[key] / len(train_loader)
    print('Train set: Average loss: ', running_loss)
    
    return running_loss

def val(model, val_loader, args):
    if args.model_name in ['Dopamine']:
        running_loss = {'x0': 0., 'out':0., 'sum':0.}
        ssim_loss_f = SSIMLoss()
    elif args.model_name in ['MANTIS']:
        running_loss = {'map': 0., 'measure':0., 'sum':0.}
    criterion_output = {"nmse_t2": 0., "nmse_pd": 0.}
    
    for batch_idx, datas in enumerate(val_loader):
        with torch.no_grad():
            qmap = datas['qmap'].to(device=args.device)
            pdmap = datas['pdmap'].to(device=args.device)
            images = datas['images'].to(device=args.device)
            measurement = datas['measurement'].to(device=args.device)
            sensmap = datas['sensmap'].to(device=args.device)
            mask = datas['mask'].to(device=args.device)
            roi = datas['roi'].to(device=args.device)
            TE_t = torch.tensor(args.TEs, device=qmap.device)
            map_target = torch.cat([pdmap, qmap / args.q_max], dim=1).squeeze(2)
            
            if args.model_name in ['Dopamine']:
                images_zf = torch.fft.ifft2(measurement, dim=(-2,-1), norm='ortho') #(B, phase, coils, H, W)
                images_zf_combine = torch.sum(images_zf * sensmap.conj(), dim=2) #(B, phase, H, W)
                images_zf_combine = images_zf_combine.unsqueeze(1)
                sgmodel = SignalModel_T2decay(TE_t, args.q_max).to('cuda')
                
                sensmap = torch.permute(sensmap, (0,2,1,3,4))
                mask = torch.permute(mask, (0,2,1,3,4))
                measurement = torch.permute(measurement, (0,2,1,3,4))
                map_x0, map_out = model(imgW_init=images_zf_combine, signal_model=sgmodel,
                                    smaps=sensmap, masks=mask, kdatas=measurement)
                map_out = torch.cat([map_out[:,:1], map_out[:,2:]], dim=1)
                image_x0 = sgmodel.CalcWeightedImage(x=map_x0.unsqueeze(1)).squeeze(1)
                # compute loss
                x0_loss = F.l1_loss(c2r(image_x0, axis=1), c2r(images_zf_combine.squeeze(1), axis=1))
                if hasattr(args, 'loss_mask') and args.loss_mask:
                    out_loss = F.l1_loss(map_out * roi, map_target * roi)
                    ssim_loss = ssim_loss_f(map_out * roi, map_target * roi)
                else:
                    out_loss = F.l1_loss(map_out, map_target)
                    ssim_loss = ssim_loss_f(map_out, map_target)
                loss = x0_loss + out_loss + 0.5 * ssim_loss
                
                running_loss['sum'] += loss.item()
                running_loss['x0'] += x0_loss.item()
                running_loss['out'] += out_loss.item()
                
            elif args.model_name in ['MANTIS']:
                images_zf = torch.fft.ifft2(measurement, dim=(-2,-1), norm='ortho')
                images_zf = images_zf.view([images_zf.shape[0], -1, images_zf.shape[-2], images_zf.shape[-1]])
                images_zf = c2r(images_zf, axis=1)
                
                map_out = model.forward(images_zf)
                image_x0 = SignalEquation_T2decay(PD=map_out[:,:1].unsqueeze(2),
                                                T2=map_out[:,1:].unsqueeze(2) * args.q_max,
                                                TE=TE_t, Nwdim=1)
                kspace_out = MRI_Aop(image_x0, sensmap, mask=mask)

                if hasattr(args, 'loss_mask') and args.loss_mask:
                    map_loss = F.mse_loss(map_out * roi, map_target * roi)
                else:
                    map_loss = F.mse_loss(map_out, map_target)
                measure_loss = F.mse_loss(torch.view_as_real(measurement), torch.view_as_real(kspace_out))
                loss = map_loss + measure_loss * args.lambda_k
                running_loss['sum'] += loss.item()
                running_loss['measure'] += measure_loss.item()
                running_loss['map'] += map_loss.item()
            else:
                raise(ValueError)

            if hasattr(args, 'loss_mask') and args.loss_mask:
                map_out = map_out * roi
                map_target = map_target * roi

            qmap_target = map_target[0, 1].cpu().numpy() * args.q_max
            pdmap_target = map_target[0, 0].cpu().numpy()
            qmap_output = map_out[0, 1].cpu().numpy() * args.q_max
            pdmap_output = map_out[0, 0].cpu().numpy()
            
            mse = np.mean((qmap_output - qmap_target) ** 2)
            norms = np.mean(qmap_target ** 2)
            criterion_output["nmse_t2"] += mse / norms
            mse = np.mean((pdmap_output - pdmap_target) ** 2)
            norms = np.mean(pdmap_target ** 2)
            criterion_output["nmse_pd"] += mse / norms

    for key in running_loss.keys():
        running_loss[key] = running_loss[key] / len(val_loader)
    for key in criterion_output.keys():
        criterion_output[key] = criterion_output[key] / len(val_loader)
    print('Test set: Average loss: ', running_loss, ' ', criterion_output)
    
    return criterion_output, running_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training command parser', add_help=False)
    parser.add_argument('-c', '--config_path', default='./config/MANTIS.yaml',
                        type=str, help='config path')
    cmd_par =  parser.parse_args()
    
    config_path = cmd_par.config_path
    args, args_model = get_args(config_path)
    
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

    # set export path
    save_sub_dir, save_label = get_savepath(args, args_model)
    save_path = os.path.join(args.save_dir, save_sub_dir, save_label)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    print('Export path: ', save_path)
    shutil.copyfile(config_path, save_path + '.yaml')
    
    start_epoch = 0
    best_score = np.finfo(np.float64).max
    if args.resume_training:
        load_ckpt(save_path, model, optimizer, scheduler)
        
    for epoch in range(start_epoch, args.epoches):
        epoch_start = time.time()
        model.train()
        loss_train = train(model, train_loader, optimizer, scheduler, epoch, args)
        model.eval()
        criterion, loss_val = val(model, val_loader, args)
        loss_val_save = {l+"_val": loss_val[l] for l in loss_val.keys()}
        
        save_dict = {'epoch':epoch, **criterion, **loss_train, **loss_val_save}
        f = open(save_path + '_metric.txt', 'a')
        f.write(json.dumps(save_dict))
        f.write('\n')
        f.close()
        if criterion["nmse_t2"] < best_score:  # save best model
            best_score = criterion["nmse_t2"]
            save_ckpt(save_path + '_best_.pth', epoch, model, best_score, optimizer, scheduler)
        print(f"time cost: {time.time()-epoch_start} s")
        save_ckpt(save_path + '_last_.pth', epoch, model, best_score, optimizer, scheduler)
        print(f"time cost: {time.time()-epoch_start} s")

    torch.save(model.state_dict(), save_path + '_alllast_.pth')
    print('over')