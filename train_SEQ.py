'''
 * Copyright (c) 2023, Toshiba Europe Ltd
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Chao Zhang
'''
import argparse
from operator import mod
import os
from re import M
from typing_extensions import Self
import ruamel.yaml as yaml
import numpy as np
import random
import time
import wandb
import datetime
import json
from PIL import Image
from pathlib import Path
from tqdm import tqdm 
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from models.my_blip import blip_decoder, blip_decoder2
from models.my_seq import LED_Base, load_clipseg

from collections import defaultdict
from data.utils import loss_func_seq, evaluate_clipseg, loss_func_aux
import utils
from utils import cosine_lr_schedule, distance_from_pixels, accuracy, lprint, confidence_from_pixels, euclid_distance_from_pixels
from data.seq_dataset import Loader
from data.utils import save_result, coco_caption_eval, convert_tensor_to_input, save_pred_seq



def train(model, data_loader, val_loader, optimizer, epoch, device, loss_func, config, args, log, wandb, tmodel):
    # train
    model.train()  
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq = len(data_loader)//5
    #print_freq = config['print_freq']
    if print_freq==0:
        print_freq=10
    lprint(f'valid freq: {print_freq}', log)

    mse_func = torch.nn.MSELoss()
    save_list = []
    for i, (info_elem, targets, maps, tokens, conversions, level, cap_len, attn_mask, cap_img) in enumerate(tqdm(data_loader)):
    
        if config['dry_run'] and i > 1:
            print('dry run break')
            break
        if config['debug'] and i>200:
            print('debug break')
            break
        
        maps = maps.to(device)       
        B, num_maps, C, H, W = maps.size()
        maps = maps.view(B * num_maps, C, H, W)
        B, num_maps, tH, tW = targets.size()
        targets = targets.view(B * num_maps, tH, tW).to(device).float()

        attn_mask = attn_mask.to(device)
        tokens = tokens.to(device).long()
        cap_len = cap_len.to(device)
        max_turn = cap_len.max().item()
        if num_maps>1:
            tokens = torch.repeat_interleave(tokens, num_maps, dim=0)
            attn_mask = torch.repeat_interleave(attn_mask, num_maps, dim=0)
            cap_len = torch.repeat_interleave(cap_len, num_maps, dim=0)
            cap_img = torch.repeat_interleave(cap_img, num_maps, dim=0)
        # teacher model
        #toutput = tmodel(maps, tokens, attn_mask, cap_len)
        output = model(maps, tokens, attn_mask, cap_len)    
        #import ipdb; ipdb.set_trace()  
        ts, BN, OC, h, w = output.shape
        if output.ndim==5:
            heatmaps = output[:,:,0]
        else:
            heatmaps = output[:,0,:,:]
        if OC>1:
            bimaps = output[:,:,1]
        else:
            bimaps = None
        
        map_pred = None

        if max_turn >= 3 and config['debug']:
        # if (i%100==0) and config['debug']:
            save_list.append(i)
            if len(save_list)<5:
                save_pred_seq(epoch, targets, maps, heatmaps, bimaps, level,
                        cap_img, cap_len,
                        B=B, num_maps=num_maps,
                        map_size=(H,W), target_size=(tH,tW),
                        heat_size=(h,w), args=args, config=config, mode='train', iter=i)
  
        # resize target to heatmap
        if tH!=h or tW!=w:
            targets = F.interpolate(
                    targets.unsqueeze(1),
                    (h, w),
                    mode="bilinear",
                ).squeeze(1).float()
        targets = targets.view(B, num_maps, h, w)

        # log-softmax loss
        if config['per_image_softmax']:
            log_heatmaps = F.log_softmax(heatmaps.view(ts*B*num_maps, -1), 1).view(ts, B, num_maps, h, w)
        else:    
            log_heatmaps = F.log_softmax(heatmaps.view(B, -1), 1).view(B, num_maps, h, w)
        
        loss = 0
        loss_loc = loss_func_seq(loss_func, log_heatmaps, targets, cap_len, config, device) * config["loss_w"] 
        loss+=loss_loc
        
        # sigmoid loss
        if config['per_pixel_sigmoid']:
            if bimaps is not None:
                sig_heatmaps = F.sigmoid(bimaps).view(ts, B, num_maps, h, w)
            else:
                sig_heatmaps = F.sigmoid(heatmaps).view(ts, B, num_maps, h, w)
            loss_aux = loss_func_aux(mse_func, sig_heatmaps, targets, cap_len, config, device) * config['aux_loss_w']
            loss+=loss_aux
        else:
            loss_aux = torch.tensor(0)
        
        if config['map_recon']:
            loss_recon = mse_func(map_pred, maps) * config['map_loss_w']
        else:
            loss_recon = torch.tensor(0)
        loss+=loss_recon
       
        if config['loss_gain'] and ts>1:
            #loss_gain = mse_func(heatmaps[0], heatmaps[-1]) * config['gain_loss_w']
            loss_gain = loss_func(log_heatmaps[0], targets).sum((1,2,3)) * config['gain_loss_w']
            loss_gain = loss_gain.sum() / targets.size(0)
            loss_gain = loss_gain*(-1)
        else:
            loss_gain = torch.tensor(0)
        loss+=loss_gain
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if ((i+1) % print_freq == 0):
            # log metrics to wandb
            if wandb is not None:
                wandb.log({"tr_loss": loss.item(),
                           "tr_loc": loss_loc.item(),
                           "tr_aux": loss_aux.item(),
                           "lr": optimizer.param_groups[0]["lr"]})
        if config['debug']:
            lprint(f'tr_loss: {loss.item():.4f},tr_loc: {loss_loc.item():.4f},tr_aux: {loss_aux.item():.4f}' )
            
        torch.cuda.empty_cache()

    return metric_logger.global_avg()

@torch.no_grad()
def evaluate(model, data_loader, optimizer, epoch, device, loss_func, config, args, mode, log, wandb):
    # evaluate
    model.eval() 
    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('le', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('acc0', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('acc5', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    mse_func = torch.nn.MSELoss()
    save_list = []
    
    le_dict = defaultdict(list)
    le_list = []
        
    for i, (info_elem, targets, maps, tokens, conversions, level, cap_len, attn_mask, cap_img) in enumerate(tqdm(data_loader)):

        # if i>=10: break
        maps = maps.to(device)       
        B, num_maps, C, H, W = maps.size()
        maps = maps.view(B * num_maps, C, H, W)
        B, num_maps, tH, tW = targets.size()
        targets = targets.view(B * num_maps, tH, tW).to(device).float()

        attn_mask = attn_mask.to(device)
        tokens = tokens.to(device).long()
        cap_len = cap_len.to(device)

        if num_maps>1:
            tokens = torch.repeat_interleave(tokens, num_maps, dim=0)
            attn_mask = torch.repeat_interleave(attn_mask, num_maps, dim=0)
            cap_len = torch.repeat_interleave(cap_len, num_maps, dim=0)
            cap_img = torch.repeat_interleave(cap_img, num_maps, dim=0)
            
        output = model(maps, tokens, attn_mask, cap_len)  

        ts, BN, OC, h, w = output.shape
        if output.ndim==5:
            heatmaps = output[:,:,0]
        else:
            heatmaps = output[:,0,:,:]
        if OC>1:
            bimaps = output[:,:,1]
        else:
            bimaps = None
        
        map_pred = None

        max_turn = cap_len.max().item()
        if args.evaluate or (max_turn >= 3 and config["debug"]):
            
            save_pred_seq(epoch, targets, maps, heatmaps, bimaps, level, cap_img, cap_len,
                        B=B, num_maps=num_maps,
                        map_size=(H,W), target_size=(tH,tW),
                        heat_size=(h,w), args=args, config=config, mode=mode, iter=i)
            save_list.append(i)
        
        if config['per_image_softmax']:
            heatmaps = F.log_softmax(heatmaps.view(ts*B*num_maps, -1), 1).view(ts, B, num_maps, h, w)
        else:    
            heatmaps = F.log_softmax(heatmaps.view(B, -1), 1).view(B, num_maps, h, w)
        
        # resize target to heatmap
        if h!=tH or w!=tW:
            targets = F.interpolate(
                    targets.unsqueeze(1),
                    (h, w),
                    mode="bilinear",
                ).squeeze(1).float()
        targets = targets.view(B, num_maps, h, w)

        def loss_func_seq(loss_func, heatmaps, targets, gt_turns, config):
            ts = heatmaps.size(0)
            loss = 0
            for t in range(ts):
                loss_turn = loss_func(heatmaps[t], targets).sum((1,2,3))
                cur_t = torch.tensor(t+1).repeat(B).to(device)
                weights = config['discount']**(gt_turns - cur_t)
                weights[gt_turns<cur_t] = 0
                loss_turn = (loss_turn * weights).sum()/targets.size(0)
                loss+=loss_turn
            return loss

        loss = loss_func_seq(loss_func, heatmaps, targets, cap_len, config)
        
        targets = targets.detach().cpu()
        heatmaps = heatmaps.detach().cpu()

        if args.eval_per_turn:
            assert B==1
            # per turn evaluation
            last_heats = []
            
            for turn_id in range(cap_len.item()):
                last_heats = heatmaps[turn_id, 0, ...].view(B, num_maps, h, w)
                le, ep = distance_from_pixels(args, num_maps, last_heats, conversions, info_elem, mode)
                save_le_name = os.path.join(args.result_dir, f'{mode}_ep_{epoch}_input_{i}_t{turn_id+1}.txt')
                np.savetxt(save_le_name, le)
            
        else:
            # final prediction evaluation
            last_heats = []
            for i, turn_id in enumerate(cap_len):
                #import ipdb; ipdb.set_trace()
                last_heats.append(heatmaps[turn_id-1, 0, ...])
            last_heats = torch.cat(last_heats)
            last_heats = last_heats.view(B, num_maps, h, w)
            
            # get metrics
            if args.use_euclid:
                le, ep = euclid_distance_from_pixels(args, num_maps, last_heats, conversions, info_elem, mode)   
            else:
                le, ep = distance_from_pixels(args, num_maps, last_heats, conversions, info_elem, mode)        
            #import ipdb; ipdb.set_trace()
            acc0 = accuracy(le, 0) 
            acc5 = accuracy(le, 5)
          
            le_list.extend(le)
            mean_le = np.mean(le)
            metric_logger.update(loss=loss.item())
            metric_logger.update(le=mean_le)
            metric_logger.update(acc0=acc0)
            metric_logger.update(acc5=acc5)
        
            # le for intermediate predictions
            if args.inter_le:
                heat_list = defaultdict(list)
                conv_list = defaultdict(list)
                info_list = defaultdict(list)
                path, levels, scan_names, episode_ids, true_viewpoints = info_elem
                
                for i, gt_turn in enumerate(cap_len):
                    for t in range(gt_turn):
                        new_elem = [path[i],levels[i],scan_names[i],episode_ids[i],true_viewpoints[i]]
                  
                        key = f'{gt_turn.item()}_{t}'
                        heat_list[key].append(heatmaps[t, i, ...])
                        conv_list[key].append(conversions[i])
                        info_list[key].append(new_elem)

                for key, val in heat_list.items():
                    val = torch.cat(val)
                    val = val.view(-1,num_maps,h,w)
                    le, ep = distance_from_pixels(
                           args, num_maps, val, conv_list[key], info_list[key], mode, inter_le=True
                    )        
                
                    le_dict[key].extend(le)
            else: # Prec Conf at gt pixel
                # for key, val in zip(cap_len, le):
                # le_dict[key.item()].append(val)
                tar_list = defaultdict(list)
                pred_list = defaultdict(list)
                for i, gt_turn in enumerate(cap_len):
                    for t in range(gt_turn):
                        key = f'{gt_turn.item()}_{t}'
                        tar_list[key].append(targets[i])
                        pred_list[key].append(heatmaps[t,i,...])
               
                for key, val in pred_list.items():
                    preds = torch.cat(val)
                    tars = torch.cat(tar_list[key])
                    pc = confidence_from_pixels(args, preds, tars, mode)
                    #import ipdb; ipdb.set_trace()
                    le_dict[key].extend(pc)

    if args.eval_per_turn:
        print('per turn le done.')
        return None, None, None

    # save le_list 
    le_savename = os.path.join(args.result_dir, f"le_{mode}_epoch{epoch}.npy")
    np.save(le_savename, le_list)
    lprint(f"le list saved to {le_savename}", log)
    # log metrics to wandb
    avg_dict = metric_logger.get_avg()
    if wandb is not None:
        wandb.log({"acc0_"+mode: avg_dict['acc0'],
                    "acc5_"+mode: avg_dict['acc5'],
                    "loss_"+mode: avg_dict['loss'],
                    "le_"+mode: avg_dict['le'],
                    })    
    eval_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()} 
    return metric_logger.global_avg(), eval_stats, le_dict




def main(args, config, wandb=None):
    #utils.init_distributed_mode(args)    
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    #### Model #### 
    print(f"Creating model {config['model_ver']}")
    if config['model_ver']==1: # Bert as base, ViT as hidden
        model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], prompt=config['prompt'], config=config)
    elif config['model_ver']==2: # ViT as base, Bert as hidden
        model = blip_decoder2(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], prompt=config['prompt'], config=config)
    elif config['model_ver']==3: # Standard ViT and Bert, new Multi
        model = LED_Base(image_size=config['image_size'], vit=config['vit'], config=config)

  
    tmodel = None
  
    #### Dataset #### 
    print("Creating LED dataset")
    loader = Loader(args, config)
    loader.build_dataset(file="train_expanded_data.json", tokenizer=model.tokenizer, image_processor=model.image_processor)
    loader.build_dataset(file="valSeen_data.json", tokenizer=model.tokenizer, image_processor=model.image_processor)
    loader.build_dataset(file="valUnseen_data.json", tokenizer=model.tokenizer, image_processor=model.image_processor)

    train_iterator = DataLoader(
        loader.datasets["train"],
        batch_size=config['batch_size'],
        shuffle=not config['debug'],
        num_workers=8,
    )
    valSeen_iterator = DataLoader(
        loader.datasets["valSeen"],
        batch_size=config['batch_size_test'],
        shuffle=False,
        num_workers=4,
    )
    valUnseen_iterator = DataLoader(
        loader.datasets["valUnseen"],
        batch_size=config['batch_size_test'],
        shuffle=False,
        num_workers=4,
    )

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'])
    loss_func = torch.nn.KLDivLoss(reduction="none")
        
    if args.evaluate:
        # load ckpt
        state_dict = torch.load(args.ckpt_path)
        epoch = state_dict['epoch']
        model.load_state_dict(state_dict['model'])
        model = model.to(device) 
        print(f'loaded state dict via {args.ckpt_path}')
        if args.use_euclid:
            log = open(os.path.join(args.result_dir, 'euc_log.txt'), 'a')
        else:
            log = open(os.path.join(args.result_dir, 'log.txt'), 'a')
        eval_modes = ['valSeen', 'valUnseen']
        for eval_i, eval_iterator in enumerate([valSeen_iterator, valUnseen_iterator]):
            if eval_i==0: 
                continue
            lprint(f"Start eval {eval_modes[eval_i]} {epoch}", log)
            import time 
            st = time.time()
            val_str, val_stat, val_led = evaluate(model, eval_iterator, optimizer, epoch, device, loss_func, config, args, mode=eval_modes[eval_i], log=log, wandb=wandb) 
            lprint(val_str, log)
            eval_time = time.time()-st 
            eval_time = eval_time / len(eval_iterator) / 60
            print(eval_time)
            
            if args.eval_per_turn:
                continue

            if args.inter_le:
                lprint('start intermediate evaluation', log)
                size_dict={}
                mean_dict={}
                # average le dict 
                for key, val in val_led.items():
                    size_dict[key] = len(val)
                    mean_dict[key] = np.mean(val)
                    lprint(f'Turn:{key} Total:{size_dict[key]} LE:{mean_dict[key]}', log)
            else:
                lprint("start fine-grained evaluation", log)
                size_dict={}
                mean_dict={}
                for key, val in val_led.items():
                    size_dict[key] = len(val)
                    mean_dict[key] = np.mean(val)
                    lprint(f'Len:{key} Total:{size_dict[key]} LE:{mean_dict[key]}', log)

        lprint('evaluation finished', log)
        return

    
    # training
    model = model.to(device) 
    log = open(os.path.join(args.result_dir, 'log.txt'), 'a')
        
    import time
    for epoch in range(0, config['max_epoch']):
        start_time = time.time()   
        lprint('----------------------------------------', log)
        #cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])      

        lprint(f"Start training {epoch}", log)
        train_str = train(model, train_iterator, valSeen_iterator, optimizer, epoch, device, loss_func, config, args, log=log, wandb=wandb, tmodel=tmodel) 
        lprint(train_str, log)

        lprint(f"Start eval valSeen {epoch}", log)
        valSeen_str, valSeen_stat, valSeen_led = evaluate(model, valSeen_iterator, optimizer, epoch, device, loss_func, config, args, mode='valSeen', log=log, wandb=wandb) 
        lprint(valSeen_str, log)

        lprint(f"Start eval valUnseen {epoch}", log)
        valUnseen_str, valUnseen_stat, valUnseen_led = evaluate(model, valUnseen_iterator, optimizer, epoch, device, loss_func, config, args, mode='valUnseen', log=log, wandb=wandb) 
        lprint(valUnseen_str, log)

   
        save_obj = {
                'model': copy.deepcopy(model.state_dict()),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
        torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))
            
        epoch_time = time.time()-start_time
        lprint(f'Epoch {epoch} time: {epoch_time/60:.4f} min', log)
        lprint('----------------------------------------', log)
        torch.cuda.empty_cache()

def make_names(args, config):
    new_name = config['project_name']
    if config["debug"]:
        new_name="debug_"+new_name
    if config['freeze_visual']:
        new_name+='_freeze'
    else:
        new_name+=f'_{config["arch"]}'
    if config['reuse_hidden']:
        new_name += '_reuse'
    if config['use_prev_est']:
        new_name+=f'_{config["fusion_prev_est"]}' 
    try:
        new_name += f'_depth{config["multi_depth"]}_dis{str(args.discount)}_{config["loss_reduce"]}'
        if config['per_pixel_sigmoid']:
            new_name+=f'_ad{config["aux_discount"]}_{config["loss_reduce_aux"]}_aw{config["aux_loss_w"]}'
            if config["new_bimap"]:
                new_name+='_newBimap'
    except: 
        pass

    if "dialog_cliping" in config and config["dialog_cliping"]:
        new_name+=f'_dialogClip'
        assert config["dialog_history"]==True
        assert config["reuse_hidden"]==False 
        assert config["use_prev_est"]==False 

    if "new_bimap" not in config:
        config["new_bimap"] = False
    
    if "to_paper" in config and config["to_paper"]:
        config["batch_size"]=1
        config["batch_size_test"]=1
    if args.use_euclid:
        new_name+=f'_euclid'

    return new_name



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/led_config.yaml')
    parser.add_argument('--output_dir', default='output')           
    parser.add_argument('--evaluate', action='store_true') 
    parser.add_argument('--use_euclid', action='store_true')    
    parser.add_argument('--inter_le', action='store_true')
    parser.add_argument('--to_paper', action='store_true')
    parser.add_argument('--eval_per_turn', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--ckpt_dir', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--map_h', default=700, type=int)
    parser.add_argument('--map_w', default=1200, type=int)
    parser.add_argument('--discount', default=0.0, type=float, help='decay factor')
    parser.add_argument('--aux_w', default=0.0, type=float, help='aux loss weight')
    parser.add_argument('--max_len', default=200, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument("--data_dir", type=str, default="./led_data/way_splits/")
    parser.add_argument("--image_dir", type=str, default="./led_data/floorplans/")
    parser.add_argument("--embedding_dir", type=str, default="./led_data/word_embeddings/")
    parser.add_argument("--connect_dir", type=str, default="./led_data/connectivity/")
    parser.add_argument(
      "--geodistance_file", type=str, default="./led_data/geodistance_nodes.json"
  )
    parser.add_argument("--ds_percent", type=float, default=1.0)
    parser.add_argument("--max_floors", type=int, default=5)
    
    args = parser.parse_args()
    
    if args.evaluate:
        assert os.path.exists(args.ckpt_dir)
        for file in os.listdir(args.ckpt_dir):
            if file.endswith(".pth"):
                args.ckpt_path = os.path.join(args.ckpt_dir, file)
                print(args.ckpt_path)
            elif file.endswith('.yaml'):
                args.config = os.path.join(args.ckpt_dir, file)
                print(args.config)
                

    global_config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    if args.to_paper:
        global_config["to_paper"]=True
    if args.discount > 0:
        global_config['discount'] = args.discount
        global_config['aux_discount'] = args.discount
    if args.aux_w > 0:
        global_config['aux_loss_w'] = args.aux_w
    global_config['project_name'] = make_names(args, global_config)

    if args.evaluate:
        global_config["batch_size_test"] = 1
    print(global_config['project_name'])


    if args.evaluate:
        args.output_dir = os.path.join(args.ckpt_dir, 'test')
    else:
        args.output_dir = os.path.join(global_config['output_dir'], global_config['project_name'])
        
    args.result_dir = os.path.join(args.output_dir, 'result')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    
    if not args.evaluate:
        import shutil
        shutil.copy(args.config, args.output_dir)
    wandb = None
    
    main(args, global_config, wandb)