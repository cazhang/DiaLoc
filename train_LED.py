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
import ruamel.yaml as yaml
import numpy as np
import random
import time
import wandb
import datetime
import json
from pathlib import Path
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from models.my_blip import blip_decoder, blip_decoder2
from models.my_model import LED_Base
from collections import defaultdict
from torchvision import transforms
from data.utils import blur_target
import utils
from utils import cosine_lr_schedule, distance_from_pixels, accuracy, lprint, euclid_distance_from_pixels
from data.led_dataset import Loader
from data.utils import save_result, coco_caption_eval, convert_tensor_to_input, save_pred

def train(model, data_loader, val_loader, optimizer, epoch, device, loss_func, config, args, log, wandb):
    # train
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train LED Epoch: [{}]'.format(epoch)
    print_freq = len(data_loader)//5
    if print_freq==0:
        print_freq=10  

    for iter, (info_elem, targets, maps, tokens, conversions, level, cap_len, attn_mask, cap_img) in enumerate(tqdm(data_loader)):
        
        maps = maps.to(device)       
        B, num_maps, C, H, W = maps.size()
        maps = maps.view(B * num_maps, C, H, W)
        B, num_maps, tH, tW = targets.size()

        #print(targets.max(), targets.min())
        targets = targets.view(B * num_maps, tH, tW).to(device).float()
            
        attn_mask = attn_mask.to(device)
        tokens = tokens.to(device)
        cap_len = cap_len.to(device)

        if num_maps>1:
            tokens = torch.repeat_interleave(tokens, num_maps, dim=0)
            attn_mask = torch.repeat_interleave(attn_mask, num_maps, dim=0)
            cap_len = torch.repeat_interleave(cap_len, num_maps, dim=0)
            cap_img = torch.repeat_interleave(cap_img, num_maps, dim=0)
        
        output = model(maps, tokens, attn_mask, cap_len)      

        BN, OC, h, w = output.shape
        heatmaps = output[:,0,:,:]
        if OC>1:
            map_pred = output[:,1::, :, :]
        else:
            map_pred = None
        if config['debug']:
            save_name = save_pred(epoch, targets, maps, heatmaps, map_pred, level, cap_img, B=B, num_maps=num_maps,
                        map_size=(H,W), target_size=(tH,tW),
                        heat_size=(h,w), args=args, config=config, mode='train', iter=iter)
        # resize target to heatmap
        #import ipdb; ipdb.set_trace()
        #print(targets.max(), targets.min())
        if tH!=h or tW!=w:
            if targets.ndim==3:
                targets = F.interpolate(targets.unsqueeze(1), (h, w),mode="bilinear").squeeze(1).float()
            else:
                targets = F.interpolate(targets, (h, w), mode="bilinear").float()
        
        targets = targets.view(B, num_maps, h, w)
        #print(targets.max(), targets.min())
        #import ipdb; ipdb.set_trace()

        if config['per_image_softmax']:
            log_heatmaps = F.log_softmax(heatmaps.view(B*num_maps, -1), 1).view(B, num_maps, h, w)
        else:    
            log_heatmaps = F.log_softmax(heatmaps.view(B, -1), 1).view(B, num_maps, h, w)

        loss = loss_func(log_heatmaps, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if ((iter+1) % print_freq == 0):
            # evaluate
            if config['debug']:
                valSeen_str, valSeen_stat = evaluate(model, val_loader, optimizer, epoch, device, loss_func, config, args, mode='valSeen', log=log, wandb=wandb) 
                lprint(valSeen_str, log)
            # log metrics to wandb
            if wandb is not None:
                wandb.log({"tr_loss": loss.item(),
                           "lr": optimizer.param_groups[0]["lr"]})
            
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
    #metric_logger.add_meter('acc10', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    mse_func = torch.nn.MSELoss()
    submission = {}
    save_freq = 100
    le_list = []
    for iter, (info_elem, targets, maps, tokens, conversions, level, cap_len, attn_mask, cap_img) in enumerate(tqdm(data_loader)):

        maps = maps.to(device)       
        B, num_maps, C, H, W = maps.size()
        maps = maps.view(B * num_maps, C, H, W)
        B, num_maps, tH, tW = targets.size()
        #tH, tW = config['map_h'], config['map_w']
        targets = targets.view(B * num_maps, tH, tW).to(device).float()
        attn_mask = attn_mask.to(device)
        tokens = tokens.to(device)
        cap_len = cap_len.to(device)

        if num_maps>1:
            tokens = torch.repeat_interleave(tokens, num_maps, dim=0)
            attn_mask = torch.repeat_interleave(attn_mask, num_maps, dim=0)
            cap_len = torch.repeat_interleave(cap_len, num_maps, dim=0)
            cap_img = torch.repeat_interleave(cap_img, num_maps, dim=0)

        output = model(maps, tokens, attn_mask, cap_len)  

        BN, OC, h, w = output.shape
        heatmaps = output[:,0]
        if OC>1:
            map_pred = output[:,1::, :, :]
        else:
            map_pred = None

        save_pred_name=None
        if (iter%save_freq==0) or args.evaluate:
            save_pred_name = save_pred(epoch, targets, maps, heatmaps, map_pred, level, cap_img, B=B, num_maps=num_maps,
                        map_size=(H,W), target_size=(tH,tW),
                        heat_size=(h,w), args=args, config=config, mode=mode, iter=iter)

        if config['per_image_softmax']:
            heatmaps = F.log_softmax(heatmaps.view(B*num_maps, -1), 1).view(B, num_maps, h, w)
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
        loss = loss_func(heatmaps, targets)
       
        heatmaps = heatmaps.view(B, num_maps, h, w).detach().cpu()

        # get metrics
        if args.use_euclid:
            le, ep = euclid_distance_from_pixels(args, num_maps, heatmaps, conversions, info_elem, mode)
        else:
            le, ep = distance_from_pixels(args, num_maps, heatmaps, conversions, info_elem, mode
        )        

        if save_pred_name is not None:
            save_le_name = save_pred_name.replace('.png','.txt')
            np.savetxt(save_le_name, le)
        for i in ep:
            submission[i[0]] = {"viewpoint": i[1]}

        le_list.extend(le)
        acc0 = accuracy(le, 0) 
        acc5 = accuracy(le, 5)
        mean_le = np.mean(le)
        metric_logger.update(loss=loss.item())
        metric_logger.update(le=mean_le)
        metric_logger.update(acc0=acc0)
        metric_logger.update(acc5=acc5)

    # save le list 
    le_savename = os.path.join(args.result_dir, f"le_{mode}_epoch{epoch}.npy")
    if mode!="test":
        np.save(le_savename, le_list)
        lprint(f"Mode: {mode} le saved to {le_savename}", log)
    # log metrics to wandb
    avg_dict = metric_logger.get_avg()
    if wandb is not None:
        wandb.log({"acc0_"+mode: avg_dict['acc0'],
                    "acc5_"+mode: avg_dict['acc5'],
                    "loss_"+mode: avg_dict['loss'],
                    "le_"+mode: avg_dict['le'],
                    })

    
    eval_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()} 
    return metric_logger.global_avg(), eval_stats, submission



class Container(object):
    pass 
        
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
        fakeObj = Container()
        fakeObj.image_mean = (0.48145466, 0.4578275, 0.40821073)
        fakeObj.image_std = (0.26862954, 0.26130258, 0.27577711)
        
        model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], prompt=config['prompt'], config=config)
        model.image_processor = fakeObj
    elif config['model_ver']==2: # ViT as base, Bert as hidden
        model = blip_decoder2(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], prompt=config['prompt'], config=config)
    elif config['model_ver']==3: # Standard ViT and Bert, new Multi
        model = LED_Base(image_size=config['image_size'], vit=config['vit'], config=config)

    #### Dataset #### 
    print("Creating LED dataset")
    loader = Loader(args, config)
    loader.build_dataset(file="train_expanded_data.json", tokenizer=model.tokenizer, image_processor=model.image_processor)
    loader.build_dataset(file="valSeen_data.json", tokenizer=model.tokenizer, image_processor=model.image_processor)
    loader.build_dataset(file="valUnseen_data.json", tokenizer=model.tokenizer, image_processor=model.image_processor)
    loader.build_dataset(file="test_data.json", tokenizer=model.tokenizer, image_processor=model.image_processor)

    train_iterator = DataLoader(
        loader.datasets["train"],
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,
    )
    valSeen_iterator = DataLoader(
        loader.datasets["valSeen"],
        batch_size=config['batch_size_test'],
        shuffle=False,
        num_workers=8,
    )
    valUnseen_iterator = DataLoader(
        loader.datasets["valUnseen"],
        batch_size=config['batch_size_test'],
        shuffle=False,
        num_workers=8,
    )
    test_iterator = DataLoader(
        loader.datasets["test"],
        batch_size=config['batch_size_test'],
        shuffle=False,
        num_workers=8,
    )
        
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'])
    loss_func = torch.nn.KLDivLoss(reduction="batchmean")

    if args.evaluate:
        state_dict = torch.load(args.ckpt_path)
        epoch = state_dict['epoch']
        model.load_state_dict(state_dict['model'])
        model = model.to(device) 
        print(f'loaded state dict via {args.ckpt_path}')
        if args.use_euclid:
            log = open(os.path.join(args.result_dir, 'euc_log.txt'), 'a')
        else:
            log = open(os.path.join(args.result_dir, 'log.txt'), 'a')
        if args.test_mode:
            eval_modes = ['test']
            eval_iters = [test_iterator]
        else:
            eval_modes=['valSeen', 'valUnseen']
            eval_iters = [valSeen_iterator, valUnseen_iterator]
        for eval_mode, eval_iterator in zip(eval_modes, eval_iters):

            lprint(f"Start eval {eval_mode} {epoch}", log)

            val_str, val_stat, submission = evaluate(model, eval_iterator, optimizer, epoch, device, loss_func, config, args, mode=eval_mode, log=log, wandb=wandb)
            if eval_mode!='test': 
                lprint(val_str, log)
            
            filename = f"{eval_mode}_submission.json"
            filename = os.path.join(args.output_dir, filename)
            json.dump(submission, open(filename, "w"), indent=3)
            print("submission saved at ", filename)

        return

    
    model = model.to(device)   

    log = open(os.path.join(args.result_dir, 'log.txt'), 'w')

    for epoch in range(0, config['max_epoch']):
        start_time = time.time()   
        lprint('----------------------------------------', log)

        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])      

        lprint(f"Start training {epoch}", log)
        train_str = train(model, train_iterator, valSeen_iterator, optimizer, epoch, device, loss_func, config, args, log=log, wandb=wandb) 
        lprint(train_str, log)

        lprint(f"Start eval valSeen {epoch}", log)
        valSeen_str, valSeen_stat, valSeen_sub = evaluate(model, valSeen_iterator, optimizer, epoch, device, loss_func, config, args, mode='valSeen', log=log, wandb=wandb) 
        lprint(valSeen_str, log)

        lprint(f"Start eval valUnseen {epoch}", log)
        valUnseen_str, valUnseen_stat, valUnseen_sub = evaluate(model, valUnseen_iterator, optimizer, epoch, device, loss_func, config, args, mode='valUnseen', log=log, wandb=wandb) 
        lprint(valUnseen_str, log)
            
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
   
    new_name += f'_depth{config["multi_depth"]}_floors{config["max_floors"]}'

    if args.use_euclid:
        new_name+=f'_euclid'
    return new_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/led_config.yaml')
    parser.add_argument('--output_dir', default='output')           
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--use_euclid', action='store_true') 
    parser.add_argument('--test_mode', action='store_true')    
    parser.add_argument('--inter_le', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--ckpt_dir', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--map_h', default=700, type=int)
    parser.add_argument('--map_w', default=1200, type=int)
    #parser.add_argument('--batch_size', default=8, type=int, help='vit: 6, frozen: 16')
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
    global_config['project_name'] = make_names(args, global_config)

    if args.evaluate:
        args.output_dir = os.path.join(args.ckpt_dir, 'test')
        global_config["batch_size"]=1
        global_config["batch_size_test"]=1

    else:
        args.output_dir = os.path.join(global_config['output_dir'], global_config['project_name'])

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(global_config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    
    if not args.evaluate:
        import shutil
        shutil.copy(args.config, args.output_dir)
    

    
    main(args, global_config)
