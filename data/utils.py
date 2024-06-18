import re
import json
import os
from PIL import Image
import torch
import torch.distributed as dist
import copy
import numpy as np
from matplotlib import pyplot as plt
import utils

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question

def save_pred(epoch, targets, maps, heatmaps, pred_maps,
                    levels, 
                    cap_img, 
                    B, num_maps, 
                    map_size,
                    target_size,
                    heat_size,
                    args,
                    config,
                    mode,
                    iter):

    # maps:      Bxmaps x 3 x H x W
    # pred_maps: Bxmaps x 3 x H x W
    # heatmaps:  Bxmaps x H x W
    # targets:   Bxmaps x tH x tW
    H, W = map_size[0], map_size[1]
    tH, tW = target_size[0], target_size[1]
    h, w = heat_size[0], heat_size[1]
    cmap = plt.get_cmap('viridis')
    map_h, map_w = config['map_h'], config['map_w']
    if config["model_ver"]==1:
        img_mean=(0.485,0.456,0.406)
        img_std=(0.229,0.224,0.225)
    elif config["model_ver"]==3:
        if config['arch']=='vit':
            img_mean = (0.5,0.5,0.5)
            img_std = (0.5,0.5,0.5)
        elif config['arch']=='res18':
            img_mean = (0.5,0.5,0.5)
            img_std = (0.5,0.5,0.5)
        elif config['arch']=='seg':
            img_mean=(0.485,0.456,0.406)
            img_std=(0.229,0.224,0.225)
    # resize 
    maps = torch.nn.functional.interpolate(
                maps,
                (map_h, map_w),
                mode="bilinear",
            ).float().detach()

    if pred_maps is not None:
        pred_maps = torch.nn.functional.interpolate(
                pred_maps,
                (map_h, map_w),
                mode="bilinear",
            ).float().detach()

    targets = torch.nn.functional.interpolate(
                targets.unsqueeze(1),
                (map_h, map_w),
                mode="bilinear",
            ).squeeze(1).float().detach()

    # softmax
    softmaps = torch.clone(heatmaps.detach())
    if config['per_image_softmax']:
        softmaps = softmaps.view(B*num_maps, -1).softmax(1).view(B*num_maps, h, w)
    else:    
        softmaps = softmaps.view(B, -1).softmax(1).view(B*num_maps, h, w)

    softmaps = torch.nn.functional.interpolate(
                softmaps.unsqueeze(1),
                (map_h, map_w),
                mode="bilinear",
            ).squeeze(1).float().detach()

    maps = maps.view(B, num_maps, 3, map_h, map_w)

    cap_h = cap_img.size(-2)
    cap_w = cap_img.size(-1)
    cap_img = cap_img.view(B, num_maps, 3, cap_h, cap_w)
    n_samples = 3
    if n_samples > B:
        n_samples = B
    if config["to_paper"]:
        assert B==1

    fig_rows = 3
    softmaps = softmaps.view(B, num_maps, map_h, map_w)
    targets = targets.view(B, num_maps, map_h, map_w)

    for i in range(n_samples):
        global_id = iter * config["batch_size_test"]+i
        if num_maps==1:
            fig_cols = 3
        else:
            fig_cols = num_maps
        fig, ax = plt.subplots(fig_rows, fig_cols, figsize=(18,7), squeeze=False)
        level = int(levels[i])

        h_max = softmaps[i].max()
        h_min = softmaps[i].min()
        if config["debug"]:
            print(h_min, h_max)
        #ax = ax.flatten()
        for j in range(num_maps):
            map_one = convert_tensor_to_input(maps[i,j], mean=img_mean, std=img_std)
            map_one = map_one.permute(1,2,0).detach().cpu().numpy()*255
            map_one = np.clip(map_one.astype(np.uint8), 0, 255)
            if config["to_paper"]:
                im = Image.fromarray(map_one)
                im.save(os.path.join(args.result_dir, f'{mode}_ep{epoch}_{global_id}_0_map.png'))
            ax[0,j].imshow(map_one)
            ax[0,j].set_title('Map')
    
            heatmap_one = softmaps[i,j].detach().cpu().numpy()
            max_one, min_one = heatmap_one.max(), heatmap_one.min()
            if config["to_paper"]:
                heatmap_one = (heatmap_one-min_one)/(max_one-min_one+1e-8)
                heatmap_one = cmap(heatmap_one)[...,:3]*255.0
                heatmap_one = heatmap_one.astype(np.uint8)
                im = Image.fromarray(heatmap_one)
                im.save(os.path.join(args.result_dir, f'{mode}_ep{epoch}_{global_id}_2pred.png'))
            ax[1,j].imshow(heatmap_one)
            ax[1,j].set_title(f'Heatmap:{max_one:.2f}')
            target_one = targets[i,j].detach().cpu().numpy()
            target_one = (target_one - np.min(target_one)) / (np.max(target_one)-np.min(target_one))
            
            target_one = cmap(target_one)[...,:3]*255.0
            target_one = np.clip((0.5*target_one + 0.5*map_one).astype(np.uint8), 0, 255)
            if config["to_paper"]:
                im = Image.fromarray(target_one)
                im.save(os.path.join(args.result_dir, f'{mode}_ep{epoch}_{global_id}_1_target.png'))

            ax[2,j].imshow(target_one)
            ax[2,j].set_title('Target')
            cap_one = cap_img[i,j].permute(1,2,0).detach().cpu().numpy()
            ax[fig_rows-1, fig_cols-1].imshow(cap_one)
        save_name = os.path.join(args.result_dir, f'{mode}_ep_{epoch}_input_{global_id}.png')
        plt.tight_layout()
        plt.savefig(save_name, dpi=150)
        del fig, ax
        return save_name

def save_pred_seq(epoch, targets, maps, heatmaps, bimaps,
                    levels, 
                    cap_img, 
                    cap_len,
                    B, num_maps, 
                    map_size,
                    target_size,
                    heat_size,
                    args,
                    config,
                    mode,
                    iter):

    # maps:      Bxmaps x 3 x H x W
    # pred_maps: Bxmaps x 3 x H x W
    # heatmaps:  turns x Bxmaps x H x W
    # targets:   Bxmaps x tH x tW
    H, W = map_size[0], map_size[1]
    tH, tW = target_size[0], target_size[1]
    h, w = heat_size[0], heat_size[1]
    ts = heatmaps.size(0)
    cmap = plt.get_cmap('viridis')
    map_h, map_w = config['map_h'], config['map_w']
    if config['arch'] in['vit','res18']:
        img_mean = (0.5,0.5,0.5)
        img_std = (0.5,0.5,0.5)
    elif config['arch']=='seg':
        img_mean=(0.485,0.456,0.406)
        img_std=(0.229,0.224,0.225)
    # resize 
    maps = torch.nn.functional.interpolate(
                maps,
                (map_h, map_w),
                mode="bilinear",
            ).float().detach()

    targets = torch.nn.functional.interpolate(
                targets.unsqueeze(1),
                (map_h, map_w),
                mode="bilinear",
            ).squeeze(1).float().detach()

    # softmax
    softmaps = torch.clone(heatmaps.detach())
    if config['per_image_softmax']:
        softmaps = softmaps.view(ts*B*num_maps, -1).softmax(1).view(ts*B*num_maps, h, w)
    else:    
        softmaps = softmaps.view(B, -1).softmax(1).view(B*num_maps, h, w)

    softmaps = torch.nn.functional.interpolate(
                softmaps.unsqueeze(1),
                (map_h, map_w),
                mode="bilinear",
            ).squeeze(1).float().detach()


    maps = maps.view(B, num_maps, 3, map_h, map_w)

    cap_h = cap_img.size(-2)
    cap_w = cap_img.size(-1)
    cap_img = cap_img.view(B, num_maps, 3, cap_h, cap_w)

    softmaps = softmaps.view(ts, B, num_maps, map_h, map_w)
    targets = targets.view(B, num_maps, map_h, map_w)

    if config['per_pixel_sigmoid']:
        if bimaps is None:
            bimaps = torch.clone(heatmaps.detach())
        else:
            bimaps = bimaps.detach()
        bimaps = bimaps.sigmoid().view(ts*B*num_maps, h, w)
        bimaps = torch.nn.functional.interpolate(
                bimaps.unsqueeze(1),
                (map_h, map_w),
                mode="bilinear",
            ).squeeze(1).float().detach()
        bimaps = bimaps.view(ts, B, num_maps, map_h, map_w)

    if "use_min_max" not in config:
        config["use_min_max"] = False
    if args.evaluate:
        config["use_min_max"] = False


    ts = cap_len.max().item()
    if ts<3:
        return

    if config["to_paper"]:
        assert B==1
        n_samples = B
    else:
        n_samples = 3
    
    if n_samples>B:
        n_samples=B
    

    fig_rows = (n_samples * 2) if config["per_pixel_sigmoid"] else n_samples
    fig_cols = ts+3
    fig, ax = plt.subplots(fig_rows, fig_cols, figsize=(18,7), squeeze=False)
    for i in range(n_samples):
        global_id = iter * config["batch_size_test"]+i
        if config["per_pixel_sigmoid"]:
            dst_row = i*2
            ax[dst_row+1,0].axis('off')
            ax[dst_row+1,1].axis('off')
        else:
            dst_row = i
        h_max = softmaps[:,i].max().cpu().numpy()
        h_min = softmaps[:,i].min().cpu().numpy()
        level = int(levels[i])
        gt_turn = cap_len[i]

        map_one = convert_tensor_to_input(maps[i,0], mean=img_mean, std=img_std)
        map_one = map_one.permute(1,2,0).detach().cpu().numpy()*255
        map_one = np.clip(map_one.astype(np.uint8), 0, 255)
        if config["to_paper"]:
            im = Image.fromarray(map_one)
            im.save(os.path.join(args.result_dir, f'{mode}_ep{epoch}_{global_id}_0_map.png'))

        ax[dst_row,0].imshow(map_one)
        ax[dst_row,0].set_title('Map') 

        target_one = targets[i,0].detach().cpu().numpy()
        target_one = (target_one - np.min(target_one)) / (np.max(target_one)-np.min(target_one))
        target_one = cmap(target_one)[...,:3]*255.0
        target_one = np.clip((0.5*target_one + 0.5*map_one).astype(np.uint8), 0, 255)
        if config["to_paper"]:
            im = Image.fromarray(target_one)
            im.save(os.path.join(args.result_dir, f'{mode}_ep{epoch}_{global_id}_1_target.png'))

        ax[dst_row,1].imshow(target_one)
        ax[dst_row,1].set_title('Target')

        for t in range(ts):
            if (t+1) <= gt_turn:
                heatmap_one = softmaps[t,i,0].detach().cpu().numpy()
                min_one, max_one = heatmap_one.min(), heatmap_one.max()
                if config["use_min_max"]:
                    ax[dst_row,t+2].imshow(heatmap_one, vmin=h_min, vmax=h_max)
                else:
                    ax[dst_row,t+2].imshow(heatmap_one)
                if config["to_paper"]:
                    heatmap_one = (heatmap_one-min_one)/(max_one-min_one+1e-8)
                    heatmap_one = cmap(heatmap_one)[...,:3]*255.0
                    heatmap_one = heatmap_one.astype(np.uint8)
                    im = Image.fromarray(heatmap_one)
                    im.save(os.path.join(args.result_dir, f'{mode}_ep{epoch}_{global_id}_2_t{t+1}.png'))

                ax[dst_row,t+2].set_title(f't={t+1} [{min_one:.2f}-{max_one:.2f}]')
                if config["per_pixel_sigmoid"]:
                    bimap_one = bimaps[t,i,0].detach().cpu().numpy()
                    ax[dst_row+1,t+2].imshow(bimap_one)
                    ax[dst_row+1,t+2].set_title(f'bimap{t+1}')
            else:
                ax[dst_row,t+2].axis('off')
                if config["per_pixel_sigmoid"]:
                    ax[dst_row+1,t+2].axis('off')
            cap_one = cap_img[i,0].permute(1,2,0).detach().cpu().numpy()
            ax[dst_row, fig_cols-1].imshow(cap_one)
            if config["per_pixel_sigmoid"]:
                ax[dst_row+1,fig_cols-1].axis('off')

    save_name = os.path.join(args.result_dir, f'{mode}_ep_{epoch}_iter_{iter}.png')
    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
      
    del fig, ax, maps, softmaps, targets, heatmaps

def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result,open(result_file,'w'))

    dist.barrier()

    if utils.is_main_process():   
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        print('result file saved to %s'%final_result_file)

    return final_result_file



from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url

def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    
    
    download_url(urls[split],coco_gt_root)
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval


##  convert tensor to input image
# mean=[0.485, 0.456, 0.406, 0.555],
# std=[0.229, 0.224, 0.225, 0.222],
def convert_tensor_to_input(tensor, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    #assert tensor.dim == 3 and tensor.size(1)==3
    tensor[0]=tensor[0]*std[0] + mean[0]
    tensor[1]=tensor[1]*std[1] + mean[1]
    tensor[2]=tensor[2]*std[2] + mean[2]
    return tensor

def blur_target(targets, config, vis=False):
    
    targets = targets.unsqueeze(1)
    
    if vis:
        fig_rows = 2
        fig_cols = 4
        fig, ax = plt.subplots(fig_rows, fig_cols, figsize=(18,7))
        for i in range(fig_cols):
            ks = np.random.randint(config['k_min'], config['k_max'],size=1)[0]*2+1
            sg = np.random.randint(config['s_min'], config['s_max'],size=1)[0]
            tar = targets[i]
            blur_op = transforms.GaussianBlur(kernel_size=(ks, ks), sigma=sg)
            tarblur = blur_op(tar)
            tarblur = tarblur / tarblur.sum()

            target_one = tar[0].detach().cpu().numpy()
            ax[0,i].imshow(target_one)
            ax[0,i].set_title('Target')
            target_one = tarblur[0].detach().cpu().numpy()
            ax[1,i].imshow(target_one)
            ax[1,i].set_title(f'K {ks} Sig{sg}')
        save_name = os.path.join(args.result_dir, f'input{iter}_blur.png')
        plt.tight_layout()
        plt.savefig(save_name, dpi=150)
        del fig
    else:
        for i in range(targets.size(0)):
            ks = np.random.randint(config['k_min'], config['k_max'],size=1)[0]*2+1
            sg = np.random.randint(config['s_min'], config['s_max'],size=1)[0]
            blur_op = transforms.GaussianBlur(kernel_size=(ks, ks), sigma=sg)
            tarblur = blur_op(targets[i])
            tarblur = tarblur / tarblur.sum()
            targets[i] = tarblur[0]
        return targets

    
# wrapped loss func for seq version
def loss_func_seq(loss_func, heatmaps, targets, gt_turns, config, device):
    # heatmaps: ts x B x num_maps x h x w
    # targets:  B x num_maps x h x w
    ts = heatmaps.size(0)
    B = targets.size(0)
    loss = 0

    if config['use_soft']:
        # blur targets
        ks = np.random.randint(config['k_min'],config['k_max'],size=1)[0]*2+1
        sg = np.random.randint(config['s_min'],config['s_max'],size=1)[0]
        blur_op = transforms.GaussianBlur(kernel_size=(ks, ks), sigma=sg)
        tarblur = blur_op(targets)
        tarblur = tarblur / tarblur.sum((1,2,3)).view(B,1,1,1)

    for t in range(ts):
        loss_turn = loss_func(heatmaps[t], targets).sum((1,2,3))
        cur_t = torch.tensor(t+1).repeat(B).to(device)
        
        if config['use_soft']:
            mask_hard = 0**(gt_turns-cur_t)
            mask_soft = 1-mask_hard
            loss_soft = loss_func(heatmaps[t], tarblur).sum((1,2,3))
            loss_soft = (loss_soft * mask_soft).sum()/targets.size(0)
            loss_turn = (loss_turn * mask_hard).sum()/targets.size(0)
            loss+= (loss_turn+loss_soft)
        else:
            weights = config['discount']**(gt_turns - cur_t)
            weights[gt_turns<cur_t] = 0
            loss_turn = loss_turn * weights
            loss+=loss_turn

    if config['loss_reduce']=='batch':
        loss = loss.sum()/targets.size(0)
    elif config['loss_reduce']=='turn':
        loss = loss.sum()/gt_turns.sum()
    elif config['loss_reduce']=='mean':
        loss = (loss / gt_turns).sum() / targets.size(0)
    else:
        ValueError(f'loss reduce {config["loss_reduce"]} not found.')
    return loss    
    
# aux loss for pixel-wise binary prediction
def loss_func_aux(loss_func, heatmaps, targets, gt_turns, config, device):
    loss = 0 
    ts = heatmaps.size(0)
    B = targets.size(0)
    masks = (targets>0).float().to(device)


    for t in range(ts):
        heatmap_turn = heatmaps[t] * masks
        loss_turn = loss_func(heatmap_turn, targets)
        cur_t = torch.tensor(t+1).repeat(B).to(device)
        weights = config['aux_discount']**(gt_turns - cur_t)
        weights[gt_turns<cur_t] = 0
        loss_turn = loss_turn * weights
        loss+=loss_turn
    #import ipdb; ipdb.set_trace()
    if config['loss_reduce_aux']=='batch':
        loss = loss.sum()/targets.size(0)
    elif config['loss_reduce_aux']=='turn':
        loss = loss.sum()/gt_turns.sum()
    elif config['loss_reduce_aux']=='mean':
        loss = (loss / gt_turns).sum() / targets.size(0)
    else:
        ValueError(f'loss reduce aux {config["loss_reduce_aux"]} not found.')
    return loss

# evaluate clipSeg

@torch.no_grad()
def evaluate_clipseg(tokenizer, data_loader, optimizer, epoch, device, loss_func, config, args, mode, log, wandb):
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    # evaluate
    model.eval() 
    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('le', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('acc0', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('acc5', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    mse_func = torch.nn.MSELoss()
    save_list = []
    for iter, (info_elem, targets, maps, tokens, conversions, level, cap_len, attn_mask, cap_img) in enumerate(data_loader):
     
        print(iter)
        B, num_maps, C, H, W = maps.size()
        assert B==1
        B, num_maps, tH, tW = targets.size()
        targets = targets.view(B * num_maps, tH, tW).float()
        image_path = info_elem[0][0]
        # attn_mask = attn_mask.to(device)
        tokens = tokens[0]
        cap_len = cap_len[0].item()

        if num_maps>1:
            tokens = torch.repeat_interleave(tokens, num_maps, dim=0)
            attn_mask = torch.repeat_interleave(attn_mask, num_maps, dim=0)
            cap_len = torch.repeat_interleave(cap_len, num_maps, dim=0)
            cap_img = torch.repeat_interleave(cap_img, num_maps, dim=0)
            
        # load image
        #import ipdb; ipdb.set_trace()
        text = ""
        image = Image.open(image_path).convert('RGB')
        for i in range(cap_len):
            text+=tokenizer.decode(tokens[i], skip_special_tokens=True)
            #if i!=(cap_len-1):
            #    text+='[LOC]'

        #prompts = text.split('[LOC]')
        prompts = []
        prompts.append(text)
        inputs = processor(text=prompts, images=[image] * len(prompts), padding=True, truncation=True, return_tensors="pt")

        outputs = model(**inputs)
        preds = outputs.logits
        
        if len(prompts)==1:
            preds = preds.unsqueeze(0)
        B, h, w = preds.shape 
        
        heatmaps = F.softmax(preds.view(B, -1), 1).view(B, h, w)
        heatmaps = heatmaps.detach().cpu()
        # plot
        image = image.resize((w,h))

        # resize target to heatmap
        if h!=tH or w!=tW:
            targets = F.interpolate(
                    targets.unsqueeze(1),
                    (h, w),
                    mode="bilinear",
                ).squeeze(1).float()
        #targets = targets.view(B, h, w)

        fig, ax = plt.subplots(len(prompts) + 2, 1, figsize=(14, 3*(len(prompts) + 1)))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(image)
        ax[1].imshow(targets[0])
        [ax[i+2].imshow(heatmaps[i]) for i in range(len(prompts))]
        [ax[i+2].text(0, -15, prompt) for i, prompt in enumerate(prompts)]

        save_name = os.path.join(args.result_dir, f'{mode}_ep_{epoch}_input_{iter}.png')
        plt.tight_layout()
        plt.savefig(save_name, dpi=150)

        if False:
            # get metrics
            last_heats = []
            for i, turn_id in enumerate(cap_len):
                last_heats.append(heatmaps[turn_id-1, i, ...])
            last_heats = torch.cat(last_heats)
            last_heats = last_heats.view(B, num_maps, h, w)
    
            le, ep = distance_from_pixels(
                args, num_maps, last_heats, conversions, info_elem, mode
            )        
            
            acc0 = accuracy(le, 0) 
            acc5 = accuracy(le, 5)
        
            mean_le = np.mean(le)
            metric_logger.update(le=mean_le)
            metric_logger.update(acc0=acc0)
            metric_logger.update(acc5=acc5)
        else:
            metric_logger.update(le=0)
            metric_logger.update(acc0=0)
            metric_logger.update(acc5=0)

    # log metrics to wandb
    avg_dict = metric_logger.get_avg()
    if wandb is not None:
        wandb.log({"acc0_"+mode: avg_dict['acc0'],
                    "acc5_"+mode: avg_dict['acc5'],
                    "loss_"+mode: avg_dict['loss'],
                    "le_"+mode: avg_dict['le'],
                    })    
    eval_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()} 
    return metric_logger.global_avg(), eval_stats

def draw_cmc(savename, le, low, high, step=0.1):
    total_cnt = len(le)
    ranks = []
    cmc_rates = []
    for bin in np.arange(low, high+step, step):
        ranks.append(bin)
        rate = np.where(le<=bin)[0].shape[0]
        rate = float(rate) / total_cnt
        cmc_rates.append(rate)


    # Plotting the CMC curve
    plt.figure(figsize=(8, 6))
    plt.plot(ranks, cmc_rates, marker='o', linestyle='-', color='b')
    plt.xlabel('Rank')
    plt.ylabel('Cumulative Match Rate')
    plt.title('Cumulative Match Characteristic (CMC) Curve')
    plt.grid(True)
    plt.xticks(ranks)
    plt.ylim(0, 1.05)  # Set appropriate limits for the y-axis
    plt.xlim(0, max(ranks) + 10)  # Set appropriate limits for the x-axis
    plt.gca().invert_xaxis()  # Invert the x-axis for better visualization
    # plt.show()
    plt.tight_layout()
    plt.savefig(savename, dpi=150)