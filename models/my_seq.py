'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
from distutils.command.config import config
from tkinter import Image
import warnings

#from zmq import device
warnings.filterwarnings("ignore")

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import math
import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
from models.vit import Block, trunc_normal_
import torchvision.models as models
from transformers import BertTokenizer, BertModel
from transformers.models.bert.configuration_bert import BertConfig

from transformers import AutoImageProcessor, ViTModel, SegformerModel
from datasets import load_dataset



class LED_Multi(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, num_patches=196, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, arch=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 use_grad_checkpointing=False, ckpt_layer=0, cross_start_layer=6, fusion='concat'):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.proj_layer = nn.Linear(in_chans, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.arch = arch

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_grad_checkpointing=(use_grad_checkpointing and i>=depth-ckpt_layer),
                add_cross_attention=(i>=cross_start_layer), layer_num=i, fusion=fusion
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, register_blk=-1, 
                encoder_hidden_states=None,
                encoder_attention_mask=None, 
                prev_estimate=None):
        B = x.shape[0]
        #import ipdb; ipdb.set_trace()
        if self.arch == 'res18':
            x = self.proj_layer(x)
            #cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            #x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed[:,:x.size(1),:]
            x = self.pos_drop(x)

        elif self.arch == 'seg':
            # 1. last hidden project to 768
            # x: B x 256 x 7 x 7
            b, c, h, w = x.shape
            x = x.flatten(2).transpose(1,2)
            x = self.proj_layer(x)
            #x = x.transpose(1,2).reshape(b, -1, h, w).contiguous()

        if encoder_attention_mask is not None:
            if encoder_attention_mask.dim()==2:
                extended_attention_mask = encoder_attention_mask[:,None,None,:]
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            else:
                raise ValueError('wrong shape of attention mask')

        for i,blk in enumerate(self.blocks):
            x = blk(x, register_blk==i, 
                    encoder_hidden_states=encoder_hidden_states, 
                    encoder_attention_mask=extended_attention_mask,
                    prev_estimate=prev_estimate)
        x = self.norm(x)
        
        return x

## 
class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states

class SegformerFuser(nn.Module):
    def __init__(self,       
                num_encoder_blocks = 4,
                hidden_sizes = [32, 64, 160, 768],
                output_dim = 768,
                image_size = (224,224)
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        self.image_size = image_size

        mlps = []
        for i in range(num_encoder_blocks):
            mlp = SegformerMLP(input_dim=hidden_sizes[i], out_dim=output_dim)
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=output_dim * num_encoder_blocks,
            out_channels=output_dim,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(output_dim)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Conv2d(output_dim, 1, kernel_size=1)

    def forward(self, encoder_hidden_states):
        # last_hidden: B x 768 x h x w
        # all_hidden: B x 32 x H x W
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
         
            #import ipdb; ipdb.set_trace()
            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=self.image_size, mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states, dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.classifier(hidden_states)

        return hidden_states




#### Encoder part for image and text
class LED_Base(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,    
                 config = None,             
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.config = config
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased", local_files_only=True)
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", local_files_only=True) 
        
        if self.config['add_loc_token']:
            num_added = self.tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]', '[LOC]']})   
            print(f'len of tokenizer:', len(self.tokenizer))
            self.loc_token_idx = self.tokenizer.additional_special_tokens_ids[1]
            print(f'loc id:', self.loc_token_idx)
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        
        if config['arch']=='vit':
            self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", local_files_only=True)
            num_patches = (config['image_h'] // 16) * (config['image_w']//16)
            in_chans=3
            if config['image_h']!=224 or config['image_w']!=224:
                self.vit_rect = True
                assert self.config['freeze_visual'] is False
                self.visual_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", local_files_only=True)
                
                embedding_size = self.visual_encoder.embeddings.position_embeddings.shape[-1]
                self.visual_encoder.embeddings.patch_embeddings.image_size = (config['image_h'], config['image_w'])
                self.visual_encoder.embeddings.patch_embeddings.num_patches = num_patches
                self.visual_encoder.embeddings.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embedding_size))
                #pos_embed = self.visual_encoder.embeddings.position_embeddings.data
                vit_state_dict = torch.load('vit_base.pth')
                #torch.save(self.visual_encoder.state_dict(), 'vit_base.pth')
                vit_state_dict['embeddings.position_embeddings'] = self.interpolate_pos_embed(vit_state_dict['embeddings.position_embeddings'], self.visual_encoder, config) 

                msg = self.visual_encoder.load_state_dict(vit_state_dict,strict=False)
            else:
                self.visual_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", local_files_only=True)

        elif config['arch'] == 'seg':
            self.image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0")
            self.image_processor.size['height'] = config['image_h']
            self.image_processor.size['width'] = config['image_w']
            self.visual_encoder = SegformerModel.from_pretrained("nvidia/mit-b0")
            self.segformer_mlp = SegformerFuser(output_dim=128, image_size=(config['image_h'], config['image_w'])) 
            num_patches = 1
            in_chans=256

        elif config['arch'] == 'res18':
            self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", local_files_only=True)
            resnet = models.resnet18(pretrained=True)
            modules = list(resnet.children())[:-4]
            
            modules.append(nn.Conv2d(128, 128, kernel_size=1, stride=2, bias=False))
            self.visual_encoder = nn.Sequential(*modules)
            
            #import ipdb; ipdb.set_trace()
            num_patches = 14*14
            in_chans = 128
            

        self.multi_encoder = LED_Multi(num_patches=num_patches,
                                        in_chans=in_chans,
                                        embed_dim=768,
                                        depth=self.config['multi_depth'],
                                        num_heads=12,
                                        cross_start_layer=0,
                                        arch = config['arch'],
                                        drop_rate=config['dropout_rate'],
                                        attn_drop_rate=config['dropout_rate'],
                                        drop_path_rate=config['dropout_rate'],
                                        fusion=config['fusion_prev_est'])

        ### Freeze model
        if config['freeze_lm']:
            print('Freezing the LM.')
            self.text_encoder.eval()      
            if config['freeze_lm_layer']=='self':
                print('Freezing self-attention in LM.')
                for name, param in self.text_encoder.bert.encoder.named_parameters():
                    if 'self' in name and 'attention' in name:
                        param.requires_grad = False
            else:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
        else:
            self.text_encoder.train()

        if config['freeze_visual']:
            print("Freezing the VM.")
            self.visual_encoder.eval()
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        else:
            self.visual_encoder.train()
    
        ## get text embedding
        self.text_embeddings = self.text_encoder.get_input_embeddings()
        for param in self.text_embeddings.parameters():
            param.requires_grad = False

        ## setup unet decoder
        if config['map_recon']:
            self.out_channels = 4
        elif config['new_bimap']:
            self.out_channels = 2
        else:
            self.out_channels = 1
       
        # decoder setup 
        if config['use_cls']:
            if config['heat_decoder'] == 'simple':
                heat_mlp = [nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 64*64)]
                self.heat_decoder = nn.Sequential(*heat_mlp)
            
            elif config['heat_decoder'] == 'linear': # inspired by Segmenter
                self.heat_decoder = nn.Linear(768, 1)
                
            elif config['heat_decoder'] == 'sep_unet':
                heat_mlp = [nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, 1024)]
                self.heat_decoder_layer1 = nn.Sequential(*heat_mlp)
                
                self.in_channels = self.out_channels = 5
                
                unet = [nn.ConvTranspose2d(self.in_channels, 32, kernel_size=2, stride=2), 
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, self.out_channels, kernel_size=3, padding=1)]
                self.unet = nn.Sequential(*unet)
            elif config['heat_decoder']=='unet':
                heat_mlp = [nn.Dropout(p=0.25), nn.Linear(768, 1024), nn.ReLU(), nn.Dropout(p=0.25),nn.Linear(1024, 1024)]
                self.heat_decoder_layer1 = nn.Sequential(*heat_mlp)
                
                unet = [#nn.Dropout(p=0.25),
                        nn.ConvTranspose2d(self.in_channels, 32, kernel_size=2, stride=2),
                        #nn.BatchNorm2d(32),
                        nn.ReLU(),
                        #nn.Dropout(p=0.25),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        #nn.BatchNorm2d(64),
                        nn.ReLU(),
                        #nn.Dropout(p=0.25),
                        nn.Conv2d(64, self.out_channels, kernel_size=3, padding=1)
                        ]
                self.unet = nn.Sequential(*unet)
                
            elif config['heat_decoder'] == 'unet_rect':
                heat_mlp = [nn.Dropout(p=0.25), nn.Linear(768, 1024), nn.ReLU(), nn.Dropout(p=0.25),nn.Linear(1024, 1344)]
                self.heat_decoder_layer1 = nn.Sequential(*heat_mlp)
                
                self.in_channels = self.out_channels = 1
                
                unet = [nn.ConvTranspose2d(self.in_channels, 32, kernel_size=2, stride=2),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, self.out_channels, kernel_size=3, padding=1)
                        ]
                self.unet = nn.Sequential(*unet)
 
        else:
            self.in_channels = 768

            if config['heat_decoder'] == 'linear': # inspired by Segmenter
                self.heat_decoder = nn.Linear(self.in_channels, self.out_channels)
            elif config['heat_decoder'] == 'unet': # unet
                unet = [nn.ConvTranspose2d(self.in_channels, 128, kernel_size=2, stride=2),
                        nn.ReLU(),
                        nn.Dropout(p=config['dropout_unet']),
                        nn.Conv2d(128, 64, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Dropout(p=config['dropout_unet']),
                        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                        nn.ReLU(),
                        nn.Dropout(p=config['dropout_unet']),
                        nn.Conv2d(32, self.out_channels, kernel_size=3, padding=1),
                        ]
                self.unet = nn.Sequential(*unet)
            elif config['heat_decoder']=='up':
                proj = [
                        nn.Upsample(size=(64,64), mode='bilinear', align_corners=True),
                        nn.Conv2d(self.in_channels,64,kernel_size=1,bias=False),
                        nn.ReLU(),
                        nn.Dropout(p=config['dropout_unet']),
                        nn.Upsample(size=(224,224), mode='bilinear', align_corners=True),
                        nn.Conv2d(64, 1, kernel_size=1,bias=False),
                        ]
                self.proj = nn.Sequential(*proj)
            elif config['heat_decoder']=='up_v2':
                proj = [
                        nn.Upsample(size=(64,64), mode='bilinear', align_corners=True),
                        nn.Conv2d(self.in_channels,128,kernel_size=3,bias=False),
                        nn.ReLU(),
                        nn.Dropout(p=config['dropout_unet']),
                        nn.Upsample(size=(112,112), mode='bilinear', align_corners=True),
                        nn.Conv2d(128,32,kernel_size=3,bias=False),
                        nn.ReLU(),
                        nn.Dropout(p=config['dropout_unet']),
                        
                        nn.Upsample(size=(224,224), mode='bilinear', align_corners=True),
                        nn.Conv2d(32, 1, kernel_size=1,bias=False),
                        ]
                self.proj = nn.Sequential(*proj)
        

    def interpolate_pos_embed(self, pos_embed_checkpoint, visual_encoder, config):        
        # interpolate position embedding
        #import ipdb; ipdb.set_trace()
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = visual_encoder.embeddings.patch_embeddings.num_patches
        num_extra_tokens = 1
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        #new_size = int(num_patches ** 0.5)
        h0 = config['image_h'] // 16
        w0 = config['image_w'] // 16
        new_scale = (h0 / orig_size, w0 / orig_size)
        if config['image_h']!=224 or config['image_w']!=224:
            # class_token and dist_token are kept unchanged
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, scale_factor=new_scale, mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            print('reshape position embedding from %d to %d'%(orig_size ** 2,h0*w0))
            
            return new_pos_embed    
        else:
            return pos_embed_checkpoint
       
    def train(self, mode=True):
        super(LED_Base, self).train(mode=mode)
        # Overwrite train() to ensure Frozen models remain frozen.
        if self.config['freeze_lm']:
            self.text_encoder.eval()
                     
        if self.config['freeze_visual']:
            self.visual_encoder.eval()
            
    def forward(self, image, text, attn_mask, turn_len):
        # text: B x turns x T (4, 9, 200)
        text = text.permute(1,0,2) # turns, B, T
        attn_mask = attn_mask.permute(1,0,2)
        B, C, H, W = image.shape
        N = B//5

       ## vision encoder 
        if self.config['arch']=='vit':
            image_output = self.visual_encoder(image)
            image_hidden = image_output.last_hidden_state
            image_hidden = image_hidden[:,1::,:] # drop CLS token
        elif self.config['arch']=='res18':
            #import ipdb; ipdb.set_trace()
            image_output = self.visual_encoder(image)
            embed_dim = image_output.size(1)
            image_hidden = image_output.view(B,embed_dim,-1).permute(0,2,1)
        elif self.config['arch']=='seg': # only last hidden is attened to text
            image_output = self.visual_encoder(image,  output_hidden_states=True, return_dict=True)
            all_hidden_states = image_output.hidden_states
            image_hidden = all_hidden_states[-1]
        else:
            ValueError('Not supported arch ')
            
        num_patches=image_hidden.size(1)
        embed_dim = 768
        
        # turn encoder
        all_heatmaps = []
        if self.config['fusion_prev_est']=='concat':
            prev_est = torch.zeros(B, num_patches, embed_dim, requires_grad=False).to(image.device)
            # prev_est = torch.normal(0, 0.02, size=(B, 196, 768), requires_grad=False).to(image.device)
        elif self.config['fusion_prev_est']=='multiply':       
            prev_est = torch.ones(B, num_patches, embed_dim, requires_grad=False).to(image.device)
        elif self.config['fusion_prev_est']=='concat_final':
            prev_est = torch.zeros(B, num_patches, 1, requires_grad=False).to(image.device)
        else:
            prev_est = None
            
        for i, (turn, mask) in enumerate(zip(text, attn_mask)): 
            # no more valid turn
            if (i+1) > turn_len.max().item():
                break
            text_embs = self.text_embeddings(turn)  # (N, T, D) using embedding?
            text_output = self.text_encoder(inputs_embeds=text_embs,
                                            attention_mask=mask,
                                            )
        
            text_hidden = text_output.last_hidden_state

            if self.config['reuse_hidden'] and i>0:
                input_embs = final_embs
            else:
                input_embs = image_hidden
            final_embs = self.multi_encoder(input_embs, 
                                        encoder_hidden_states = text_hidden,
                                        encoder_attention_mask = mask,
                                        prev_estimate = prev_est,
                                        )

            # processing segformer output
            if self.config['arch']=='seg':
                h=w = int(math.sqrt(final_embs.size(1)))
                final_embs = final_embs.view(B,h,w,-1).permute(0,3,1,2)
                all_hidden_states = list(all_hidden_states)
                all_hidden_states[-1] = final_embs
                heatmap = self.segformer_mlp(all_hidden_states)
                #import ipdb; ipdb.set_trace()
                return heatmap
            # use patch tokens of ViT to get heatmap
            if self.config['arch']=='vit':
                h0 = self.config['image_h'] // 16
                w0 = self.config['image_w'] // 16
            elif self.config['arch']=='res18':
                h0 = 14
                w0 = 14

            loc_embed = final_embs # B x 196 x D 

            if self.config['heat_decoder'] == 'linear':
                heatmap = self.heat_decoder(loc_embed)
                heatmap = heatmap.reshape(-1, h0, w0, self.out_channels).permute(0,3,1,2)
                heatmap = F.interpolate(heatmap, size=(self.config['image_h'], self.config['image_w']), mode='bilinear')
            
            elif self.config['heat_decoder'] == 'unet':
                heatmap = loc_embed.reshape(-1, h0, w0, self.in_channels).permute(0,3,1,2)
                heatmap = self.unet(heatmap) # x 4
                #heatmap = F.interpolate(heatmap, size=(self.config['image_h'], self.config['image_w']), mode='bilinear') 
            
            # update image_hidden with new state
            #image_hidden[:, 1::] += loc_embed
            #import ipdb; ipdb.set_trace()
            if self.config['use_prev_est']:
                if self.config['fusion_prev_est']=='concat':
                    prev_est = final_embs.detach()
                elif self.config['fusion_prev_est']=='multiply':
                    prev_est = final_embs.detach()
                elif self.config['fusion_prev_est']=='concat_final':
                    prev_est = heatmap.detach()
                    prev_est = F.interpolate(prev_est, size=(h0, w0), mode='bilinear') 
                    prev_est = prev_est.view(-1,1,h0*w0).permute(0,2,1)
            else:
                prev_est = None
                    
            all_heatmaps.append(heatmap)
        
        all_heatmaps = torch.stack(all_heatmaps)
        del prev_est
        return all_heatmaps
        
def mini_test():
    X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    Y = torch.tensor([[0], [1], [2]])

    # Get the corresponding values from X using Y
    corresponding_values = X[torch.arange(X.size(0)), Y.squeeze()]

    print(corresponding_values)



if __name__ == '__main__':
    mini_test()
