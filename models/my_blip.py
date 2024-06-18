'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
from distutils.command.config import config
import warnings
warnings.filterwarnings("ignore")

from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import copy
import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file


class BLIP_Base(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                 
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)  

        
    def forward(self, image, caption, mode):
        
        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device) 
        
        if mode=='image':    
            # return image features
            image_embeds = self.visual_encoder(image)             
            return image_embeds
        
        elif mode=='text':
            # return text features
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')  
            return text_output.last_hidden_state
        
        elif mode=='multimodal':
            # return multimodel features
            image_embeds = self.visual_encoder(image)    
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)      
            
            text.input_ids[:,0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )              
            return output.last_hidden_state
        
        
        
class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 config = None,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   

        self.config = config
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        
        
        self.text_decoder = BertLMHeadModel(config=med_config)    
        
        #### Add token
        num_added = self.tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]', '[LOC]']})   
        print(f'len of tokenizer:', len(self.tokenizer))
        print(f'loc id:', self.tokenizer.additional_special_tokens_ids[1])
        self.text_decoder.resize_token_embeddings(len(self.tokenizer)) 
        self.loc_token_idx = self.tokenizer.additional_special_tokens_ids[1]
    
  
        #import ipdb; ipdb.set_trace()
        ### Freeze model
        if config['freeze_lm']:
            print('Freezing the LM.')
            self.text_decoder.eval()      
            if config['freeze_lm_layer']=='self':
                print('Freezing self-attention in LM.')
                for name, param in self.text_decoder.bert.encoder.named_parameters():
                    if 'self' in name and 'attention' in name:
                        param.requires_grad = False
            else:
                for param in self.text_decoder.parameters():
                    param.requires_grad = False
        
        else:
            self.text_decoder.train()

        if config['freeze_visual']:
            print("Freezing the VM.")
            self.visual_encoder.eval()
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        else:
            self.visual_encoder.train()
            
        self.input_embeddings = self.text_decoder.get_input_embeddings()
        
        for param in self.input_embeddings.parameters():
            param.requires_grad = True
            
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        if config['heat_decoder'] == 'simple':
            self.heat_decoder = nn.Linear(768, 384*384)
        
        #heat_mlp = [nn.Linear(768, 128*128), nn.ReLU(),nn.Linear(128*128, 384*384),nn.Dropout(0.5)]
        #self.heat_decoder = nn.Sequential(*heat_mlp)
        else:
            heat_mlp = [nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 1024)]
            self.heat_decoder_layer1 = nn.Sequential(*heat_mlp)
            
            in_channels = out_channels = 1
            
            self.upconv = nn.ConvTranspose2d(in_channels, 32, kernel_size=2, stride=2)
            
            self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU()


        
    def train(self, mode=True):
        super(BLIP_Decoder, self).train(mode=mode)
        # Overwrite train() to ensure Frozen models remain frozen.
        if self.config['freeze_lm']:
            self.text_decoder.eval()
                     
        if self.config['freeze_visual']:
            self.visual_encoder.eval()
        
    def forward(self, image, text, attn_mask, cap_len):
        
        B, C, H, W = image.shape
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        #tokens = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
        
        #text = self.tokenizer.bos_token_id
        #import ipdb; ipdb.set_trace()
        input_embs = self.input_embeddings(text)  # (N, T, D)
        
        if False:
            input_embs = torch.cat([input_embs, image_embeds], axis=1)
            attn_mask = torch.cat([attn_mask, image_atts], axis=1)
            
            
            decoder_output = self.text_decoder(inputs_embeds=input_embs, 
                                           attention_mask = attn_mask,                  
                                           labels = None,
                                           return_dict = True,   
                                           output_hidden_states = True,
                                          )  
        else:
        
            decoder_output = self.text_decoder(inputs_embeds=input_embs, 
                                               attention_mask = attn_mask, 
                                               encoder_hidden_states = image_embeds,
                                               encoder_attention_mask = image_atts,                  
                                               labels = None,
                                               return_dict = True,   
                                               output_hidden_states = True,
                                              )
        # loss_lm = decoder_output.loss
        if True:
            #import ipdb; ipdb.set_trace()
            last_hidden_states = decoder_output.hidden_states[-1]
            B, T, D = last_hidden_states.shape
            loc_indices = cap_len-2 # second to the last token
            
            loc_indices = loc_indices.unsqueeze(1).unsqueeze(2).expand(B, 1, D)
            loc_embed = last_hidden_states.gather(1, loc_indices).squeeze(1)
            
            if self.config['heat_decoder']=='simple':
                heatmap = self.heat_decoder(loc_embed)
                heatmap = heatmap.view(B, H, W)
            else: 
                #import ipdb; ipdb.set_trace()
                heatmap = self.heat_decoder_layer1(loc_embed)
                heatmap = heatmap.view(B, 1, 32, 32)
                heatmap = self.upconv(heatmap)
                heatmap = self.relu(heatmap)
                heatmap = self.conv1(heatmap)
                heatmap = self.relu(heatmap)
                heatmap = self.conv2(heatmap)
                
                #heatmap = heatmap.squeeze(1)
            #import ipdb; ipdb.set_trace()
            
            #heatmap = F.log_softmax(heatmap.view(B, -1), 1).view(B, H, W)
            return heatmap
        return decoder_output
        

    def pred_heatmap(self, loc_embed):
        #print('do pred here')
        output = self.heat_decoder(loc_embed)
        return output



    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        
        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device) 
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 

        if sample:
            #nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            
                                                  **model_kwargs)
        else:
            #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)            
            
        captions = []    
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(self.prompt):])
        return captions
    

#def blip_decoder(pretrained='',**kwargs):
#    model = BLIP_Decoder(**kwargs)
#    if pretrained:
#        model,msg = load_checkpoint(model,pretrained)
#        #assert(len(msg.missing_keys)==0)
#    return model 

     
# use vit as main network, bert output used as encoder_hidden_states for x-attention, CLS token of vit used for heatmap genearation 

class BLIP_Decoder_v2(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 config = None,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   

        self.config = config
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)    
        
        #### Add token
        if config['add_loc_token']:
            num_added = self.tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]', '[LOC]']})   
            print(f'len of tokenizer:', len(self.tokenizer))
            print(f'loc id:', self.tokenizer.additional_special_tokens_ids[1])
            self.text_decoder.resize_token_embeddings(len(self.tokenizer)) 
            self.loc_token_idx = self.tokenizer.additional_special_tokens_ids[1]
        else:
            self.loc_token_idx = None
    
  
        #import ipdb; ipdb.set_trace()
        ### Freeze model
        if config['freeze_lm']:
            print('Freezing the LM.')
            self.text_decoder.eval()      
            if config['freeze_lm_layer']=='self':
                print('Freezing self-attention in LM.')
                for name, param in self.text_decoder.bert.encoder.named_parameters():
                    if 'self' in name and 'attention' in name:
                        param.requires_grad = False
            else:
                for param in self.text_decoder.parameters():
                    param.requires_grad = False
        
        else:
            self.text_decoder.train()

        if config['freeze_visual']:
            print("Freezing the VM.")
            self.visual_encoder.eval()
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        else:
            self.visual_encoder.train()
            
        self.input_embeddings = self.text_decoder.get_input_embeddings()
        
        for param in self.input_embeddings.parameters():
            param.requires_grad = False
            
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        self.text_proj = nn.Linear(768, 768)
        if config['heat_decoder'] == 'simple':
            self.heat_decoder = nn.Linear(768, 64*64)
        
        #heat_mlp = [nn.Linear(768, 128*128), nn.ReLU(),nn.Linear(128*128, 384*384),nn.Dropout(0.5)]
        #self.heat_decoder = nn.Sequential(*heat_mlp)
        else:
            heat_mlp = [nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 1024)]
            self.heat_decoder_layer1 = nn.Sequential(*heat_mlp)
            
            in_channels = out_channels = 1
            
            self.upconv = nn.ConvTranspose2d(in_channels, 32, kernel_size=2, stride=2)
            
            self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU()


        
    # def train(self, mode=True):
    #     super(BLIP_Decoder_v2, self).train(mode=mode)
    #     # Overwrite train() to ensure Frozen models remain frozen.
    #     if self.config['freeze_lm']:
    #         self.text_decoder.eval()
                     
    #     if self.config['freeze_visual']:
    #         self.visual_encoder.eval()
        
    def forward(self, image, text, attn_mask, cap_len):
        
        B, C, H, W = image.shape
        
        
        #text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
        
        #text = self.tokenizer.bos_token_id
        #import ipdb; ipdb.set_trace()
        input_embs = self.input_embeddings(text)  # (N, T, D) using embedding?
        decoder_output = self.text_decoder(inputs_embeds=input_embs,
                                        attention_mask = attn_mask,
                                        labels=None,
                                        return_dict=True,
                                        output_hidden_states=True,
                                        mode=self.config['bert_mode'])
        
        last_hidden_states = decoder_output.hidden_states[-1]
        last_hidden_states = self.text_proj(last_hidden_states)

        image_embeds = self.visual_encoder(image, encoder_hidden_states=last_hidden_states, 
                                            encoder_attention_mask=attn_mask)

        #image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
    
        
        #loc_embed = image_embeds[:, 0].squeeze(1)
        # loss_lm = decoder_output.loss
        if True:
            #import ipdb; ipdb.set_trace()
            # last_hidden_states = decoder_output.hidden_states[-1]
            # B, T, D = last_hidden_states.shape
            # loc_indices = cap_len-2 # second to the last token
            
            # loc_indices = loc_indices.unsqueeze(1).unsqueeze(2).expand(B, 1, D)
            # loc_embed = last_hidden_states.gather(1, loc_indices).squeeze(1)
            #import ipdb; ipdb.set_trace()
            if False:
                loc_embed = image_embeds[:, 0].squeeze(1)
            else:
                loc_embed = image_embeds[:,1::].mean(1)
            
            if self.config['heat_decoder']=='simple':
                heatmap = self.heat_decoder(loc_embed)
                heatmap = heatmap.view(B, 64, 64)
            else: 
                #import ipdb; ipdb.set_trace()
                heatmap = self.heat_decoder_layer1(loc_embed)
                heatmap = heatmap.view(B, 1, 32, 32)
                heatmap = self.upconv(heatmap)
                heatmap = self.relu(heatmap)
                heatmap = self.conv1(heatmap)
                heatmap = self.relu(heatmap)
                heatmap = self.conv2(heatmap)
                
                heatmap = heatmap.squeeze(1)
            #import ipdb; ipdb.set_trace()
            
            #heatmap = F.log_softmax(heatmap.view(B, -1), 1).view(B, H, W)
            return heatmap
        return decoder_output
        

    def pred_heatmap(self, loc_embed):
        #print('do pred here')
        output = self.heat_decoder(loc_embed)
        return output

    
def blip_decoder2(pretrained='',**kwargs):
    model = BLIP_Decoder_v2(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        #assert(len(msg.missing_keys)==0)
    return model 

def blip_decoder(pretrained='',**kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        #assert(len(msg.missing_keys)==0)
    return model 
  
    
def blip_feature_extractor(pretrained='',**kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model        

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0,
                cross_start_layer=6):
        
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate,
                                           cross_start_layer=cross_start_layer
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg
    
