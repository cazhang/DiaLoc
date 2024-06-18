from distutils.command.config import config
from pyexpat import model
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
import torch
import numpy as np
import copy
import json
import random
import os
#from led import utils
from PIL import Image, ImageDraw, ImageFont, ImageOps
import re
#from nltk.tokenize import word_tokenize
import torchvision.transforms.functional as TF


def create_image_of_text(text: str, width: int = 224, nrows: int = 2, color=(255, 255, 255), font=None) -> torch.Tensor:
  """Creates a (3, nrows * 14, width) image of text.

  Returns:
    cap_img: (3, 14 * nrows, width) image of wrapped text.
  """
  height = 12
  padding = 5
  effective_width = width - 2 * padding
  # Create a black image to draw text on.
  cap_img = Image.new('RGB', (effective_width * nrows, height), color = (0, 0, 0))
  draw = ImageDraw.Draw(cap_img)
  draw.text((0, 0), text, color, font=font or ImageFont.load_default())
  cap_img = TF.convert_image_dtype(TF.pil_to_tensor(cap_img), torch.float32)  # (3, height, W * nrows)
  cap_img = torch.split(cap_img, effective_width, dim=-1)  # List of nrow elements of shape (3, height, W)
  cap_img = torch.cat(cap_img, dim=1)  # (3, height * nrows, W)
  # Add zero padding.
  cap_img = torch.nn.functional.pad(cap_img, [padding, padding, 0, padding])
  return cap_img

class RandomResizeCrop:
    def __init__(self, size=[224, 224], scale=[0.75, 1.0], ratio=[0.95, 1.0]):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image, target):
        params = transforms.RandomResizedCrop.get_params(image, self.scale, self.ratio)
        image = TF.resized_crop(image, *params, self.size)
        target = TF.resized_crop(target, *params, self.size)
        
        return image, target

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, image, target):
        angle = random.choice(self.angles)
        image = TF.rotate(image, angle)
        target = TF.rotate(target, angle)
        return image, target

class LEDDataset(Dataset):
    def __init__(
        self,
        mode,
        args,
        config,
        texts,
        mesh_conversions,
        locations,
        viewPoint_location,
        dialogs,
        scan_names,
        levels,
        annotation_ids,
        tokenizer,
        image_processor,
        max_turns,
    ):
        self.mode = mode
        self.args = args
        self.config = config
        self.image_size = [
            3,
            config['image_h'],
            config['image_w'],
        ]
        
        self.target_size = [
            config['map_h'],
            config['map_w'],
        ]
        self.max_turns = max_turns
        self.dialog_history = config['dialog_history']

        if 'train' in self.mode:
            self.max_floors = config['max_floors']
        else:
            self.max_floors = config['max_floors_test']

        self.tokenizer = tokenizer
        self.texts = texts
        self.mesh_conversions = mesh_conversions   ## what is this?
        self.locations = locations
        self.viewPoint_location = viewPoint_location
        if len(dialogs)==2:
            self.dialogs = dialogs[0]
            self.gpt_dialogs = dialogs[1]
        else:
            self.dialogs = dialogs
            self.gpt_dialogs=None
        self.scan_names = scan_names
        self.levels = levels
        self.annotation_ids = annotation_ids
        self.mesh2meters = json.load(open(args.image_dir + "pix2meshDistance.json"))  ## what is this?
        self.max_len = args.max_len
        self.loc_token_idx = 30523
        self.font = None
        self.resize_ratio = 224/700

        self.RndCropResizeOp = RandomResizeCrop(size=(self.image_size[1], self.image_size[2]))
        self.RndRotateOp = MyRotationTransform(angles=[0, 180])
        self.ColorOp = transforms.ColorJitter(brightness=0.5, hue=0.1, saturation=0.1)
        self.NormalizeOp = transforms.Normalize(
                    mean=image_processor.image_mean,
                    std=image_processor.image_std,
                )

        self.preprocess_data_aug = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.5, hue=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=image_processor.image_mean,
                    std=image_processor.image_std,
                ),
            ]
        )

        self.preprocess_target_aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        self.preprocess_target = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )


        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=image_processor.image_mean,
                    std=image_processor.image_std,
                ),
            ]
        )

    def gather_all_floors(self, index, location, mesh_conversion):
        all_maps = torch.zeros(
            self.max_floors,
            self.image_size[0],
            self.image_size[1],
            self.image_size[2],
        )
        all_tars = torch.zeros(
            self.max_floors,
            self.image_size[1],
            self.image_size[2],
        )

        img_paths = []

        target = self.create_target(index, location, mesh_conversion)

        all_conversions = torch.zeros(self.max_floors, 1)
        sn = self.scan_names[index]
        floors = self.mesh2meters[sn].keys()

        if self.max_floors==1:
            floors = self.levels[index]

        #import ipdb; ipdb.set_trace()
        for enum, f in enumerate(floors):
            img_path = "{}floor_{}/{}_{}.png".format(self.args.image_dir, f, sn, f)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.image_size[2], self.image_size[1]))

            img_paths.append(img_path)
            if self.config['light_aug']: # original LED
                if "train" in self.mode:
                    all_maps[enum, :, :, :] = self.preprocess_data_aug(img)[:3, :, :]
                    #new_tar[enum] = self.preprocess_target_aug(target[enum].unsqueeze(0)*255).squeeze(0)
                else:
                    all_maps[enum, :, :, :] = self.preprocess(img)[:3, :, :]
                    #new_tar[enum] = self.preprocess_target(target[enum].unsqueeze(0)).squeeze(0)
                all_tars = target
            else:
                tar = target[enum].float()
                # tar = TF.to_tensor(img)[0] #debug
                tar = TF.to_pil_image(tar, 'F')
                #tar_tensor = TF.to_tensor(tar) 
                tar = tar.resize((self.image_size[2], self.image_size[1]))
                #tar = TF.resize(tar, (self.image_size[1], self.image_size[2]))
                # call transform
                img, tar = self.my_transform(img, tar, [self.RndCropResizeOp])
                all_maps[enum,...] = img
                all_tars[enum,...] = tar 

            all_conversions[enum, :] = self.mesh2meters[sn][f]["threeMeterRadius"] / 3.0
        return all_maps, all_tars, all_conversions, img_paths

    def get_info(self, index, path):
        info_elem = [
            path,
            self.levels[index],
            self.scan_names[index],
            self.annotation_ids[index],
            self.viewPoint_location[index],
        ]
        return info_elem

    def create_target(self, index, location, mesh_conversion):

        if self.max_floors>1:
            tar_floor = int(self.levels[index])
        else:
            tar_floor = 0
        gaussian_target = np.zeros(
            (self.max_floors, self.target_size[0], self.target_size[1])
        )
        gaussian_target[tar_floor, location[0], location[1]] = 1
        gaussian_target[tar_floor, :, :] = gaussian_filter(
            gaussian_target[tar_floor, :, :], sigma=mesh_conversion,
        )
        gaussian_target[tar_floor, :, :] = (
            gaussian_target[tar_floor, :, :]
            / gaussian_target[tar_floor, :, :].sum()
        )
        gaussian_target = torch.tensor(gaussian_target)
        return gaussian_target
  

    def my_transform(self, image, target, more_trans=[]):
        # both maps and targets are PIL image: n_floors x 224 x 224

        if 'train' in self.mode:
            # Color jittering
            image = self.ColorOp(image)
            if len(more_trans)<2:
                # Random horizontal flipping
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    target = TF.hflip(target)
                # Random vertical flipping
                if random.random() > 0.5:
                    image = TF.vflip(image)
                    target = TF.vflip(target)

            # crop and rotate
            if len(more_trans)>0:
                for trans in more_trans:
                    image, target = trans(image, target)

        # Transform to tensor
        image = TF.to_tensor(image)
        image = self.NormalizeOp(image)
        target = TF.to_tensor(target)
        target[target<0] = 0
        target = target / (target.sum()+1e-8)
        return image, target


    def __getitem__(self, index):
        location = copy.deepcopy(self.locations[index])
        location = np.round(np.asarray(location) * self.args.ds_percent).astype(int)
        mesh_conversion = self.mesh_conversions[index] * self.args.ds_percent

        # process texts
        #import ipdb; ipdb.set_trace()
        if self.gpt_dialogs is None or len(self.gpt_dialogs)==0:
            text = str(self.dialogs[index])
        else:
            rnd_num = torch.rand(1).item()
            #print(rnd_num)
            if rnd_num>0.5:
                text = str(self.dialogs[index])
            else:
                text = str(self.gpt_dialogs[index])
        #text += '[LOC]'
        turns = text.split('[LOC]') 
        #import ipdb; ipdb.set_trace()
        tokens_list = []
        mask_list = []
        past_turn = ''
        for turn in turns:
            if len(turn)==0:
                break
            if self.dialog_history:
                turn = past_turn+turn
                past_turn = turn 

            tokenized_data = self.tokenizer(
            turn,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_len)

            tokens = tokenized_data.input_ids[0]
            attention_mask = tokenized_data.attention_mask[0]

            tokens_list.append(tokens)
            mask_list.append(attention_mask)

        num_turns = len(tokens_list)

        if "train" in self.mode and self.config["dialog_cliping"]:
            assert self.config["dialog_history"]==True
            num_turns_clip = random.randint(1, num_turns)
            assert num_turns_clip>=1
            num_turns = num_turns_clip
            tokens_list = tokens_list[:num_turns]
            mask_list = mask_list[:num_turns]
        
        fake_tokens = torch.zeros(self.max_len)
        fake_mask = torch.zeros(self.max_len)

        for turn in range(num_turns, self.max_turns+1):
            tokens_list.append(fake_tokens)
            mask_list.append(fake_mask)

        tokens_list = torch.stack(tokens_list)
        mask_list = torch.stack(mask_list)

        self.font = self.font or ImageFont.load_default()
        cap_img = create_image_of_text(text, width=224, nrows=8, font=self.font)

        if tokens[-1] not in [self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
            tokens[-1] = self.loc_token_idx

        #import ipdb; ipdb.set_trace()
        maps, target, conversions, img_paths = self.gather_all_floors(index, location, mesh_conversion)
        level = self.levels[index]
        #target = self.create_target(index, location, mesh_conversion)
        info_elem = self.get_info(index, img_paths[0])

        #print(f'tokens ',tokens.shape, 'maps ', maps.shape, 'tar ', target.shape, 'conv ', conversions, 'level ', level)
        
        return (
            info_elem,
            target,
            maps,
            tokens_list,
            conversions,
            level,
            num_turns,
            mask_list,
            cap_img
        )

    def __len__(self):
        return len(self.annotation_ids)



class Loader:
    def __init__(self, args, config):
        self.mesh2meters = json.load(open(args.image_dir + "pix2meshDistance.json"))
        self.vocab = Vocabulary()
        self.max_length = 0
        self.max_dialog_length = 0
        self.datasets = {}
        self.args = args
        self.config = config

    def load_image_paths(self, data, mode):
        episode_ids, scan_names, levels, mesh_conversions, dialogs, turns = [], [], [], [], [], []
       
        gpt_dialogs = []
        for i, data_obj in enumerate(data):
            episode_ids.append(data_obj["episodeId"])
            scan_names.append(data_obj["scanName"])
            turns.append(len(data_obj['dialogArray'])//2)
            if self.config['add_loc_token']:
                dialogs.append(self.add_tokens_loc(data_obj["dialogArray"]))
            else:
                dialogs.append(self.add_tokens(data_obj["dialogArray"]))
            
            if 'use_gpt_dialog' not in self.config:
                self.config['use_gpt_dialog']=False
            if mode=='train' and self.config['use_gpt_dialog']:
                gpt_filename = os.path.join(self.args.data_dir, self.config['gpt_diag_dir'], f'gpt{i}.json')
                gptdata = json.load(open(gpt_filename))
                if self.config['add_loc_token']:
                    gpt_dialogs.append(self.add_tokens_loc(gptdata["gptDialogArray"][0]))
                else:
                    gpt_dialogs.append(self.add_tokens(gptdata["gptDialogArray"][0]))

            level = 0
            if mode != "test":
                level = str(data_obj["finalLocation"]["floor"])
            levels.append(level)
            mesh_conversions.append(
                self.mesh2meters[data_obj["scanName"]][str(level)]["threeMeterRadius"]
                / 3.0
            )
        return episode_ids, scan_names, levels, mesh_conversions, dialogs, turns, gpt_dialogs

    def add_tokens(self, message_arr):
        new_dialog = ""
        for enum, message in enumerate(message_arr):
            if enum % 2 == 1:
                new_dialog += message + " "
            else:
                new_dialog += message + " "

        return new_dialog

    def add_tokens_loc(self, message_arr):
        new_dialog = ""
        for enum, message in enumerate(message_arr):
            if enum % 2 == 1:
                new_dialog += message + "[LOC]"
            else:
                new_dialog += message + " "

        return new_dialog

    def load_locations(self, data, mode):
        if "test" in mode:
            return [[0, 0] for _ in data], ["" for _ in data]

        x = [
            [
                data_obj["finalLocation"]["pixel_coord"][1],
                data_obj["finalLocation"]["pixel_coord"][0],
            ]
            for data_obj in data
        ]

        y = [data_obj["finalLocation"]["viewPoint"] for data_obj in data]

        return x, y

    def build_pretrained_vocab(self, texts):

        self.vocab.word2idx = json.load(open(self.args.embedding_dir + "word2idx.json"))
        self.vocab.idx2word = json.load(open(self.args.embedding_dir + "idx2word.json"))
        ids = []
        seq_lengths = []

        for text in texts:
            text = re.sub(r"\.\.+", ". ", text)
            line_ids = []

            words = word_tokenize(text.lower())
            words=[word.lower() for word in words if word.isalpha()]

            self.max_length = max(self.max_length, len(words))
            for word in words:
                line_ids.append(self.vocab.word2idx[word])
            ids.append(line_ids)
            seq_lengths.append(len(words))
        text_ids = np.array([row + [0] * (self.max_length - len(row)) for row in ids])
        return text_ids, seq_lengths

    def build_dataset(self, file, tokenizer, image_processor):
        mode = file.split("_")[0]
        print("[{}]: Loading JSON file...".format(mode))
        data = json.load(open(self.args.data_dir + file))#[:20]
        #import ipdb; ipdb.set_trace()

        print("[{}]: Using {} samples".format(mode, len(data)))
        locations, viewPoint_location = self.load_locations(data, mode)
        (
            episode_ids,
            scan_names,
            levels,
            mesh_conversions,
            dialogs,
            turns,
            gpt_dialogs,
        ) = self.load_image_paths(data, mode)

        print(f'max turns: {max(turns)}, min turns: {min(turns)}, mean: {np.mean(turns)}')
        texts = copy.deepcopy(dialogs)
        max_turns = max(turns)
        seq_lengths = None
        print("[{}]: Building dataset...".format(mode))

        
        if mode=='train':
            dialogs_list = []
            dialogs_list.append(dialogs)
            dialogs_list.append(gpt_dialogs)
        else:
            dialogs_list = dialogs
        dataset = LEDDataset(
            mode,
            self.args,
            self.config,
            texts,
            mesh_conversions,
            locations,
            viewPoint_location,
            dialogs_list,
            scan_names,
            levels,
            episode_ids,
            tokenizer,
            image_processor,
            max_turns,
        )
        self.datasets[mode] = dataset
        print("[{}]: Finish building dataset...".format(mode))


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}

    def add_word(self, word, mode):
        if word not in self.word2idx and mode in ("train"):
            idx = len(self.idx2word)
            self.idx2word[idx] = word
            self.word2idx[word] = idx
            return word
        elif word not in self.word2idx and mode != "train":
            return "<unk>"
        else:
            return word

    def __len__(self):
        return len(self.idx2word)