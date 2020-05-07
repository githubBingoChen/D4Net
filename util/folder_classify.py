import os

import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
from PIL import Image
from torch.utils import data
import json
from config import data_root




def make_dataset(mode):
    assert (mode in ['train', 'val'])

    dataset_path = os.path.join(data_root, mode)

    items = []
    for dir_name in os.listdir(dataset_path):

        lace_path = os.path.join(dataset_path, dir_name)
        if os.path.isdir(lace_path):

            json_path = os.path.join(lace_path, 'pair_infos.json')
            if not os.path.isfile(json_path):
                continue
            h_json = open(json_path, 'r')
            f_json = json.load(h_json)

            img_list = [os.path.splitext(f)[0] for f in os.listdir(lace_path) if 'img' in f]

            for img_name in img_list:
                img_path1 = os.path.join(lace_path, img_name + '.JPG')
                img_path2 = os.path.join(lace_path, 'ref' + img_name[3:] + '.JPG')

                if (not os.path.isfile(img_path1)) or (not os.path.isfile(img_path2)) or (img_name not in f_json.keys()):
                    continue

                item = (dir_name, img_path1, img_path2, f_json[img_name]['gt'])
                items.append(item)

            h_json.close()

    print(mode, ' dataset filenum: ', len(items))
    return items


class PairLoader(data.Dataset):
    def __init__(self, mode, transform):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        dir_name, img_path1, img_path2, gt = self.imgs[index]

        target_ori = Image.open(img_path1).convert('RGB')
        ref_ori = Image.open(img_path2).convert('RGB')

        if self.transform is not None:
            target_img = self.transform(target_ori)
            ref_img = self.transform(ref_ori)

        if gt == 1:
            gt = 1
        else:
            gt = 0
        gt = int(gt)

        return target_img, ref_img, gt

    def __len__(self):
        return len(self.imgs)

