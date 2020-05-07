import os

import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
from PIL import Image
from torch.utils import data
import json


input_root = '/media/b3-542/0DFD5CD11721CA55/mvtec_anomaly_detection'


def make_dataset():


    items = []
    dirs = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]
    for dir_name in sorted(dirs)[3:4]:
    # for dir_name in sorted(os.listdir(input_root))[1:2]:
        print(dir_name)
        # os.path.join(input_root, dir_name, 'test')
        img_root = os.path.join(input_root, dir_name, 'test')

        json_path = os.path.join(input_root, '%s.json' % (dir_name))
        if not os.path.isfile(json_path):
            continue
        h_json = open(json_path, 'r')
        f_json = json.load(h_json)

        img_list = []
        for img_dir in sorted(os.listdir(img_root)):
            # if 'good' not in img_dir:
            if 'good' in img_dir:
                continue
            for img_name in os.listdir(os.path.join(img_root, img_dir)):
                img_list.append(os.path.join(img_root, img_dir, img_name))

        img_list = sorted(img_list)
        for img_name in img_list:
            img_path1 = img_name
            # if img_path1 in delete_list:
            #     continue
            img_path2 = f_json[img_name]

            if (not os.path.isfile(img_path1)) or (not os.path.isfile(img_path2)) or (img_name not in f_json.keys()):
                continue

            gt = 0
            if 'good' in img_path1:
                gt = 1

            item = (img_path1, img_path2, gt)
            items.append(item)

        h_json.close()

    print('dataset filenum: ', len(items))
    return items


class PairLoader(data.Dataset):
    def __init__(self, transform):
        self.imgs = make_dataset()
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))
        self.transform = transform

    def __getitem__(self, index):
        img_path1, img_path2, gt = self.imgs[index]

        target_ori = Image.open(img_path1).convert('RGB')
        ref_ori = Image.open(img_path2).convert('RGB')

        if self.transform is not None:
            target_img = self.transform(target_ori)
            ref_img = self.transform(ref_ori)

        gt = int(gt)

        return target_img, ref_img, gt

    def __len__(self):
        return len(self.imgs)