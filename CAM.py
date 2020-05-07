# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import torch
import cv2
import os
import pdb

from resnext import D4Net



class Net(nn.Module):
    def __init__(self, num_cls):
        super(Net, self).__init__()
        self.net = D4Net(num_cls=2)
        self.layer0 = self.net.layer0
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

        # manus
        pretrained_path = os.path.join('./ckpt', 'd4net',
                                       'd4net.pth')
        self.net.load_state_dict(torch.load(pretrained_path))

    def forward(self, x1, x2):
        output = self.net(x1, x2)

        layer0_x1 = self.layer0(x1)
        layer1_x1 = self.layer1(layer0_x1)
        layer2_x1 = self.layer2(layer1_x1)
        layer3_x1 = self.layer3(layer2_x1)
        layer4_x1 = self.layer4(layer3_x1)

        layer0_x2 = self.layer0(x2)
        layer1_x2 = self.layer1(layer0_x2)
        layer2_x2 = self.layer2(layer1_x2)
        layer3_x2 = self.layer3(layer2_x2)
        layer4_x2 = self.layer4(layer3_x2)
        difference = layer4_x2 - layer4_x1

        return output, difference



img_root = 'the path to image directory'
img_name = ''
ref_name = ''

net = Net(num_cls=2)
finalconv_name = 'layer4'

net.eval()
with torch.no_grad():

    # hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 512x512
        size_upsample = (512, 512)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam


    preprocess = transforms.Compose([
       transforms.Resize((224,224)),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    img_path = os.path.join(img_root, img_name)
    ref_path = os.path.join(img_root, ref_name)
    img = Image.open(img_path)
    ref = Image.open(ref_path)

    img_tensor = preprocess(img)
    ref_tensor = preprocess(ref)
    img_variable = Variable(img_tensor.unsqueeze(0))
    ref_variable = Variable(ref_tensor.unsqueeze(0))

    logit, features_blobs = net(img_variable, ref_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    CAMs = returnCAM(features_blobs, weight_softmax, [idx[0]])


    # render the CAM and output
    # print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite( os.path.join('visual_result', 'result.jpg'), result)

