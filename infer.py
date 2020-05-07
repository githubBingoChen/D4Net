import datetime
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import time

from config import *
from resnext import D4Net
# from util.folder_mvtec import PairLoader
from util.folder_classify import PairLoader
import json
from util.misc import AvgMeter, check_mkdir, cal_metric
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ckpt_path = './ckpt'

args = {
    'exp_name': 'd4net',
    'val_batch_size': 157,
    'snapshot': 'd4net.pth',
}

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_set = PairLoader(mode='val', transform=val_transform)
# val_set = PairLoader(transform=val_transform)
val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=12, drop_last=False)

def main():

    snapshot_path = os.path.join(ckpt_path, args['exp_name'], args['snapshot'])

    net = D4Net(num_cls=2)
    net.load_state_dict(torch.load(snapshot_path))
    net.cuda().eval()

    with torch.no_grad():

        val_length = len(val_loader)
        acc_meter = AvgMeter()
        precision_meter = AvgMeter()
        recall_meter = AvgMeter()
        f1_meter = AvgMeter()

        for vi, data in enumerate(val_loader, 0):
            print '%d/%d' % (vi + 1, val_length)
            test_images, ref_images, labels = data

            val_batch_size = labels.size(0)

            test_images = Variable(test_images).cuda()
            ref_images = Variable(ref_images).cuda()
            labels = Variable(labels).cuda()

            outputs, _ = net(test_images, ref_images)

            accuracy, precision, recall, f1_score = cal_metric(outputs.data, labels.data)
            acc_meter.update(accuracy, val_batch_size)
            precision_meter.update(precision, val_batch_size)
            recall_meter.update(recall, val_batch_size)
            f1_meter.update(f1_score, val_batch_size)


        log1 = args['exp_name'] + 'mvtec'
        log2 = '\nacc:  precision:  recall:  f1:  FPS\n'
        log3 = '%.4f \t %.4f \t %.4f \t %.4f\n' % (
        acc_meter.avg, precision_meter.avg, recall_meter.avg, f1_meter.avg)

        with open('experiment_result.txt', 'a') as f:
            f.write(log1 + log2 + log3 + '\n\n')

if __name__ == '__main__':
    main()