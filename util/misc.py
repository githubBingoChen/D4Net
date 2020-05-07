import numpy as np
import torch
import cv2
from skimage.measure import compare_psnr, compare_ssim
import os
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, precision_recall_curve



class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)




# def calc_psnr(im1, im2):
#     im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
#     im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
#     return compare_psnr(im1_y, im2_y)

def calc_ssim(im1, im2):
    # im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    return compare_ssim(im1_y, im2_y)


def cal_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.div_(batch_size).item())
    return res


def cal_metric(output, target):
    maxk = 1
    _, pred = output.topk(maxk, 1, True, True)

    accuracy = accuracy_score(target, pred)
    precision = precision_score(target, pred, pos_label=0)

    recall = recall_score(target, pred, pos_label=0)
    f1_value = f1_score(target, pred, pos_label=0)
    return accuracy, precision, recall, f1_value


def cal_pr_curve(pred, target):
    precision, recall, thresholds = precision_recall_curve(target, pred, pos_label=0)
    return precision, recall, thresholds


def cal_metric_with_path(output, target, image_path):
    maxk = 1
    _, pred = output.topk(maxk, 1, True, True)

    for i in range(len(pred)):
        if (target[i] == 0 and  pred[i] != target[i]):
            print image_path[i]
            with open('failure_recall_error_record.txt', 'a') as f:
                f.write(image_path[i] + '\n')

    accuracy = accuracy_score(target, pred)
    precision = precision_score(target, pred, pos_label=0)

    recall = recall_score(target, pred, pos_label=0)
    f1_value = f1_score(target, pred, pos_label=0)
    return accuracy, precision, recall, f1_value





def get_center_loss(centers, features, target, alpha):
    batch_size = target.size(0)
    features_dim = features.size(1)
    target_expand = target.view(batch_size, 1).expand(batch_size, features_dim)
    centers_var = Variable(centers)
    centers_batch = centers_var.gather(0, target_expand)
    criterion = nn.MSELoss()
    center_loss = criterion(features, centers_batch)

    diff = centers_batch - features
    unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True,
                                                           return_counts=True)
    appear_times = torch.from_numpy(unique_count).gather(0, torch.from_numpy(unique_reverse))
    appear_times_expand = appear_times.view(-1, 1).expand(batch_size, features_dim).type(torch.FloatTensor)
    diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)
    diff_cpu *= alpha
    for i in range(batch_size):
        centers[target.data[i]] -= diff_cpu[i].type(centers.type())

    return center_loss, centers