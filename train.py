import datetime
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from config import *
from resnext import D4Net
from util.folder_classify import PairLoader
import json
from util.misc import AvgMeter, check_mkdir, cal_accuracy


ckpt_path = './ckpt'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = {
    'exp_name': 'd4net',
    'margin1': 0.2,
    'margin2': 0.5,
    'iter_num': 20000,
    'train_batch_size': 16,
    'val_batch_size': 160,
    'last_iter': 0,
    'val_print_freq': 1000,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'resize_size': (224, 224),
    'snapshot': ''
}

input_size, _ = args['resize_size']

train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1),
        transforms.Resize(args['resize_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_transform = transforms.Compose([
    transforms.Resize(args['resize_size']),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# mode, transform

train_set = PairLoader(mode='train', transform=train_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], shuffle=True, num_workers=8, drop_last=True)

val_set = PairLoader(mode='val', transform=val_transform)
val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=12)

criterion = nn.CrossEntropyLoss().cuda()

def main():
    try:
        train(args['exp_name'])
    except:
        print 'exception'


def train(exp_name):

    log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')

    net = D4Net(num_cls=2)
    net.cuda().train()

    best_record = {}

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    if len(args['snapshot']) == 0:
        best_record['loss'] = 1e10
        best_record['iter'] = 0
        best_record['lr'] = args['lr']
    else:
        print('training resumes from \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, args['exp_name'], args['snapshot'] + '.pth')))
        optimizer.load_state_dict(
            torch.load(os.path.join(ckpt_path, args['exp_name'], args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

        best_record['loss'] = 1e10
        best_record['iter'] = 0
        best_record['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, args['exp_name']))
    with open(log_path, 'w') as f:
        f.write(str(args) + '\n\n')
    print 'start to train'

    curr_iter = args['last_iter']

    while True:
        class_loss_meter = AvgMeter()
        margin_loss_meter = AvgMeter()
        margin_loss2_meter = AvgMeter()
        acc_meter = AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            test_images, ref_images, labels = data
            batch_size = labels.size(0)

            test_images = Variable(test_images).cuda()
            ref_images = Variable(ref_images).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()

            outputs, cos_similarity = net(test_images, ref_images)

            class_loss = criterion(outputs, labels)

            pos_number = cos_similarity[labels == 1].size(0)

            if pos_number != batch_size:

                pos_cos_similarity = torch.min(cos_similarity[labels == 1])
                neg_cos_similarity = torch.max(cos_similarity[labels == 0])

                margin_loss = torch.max(Variable(torch.zeros_like(neg_cos_similarity)).cuda(), -torch.abs(neg_cos_similarity - pos_cos_similarity) + args['margin2'])
                margin2_loss = torch.max(Variable(torch.zeros_like(neg_cos_similarity)).cuda(), torch.abs(1 - pos_cos_similarity) -  args['margin1'])

                loss = margin_loss + class_loss + margin2_loss
                margin_loss_meter.update(margin_loss)
                margin_loss2_meter.update(margin2_loss)
            else:
                loss = class_loss

            loss.backward()
            optimizer.step()

            accuracy = cal_accuracy(outputs.data, labels.data)
            acc_meter.update(accuracy[0], batch_size)
            class_loss_meter.update(class_loss.item(), batch_size)

            log = '[iter %d], [class_loss %.4f], [margin_loss %.4f], [pos_1_diff %.4f], [train_acc %.4f], [lr %.8f]' % (
                curr_iter+1, class_loss_meter.avg, margin_loss_meter.avg, margin_loss2_meter.avg, acc_meter.avg, optimizer.param_groups[1]['lr'])
            print log
            with open(log_path, 'a') as f:
                f.write(log + '\n')

            if (curr_iter + 1) % args['val_print_freq'] == 0:

                val_loss, val_acc = validate(net)
                log = 'iter_%d_loss_%.4f_valacc_%.4f_lr_%.8f' % (
                    curr_iter + 1, val_loss, val_acc, optimizer.param_groups[1]['lr'])
                print '--------------------------------------------------------------------------------'
                print log

                if best_record['loss'] > val_loss:
                    best_record['loss'] = val_loss
                    best_record['valacc'] = val_acc
                    best_record['iter'] = curr_iter
                    best_record['lr'] = optimizer.param_groups[1]['lr']

                    torch.save(net.state_dict(), os.path.join(ckpt_path, args['exp_name'], log + '.pth'))
                    torch.save(optimizer.state_dict(),
                               os.path.join(ckpt_path, args['exp_name'], log + '_optim.pth'))

                with open(os.path.join(ckpt_path, args['exp_name'] + ".txt"), "a") as f:
                    f.write(log + '\n')

                print '[best]: [iter %d], [val_loss %.4f], [val_acc %.4f], [lr %.8f]' % (
                    best_record['iter'] + 1, best_record['loss'], best_record['valacc'], best_record['lr']
                )
                print '--------------------------------------------------------------------------------'

            curr_iter +=1
            if curr_iter > args['iter_num'] :
                return


def validate(net):
    print 'validating...'
    net.eval()

    val_length = len(val_loader)
    with torch.no_grad():
        loss_meter = AvgMeter()
        acc_meter = AvgMeter()
        for vi, data in enumerate(val_loader, 0):
            print '%d/%d' % (vi+1, val_length)
            test_images, ref_images, labels = data

            val_batch_size = labels.size(0)

            test_images = Variable(test_images).cuda()
            ref_images = Variable(ref_images).cuda()
            labels = Variable(labels).cuda()

            outputs, _ = net(test_images, ref_images)
            loss = criterion(outputs, labels)
            loss_meter.update(loss.item(), val_batch_size)

            accuracy = cal_accuracy(outputs.data, labels.data)
            acc_meter.update(accuracy[0], val_batch_size)

    net.train()
    return loss_meter.avg, acc_meter.avg


if __name__ == '__main__':
    main()