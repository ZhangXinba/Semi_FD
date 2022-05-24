from __future__ import print_function
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import datasets
import data
import distiller
import load_settings
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir,
                        args):
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)

#    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])
    train_transformation = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                             np.array([63.0, 62.1, 66.7]) / 255.0)
    ])

    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    if args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
    #        print('1111111111111111111111111111111111111111111111111111111111111111111111111111111')
    #        print(labels)
        labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)
        #print('22222222222222222222222222222222222222222222222222222222222222222222222222222222')
        #print(labeled_idxs)
        #print(unlabeled_idxs)

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.labeled_batch_size:
        batch_sampler = data.TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                             np.array([63.0, 62.1, 66.7]) / 255.0),
    ])

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transformation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, eval_loader

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='CIFAR-10 training')
#parser.add_argument('--data_path', type=str, default='../data')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                        choices=datasets.__all__,
                        help='dataset: ' +
                            ' | '.join(datasets.__all__) +
                            ' (default: cifar10)')
parser.add_argument('--train-subdir', type=str, default='train',
                        help='the subdirectory inside the data directory that contains the training data')
parser.add_argument('--eval-subdir', type=str, default='val',
                        help='the subdirectory inside the data directory that contains the evaluation data')
parser.add_argument('--labels', default=None, type=str, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')
parser.add_argument('--exclude-unlabeled', default=False, type=str2bool, metavar='BOOL',
                        help='exclude unlabeled examples from the training set')
parser.add_argument('--labeled-batch-size', default=None, type=int,
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
parser.add_argument('--paper_setting', default='a', type=str)
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--data_path',default=None, type=str, help='Path')
args = parser.parse_args()

gpu_num = 0
use_cuda = torch.cuda.is_available()
dataset_config = datasets.__dict__[args.dataset]()
num_classes = dataset_config.pop('num_classes')
train_loader, eval_loader = create_data_loaders(**dataset_config, args=args)
#trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
#testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)
#testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Model
t_net, s_net, args = load_settings.load_paper_settings(args)

# Module for distillation
d_net = distiller.Distiller(t_net, s_net)

print('the number of teacher model parameters: {}'.format(sum([p.data.nelement() for p in t_net.parameters()])))
print('the number of student model parameters: {}'.format(sum([p.data.nelement() for p in s_net.parameters()])))

if use_cuda:
    torch.cuda.set_device(0)
    d_net.cuda()
    s_net.cuda()
    t_net.cuda()
    cudnn.benchmark = True

criterion_CE = nn.CrossEntropyLoss()

# Training
def train_with_distill(d_net, epoch):
    epoch_start_time = time.time()
    print('\nDistillation epoch: %d' % epoch)

    d_net.train()
    d_net.s_net.train()
    d_net.t_net.train()

    train_loss = 0
    correct = 0
    total = 0

    global optimizer
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #print("targets: {}".format(targets))
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        batch_size = inputs.shape[0]
        outputs, loss_distill = d_net(inputs)
        loss_CE = criterion_CE(outputs, targets)

        loss = loss_CE + loss_distill.sum() / batch_size / 1000

        loss.backward()
        optimizer.step()

        train_loss += loss_CE.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()

        b_idx = batch_idx

    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

    return train_loss / (b_idx + 1)

def test(net):
    epoch_start_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(eval_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion_CE(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('Test \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (b_idx + 1), correct / total


print('Performance of teacher network')
test(t_net)

for epoch in range(args.epochs):
    if epoch is 0:
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif epoch is (args.epochs // 2):
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr / 10, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif epoch is (args.epochs * 3 // 4):
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr / 100, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    train_loss = train_with_distill(d_net, epoch)
    test_loss, accuracy = test(s_net)
