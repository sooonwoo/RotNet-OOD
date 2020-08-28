import argparse
import random 
from tqdm import tqdm 

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader as dataloader
import torchvision.datasets as datasets

from sklearn.metrics import roc_auc_score

from models.allconv import AllConvNet
from models.wrn_prime import WideResNet
from RotDataset import RotDataset
from utils import * 

def arg_parser():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--method', type=str, default='rot', help='rot, msp')
    parser.add_argument('--ood_dataset', type=str, default='cifar100', help='cifar100 | svhn')
    parser.add_argument('--num_workers', type=int, default=8)

    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--test_bs', type=int, default=200)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--rot-loss-weight', type=float, default=0.5, help='Multiplicative factor on the rot losses')

    # WRN Architecture
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')


    args = parser.parse_args()

    return args 


def main():
    # arg parser 
    args = arg_parser()
    
    # set seed 
    set_seed(args.seed)  
    
    # dataset 
    id_testdata = datasets.CIFAR10('./data/', train=False, download=True)
    id_testdata = RotDataset(id_testdata, train_mode=False)

    if args.ood_dataset == 'cifar100':
        ood_testdata = datasets.CIFAR100('./data/', train=False, download=True)
    elif args.ood_dataset == 'svhn':
        ood_testdata = datasets.SVHN('./data/', split='test', download=True)
    else:
        raise ValueError(args.ood_dataset)
    ood_testdata = RotDataset(ood_testdata, train_mode=False)
    
    # data loader  
    id_test_loader = dataloader(id_testdata, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    ood_test_loader = dataloader(ood_testdata, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
  
    # load model
    num_classes = 10
    model = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    model.rot_head = nn.Linear(128, 4)
    model = model.cuda()
    model.load_state_dict(torch.load('./models/trained_model_{}.pth'.format(args.method)))

    TODO:
    # 1. calculate ood score by two methods(MSP, Rot)

    # 2. calculate AUROC by using ood scores 
    

if __name__ == "__main__":
    main()
