import argparse
from tqdm import tqdm 

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as dataloader

from utils import * 
from models.allconv import AllConvNet
from models.wrn_prime import WideResNet
from RotDataset import RotDataset

def arg_parser():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--method', type=str, default='rot', help='rot, msp')
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


def train(args, epoch, model, train_loader, optimizer, lr_scheduler):
    model.train()
    train_loss = 0.
  
    for x_tf_0, x_tf_90, x_tf_180, x_tf_270, batch_y in tqdm(train_loader):  
        batch_size = x_tf_0.shape[0]

        batch_x = torch.cat([x_tf_0, x_tf_90, x_tf_180, x_tf_270], 0).cuda()    # batch_x: [bs*4, 3, 32, 32]
        batch_y = batch_y.cuda()                                                # batch_y: [bs]        
        batch_rot_y = torch.cat((                                               # batch_rot_y: [bs*4]
            torch.zeros(batch_size),
            torch.ones(batch_size),
            2 * torch.ones(batch_size),
            3 * torch.ones(batch_size)
        ), 0).long().cuda()

        optimizer.zero_grad()

        logits, pen = model(batch_x)
        
        classification_logits = logits[:batch_size]
        rot_logits  = model.rot_head(pen)

        # classification loss(only using not rotated images)
        classification_loss = F.cross_entropy(classification_logits, batch_y)
        # rotation loss
        rot_loss = F.cross_entropy(rot_logits, batch_rot_y)  
        
        # use self-supervised rotation loss 
        if args.method == 'rot':
            loss = classification_loss + args.rot_loss_weight * rot_loss 
        # baseline, maximum softmax probability
        elif args.method == 'msp':
            loss = classification_loss

        loss.backward()
        optimizer.step()

        train_loss += loss 

    return train_loss / len(train_loader)


def test(args, model, test_loader):
    model.eval()

    with torch.no_grad():
        loss = 0.
        acc = 0.

        for x_tf_0, x_tf_90, x_tf_180, x_tf_270, batch_y in tqdm(test_loader):  
            batch_size = x_tf_0.shape[0]

            batch_x = torch.cat([x_tf_0, x_tf_90, x_tf_180, x_tf_270], 0).cuda()
            batch_y = batch_y.cuda()
            
            batch_rot_y = torch.cat((
                torch.zeros(batch_size),
                torch.ones(batch_size),
                2 * torch.ones(batch_size),
                3 * torch.ones(batch_size)
            ), 0).long().cuda()
            
            logits, pen = model(batch_x)
            
            classification_logits = logits[:batch_size]
            rot_logits  = model.rot_head(pen)

            classification_loss = F.cross_entropy(classification_logits, batch_y)
            rot_loss = F.cross_entropy(rot_logits, batch_rot_y)  
            
            # use self-supervised rotation loss 
            if args.method == 'rot':
                loss += classification_loss + args.rot_loss_weight * rot_loss 
            # baseline, maximum softmax probability
            elif args.method == 'msp':
                loss += classification_loss

            # accuracy
            pred = classification_logits.data.max(1)[1]
            acc += pred.eq(batch_y.data).sum().item()
        
        return loss / len(test_loader), acc / len(test_loader.dataset)


def main():
    # arg parser 
    args = arg_parser()
    
    # set seed 
    set_seed(args.seed)  
  
    # dataset 
    id_traindata = datasets.CIFAR10('./data/', train=True, download=True)
    id_testdata = datasets.CIFAR10('./data/', train=False, download=True)

    id_traindata = RotDataset(id_traindata, train_mode=True)
    id_testdata = RotDataset(id_testdata, train_mode=False)

    # data loader  
    if args.method == 'rot' or args.method == 'msp':
        train_loader = dataloader(id_traindata, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    else:
        raise ValueError(args.method)

    test_loader = dataloader(id_testdata, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # model 
    num_classes = 10
    model = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    model.rot_head = nn.Linear(128, 4)
    model = model.cuda()
    
    # optimizer 
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True
    )

    # training 
    for epoch in range(1, args.epochs+1):
        
        train_loss = train(args, epoch, model, train_loader, optimizer, lr_scheduler=None)
        test_loss, test_acc = test(args, model, test_loader)

        print('epoch:{}, train_loss:{}, test_loss:{}, test_acc:{}'.format(epoch, round(train_loss.item(), 4), round(test_loss.item(), 4), round(test_acc, 4)))
        torch.save(model.state_dict(), './trained_model_{}.pth'.format(args.method))

if __name__ == "__main__":
   main()