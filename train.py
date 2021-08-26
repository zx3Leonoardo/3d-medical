from typing import OrderedDict
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch.optim as optim
from dataset.dataset import *
from torch.utils.data import DataLoader
import torch.nn as nn

import os
import numpy as np
import argparse
from tqdm import tqdm

from config.config import *
from model.Unet import *
from .utils import common, logger, losses, metrics, weights_init

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", required=True)
    parser.add_argument("--batch_size")
    args = parser.parse_args()
    return args

def train(model, epoch, train_loader, device, optimizer, loss_func, args):
    print("=====Epoch:{}=====lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(args.n_labels)
    
    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.float(), target.float()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        
        ce_loss = loss_func[0](output, target)
        print('ce loss: {}'.format(ce_loss.item()))

        output = torch.softmax(output, 1)
        dsc_loss = loss_func[1](output, target)
        print('dsc loss: {}'.format(dsc_loss.item()))

        loss = ce_loss * args.alpha + dsc_loss * args.beta
        print('total loss: {}'.format(loss.item()))
        loss.backward()
        optimizer.step()

        train_dice.update(output, target)
        train_loss.update(loss.item(), data.size(0)) #not clamp loss

        val_log = OrderedDict({'Train_Loss':train_loss.avg, 'Train_dice':train_dice.avg[1]})
    return val_log


def val(model, val_loader, loss_func, args, device):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(args.n_labels)
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.float(), target.float()
            data, target = data.to(device), target.to(device)
            output = model(data)
            ce_loss = loss_func[0](output, target)

            output = torch.softmax(output, 1)
            dsc_loss = loss_func[1](output, target)
            loss = ce_loss * args.alpha + dsc_loss * args.beta

            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss':val_loss.avg, 'Val_dice':val_dice.avg.mean()})
    return val_log
        

def main():
    args = arg
    # save path
    save_path = os.path.join(args.save)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # set device
    device = torch.device('cuda:{}'.format(args.gpu_ids[0]))

    #dataset
    trainset = coronary_dataset(args, "training_ids")
    testset = coronary_dataset(args, "test_ids")
    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.threads, shuffle=False)
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.threads, shuffle=False)

    #model
    model = unSymUnet(in_channel=1, out_channel=args.n_labels, training=True)
    weights_init.init_model(model)
    common.print_network(model)
    model.to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # loss
    weight = torch.Tensor(50 * torch.ones((1, args.n_labels, 1, 1, 1)))
    weight[:, 0, :] = 1
    weight[:, -2, :] = 6
    weight[:, -3, :] = 6
    weight = weight.to(device)

    loss = [nn.BCEWithLogitsLoss(pos_weight=weight), losses.losses.logDiceloss()]

    # log
    log = logger.TrainLogger(save_path, "train_log")

    best_res = [0, 0]
    trigger = 0
    alpha = 0.4 # deep superision decay ratio

    for epoch in range(1,args.epochs+1):
        common.adjust_lr_v1(optimizer, epoch, args)
        train_log = train(model, epoch, train_loader, device, optimizer, loss, args)
        val_log = val(model, test_loader, loss, args, device)
        log.update(epoch, train_log, val_log)

        # save checkpoint
        state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(),'epoch':epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_dice'] > best_res[1]:
            print('Saving Best Model.')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best_res[0] = epoch
            best_res[1] = val_log['Val_dice']
            trigger = 0
        print('Best performance at Epoch {}|{}'.format(best_res[0], best_res[1]))

        # deep supvision ratio decay
        # if epoch % 30 == 0:
        #     alpha *= 0.8

        # early stop
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping.")
                break
        torch.cuda.empty_cache()

if __name__=='__main__':
    main()

