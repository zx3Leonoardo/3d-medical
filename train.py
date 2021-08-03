from random import shuffle
from typing import OrderedDict
from numpy.lib.twodim_base import tri
from torch._C import device
import torch.optim as optim
from dataset.dataset import coronary_dataset
import torch
from torch.utils.data import DataLoader
import json
import os
import argparse
from config.config import *
from model.Unet import Unet
from tqdm import tqdm
from .utils import common, logger, losses, metrics, weights_init

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", required=True)
    parser.add_argument("--batch_size")
    args = parser.parse_args()
    return args

def train(model, epoch, train_loader, device, optimizer, loss_func, n_labels, alpha):
    print("=====Epoch:{}=====lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)
    
    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.float(), target.long()
        target = common.one_hot_3d(target, n_labels)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss0 = loss_func(output[0], target)
        loss1 = loss_func(output[1], target)
        loss2 = loss_func(output[2], target)
        #loss3 = loss_func(output[3], target)
        #loss = loss3 + alpha*(loss0+loss1+loss2)
        loss = loss2+alpha*(loss0+loss1)
        loss.backward()
        optimizer.step()

        train_dice.update(output[2], target)
        train_loss.update(loss2.item(), data.size(0)) #not clamp loss

        val_log = OrderedDict({'Train_Loss':train_loss.avg, 'Train_dice':train_dice.avg[1]})
    return val_log


def val(model, val_loader, loss_func, n_labels, device):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.float(), target.float()
            target = common.one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)

            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss':val_loss.avg, 'Val_dice':val_dice.avg})
    return val_log
        

def main():
    args = arg
    # save path
    save_path = os.path.join(args.save)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    device = torch.device('cuda')

    #dataset
    trainset = coronary_dataset(args, "training_ids")
    testset = coronary_dataset(args, "test_ids")
    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.threads, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.threads, shuffle=True)

    #model
    model = Unet(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    weights_init.init_weights(model, 'normal')
    common.print_network(model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # loss
    loss = losses.DiceLoss()

    # log
    log = logger.TrainLogger(save_path, "train_log")

    best_res = [0, 0]
    trigger = 0
    alpha = 0.4 # deep superision decay ratio

    for epoch in range(1,args.epochs+1):
        common.adjust_lr_v1(optimizer, epoch, args)
        train_log = train(model, epoch, train_loader, device, optimizer, loss, args.n_labels, alpha)
        val_log = val(model, test_loader, loss, args.n_labels, device)
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
        if epoch % 30 == 0:
            alpha *= 0.8
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping.")
                break
        torch.cuda.empty_cache()

if __name__=='__main__':
    main()

