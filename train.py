from random import shuffle
import torch.optim as optim
from dataset.dataset import coronary_dataset
import torch
from torch.utils.data import Dataloader
import json
import os
import argparse
from config.config import *
from model.Unet import Unet
from utils import weights_init,common

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", required=True)
    parser.add_argument("--batch_size")
    args = parser.parse_args()
    return args

def main():
    args = arg
    # save path
    save_path = os.path.join(args.save)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    device = torch.device('cuda')

    #dataset
    trainset = coronary_dataset(args, "training")
    testset = coronary_dataset(args, "test")
    train_loader = Dataloader(trainset, batch_size=args.batch_size, num_workers=args.threads, shuffle=True)
    test_loader = Dataloader(testset, batch_size=args.batch_size, num_workers=args.threads, shuffle=True)

    #model
    model = Unet(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    weights_init.init_weights(model, 'normal')
    common.print_network(model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # loss
    loss = 


