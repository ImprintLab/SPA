# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Jiayuan Zhu
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import cfg
import function
from conf import settings
#from models.discriminatorlayer import discriminator
from dataset import *
from utils import *
from models.sam.modeling import EMWeights, EMMeanVariance

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
if args.pretrain:
    weights = torch.load(args.pretrain)
    net.load_state_dict(weights,strict=False)

net.EM_weights = EMWeights(n_components=16).to(GPUdevice)
net.EM_mean_variance = EMMeanVariance(se_dim = 256, pe_dim = 256, n_components=16).to(GPUdevice)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) 

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)


'''segmentation data'''
transform_train = transforms.Compose([
    transforms.Resize((args.image_size,args.image_size)),
    transforms.ToTensor(),
])

transform_train_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_test_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size)),
    transforms.ToTensor(),
])


if args.dataset == 'REFUGE':
    '''REFUGE data'''
    refuge_train_dataset = REFUGE(args, args.data_path, transform = transform_train, mode = 'Training')
    refuge_test_dataset = REFUGE(args, args.data_path, transform = transform_test, mode = 'Test')

    nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=2, pin_memory=True)
    nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True)
    '''end'''

'''checkpoint path and tensorboard'''
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
#use tensorboard
if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)
writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))

#create checkpoint folder to save model
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begain training'''
best_dice = 0.0

for epoch in range(settings.EPOCH):
        
    net.train()
    time_start = time.time()
    loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
    logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
    time_end = time.time()
    print('time_for_training ', time_end - time_start)

    net.eval()
    if epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:

        time_start = time.time()
        tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, selected_rater_df_path=False)
        logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_validation ', time_end - time_start)

        if  max(edice) > best_dice:
            best_dice = max(edice)
            is_best = True

            save_checkpoint({
            'epoch': epoch + 1,
            'model': args.net,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'path_helper': args.path_helper,
            'EM_weights': net.EM_weights.weights,  
            'EM_means': net.EM_mean_variance.means,  
            'EM_variances': net.EM_mean_variance.variances,  
        }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")
        else:
            is_best = False

writer.close()
