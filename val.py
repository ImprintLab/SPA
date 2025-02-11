# train.py
#!/usr/bin/env	python3

""" valuate network using pytorch
    Jiayuan Zhu
"""

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from dataset import *
from conf import settings
import time
import cfg
from torch.utils.data import DataLoader
from utils import *
import function
import pandas as pd
from models.sam.modeling import EMWeights, EMMeanVariance


args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
net.EM_weights = EMWeights(n_components=16).to(GPUdevice)
net.EM_mean_variance = EMMeanVariance(se_dim = 256, pe_dim = 256, n_components=16).to(GPUdevice)

'''load pretrained model'''
assert args.weights != 0
print(f'=> resuming from {args.weights}')
assert os.path.exists(args.weights)
checkpoint_file = os.path.join(args.weights)
assert os.path.exists(checkpoint_file)
loc = 'cuda:{}'.format(args.gpu_device)
checkpoint = torch.load(checkpoint_file, map_location=loc)
start_epoch = checkpoint['epoch'] - 1

state_dict = checkpoint['state_dict']
net.load_state_dict(state_dict)
net.EM_weights.weights = checkpoint['EM_weights']
net.EM_mean_variance.means = checkpoint['EM_means']
net.EM_mean_variance.variances = checkpoint['EM_variances']


args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)

'''segmentation data'''
transform_train = transforms.Compose([
    transforms.Resize((args.image_size,args.image_size)),
    transforms.ToTensor(),
])

transform_train_seg = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((args.image_size,args.image_size)),
])

transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_test_seg = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((args.image_size, args.image_size)),
    
])

'''data end'''
if args.dataset == 'REFUGE':
    '''REFUGE data'''
    refuge_train_dataset = REFUGE(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training')
    refuge_test_dataset = REFUGE(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test')

    nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    '''end'''

'''begain valuation'''
net.eval()
tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, start_epoch, net, selected_rater_df_path=False)
logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {start_epoch}.')

    