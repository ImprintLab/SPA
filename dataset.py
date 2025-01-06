""" train and test dataset

author Jiayuan Zhu
"""
import os
import pickle
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from monai.transforms import LoadImage, LoadImaged, Randomizable
from PIL import Image
from skimage import io
from skimage.transform import rotate
from torch.utils.data import Dataset

from utils import random_click, random_click_LIDC
from itertools import combinations
import nibabel as nib



class REFUGE(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):
        self.data_path = data_path
        self.subfolders = [f.path for f in os.scandir(os.path.join(data_path, mode + '-400')) if f.is_dir()]
        # if mode == 'Training':
        #     self.subfolders += [f.path for f in os.scandir(os.path.join(data_path, 'Validation' + '-400')) if f.is_dir()]
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, index):

        """Get the images"""
        subfolder = self.subfolders[index]
        name = subfolder.split('/')[-1]

        # raw image and raters path
        img_path = os.path.join(subfolder, name + '_cropped.jpg')
        multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '_cropped.jpg') for i in range(1, 8)]

        # raw image and raters images
        img = Image.open(img_path).convert('RGB')
        multi_rater_cup = [Image.open(path).convert('L') for path in multi_rater_cup_path]

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            multi_rater_cup = [torch.as_tensor((self.transform(single_rater) >=0.5).float(), dtype=torch.float32) for single_rater in multi_rater_cup]
            multi_rater_cup = torch.stack(multi_rater_cup, dim=0)

            torch.set_rng_state(state)

        if self.prompt == 'click':
            point_label_cup, pt_cup, selected_rater_cup, not_selected_rater_cup, selected_rater_mask_cup, selected_rater_mask_cup_ori = random_click(multi_rater_cup, self.mask_size)
            


        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'multi_rater': multi_rater_cup, 
            'p_label': point_label_cup,
            'pt':pt_cup, 
            'selected_rater': selected_rater_cup, 
            'not_selected_rater': not_selected_rater_cup, 
            'mask': selected_rater_mask_cup, 
            'mask_ori': selected_rater_mask_cup_ori,
            'image_meta_dict':image_meta_dict,
        }
    