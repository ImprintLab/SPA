
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from skimage import io
from tqdm import tqdm

import cfg
from utils import *
import pandas as pd
from sklearn.cluster import KMeans


args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones([1]).cuda(device=GPUdevice)*2)
seed = torch.randint(1,11,(args.b,7))
torch.backends.cudnn.benchmark = True

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, runs=6, vis = 50):
    epoch_loss = 0
    ind = 0
    
    num_sample = 48
    n_clusters = 4

    # train mode
    net.train()
    optimizer.zero_grad()
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        
        for pack in train_loader:
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)

            if 'multi_rater' in pack:
                multi_rater = pack['multi_rater'].to(dtype = torch.float32, device = GPUdevice)

            if 'pt' in pack:
                pt = pack['pt'].unsqueeze(1)
                point_labels = pack['p_label'].unsqueeze(1)

            if 'selected_rater' in pack:
                selected_rater = pack['selected_rater']
                masks_all = pack['mask'].unsqueeze(1).repeat(1, runs, 1, 1, 1).to(device = GPUdevice)
                masks_ori_all = pack['mask_ori'].unsqueeze(1).repeat(1, runs, 1, 1, 1).to(device = GPUdevice)
                
            name = pack['image_meta_dict']['filename_or_obj']
            
            ind += 1
            b_size,c,w,h = imgs.size()
                   
            coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)

            '''init'''
            imgs = imgs.to(dtype = torch.float32, device = GPUdevice)            
            weights = torch.tensor(net.EM_weights.weights, dtype=torch.float, device = GPUdevice)
            pred_masks_weights_list = weights.unsqueeze(0).repeat(imgs.size(0), 1) 
            means = torch.tensor(net.EM_mean_variance.means, dtype=torch.float, device = GPUdevice)
            variances = torch.tensor(net.EM_mean_variance.variances, dtype=torch.float, device = GPUdevice)
            last_pred = None


            for run in range(runs):
                masks = masks_all[:, run, :, :, :] 
                masks_ori = masks_ori_all[:, run, :, :, :] 

                
                '''Train image encoder, combine net(inside image encoder), mask decoder'''
                # prompt encoder
                for n, value in net.prompt_encoder.named_parameters(): 
                    value.requires_grad = False
                se, de = net.prompt_encoder( 
                    points=(coords_torch, labels_torch),
                    boxes=None,
                    masks=last_pred,
                )
                pe = net.prompt_encoder.get_dense_pe().to(device = GPUdevice)

                # EM_mean_variance
                for n, value in net.EM_mean_variance.named_parameters(): 
                    value.requires_grad = False
                means, variances = net.EM_mean_variance(se, pe)

                # EM_weights
                for n, value in net.EM_weights.named_parameters(): 
                    value.requires_grad = False
                weights= net.EM_weights(pred_masks_weights_list)
                weights = weights.mean(axis=0).to(device = GPUdevice)
                weights /= weights.sum() 
                
                # image encoder & combine net
                for n, value in net.image_encoder.named_parameters(): 
                    value.requires_grad = True
                imge_list = net.image_encoder(imgs, weights, means, variances, num_sample=num_sample)  
                
                
                # mask decoder
                for n, value in net.mask_decoder.named_parameters(): 
                    value.requires_grad = True

                pred_list_last_pred = []
                pred_list_image_size = []
                pred_list_output_size = []
                for i in range(len(imge_list)):
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge_list[i],
                        image_pe=pe, 
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=(args.multimask_output > 1),
                    )
                    # for last_pred
                    pred_list_last_pred.append(pred)

                    # Resize to the image size
                    pred_image_size = F.interpolate(pred,size=(args.image_size, args.image_size)) 
                    # standardlise before cluster
                    if torch.max(pred_image_size) > 1 or torch.min(pred_image_size) < 0:
                        pred_image_size = torch.sigmoid(pred_image_size)
                    pred_list_image_size.append(pred_image_size)

                    # Resize to the output size
                    pred_output_size = F.interpolate(pred,size=(args.out_size, args.out_size))
                    pred_list_output_size.append(pred_output_size)


                # result for last_pred
                pred_list_last_pred = torch.stack(pred_list_last_pred, dim=0)
                last_pred = torch.mean(pred_list_last_pred, dim=0).detach().clone()

                # result for output_size
                pred_list_output_size = torch.stack(pred_list_output_size, dim=0)
                output = torch.mean(pred_list_output_size, dim=0)

                #pred_list_output = (pred_list_output> 0.5).float()
                loss = lossfunc(output, masks)

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


                # generate multiple options: from 2D to 1D
                flattened_pred_list = [pred.detach().cpu().numpy().flatten() for pred in pred_list_image_size]  
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(flattened_pred_list)
                target_group = kmeans.predict([masks_ori.cpu().numpy().flatten()])[0] 

                flag_select = (kmeans.labels_ == target_group)
                exclusive_list = [single_imge for single_imge, flag in zip(pred_list_image_size, flag_select) if not flag]
                select_list = [single_imge for single_imge, flag in zip(pred_list_image_size, flag_select) if flag]

                exclusive_list = torch.stack(exclusive_list, dim=0) 
                exclusive_list_mean = torch.mean(exclusive_list, dim=0)

                select_list = torch.stack(select_list, dim=0) 
                select_list_mean = torch.mean(select_list, dim=0) 


                # find pt,label for training mean & variance
                pt_temp_list = []
                point_labels_temp_list = []

                for i in range(select_list_mean.size(0)):

                    flat_diff = torch.abs(select_list_mean[i,0]-exclusive_list_mean[i,0]).view(-1)
                    top_values, top_indices = torch.topk(flat_diff, 20) 

                    top_2D_indices = [torch.tensor([(torch.div(index, select_list_mean.size(2), rounding_mode='floor')).item(), (index % select_list_mean.size(3)).item()]) for index in top_indices]
                    potential_selected = torch.stack(top_2D_indices, dim=0)  
                    select_index = torch.tensor(np.random.randint(len(potential_selected), size = 1))[0]

                    pt_temp = potential_selected[select_index]
                    point_labels_temp = masks_ori[i, 0, pt_temp[0], pt_temp[1]]
                    pt_temp_list.append(pt_temp)
                    point_labels_temp_list.append(point_labels_temp)

                pt_temp = torch.stack(pt_temp_list, dim=0).to(device=GPUdevice, dtype=torch.float)
                point_labels_temp = torch.stack(point_labels_temp_list, dim=0).to(device=GPUdevice, dtype=torch.int)
                coords_torch = torch.cat((coords_torch, pt_temp.unsqueeze(1)), dim=1)
                labels_torch = torch.cat((labels_torch, point_labels_temp.unsqueeze(1)), dim=1)


                # calculate current weights (output size)
                pred_masks_weights_list = []
                for i in range(imgs.size(0)):
                    pred_masks_weights_list.append(net.EM_weights.compute_weights(
                        torch.flatten(select_list_mean[i].detach().clone()), weights, means, variances))
                pred_masks_weights_list = torch.stack(pred_masks_weights_list, dim=0) #(batch_size, n_components)
                            

                #-----------------------------------------------------------------------

                """train prompt_encoder & EM_mean_variance"""
                # prompt encoder
                for n, value in net.prompt_encoder.named_parameters(): 
                    value.requires_grad = True
                se, de = net.prompt_encoder(
                    points=(coords_torch, labels_torch),
                    boxes=None,
                    masks=last_pred,
                )
                pe = net.prompt_encoder.get_dense_pe().to(device = GPUdevice) 

                # EM_mean_variance
                for n, value in net.EM_mean_variance.named_parameters(): 
                    value.requires_grad = True
                means, variances = net.EM_mean_variance(se, pe)

                # EM_weights
                for n, value in net.EM_weights.named_parameters(): 
                    value.requires_grad = False
                weights= net.EM_weights(pred_masks_weights_list)
                weights = weights.mean(axis=0).to(device = GPUdevice)
                weights /= weights.sum() # avoid 0.99999 not sum to 1

                # image encoder & combine net
                for n, value in net.image_encoder.named_parameters(): 
                    value.requires_grad = False
                imge_list = net.image_encoder(imgs, weights, means, variances, num_sample=num_sample)  

                # mask decoder
                for n, value in net.mask_decoder.named_parameters(): 
                    value.requires_grad = False

                pred_list_output = []
                for i in range(len(imge_list)):
                    pred, _ = net.mask_decoder(
                        image_embeddings=imge_list[i],
                        image_pe=pe, 
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de, 
                        multimask_output=(args.multimask_output > 1),
                    )
                    # Resize to the ordered output size
                    pred_list_output.append(F.interpolate(pred,size=(args.out_size,args.out_size)))
            
         
                # result for out size pred
                pred_list_output = torch.stack(pred_list_output, dim=0)
                pred_list_output = torch.mean(pred_list_output, dim=0)  

                loss = lossfunc(pred_list_output, masks)

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


                #-----------------------------------------------------------------------
                """train EM_weights"""
                # prompt encoder
                for n, value in net.prompt_encoder.named_parameters(): 
                    value.requires_grad = False
                se, de = net.prompt_encoder(
                    points=(coords_torch, labels_torch),
                    boxes=None,
                    masks=last_pred,
                )
                pe = net.prompt_encoder.get_dense_pe().to(device = GPUdevice) 

                # EM_mean_variance
                for n, value in net.EM_mean_variance.named_parameters(): 
                    value.requires_grad = False
                means, variances = net.EM_mean_variance(se, pe)


                # calculate current weights (output size)
                pred_masks_weights_list_train = []
                true_masks_weights_list_train = []
                for i in range(imgs.size(0)):
                    pred_masks_weights_list_train.append(net.EM_weights.compute_weights(
                        torch.flatten(select_list_mean[i].detach().clone()), weights, means, variances))
                    true_masks_weights_list_train.append(net.EM_weights.compute_weights(
                        torch.flatten(masks_ori[i]), weights, means, variances))
                    
                pred_masks_weights_list_train = torch.stack(pred_masks_weights_list_train, dim=0) #(batch_size, n_components)
                true_masks_weights_list_train = torch.stack(true_masks_weights_list_train, dim=0) #(batch_size, n_components)

                # EM_weights
                for n, value in net.EM_weights.named_parameters(): 
                    value.requires_grad = True
                updated_weights_list = net.EM_weights(pred_masks_weights_list_train)  

                loss = nn.MSELoss()(updated_weights_list, true_masks_weights_list_train)

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                epoch_loss += loss.item()

                loss.backward()                    
                optimizer.step()
                optimizer.zero_grad()

            pbar.update()

    return epoch_loss/len(train_loader)


def validation_sam(args, val_loader, epoch, net: nn.Module, runs=6, selected_rater_df_path = False):
    # eval mode
    net.eval()

    n_val = len(val_loader)  # the number of batch
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    
    num_sample = 48
    n_clusters = 4

    total_loss_list = np.zeros(runs)
    total_eiou_list = np.zeros(runs)
    total_dice_list = np.zeros(runs)

    lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:

        for ind, pack in enumerate(val_loader):
            
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']

            if 'multi_rater' in pack:
                multi_rater = pack['multi_rater'].to(dtype = torch.float32, device = GPUdevice) 
                
            if selected_rater_df_path != False: 
                selected_rater, masks_all, masks_ori_all = selected_rater_from_df(args, multi_rater, name, selected_rater_df_path, epoch)
                masks_all = masks_all.unsqueeze(1).repeat(1, runs, 1, 1, 1).to(device = GPUdevice)
                masks_ori_all = masks_ori_all.unsqueeze(1).repeat(1, runs, 1, 1, 1).to(device = GPUdevice)

                pt = pack['pt'].unsqueeze(1)
                point_labels = pack['p_label'].unsqueeze(1)

            else:
                pt = pack['pt'].unsqueeze(1)
                point_labels = pack['p_label'].unsqueeze(1)

                selected_rater = pack['selected_rater']
                masks_all = pack['mask'].unsqueeze(1).repeat(1, runs, 1, 1, 1).to(device = GPUdevice)
                masks_ori_all = pack['mask_ori'].unsqueeze(1).repeat(1, runs, 1, 1, 1).to(device = GPUdevice)
                
            ind += 1
            coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)

            '''init'''
            imgs = imgs.to(dtype = torch.float32, device = GPUdevice)

            weights = torch.tensor(net.EM_weights.weights, dtype=torch.float, device = GPUdevice)
            pred_masks_weights_list = weights.unsqueeze(0).repeat(imgs.size(0), 1) # repeate with batch_size
            means = torch.tensor(net.EM_mean_variance.means, dtype=torch.float, device = GPUdevice)
            variances = torch.tensor(net.EM_mean_variance.variances, dtype=torch.float, device = GPUdevice)

            
            last_pred = None
            
            '''test'''
            with torch.no_grad():
                # prompt encoder
                se, de = net.prompt_encoder( 
                    points=(coords_torch, labels_torch),
                    boxes=None,
                    masks=None,
                )
                pe = net.prompt_encoder.get_dense_pe().to(device = GPUdevice) 
                    
                # EM_mean_variance
                means, variances = net.EM_mean_variance(se, pe)



            for run in range(runs):
                masks = masks_all[:, run, :, :, :] 
                masks_ori = masks_ori_all[:, run, :, :, :] 
                # showp = coords_torch[:,run,:]

                with torch.no_grad():

                    # EM_weights
                    weights= net.EM_weights(pred_masks_weights_list)
                    weights = weights.mean(axis=0).to(device = GPUdevice)
                    weights /= weights.sum() # avoid 0.99999 not sum to 1

                    # image encoder & combine net
                    imge_list = net.image_encoder(imgs, weights, means, variances, num_sample=num_sample) 
                    
                    # mask decoder
                    pred_list_last_pred = []
                    pred_list_image_size = []
                    pred_list_output_size = []
                    for i in range(len(imge_list)):
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge_list[i],
                            image_pe=pe, 
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de, 
                            multimask_output=(args.multimask_output > 1),
                        )

                        # for last_pred
                        pred_list_last_pred.append(pred)

                        # Resize to the image size
                        pred_image_size = F.interpolate(pred,size=(args.image_size, args.image_size)) 
                        # standardlise before cluster
                        if torch.max(pred_image_size) > 1 or torch.min(pred_image_size) < 0:
                            pred_image_size = torch.sigmoid(pred_image_size)
                        pred_list_image_size.append(pred_image_size)

                        # Resize to the output size
                        pred_output_size = F.interpolate(pred,size=(args.out_size, args.out_size))
                        pred_list_output_size.append(pred_output_size)


                    # result for last_pred
                    pred_list_last_pred = torch.stack(pred_list_last_pred, dim=0)
                    last_pred = torch.mean(pred_list_last_pred, dim=0).detach().clone()

                    # result for output_size
                    pred_list_output_size = torch.stack(pred_list_output_size, dim=0)
                    output = torch.mean(pred_list_output_size, dim=0)
                    output = (output> 0.5).float()


                    temp = eval_seg(output, masks, threshold)
                    loss = lossfunc(output, masks)

                    total_loss_list[run] += loss.item()
                    total_eiou_list[run] += temp[0]
                    total_dice_list[run] += temp[1]
                    
                    
                    '''vis images'''
                    if ind % args.vis == 0:
                        namecat = 'Test'
                        for na in name:
                            namecat = namecat + na + '+' #.split('/')[-1].split('.')[0] + '+'
                        vis_image(imgs,output,masks.clone(), os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '+iteration+'+str(run+1)+ '.jpg'), reverse=False, points=None)



                    """find the output for specific cluster to remind user"""
                    # from 2D to 1D 
                    flattened_pred_list = [pred.cpu().numpy().flatten() for pred in pred_list_image_size]
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(flattened_pred_list)
                    target_group = kmeans.predict([masks_ori.cpu().numpy().flatten()])[0]


                    for cluster in range(n_clusters):
                        flag_select = (kmeans.labels_ == cluster)

                        temp_exclusive_list = [single_imge for single_imge, flag in zip(pred_list_image_size, flag_select) if not flag]
                        temp_select_list = [single_imge for single_imge, flag in zip(pred_list_image_size, flag_select) if flag]

                        temp_exclusive_list = torch.stack(temp_exclusive_list, dim=0) 
                        temp_exclusive_list_mean = torch.mean(temp_exclusive_list, dim=0)
                            
                        temp_select_list = torch.stack(temp_select_list, dim=0) 
                        temp_select_list_mean = torch.mean(temp_select_list, dim=0) 
                        
                        plot_image = F.interpolate(temp_select_list_mean,size=(args.out_size,args.out_size))
                        plot_image = (plot_image> 0.5).float()

                        '''vis images'''
                        if ind % args.vis == 0:
                            namecat = 'Test'
                            for na in name:
                                namecat = namecat + na + '+' #.split('/')[-1].split('.')[0] + '+'
                            #vis_image(imgs,plot_image,masks.clone(), os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '+iteration+'+str(run+1)+ '+cluster'+ str(cluster)+'.jpg'), reverse=False, points=None)


                        # only find pt, label for training weights, mean & variance
                        if cluster == target_group:

                            final_select_list_mean = temp_select_list_mean

                            pt_temp_list = []
                            point_labels_temp_list = []
                            
                            for i in range(temp_select_list_mean.size(0)):
                        
                                flat_diff = torch.abs(temp_select_list_mean[i,0]-temp_exclusive_list_mean[i,0]).view(-1)
                                top_values, top_indices = torch.topk(flat_diff, 20) # Get the indices of the top 20 differences

                                top_2D_indices = [torch.tensor([(torch.div(index, temp_select_list_mean.size(2), rounding_mode='floor')).item(), (index % temp_select_list_mean.size(3)).item()]) for index in top_indices]
                                potential_selected = torch.stack(top_2D_indices, dim=0)  
                                select_index = torch.tensor(np.random.randint(len(potential_selected), size = 1))[0]

                                pt_temp = potential_selected[select_index]
                                point_labels_temp = masks_ori[i, 0, pt_temp[0], pt_temp[1]]
                                pt_temp_list.append(pt_temp)
                                point_labels_temp_list.append(point_labels_temp)


                            pt_temp = torch.stack(pt_temp_list, dim=0).to(device=GPUdevice)
                            point_labels_temp = torch.stack(point_labels_temp_list, dim=0).to(device=GPUdevice)
                            coords_torch = torch.cat((coords_torch, pt_temp.unsqueeze(1)), dim=1).to(dtype=torch.float)
                            labels_torch = torch.cat((labels_torch, point_labels_temp.unsqueeze(1)), dim=1).to(dtype=torch.int)
                            #showp = pt_temp

                    # prompt encoder
                    se, de = net.prompt_encoder( 
                        points=(coords_torch, labels_torch),
                        boxes=None,
                        masks=None,
                    )
                    pe = net.prompt_encoder.get_dense_pe().to(device = GPUdevice) 
                    
                    # EM_mean_variance
                    means, variances = net.EM_mean_variance(se, pe)

                    # calculate current weights (output size)
                    pred_masks_weights_list = []
                    for i in range(imgs.size(0)):
                        pred_masks_weights_list.append(net.EM_weights.compute_weights(
                            torch.flatten(final_select_list_mean[i]), weights, means, variances))
                    pred_masks_weights_list = torch.stack(pred_masks_weights_list, dim=0) 
                        

            pbar.update()

    return total_loss_list/ n_val , tuple([total_eiou_list/n_val, total_dice_list/n_val])

