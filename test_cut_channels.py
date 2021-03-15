import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer, ChannelCut
from torch.utils.data import DataLoader 

from retinanet import csv_eval

from torchsummary import summary
import time
from tqdm import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter
import os
import csv
from retinanet.csvLogger import CSVLogger

from retinanet.config import Config

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def separate_for_all(data):
    imgs = data['img']
    if torch.cuda.is_available():
        sep_imgs = [imgs[:,a*3:(a+1)*3,:,:].cuda().float() for a in range(2)]
    else:
        sep_imgs = [imgs[:,a*3:(a+1)*3,:,:] for a in range(2)]
    data['img'] = sep_imgs
    return data

def main(args=None):

    C = Config(args)

    # Get repartition IDs
    repartitions = []
    for r in range(C.k_fold_cross_validation):
        if os.path.exists(C.repartition_path+str(r)+'.csv'):
            with open(C.repartition_path+str(r)+'.csv', 'r') as f:
                repartitions.append([l[0] for l in list(csv.reader(f))])
        else:
            raise

    folder_list = [(i+C.k_config)%C.k_fold_cross_validation for i in range(C.k_fold_cross_validation)]

    if C.step==1:
        new_folder_list = folder_list[:-1]
        for i in range(C.sub_k_config):
            new_folder_list.insert(0, new_folder_list.pop(-1))
        print("Training_ids for Fine Tuning : {}".format(new_folder_list[:-1]))
        print("Valid_id for Fine Tuning : {}".format(new_folder_list[-1]))
        print("Valid_id for Step 2 : {}".format(folder_list[-1]))

        train_ids = [j for i in [repartitions[s] for s in new_folder_list[:-1]] for j in i]
        val_ids = repartitions[new_folder_list[-1]]

    elif C.step==2:
        train_ids = [j for i in [repartitions[s] for s in folder_list[:-1]] for j in i]
        val_ids = repartitions[folder_list[-1]]
        print("Training_ids : {}".format(folder_list[:-1]))
        print("Valid_ids : {}".format(folder_list[-1]))


    


 


    cutted_channels = [[],['_Cl.png'],['_Dp.png'],['_Of.png'],['_Vl.png'],['_Dp.png','_Of.png','_Vl.png']]
    best_retinanet=None
    for cc in cutted_channels:
        
        dataset_val = CSVDataset(train_file=C.simple_label_file, class_list=C.csv_classes, channels_ind = C.channels_ind, 
                                     ids = val_ids,transform=transforms.Compose([ChannelCut(C.channels_ind,cc),
                                                                                 Normalizer(C.channels_ind), Resizer(C.image_min_size)]))

        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=2, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val) 
            
        if best_retinanet is None:
            # Create the model
            if C.backbone == 'resnet18':
                best_retinanet = model.resnet18(num_classes=dataset_val.num_classes(), pretrained=True, color_mode = C.color_mode,
                                                fusion_type=C.fusion_type, step=C.step, evaluate=True)
            elif C.backbone == 'resnet34':
                best_retinanet = model.resnet34(num_classes=dataset_val.num_classes(), pretrained=True, color_mode = C.color_mode,
                                                fusion_type=C.fusion_type, step=C.step, evaluate=True)
            elif C.backbone == 'resnet50':
                best_retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True, color_mode = C.color_mode,
                                                fusion_type=C.fusion_type, step=C.step, evaluate=True)
            elif C.backbone == 'resnet101':
                best_retinanet = model.resnet101(num_classes=dataset_val.num_classes(), pretrained=True, color_mode = C.color_mode,
                                                 fusion_type=C.fusion_type, step=C.step, evaluate=True)
            elif C.backbone == 'resnet152':
                best_retinanet = model.resnet152(num_classes=dataset_val.num_classes(), pretrained=True, color_mode = C.color_mode,
                                                 fusion_type=C.fusion_type, step=C.step, evaluate=True)
            else:
                raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    
            best_retinanet.load_state_dict(torch.load(C.best_model_path))

            
            if torch.cuda.is_available():
                best_retinanet = best_retinanet.cuda()            
        if all([c in C.channels_ind for c in cc]):
            mAP = csv_eval.evaluate(dataset_val, best_retinanet,C,cc)

if __name__ == '__main__':
    main()
