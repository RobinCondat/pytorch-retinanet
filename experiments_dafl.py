import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model_experiment_2 as model
from retinanet.dataloader2 import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
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
from matplotlib import pyplot as plt 
import cv2
from retinanet.config_experiment_2 import Config

import time
import progressbar

from retinanet import losses
from retinanet import losses_dafl as new_losses

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
    print('------------------------------------------\n')
    print("ID : {}".format(C.ID))
    print("CSV classes : {}".format(C.csv_classes))
    print("Training file train : {}".format(C.simple_label_file))
    print("Ignore label : {}".format(C.ignore_class))
    print('------------------------------------------\n')
    # Get repartition IDs
    repartitions = []
    for r in range(C.k_fold_cross_validation):
        if os.path.exists(C.repartition_path+str(r)+'.csv'):
            with open(C.repartition_path+str(r)+'.csv', 'r') as f:
                repartitions.append([l[0] for l in list(csv.reader(f))])
        else:
            raise

    folder_list = [(i+C.k_config)%C.k_fold_cross_validation for i in range(C.k_fold_cross_validation)]


    train_ids = [j for i in [repartitions[s] for s in folder_list[:-1]] for j in i]
    val_ids = repartitions[folder_list[-1]]
    print("Training_ids : {}".format(folder_list[:-1]))
    print("Valid_ids : {}".format(folder_list[-1]))

    # Remove datas depending on their prefixes (from which datasets they came from)
    train_ids = [t for t in train_ids if t[:3] in C.data_prefixes]
    val_ids = [v for v in val_ids if v[:3] in C.data_prefixes]
      
    # Create the data loaders
    dataset_train = CSVDataset(train_file=C.simple_label_file, class_list=C.csv_classes, channels_ind = C.channels_ind, ids = train_ids,
                               transform=transforms.Compose([Normalizer(C.channels_ind), 
                                                             Augmenter(min_rotation=C.min_rotation,
                                                                       max_rotation=C.max_rotation,
                                                                       min_translation=C.min_translation,
                                                                       max_translation=C.max_translation,
                                                                       min_shear=C.min_shear,
                                                                       max_shear=C.max_shear,
                                                                       min_scaling=C.min_scaling,
                                                                       max_scaling=C.max_scaling,
                                                                       flip_x_chance=C.flip_x_chance,
                                                                       flip_y_chance=C.flip_y_chance), 
                                                             Resizer(C.image_min_size)]))
    
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=C.batch_size, drop_last=False, steps=C.steps)
    dataloader_train = DataLoader(dataset_train, num_workers=C.workers, collate_fn=collater, batch_sampler=sampler)

    print("Num_classes train : {}".format(dataset_train.num_classes()))
    # Create the model
    if C.backbone == 'resnet18':
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True, color_mode = C.color_mode, fusion_type=C.fusion_type, step=1, evaluate=False,ignore_class=C.ignore_class,dataset=C.dataset)
    elif C.backbone == 'resnet34':
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True, color_mode = C.color_mode, fusion_type=C.fusion_type, step=1, evaluate=False,ignore_class=C.ignore_class,dataset=C.dataset)
    elif C.backbone == 'resnet50':
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True, color_mode = C.color_mode, fusion_type=C.fusion_type, step=1, evaluate=False,ignore_class=C.ignore_class,dataset=C.dataset)
    elif C.backbone == 'resnet101':
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True, color_mode = C.color_mode, fusion_type=C.fusion_type, step=1, evaluate=False,ignore_class=C.ignore_class,dataset=C.dataset)
    elif C.backbone == 'resnet152':
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True, color_mode = C.color_mode, fusion_type=C.fusion_type, step=1, evaluate=False,ignore_class=C.ignore_class,dataset=C.dataset)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    
    if C.transfer_learning and C.backbone == 'resnet50':
        for weights in C.weights:
            print(os.path.exists(weights))
            retinanet.load_state_dict(torch.load(weights)['model_state_dict'],strict=False)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()
    
    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False

    optimizer = optim.Adam(retinanet.parameters(), lr=C.lr)
    #torch.nn.utils.clip_grad_norm_(retinanet.parameters(),0.001)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    retinanet.train()
    retinanet.module.freeze_bn()
    
    dataloaders = {'train':dataloader_train}

    print('Num training images: {}'.format(len(dataset_train)))
    
    nb_imgs = {ch:0 for ch in C.data_prefixes}
    print("Len dataloader : {}".format(len(dataloader_train)))

    phases = ['train']
    
    if C.load_model:
        checkpoint = torch.load(C.model_path)
        retinanet.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        C.init_epoch = checkpoint['epoch']
        C.best_loss = checkpoint['best_valid_loss']
    epoch_num=0

    epoch_loss = {k:[] for k in phases}
    class_losses = {k:[] for k in phases}
    reg_losses = {k:[] for k in phases}
    
    focalLoss = losses.FocalLoss()
    focalLoss2 = new_losses.FocalLoss()
    classes = ["Car","Truck","Pedestrian","Bike","Bus","Rider","Train","Motorcycle","DontCare"]
    colors = {"Car":(0,255,0),
              "Truck":(255,0,0),
              "Pedestrian":(0,0,255),
              "Bike":(255,255,0),
              "Bus":(0,255,255),
              "Rider":(255,0,255),
              "Train":(60,60,60),
              "Motorcycle":(255,255,255),
              "DontCare":(0,0,0)}

    for phase in phases:
        print("")
        prefix = 'val_' if phase=='val' else ''
        
        retinanet.train(phase=='train')
                
        for iter_num,data in enumerate(dataloaders[phase]):
            
            if C.color_mode=='ALL':
                data=separate_for_all(data)
            else:
                if torch.cuda.is_available():
                    data['img'] = data['img'].cuda().float()
            optimizer.zero_grad()
            data['annot'] = torch.Tensor([[[0.0000,0.0000,1280.0000,720.0000,8.0]]]).cuda()
            classification, regression, anchors, annotations, results = retinanet([data['img'], data['annot']])
            
            scores, labels, boxes = results

            # Imshow img with annotations
            mean = np.array([[[0.485, 0.456, 0.406]]])
            std = np.array([[[0.229, 0.224, 0.225]]])
            img = data['img'][0,:,:,:].permute(1,2,0).cpu().numpy()
            img = ((img*std+mean)*255).astype(np.uint8)
            output = img.copy()
            for i in range(data["annot"].shape[1]):
              annot = data['annot'][0,i,:].cpu().numpy()
              x,y,h,w = annot[:4]
              classe = classes[int(annot[4])]
              cv2.rectangle(output,(x,y),(h,w),colors[classe],2)
            print("Dataset : {}".format(data['dataset']))
            plt.figure(figsize=(20,20))
            plt.imshow(output)
            plt.title("Ground Truth")
            plt.show();

            scores = scores.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            boxes  = boxes.cpu().detach().numpy()

            # correct boxes for image scale
            boxes /= data['scale']      
            # select indices which have a score above the threshold
            indices = np.where(scores > 0.05)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:100]

                # select detections
                image_boxes      = boxes[indices[scores_sort], :]
                image_scores     = scores[scores_sort]
                image_labels     = labels[indices[scores_sort]]
            else:
                image_boxes = None
                image_scores = None
                image_labels = None
            output_2 = img.copy()
            for i in range(len(image_scores)):
              x,y,h,w = image_boxes[i,:]
              score = image_scores[i]
              classe = classes[image_labels[i]]
              cv2.rectangle(output_2,(x,y),(h,w),colors[classe],2)
            plt.figure(figsize=(20,20))
            plt.imshow(output_2)
            plt.title("Ground Truth")
            plt.show();
            print(x,y,h,w)
            print(score)
            print(classe)
            """
            print("Test 1 : Prédiction parfaite (Car en GT aux mêmes coordonnées que la prédiction)")
            print("Loss Ignore IoU")
            annot_1 = torch.Tensor([[[991.4912,37.74562,1048.5088,66.25438,0.0]]]).cuda()
            class_1_IoU,reg_1_IoU = focalLoss(classification, regression, anchors, annot_1, ignore_index = 8)
            print("Class loss : {}".format(class_1_IoU.cpu().numpy()[0]))
            print("Reg loss : {}".format(reg_1_IoU.cpu().numpy()[0]))

            print("Loss Ignore IoA")
            class_1_IoA,reg_1_IoA = focalLoss2(classification, regression, anchors, annot_1, ignore_index = 8)
            print("Class loss : {}".format(class_1_IoA.cpu().numpy()[0]))
            print("Reg loss : {}".format(reg_1_IoA.cpu().numpy()[0]))
            """

    

if __name__ == '__main__':
    main()
