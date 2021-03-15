import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
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

from retinanet.config import Config
from torch.optim.lr_scheduler import MultiplicativeLR

import time
import progressbar

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

    train_ids = [j for i in [repartitions[s] for s in folder_list[:-1]] for j in i]
    print("Training_ids : {}".format(folder_list[:-1]))

    # Remove datas depending on their prefixes (from which datasets they came from)
    train_ids = [t for t in train_ids if t[:3] in C.data_prefixes]
      
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
    

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=C.batch_size, drop_last=False, steps={ch:2000//(len(C.data_prefixes)) for ch in C.data_prefixes})
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
            retinanet.load_state_dict(torch.load(weights),strict=False)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()
    
    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-6)
    
    lmbda = lambda epoch: 1.005
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    
    retinanet.train()
    retinanet.module.freeze_bn()
    
    print('Num training images: {}'.format(len(dataset_train)))
    
    nb_imgs = {ch:0 for ch in C.data_prefixes}
    print("Len dataloader : {}".format(len(dataloader_train)))

    writer = SummaryWriter(C.tensorboard_dir)

    logger = CSVLogger(C.log_filename,a=True)

    if C.load_model:
        checkpoint = torch.load(C.model_path)
        retinanet.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        C.init_epoch = checkpoint['epoch']
        C.best_loss = checkpoint['best_valid_loss']
   
    t_init = time.time()
    

    epoch_loss = []
    class_losses = []
    reg_losses = []

    pbar = tqdm(dataloader_train,file=sys.stdout,position=0,desc="train",
                postfix={'loss':'0.0000',
                         'class_loss':'0.0000',
                         'reg_loss':'0.0000'})

    retinanet.train(True)

    for iter_num,data in enumerate(dataloader_train):

        if C.color_mode=='ALL':
            data=separate_for_all(data)
        else:
            if torch.cuda.is_available():
                data['img'] = data['img'].cuda().float()
        optimizer.zero_grad()
        classification_loss, regression_loss = retinanet([data['img'], data['annot']])

        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        loss = classification_loss + regression_loss

        if bool(loss == 0):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
        optimizer.step()

        epoch_loss.append(float(loss))

        class_losses.append(float(classification_loss))
        reg_losses.append(float(regression_loss))
        #print(scheduler.get_last_lr())
        pbar.set_postfix({'loss':'{:.4f}'.format(np.mean(epoch_loss)),
                          'class_loss':'{:.4f}'.format(np.mean(class_losses)),
                          'reg_loss':'{:.4f}'.format(np.mean(reg_losses)),
                          'lr':'{:.7f}'.format(scheduler.get_last_lr()[-1])},
                         refresh=False)
        print('')
        pbar.update(1)
        # Ecriture dans fichier tensorboard
        writer.add_scalar('lr',scheduler.get_last_lr()[-1])
        writer.add_scalar('loss', np.mean(epoch_loss),iter_num)
        writer.add_scalar('class_loss', np.mean(class_losses),iter_num)
        writer.add_scalar('reg_loss', np.mean(reg_losses),iter_num)

        # Ecriture dans fichier log
        logger.write(iter_num,epoch_loss,class_losses,reg_losses,lr=scheduler.get_last_lr(),a=True)
        
        del classification_loss
        del regression_loss
        scheduler.step()
        
    pbar.close()

    # Resume de l'epoch
    t_end = time.time()

    writer.close()      

if __name__ == '__main__':
    main()
