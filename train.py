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
    
    dataset_val = CSVDataset(train_file=C.simple_label_file, class_list=C.csv_classes, channels_ind = C.channels_ind,ids = val_ids,transform=transforms.Compose([Normalizer(C.channels_ind), Resizer(C.image_min_size)]))

    dataset_eval = CSVDataset(train_file=C.simple_label_file, class_list=C.csv_classes, channels_ind = C.channels_ind,ids = val_ids,transform=transforms.Compose([Normalizer(C.channels_ind), Resizer(C.image_min_size)]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=C.batch_size, drop_last=False, steps=C.steps)
    dataloader_train = DataLoader(dataset_train, num_workers=C.workers, collate_fn=collater, batch_sampler=sampler)
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=C.batch_size, drop_last=False) #, steps = 100)
    dataloader_val = DataLoader(dataset_val, num_workers=C.workers, collate_fn=collater, batch_sampler=sampler_val)  
    print("Num_classes train : {}".format(dataset_train.num_classes()))
    print("Num_classes val : {}".format(dataset_val.num_classes()))
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

    optimizer = optim.Adam(retinanet.parameters(), lr=C.lr)
    #torch.nn.utils.clip_grad_norm_(retinanet.parameters(),0.001)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    retinanet.train()
    retinanet.module.freeze_bn()
    
    dataloaders = {'train':dataloader_train, 'val':dataloader_val}

    print('Num training images: {}'.format(len(dataset_train)))
    print('Num valid images: {}'.format(len(dataset_val)))
    
    nb_imgs = {ch:0 for ch in C.data_prefixes}
    print("Len dataloader : {}".format(len(dataloader_train)))

    phases = ['train','val']
    
    writer = SummaryWriter(C.tensorboard_dir)

    logger = CSVLogger(C.log_filename)

    if C.load_model:
        checkpoint = torch.load(C.model_path)
        retinanet.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        C.init_epoch = checkpoint['epoch']
        C.best_loss = checkpoint['best_valid_loss']
   
    #print(retinanet.module.conv1_RGB.weight.requires_grad)
    #print(retinanet.module.FU_2.conv_RGB.weight.requires_grad)
    #print(retinanet.module.regressionModel.output.weight.requires_grad)
    
    for epoch_num in range(C.init_epoch,C.epochs):
        t_init = time.time()
        print('Epoch {}/{}'.format(epoch_num+1,C.epochs))

        epoch_loss = {k:[] for k in phases}
        class_losses = {k:[] for k in phases}
        reg_losses = {k:[] for k in phases}


        for phase in phases:
            print("")
            prefix = 'val_' if phase=='val' else ''
            pbar = tqdm(dataloaders[phase],file=sys.stdout,position=0,desc=phase,
                        postfix={prefix+'loss':'0.0000',
                                 prefix+'class_loss':'0.0000',
                                 prefix+'reg_loss':'0.0000'})
            
            retinanet.train(phase=='train')
            #retinanet.module.freeze_bn()
                    
            '''
            for iter_num in range(500):
                data = dataloaders[phase][iter_num]
            '''
            for iter_num,data in enumerate(dataloaders[phase]):
                
                if C.color_mode=='ALL':
                    data=separate_for_all(data)
                    #print(data['img'][0])
                    #print(data['img'][1])
                else:
                    if torch.cuda.is_available():
                        data['img'] = data['img'].cuda().float()
                optimizer.zero_grad()
                #print(data['annot'][:,:,4].size())
                #print(data['annot'][:,:,4])
                classification_loss, regression_loss = retinanet([data['img'], data['annot'],data['dataset']])
                 
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                    optimizer.step()

                epoch_loss[phase].append(float(loss))
             
                class_losses[phase].append(float(classification_loss))
                reg_losses[phase].append(float(regression_loss))
                               
                pbar.set_postfix({prefix+'loss':'{:.4f}'.format(np.mean(epoch_loss[phase])),
                                  prefix+'class_loss':'{:.4f}'.format(np.mean(class_losses[phase])),
		                  prefix+'reg_loss':'{:.4f}'.format(np.mean(reg_losses[phase]))},
                                 refresh=False)
                print('')
                pbar.update(1)
               
                del classification_loss
                del regression_loss
                #except Exception as e:
                #    print(e)
                #    continue
            pbar.close()

        # Resume de l'epoch
        t_end = time.time()
        print("\n\nEpoch {}/{} total time : {}s - loss : {:.4f} - class_loss {:.4f} - reg_loss : {:.4f} - val_loss : {:.4f} - val_class_loss {:.4f} - val_reg_loss : {:.4f}\n".format(epoch_num+1,C.epochs,int(t_end-t_init),np.mean(epoch_loss['train']),np.mean(class_losses['train']),np.mean(reg_losses['train']),np.mean(epoch_loss['val']),np.mean(class_losses['val']),np.mean(reg_losses['val'])))

        # Ecriture dans fichier tensorboard
        writer.add_scalar('loss', np.mean(epoch_loss['train']),epoch_num)
        writer.add_scalar('class_loss', np.mean(class_losses['train']),epoch_num)
        writer.add_scalar('reg_loss', np.mean(reg_losses['train']),epoch_num)
        writer.add_scalar('val_loss', np.mean(epoch_loss['val']),epoch_num)
        writer.add_scalar('val_class_loss', np.mean(class_losses['val']),epoch_num)
        writer.add_scalar('val_reg_loss', np.mean(reg_losses['val']),epoch_num)
        
        # Ecriture dans fichier log
        logger.write(epoch_num,epoch_loss,class_losses,reg_losses)
        
        # Sauvegarde du meilleur modèle
        if np.mean(epoch_loss['val']) < C.best_loss:
            C.best_loss = np.mean(epoch_loss['val'])
            torch.save(retinanet.module.state_dict(), C.best_model_path)
        
        '''
        if np.mean(class_losses['val']) < best_class_loss:
            best_class_loss = np.mean(class_losses['val'])
            torch.save(retinanet.module.state_dict(), C.best_model_path.replace('.pt','_cl.pt'))        
        '''

        #scheduler.step(np.mean(epoch_loss))

        torch.save({'epoch': epoch_num+1,
                    'model_state_dict':retinanet.module.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'best_valid_loss':C.best_loss}, C.model_path)


    
    writer.close()
    
    print('Evaluating dataset')
    
    # Create the model
    if C.backbone == 'resnet18':
        best_retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True, color_mode = C.color_mode, fusion_type=C.fusion_type, step=1, evaluate=True,ignore_class=C.ignore_class,dataset=C.dataset)
    elif C.backbone == 'resnet34':
        best_retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True, color_mode = C.color_mode, fusion_type=C.fusion_type, step=1, evaluate=True,ignore_class=C.ignore_class,dataset=C.dataset)
    elif C.backbone == 'resnet50':
        best_retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True, color_mode = C.color_mode, fusion_type=C.fusion_type, step=1, evaluate=True,ignore_class=C.ignore_class,dataset=C.dataset)
    elif C.backbone == 'resnet101':
        best_retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True, color_mode = C.color_mode, fusion_type=C.fusion_type, step=1, evaluate=True,ignore_class=C.ignore_class,dataset=C.dataset)
    elif C.backbone == 'resnet152':
        best_retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True, color_mode = C.color_mode, fusion_type=C.fusion_type, step=1, evaluate=True,ignore_class=C.ignore_class,dataset=C.dataset)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

 
    best_retinanet.load_state_dict(torch.load(C.best_model_path))

    if use_gpu:
        if torch.cuda.is_available():
            best_retinanet = best_retinanet.cuda()

    
    mAP = csv_eval.evaluate(dataset_eval, best_retinanet,C,[],save_path=C.map_dir,ignore_class=True)
      

if __name__ == '__main__':
    main()
