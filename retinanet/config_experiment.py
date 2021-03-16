import configparser
import argparse
import numpy as np
import keras
import os
import csv
import random
import string
import pandas as pd
from .csvLogger import CSVLogger

def RandomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

class Config:
    def __init__(self,args=None):
        
        args = parser_args(args)
        
        self.ID = "{}_{}_{}".format(args.ID,args.classes,args.lr)
        self.family = args.FAMILY
        self.backbone = args.backbone
        self.transfer_learning = args.transfer_learning
        self.color_mode = args.color_mode
        self.k_fold_cross_validation = args.k_fold_cross_validation
        self.k_config = args.k_config
        self.sub_k_config = args.sub_k_config
        self.batch_size = args.batch_size
        self.image_min_size = {'KIT':600,
                               'BDD':720,
                               'CIT':720,
                               'COA':720,
                               'WAY':720}

        self.base_path = args.database
        self.home_dir = args.home_dir
        self.fusion_type = args.fusion_type
        self.lr = args.lr
        self.epochs = args.epoch
        self.transform = args.data_aug
        self.ignore_class = bool(args.ignore)
        self.classes = args.classes
        self.dataset = args.datasets
        self.data_prefixes = PREFIXES[args.datasets]

        if (self.fusion_type == 0) and (self.color_mode in ['RGB','DOL']):
            pass
        elif (self.fusion_type !=0) and (self.color_mode=='ALL'):
            pass
        else:
            raise ValueError(f"Incompatible fusion_type {self.fusion_type} and color_mode {self.color_mode}")

        # All transformations
        if self.transform==1:
            self.min_rotation=-0.1
            self.max_rotation=0.1
            self.min_translation=(-0.1,-0.1)
            self.max_translation=(0.1,0.1)
            self.min_shear=-0.1
            self.max_shear=0.1
            self.min_scaling=(0.9,0.9)
            self.max_scaling=(1.1,1.1)
            self.flip_x_chance=0.5
            self.flip_y_chance=0
        # No transformations
        else:
            self.min_rotation=0
            self.max_rotation=0
            self.min_translation=(0,0)
            self.max_translation=(0,0)
            self.min_shear=0
            self.max_shear=0
            self.min_scaling=(1,1)
            self.max_scaling=(1,1)
            self.flip_x_chance=0
            self.flip_y_chance=0
        if self.classes == 'PED':
            self.csv_classes = self.base_path+"classes_ped.csv"
        else:
            self.csv_classes = self.base_path+"classes_full_{}.csv".format(args.datasets)
        
        self.nb_channels = NB_CHANNELS[self.color_mode]

        self.channels_ind = COLOR_MODE[self.color_mode]

        self.steps = None #{ch:(10000//self.batch_size)//(len(self.data_prefixes)) for ch in self.data_prefixes}
              
        self.freeze_backbone = False
        self.workers = 4
        
        # Robutness parameter
        self.channels_rscr = {'Cl':args.rscr_cl, # /100
                              'Dp':args.rscr_dp, # /100
                              'Of':args.rscr_of, # /100
                              'Vl':args.rscr_vl} # /100
        
        self.model_path = self.home_dir+'models//'+self.family+'//'+self.ID+'.pt'
        self.best_model_path = self.home_dir+'models//'+self.family+'//'+self.ID+'_bestModel.pt'

        os.makedirs(self.home_dir+'models//'+self.family+'//',exist_ok=True)
              
        self.repartition_path = self.home_dir+'repartitions//'+str(self.k_fold_cross_validation)+'_fold_cross_validation//'
        
        self.result_images_path = self.home_dir+'results_images//'+self.family+'//'+self.ID+'//'
        self.result_labels_path = self.home_dir+'results_labels//'+self.family+'//'+self.ID+'//'
        
        os.makedirs(self.result_images_path,exist_ok=True)
        os.makedirs(self.result_labels_path,exist_ok=True)        
        if self.classes == 'PED':
            self.simple_label_file = self.base_path+'training_labels_ped.txt'
        else:
            self.simple_label_file = self.base_path+'training_labels_full_2.txt'
        
        self.simple_label_test_file = self.base_path+'testing_labels.txt'
       
        self.log_filename = self.home_dir+'logs//'+self.family+'//'+self.ID+'.csv'

        self.tensorboard_dir = self.home_dir+'tensorboard//'+self.family+'//'+self.ID+'//'
 	
        self.map_dir = self.home_dir+'mAP//'+self.family+'//'
        
        os.makedirs(self.map_dir,exist_ok=True)
        os.makedirs(self.home_dir+'logs//'+self.family+'//',exist_ok=True)
        os.makedirs(self.tensorboard_dir,exist_ok=True)
	
        self.weights=None
        
        if os.path.exists(self.model_path) and os.path.exists(self.log_filename):
            self.load_model = True
            self.weights = [self.model_path]
            #self.init_epoch,self.best_loss = CSVLogger(self.log_filename).get_init_param()

        else:
            self.load_model = False
            self.init_epoch = 0
            self.best_loss = np.inf
            if self.transfer_learning:
                if args.weights == None or args.weights == 'None':
                    self.weights=[self.home_dir+'models/coco.pt']
                else:
                    self.weights=[self.best_model_path.replace(self.ID,args.weights)] #[self.home_dir+'models/{}_k{}.pt'.format(color_mode,self.k_config) for color_mode in ['RGB','DOL']]

def parser_args(args):

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ID",help="Experiment Name",default=RandomString(8),type=str)
    
    parser.add_argument("--FAMILY",help="Family of experiments",default=RandomString(8),type=str)

    parser.add_argument("--backbone",help="Backbone of RetinaNet",default="resnet50",type=str)
    parser.add_argument("--transfer_learning",help="Use of Transfer learning (bool)",default=True,type=int)
    parser.add_argument("--color_mode",help="Channels to use for detection",default="RGB",type=str)
    
    parser.add_argument("--k_fold_cross_validation",help="Number of folders for cross validation (k)",default=0,type=int)
    parser.add_argument("--k_config",help="Configuration number for cross validation",default=0,type=int)
    
    parser.add_argument("--sub_k_config",help="Configuration number for sub-CV (fine tuning)",default=0,type=int)

    parser.add_argument("--batch_size",help="Batch size",default = 4,type=int)
    parser.add_argument("--datasets",help="Datasets for training ('KIT','CIT','BDD','FIN','WAY' or 'MIX')",default='KIT',type=str)
   
    parser.add_argument("--database",help="Database Path",default="/home/2017018/rconda01/",type=str)
    parser.add_argument("--home_dir",help="Home directory",default="/home/2017018/rconda01/",type=str)   
    
    parser.add_argument("--fusion_type",help="Gate Fusion Unit Option (1: Stacked Fusion, 2: Gated Fusion)",type=int,default=1)
    
    parser.add_argument("--epoch",help="Number of iterations",default=1,type=int)

    parser.add_argument("--lr",help="Initial learning rate",default=1e-3,type=float)

    parser.add_argument("--rscr_cl",help="Random Signal Cut Rate for Cl", default=0,type=float)

    parser.add_argument("--rscr_dp",help="Random Signal Cut Rate for Dp", default=0,type=float)

    parser.add_argument("--rscr_of",help="Random Signal Cut Rate for Of", default=0,type=float)

    parser.add_argument("--rscr_vl",help="Random Signal Cut Rate for Vl", default=0,type=float)
            
    parser.add_argument('--data_aug', help='Use of Data Augmentation', default=True,type=int)

    parser.add_argument('--weights', help='Name of weights for transfer learning (default COCO if transfer learning is True)', default=None)    
    parser.add_argument('--ignore', help='Ignore class (True if there is any)', default=False, type=int)
   
    parser.add_argument('--classes',default='FULL',type=str)

    args = parser.parse_args(args)

    return args
               

        
        
COLOR_MODE = {'RGB':['_Cl.png'],
              'DOL':['_Dp.png','_Of.png','_Vl.png'],
              'ALL':['_Cl.png','_Dp.png','_Of.png','_Vl.png']
              }

NB_CHANNELS = {'RGB':[3],
               'DOL':[3],
               'ALL':[3,3]}

LABEL_COLORS = {'Pedestrian':(0,255,0),
                        'Car':(0,0,255),
                        'Truck':(255,0,0),
                        'Van':(255,255,0),
                        'DontCare':(0,0,0),
                        'Person_sitting':(0,255,255), 
                        'Cyclist':(255,0,255), 
                        'Tram':(128,128,128),
                        'Misc':(255,255,255)}

PREFIXES = {'KIT':['KIT'],
            'BDD':['BDD'],
            'CIT':['CIT','COA'],
            'WAY':['WAY'],
            'MIX':['BDD','CIT','COA','WAY'],
            'FIN':['CIT']}
