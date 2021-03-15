import os
import numpy as np
class CSVLogger:
    def __init__(self,filename,separator=',',a=False):
        self.filename = filename
        self.separator = separator
        if not os.path.exists(self.filename):
            with open(self.filename,'w') as f:
                if a:
                    f.write(self.separator.join(['epoch','loss','class_loss','reg_loss','lr\n']))
                else:
                    f.write(self.separator.join(['epoch','loss','class_loss','reg_loss','val_loss','val_class_loss','val_reg_loss','lr\n']))

            
    def write(self,epoch_num,epoch_loss,classes_losses,reg_losses,lr=1e-5,a=False):
        with open(self.filename,'a') as f:
            if a:
                line = [epoch_num,np.mean(epoch_loss),np.mean(classes_losses),np.mean(reg_losses),lr[0]]
            else:
                line = [epoch_num,np.mean(epoch_loss['train']),np.mean(classes_losses['train']),np.mean(reg_losses['train']),
                       np.mean(epoch_loss['val']),np.mean(classes_losses['val']),np.mean(reg_losses['val']),lr]
            line = [str(a) for a in line]
            f.write(self.separator.join(line)+'\n')
    
    def get_init_param(self):
        with open(self.filename,'r') as csv:
            csvFile = [line for line in csv]
            init_epoch = len(csvFile)-1
            index_val_loss = csvFile[0].split(self.separator).index('val_loss')
            val_loss_history = [float(l.split(self.separator)[index_val_loss]) for l in csvFile[1:]]
            best_init = min(val_loss_history)
        return init_epoch,best_init
