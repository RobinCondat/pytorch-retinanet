from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

import skimage.io
import skimage.transform
import skimage.color
import skimage

from .transform import random_transform, TransformParameters, adjust_transform_for_image, apply_transform, transform_aabb

from PIL import Image


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, channels_ind, ids, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        
       
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform
        self.channels_ind = channels_ind

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes, ids)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
        #self.image_names = [a+'//'+a.split('//')[-1]+'_Cl.png' for a in list(self.image_data.keys())]
        self.image_names = list(self.image_data.keys())
        #print(self.image_names)

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')


    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        channels = []

        a = self.image_names[image_index]
        for ch in self.channels_ind:

            img = skimage.io.imread(a+'//'+a.split('//')[-1]+ch,as_gray=ch!='_Cl.png')

            channels.append(img)

        if len(channels)>1:
            img = np.dstack(tuple(channels))
        else:
            img = channels[0]
        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 5))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes, ids):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1
            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)
            #print(img_file.split('//')[-1])
            #print(ids)
            if img_file.split('//')[-1] in ids:
            
                if img_file not in result:
                    result[img_file] = []

                # If a row contains only an image path, it's an image without annotations.
                if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                    continue

                x1 = self._parse(x1, float, 'line {}: malformed x1: {{}}'.format(line))
                y1 = self._parse(y1, float, 'line {}: malformed y1: {{}}'.format(line))
                x2 = self._parse(x2, float, 'line {}: malformed x2: {{}}'.format(line))
                y2 = self._parse(y2, float, 'line {}: malformed y2: {{}}'.format(line))

                # Check that the bounding box is valid.
                if x2 <= x1:
                    raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
                if y2 <= y1:
                    raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

                # check if the current class name is correctly present
                if class_name not in classes:
                    raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

                result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        a = self.image_names[image_index]
        image = Image.open(a+'//'+a.split('//')[-1]+'_Cl.png')
            
        return float(image.width) / float(image.height)


def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    channels = [int(s.shape[2]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()
    max_channels = np.array(channels).max()
    padded_imgs = torch.zeros(batch_size, max_width, max_height, max_channels)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)
    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self,min_side=None):
        self.min_side = min_side

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        
        rows, cols, cns = image.shape

        if self.min_side is not None:

            smallest_side = min(rows, cols)

            # rescale the image so the smallest side is min_side
            scale = self.min_side / smallest_side

            # resize the image with the computed scale
            image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
            rows, cols, cns = image.shape
            
            annots[:, :4] *= scale       
        
        else:
            scale = 1

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}

class ChannelCut(object):
    """ Cut one or several channels for evaluation"""
    
    def __init__(self,channels_ind, channels_to_cut):
        self.channels_ind = channels_ind
        self.channels_to_cut = channels_to_cut
        
    def __call__(self,sample):
        image, annots = sample['img'], sample['annot']
        i = 0
        for ch in self.channels_ind:
            if ch in self.channels_to_cut:
                if ch=='_Cl.png':
                    image[:,:,i:i+3]=0
                else:
                    image[:,:,i:i+1]=0
            if ch=='_Cl.png':
                i+=3
            else:
                i+=1
        return {'img': image, 'annot': annots}
        
    
class Augmenter(object):
    
    def __init__(self,
                 min_rotation=0,
                 max_rotation=0,
                 min_translation=(0,0),
                 max_translation=(0,0),
                 min_shear=0,
                 max_shear=0,
                 min_scaling=(1,1),
                 max_scaling=(1,1),
                 flip_x_chance=0,
                 flip_y_chance=0):
        self.transform_parameters   = TransformParameters()
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        self.min_translation = min_translation
        self.max_translation = max_translation
        self.min_shear = min_shear
        self.max_shear = max_shear
        self.min_scaling = min_scaling
        self.max_scaling = max_scaling
        self.flip_x_chance = flip_x_chance
        self.flip_y_chance = flip_y_chance
        
    def __call__(self, sample):
        next_transform = random_transform(min_rotation=self.min_rotation,
                                          max_rotation=self.max_rotation,
                                          min_translation=self.min_translation,
                                          max_translation=self.max_translation,
                                          min_shear=self.min_shear,
                                          max_shear=self.max_shear,
                                          min_scaling=self.min_scaling,
                                          max_scaling=self.max_scaling,
                                          flip_x_chance=self.flip_x_chance,
                                          flip_y_chance=self.flip_y_chance)

        image, annots = sample['img'], sample['annot']
        transform = adjust_transform_for_image(next_transform, image, self.transform_parameters.relative_translation)

        # apply transformation to image
        result_image = apply_transform(transform, image, self.transform_parameters)

        # Transform the bounding boxes in the annotations.
        for index in range(annots[:,:-1].shape[0]):
            annots[:,:-1][index, :] = transform_aabb(transform, annots[:,:-1][index, :])         
        
        return {'img': result_image, 'annot': annots}


class Normalizer(object):

    def __init__(self,channels_ind):
        
        self.means = {'_Cl.png':np.array([[[0.485, 0.456, 0.406]]]),
                     '_Dp.png':np.array([[[0.137]]]),
                     '_Of.png':np.array([[[0.113]]]),
                     '_Vl.png':np.array([[[0.274]]])}
        
        self.stds = {'_Cl.png':np.array([[[0.229, 0.224, 0.225]]]),
                     '_Dp.png':np.array([[[0.091]]]),
                     '_Of.png':np.array([[[0.118]]]),
                     '_Vl.png':np.array([[[0.274]]])}
        
        if len(channels_ind)==1:
            self.mean = self.means(channels_ind[0])
            self.std = self.stds(channels_ind[0])
        else:
            self.mean = np.concatenate([self.means[ch] for ch in channels_ind],axis=-1)
            self.std = np.concatenate([self.stds[ch] for ch in channels_ind],axis=-1)
            
    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))
        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
