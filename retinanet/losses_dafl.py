import numpy as np
import torch
import torch.nn as nn
from retinanet.config_experiment_2 import INDEXES_MIX, VEHICLE_INDEXES

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

def cal_ioa(a, b):
    # Intersection over Area (for ignore regions)
    area = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]),dim=1)
    area = torch.clamp(area, min=1e-8)

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    intersection = iw * ih

    IoA = intersection / area

    return IoA


class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations, dataset,ignore_index=None):

        classes_from_other_datasets = [i for i in range(classifications.shape[-1]) if i not in INDEXES_MIX[dataset]]
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]
        num_anchors = anchor.shape[0]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights
        print(classifications.shape)

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            
            # Ignore class from other datasets
            classification[:,classes_from_other_datasets]=0

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification

                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().cuda())
                    
                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification

                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())
                    
                continue

            # Filter ignore class (via ignore_index)
            if ignore_index is not None:
                # On sépare ici les annotations en 2 objets : 
                # - bbox_annotation (pour tous les objets à détecter) 
                # - ignore_annotation (pour toutes les régions à ignorer)
                ignore_annotation = bbox_annotation[bbox_annotation[:,4] == ignore_index]
                bbox_annotation = bbox_annotation[bbox_annotation[:,4] != ignore_index]

            if bbox_annotation.shape[0] != 0:
                IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations_to_detect
                IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1
            else:
                IoU_max = None
                IoU_argmax = None
            
            if ignore_index is not None:
                # On calcule ici l'intersection over area : 
                # tous les anchors ayant une IoA avec une région à ignorer supérieure à 0.5 seront ignorées pour la suite
                if ignore_annotation.shape[0] !=0:
                    IoA = cal_ioa(anchors[0, :, :], ignore_annotation[:, :4]) # num_anchors x num_annotations_to_ignore            
                    IoA_max, IoA_argmax = torch.max(IoA, dim=1) # num_anchors x 1
                else:
                    IoA_max = None
                    IoA_argmax = None
            
            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()

            if IoU_max is not None:
                targets[torch.lt(IoU_max, 0.4), :] = 0
            else:
                targets = targets*0
                
            if ignore_index is not None:
                if IoA_max is not None:
                    ignore_indices = torch.ge(IoA_max, 0.5)
                else:
                    ignore_indices = (torch.ones((num_anchors)) * 0).type(torch.ByteTensor)
            if IoU_max is not None:
                positive_indices = torch.ge(IoU_max, 0.5)
                num_positive_anchors = positive_indices.sum()
            
            else:
                positive_indices = (torch.ones((num_anchors)) * 0).type(torch.ByteTensor)
                num_positive_anchors = torch.tensor(0)
           
            if ignore_index is not None:
                if ignore_indices is not None:
                    targets[ignore_indices, :] = -1
            
            if IoU_argmax is not None:
                assigned_annotations = bbox_annotation[IoU_argmax, :]
                targets[positive_indices, :] = 0
                targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
            
            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))
            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))
   
            # compute the loss for regression

            if num_positive_anchors > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)

    
