from __future__ import print_function
from retinanet.dataloader2 import UnNormalizer
import numpy as np
import json
import os
import matplotlib
import matplotlib.pyplot as plt
import torch
import cv2
import pandas as pd

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_ignore_overlap(a,b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (N, 4) ndarray of float
    Returns:
    --------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    area = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1)

    area = np.maximum(area, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / area


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(dataset, retinanet, C, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]
    unnormalize = UnNormalizer(C.channels_ind)
    retinanet.eval()
    #if save_path is not None:
    #    os.makedirs(save_path+C.ID+"/results/",exist_ok=True)
    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes  = boxes.cpu().numpy()

            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes      = boxes[indices[scores_sort], :]
                image_scores     = scores[scores_sort]
                image_labels     = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')
            """
            if save_path is not None:
                output = (255*unnormalize(data['img'].numpy().copy())).astype(np.uint8)
                if indices.shape[0]>0:
                    for box,score,label in zip(image_boxes,image_scores,image_labels):
                        x,y,h,w = box*scale
                        cv2.rectangle(output,(int(x),int(y)),(int(h),int(w)),(0,255,0),2)
                        cv2.putText(output, "{:.2f}".format(score), (int(x), int(y) - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                        cv2.putText(output, "{:.2f}".format(score), (int(x), int(y) - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                cv2.imwrite(save_path+"results/"+data['name']+'.png',output)
            """
    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def evaluate(
    generator,
    retinanet,
    C, # config
    channel_cut=[], 
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None,
    ignore_class=False
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save precision recall curve of each label.
    # Returns
        A dict mapping class names to mAP scores.
    """



    # gather all detections and annotations

    all_detections     = _get_detections(generator, retinanet, C, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)
    #print("Nb de piétons annotés : {}".format(sum(all_annotations[i][0].shape[0] for i in range(len(generator)))))
    #print("Nb de piétons detectés : {}".format(sum(all_detections[i][0].shape[0] for i in range(len(generator)))))

    total_instances = []
    average_precisions = {}
    recalls = {}
    precisions = {}
    TPs = {}
    FPs = {}
    nb_classes = generator.num_classes()-1
    ignore_index = nb_classes
    
    for label in range(nb_classes):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(generator)):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []
            ignore_annotations = all_annotations[i][ignore_index]

            for d in detections:
                if not ignore_class:
                    scores = np.append(scores,d[4])
                    # Pas de classe à ignorer
                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue

                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

                else:
                    # Classe à ignorer
                    # Calcul overlap detection avec la classe à détecter
                    if annotations.shape[0]!=0:
                        overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                        assigned_annotation = np.argmax(overlaps, axis=1)
                        max_overlap         = overlaps[0, assigned_annotation]
                    else:
                        assigned_annotation = None
                        max_overlap = 0            

                    # Calcul overlap detection avec ignore regions
                    if ignore_annotations.shape[0]!=0:
                        overlaps_ignore = compute_ignore_overlap(np.expand_dims(d, axis=0), ignore_annotations)
                        assigned_ignore_annotation = np.argmax(overlaps_ignore, axis=1)
                        max_ignore_overlap = overlaps_ignore[0, assigned_ignore_annotation]
                    else:
                        max_ignore_overlap = 0


                     

                    if max_overlap>= iou_threshold:
                        # Détection proche d'une annotation classe
                        scores = np.append(scores,d[4])
                        if assigned_annotation not in detected_annotations:
                            # Bonne détection pas déjà détectée => True Positive
                            false_positives = np.append(false_positives, 0)
                            true_positives = np.append(true_positives, 1)
                            detected_annotations.append(assigned_annotation)
                        else:
                            # Annotation déjà détectée => False Positive
                            false_positives = np.append(false_positives, 1)
                            true_positives = np.append(true_positives, 0)
                    elif max_ignore_overlap >= 0.75:
                        # Détection dans une région à ignorer
                        continue
                    else:
                        # Aucune annotation => False Positive
                        scores = np.append(scores,d[4])
                        false_positives = np.append(false_positives ,1)
                        true_positives = np.append(true_positives, 0)

                    """
                    if max_overlap>=max_ignore_overlap:
                        scores = np.append(scores,d[4])
                        # La détection est plus proche d'une annotation classe qu'une région à ignorer OU il n'y a aucune annotation
                        if assigned_annotation == None:
                            # Aucune annotation => Faux positif
                            false_positives = np.append(false_positives, 1)
                            true_positives = np.append(true_positives, 0)
                        elif max_overlap>= iou_threshold and assigned_annotation not in detected_annotations:
                            # Bonne détection pas déjà détectée => True Positif
                            false_positives = np.append(false_positives, 0)
                            true_positives = np.append(true_positives, 1)
                            detected_annotations.append(assigned_annotation)
                        else:
                            # Détection lointaine ou annotation déjà détectée => False positive
                            false_positives = np.append(false_positives, 1)
                            true_positives = np.append(true_positives, 0)
                    else:
                        # La détection est plus proche d'une région à ignorer qu'une annotation classe
                        if max_ignore_overlap >= iou_threshold:
                            # La détection est très proche d'une région à ignorer : On ignore la détection
                            continue
                        else:
                            scores = np.append(scores,d[4])
                            # La détection est loin d'une région à ignorer : On compte la détection comme un faux positif
                            false_positives = np.append(false_positives ,1)
                            true_positives = np.append(true_positives, 0)
                    """

        total_instances.append(num_annotations)
        # no annotations -> AP for this class is 0 (is this correct?)
        # no detections -> AP for this class is 0
        if num_annotations == 0 or len(scores)==0:
            average_precisions[label] = 0, num_annotations
            precisions[label] = [0.0]
            recalls[label] = [0.0]
            TPs[label] = np.array([0])
            FPs[label] = np.array([0])
            continue
        #print("Num annotations : {}".format(num_annotations))
        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]
        #print("len FP : {}".format(len(false_positives)))
        
        
        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)
        #print("TP max : {}".format(true_positives.max()))
        #print("FP max : {}".format(false_positives.max()))
        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
        recalls[label] = recall
        precisions[label] = precision
        TPs[label] = true_positives
        FPs[label] = false_positives

    # TMP
    all_precisions = []
    ID = C.ID
    os.makedirs(save_path+C.ID+'//',exist_ok=True)
    print('\nmAP:')
    for label in range(nb_classes):
        label_name = generator.label_to_name(label)
        print("---------")
        print('{}: {}'.format(label_name, average_precisions[label][0]))
        print("Precision: ",precisions[label][-1])
        print("Recall: ",recalls[label][-1])
        print("Num annotations: {}".format(average_precisions[label][1]))
        print("Num detections: {}".format(len(TPs[label])))
        print("TP: {}".format(TPs[label].max()))
        print("FP: {}\n".format(FPs[label].max()))

        all_precisions.append(average_precisions[label][0])

        if save_path!=None:
            matplotlib.use('Agg')
            plt.figure()
            plt.xlim((0,1.1))
            plt.ylim((0,1))
            plt.plot(recalls[label],precisions[label])
            # naming the x axis 
            plt.xlabel('Recall') 
            # naming the y axis 
            plt.ylabel('Precision') 

            # giving a title to my graph 
            plt.title('Precision Recall curve') 

            # function to show the plot
            plt.savefig(save_path+C.ID+'//'+label_name+'_precision_recall.jpg')
    print("Debug : ")
    print("All precisions : {}".format(all_precisions))
    print("Total instances : {}".format(total_instances))
    mAPs = pd.DataFrame([['{:.4f}'.format(ap[0]) for ap in average_precisions.values()]+['{:.4f}'.format(sum(all_precisions) / sum(x > 0 for x in total_instances))]],columns = [generator.label_to_name(ap) for ap in average_precisions.keys()]+['mAP'])
    mAPs.to_csv(save_path+C.ID+'//'+ID+'.csv',sep='\t')

    precision_recall_ped = pd.DataFrame([precisions[0],recalls[0]])
    precision_recall_ped.to_csv(save_path+C.ID+'//precision_recall.csv',sep='\t')

    return average_precisions
