# --------------------------------------------------------
# Zhefan Ye
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import time
import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from scipy.misc import imread
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections
from model.utils.blob import im_list_to_blob

from options import parse_args
from net_opts import init_net


try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def det_get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


if __name__ == '__main__':

    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception(
            'There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    classes = np.asarray(['__background__',  # always index 0
                          'tide', 'spray_bottle', 'waterpot', 'sugar',
                          'red_bowl', 'clorox', 'sunscreen', 'downy', 'salt',
                          'toy', 'detergent', 'scotch_brite', 'coke',
                          'blue_cup', 'ranch'])

    fasterRCNN = init_net(False, classes, args)

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data, volatile=True)
    im_info = Variable(im_info, volatile=True)
    num_boxes = Variable(num_boxes, volatile=True)
    gt_boxes = Variable(gt_boxes, volatile=True)

    fasterRCNN.cuda()
    fasterRCNN.eval()

    start = time.time()
    max_per_image = 100
    vis = True

    imglist = [f for f in os.listdir(args.image_dir) if f.endswith(".jpg")]
    num_images = len(imglist)

    print('Loaded Photo: {} images.'.format(num_images))

    for i in range(num_images):

        # Load the demo image
        im_file = os.path.join(args.image_dir, imglist[i])
        # im = cv2.imread(im_file)
        im_in = np.array(imread(im_file))
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)

        im = im_in

        blobs, im_scales = det_get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        # transform to pytorch image
        im_data_pt = torch.from_numpy(im_blob)
        ih, iw = im_data_pt.size(1), im_data_pt.size(2)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2).contiguous().view(3, ih, iw)
        im_data_pt = normalize(im_data_pt.div(255)) # normalize img
        im_data_pt = im_data_pt.unsqueeze(0)

        im_info_pt = torch.from_numpy(im_info_np)

        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()

        det_tic = time.time()

        rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev

                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                    + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(
                    1, -1, 4 * len(classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im2show = np.copy(im[:, :, ::-1]) # rgb -> bgr for save
        for j in xrange(1, len(classes)):
            inds = torch.nonzero(scores[:, j] > args.cls_thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)

                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(
                        im2show, classes[j], cls_dets.cpu().numpy(), args.cls_thresh)

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r'
              .format(i + 1, num_images, detect_time, nms_time))
        if vis:
            if not os.path.exists(os.path.join(args.image_dir, 'det')):
                os.makedirs(os.path.join(args.image_dir, 'det'))
            result_path = os.path.join(
                args.image_dir, 'det', imglist[i][:-4] + "_det.jpg")
            cv2.imwrite(result_path, im2show)
