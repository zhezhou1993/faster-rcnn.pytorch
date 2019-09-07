# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from faster_rcnn_object_detector.srv import *
from faster_rcnn_object_detector.msg import *
import rospy

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="data/models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)
  parser.add_argument('--cls_thresh', default=0.5, type=float,
                      help='classifier threshold')

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

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

def image_process_bbox_with_nms(im):
  resp = ImageToBBoxResponse()

  blobs, im_scales = _get_image_blob(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"
  im_blob = blobs
  im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

  im_data_pt = torch.from_numpy(im_blob)
  im_data_pt = im_data_pt.permute(0, 3, 1, 2)
  im_info_pt = torch.from_numpy(im_info_np)

  im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
  im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
  gt_boxes.data.resize_(1, 1, 5).zero_()
  num_boxes.data.resize_(1).zero_()

  # pdb.set_trace()
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
       if args.class_agnostic:
           box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                      + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
           box_deltas = box_deltas.view(1, -1, 4)
       else:
           box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                      + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
           box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

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
     im2show = np.copy(im)
  for j in xrange(1, len(pascal_classes)):
     inds = torch.nonzero(scores[:,j]>thresh).view(-1)
     # if there is det
     if inds.numel() > 0:
       cls_scores = scores[:,j][inds]
       all_scores = scores[inds.cpu().numpy(),:]
       all_scores = all_scores[:,1:]
       _, order = torch.sort(cls_scores, 0, True)
       if args.class_agnostic:
         cls_boxes = pred_boxes[inds, :]
       else:
         cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

       cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
       # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
       cls_dets = cls_dets[order]
       all_scores = all_scores[order]
       keep = nms(cls_dets, cfg.TEST.NMS)
       cls_dets = cls_dets[keep.view(-1).long()]
       all_scores = all_scores[keep.view(-1).long()]
       if vis:
         im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), args.cls_thresh)

       dets = cls_dets.cpu().numpy()
       class_name = pascal_classes[j]

       Object = ObjectInfo()
       Object.label = class_name
       Object.all_label = pascal_classes[1:]
       for i in range(np.minimum(10, dets.shape[0])):
         bbox = tuple(int(np.round(x)) for x in dets[i, :4])
         score = dets[i, -1]
         if score > args.cls_thresh:
           BBox = BBoxInfo()
           BBox.bbox_xmin = int(bbox[0])
           BBox.bbox_ymin = int(bbox[1])
           BBox.bbox_xmax = int(bbox[2])
           BBox.bbox_ymax = int(bbox[3])
           BBox.label = pascal_classes[1:]
           BBox.score = all_scores[i,:].tolist()
           BBox.max_label = class_name
           BBox.max_score = score

           resp.bbox.append(BBox)

  misc_toc = time.time()
  nms_time = misc_toc - misc_tic
  if vis:
      cv2.imshow('test', im2show)
      cv2.waitKey(1)

  sys.stdout.write('im_detect: {:.3f}s {:.3f}s   \r' \
                    .format(detect_time, nms_time))
  sys.stdout.flush()

  print(resp)
  return resp

def image_process(im):
  resp = ImageToObjectResponse()

  blobs, im_scales = _get_image_blob(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"
  im_blob = blobs
  im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

  im_data_pt = torch.from_numpy(im_blob)
  im_data_pt = im_data_pt.permute(0, 3, 1, 2)
  im_info_pt = torch.from_numpy(im_info_np)

  im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
  im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
  gt_boxes.data.resize_(1, 1, 5).zero_()
  num_boxes.data.resize_(1).zero_()

  # pdb.set_trace()
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
       if args.class_agnostic:
           box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                      + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
           box_deltas = box_deltas.view(1, -1, 4)
       else:
           box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                      + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
           box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

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
     im2show = np.copy(im)
  for j in xrange(1, len(pascal_classes)):
     inds = torch.nonzero(scores[:,j]>thresh).view(-1)
     # if there is det
     if inds.numel() > 0:
       cls_scores = scores[:,j][inds]
       _, order = torch.sort(cls_scores, 0, True)
       if args.class_agnostic:
         cls_boxes = pred_boxes[inds, :]
       else:
         cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

       cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
       # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
       cls_dets = cls_dets[order]
       keep = nms(cls_dets, cfg.TEST.NMS)
       cls_dets = cls_dets[keep.view(-1).long()]
       if vis:
         im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), args.cls_thresh)

       dets = cls_dets.cpu().numpy()
       class_name = pascal_classes[j]

       Object = ObjectInfo()
       Object.label = class_name
       Object.all_label = pascal_classes[1:]
       for i in range(np.minimum(10, dets.shape[0])):
         bbox = tuple(int(np.round(x)) for x in dets[i, :4])
         score = dets[i, -1]
         if score > args.cls_thresh:
           Object.bbox_xmin.append(int(bbox[0]))
           Object.bbox_ymin.append(int(bbox[1]))
           Object.bbox_xmax.append(int(bbox[2]))
           Object.bbox_ymax.append(int(bbox[3]))
           Object.score.append(score)

       resp.objects.append(Object)

  misc_toc = time.time()
  nms_time = misc_toc - misc_tic

  if vis:
      cv2.imshow('test', im2show)
      cv2.waitKey(1)

  sys.stdout.write('im_detect: {:.3f}s {:.3f}s   \r' \
                    .format(detect_time, nms_time))
  sys.stdout.flush()

  return resp

def handle_image_bbox(req):
  print("Received Image, start Detection")
  image = req.image
  bridge = CvBridge()
  image = bridge.imgmsg_to_cv2(image, "bgr8")

  return image_process_bbox_with_nms(image)

def handle_image_objects(req):
  print("Received Image, start Detection")
  image = req.image
  bridge = CvBridge()
  image = bridge.imgmsg_to_cv2(image, "bgr8")

  return image_process(image)

def bbox_detection_server():
  rospy.init_node('object_detection_server')
  s = rospy.Service('object_detection', ImageToObject, handle_image_objects)
  #s = rospy.Service('object_detection', ImageToBBox, handle_image_bbox)
  print("Ready for Object Detection")
  rospy.spin()

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == 'coco':
    args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  #pascal_classes = np.asarray(['__background__',
  #                     'aeroplane', 'bicycle', 'bird', 'boat',
  #                     'bottle', 'bus', 'car', 'cat', 'chair',
  #                     'cow', 'diningtable', 'dog', 'horse',
  #                     'motorbike', 'person', 'pottedplant',
  #                     'sheep', 'sofa', 'train', 'tvmonitor'])


  if args.dataset == 'progress':
    pascal_classes = np.asarray(['__background__', # always index 0
            'apple', 'bowl', 'cereal', 'coke', 'cup', 'milk', 'pringle', 'table', 'shampoo',
            'alumn_cup', 'dispenser', 'loofah', 'rack', 'brown_table'])
  elif args.dataset == 'glassloc_demo':
     pascal_classes = np.asarray(['__background__', # always index 0
           'head', 'tail'])
  elif args.dataset == 'magna_demo':
     pascal_classes = np.asarray(['__background__', # always index 0
           'red hat', 'bag', 't-shirt', 'golf box'])
  elif args.dataset == 'lfd_demo':
     pascal_classes = np.asarray(['__background__', # always index 0
           'handle', 'feather', 'mfc_top', 'mfc_bottom'])
  elif args.dataset == 'coco':
    pascal_classes = np.asarray(['__background__', # always index 0
          'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
          'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
          'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
          'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
          'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
          'hair drier', 'toothbrush'])
  elif args.dataset == 'magna':
    pascal_classes = np.asarray(['__background__', # always index 0
          'red hat', 'apple charger', 'umbrella', 'golf box', 't-shirt', 'pen'])

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')

  # pdb.set_trace()

  print("load checkpoint %s" % (load_name))

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)

  if args.cuda > 0:
    cfg.CUDA = True

  if args.cuda > 0:
    fasterRCNN.cuda()

  fasterRCNN.eval()

  start = time.time()
  max_per_image = 100
  thresh = 0.05
  vis = False

  bbox_detection_server()
