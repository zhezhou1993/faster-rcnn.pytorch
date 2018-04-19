#!/usr/bin/env python

"""
Zhefan Ye
"""

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

import rospy
from cv_bridge import CvBridge

from density_pyramid_detector_ros_wrapper.srv import *


class options:
    """ param class """

    def __init__(self):
        self.classes = np.asarray(['__background__',  # always index 0
                                   'tide', 'spray_bottle', 'waterpot', 'sugar',
                                   'red_bowl', 'clorox', 'sunscreen', 'downy', 'salt',
                                   'toy', 'detergent', 'scotch_brite', 'coke',
                                   'blue_cup', 'ranch'])
        self.cuda = True


class FasterRCNNDetector(object):

    def __init__(self):
        rospy.init_node('density_pyramid_detector_node')

        self.opts = options()
        rospy.loginfo('Loading model...')
        checkpoint = torch.load(os.path.join(
            '/home/sui/workspace/icra18/checkpoint/ckpt_96.pkl'))
        self.net = checkpoint['net']
        rospy.loginfo('Done')
        self.net.eval()

        self.detector_server = rospy.Service('density_pyramid_detection',
                                             DensityPyramidDetection,
                                             self.image_cb)
        rospy.loginfo("Ready to detect density pyramid")

        rospy.spin()

    def image_cb(self, req):
        rospy.loginfo("Received detection request")

        image = req.image
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(image, "rgb8")

        self.opts.out_path = "/home/sui/Data/rss_experiments/" + \
            str(req.scene_name.data) + '/output/'

        return self.detect(image, self.opts)

    def det_get_image_blob(self, im):
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

    def detect(self, image, opts):
        pass
