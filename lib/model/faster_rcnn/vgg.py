from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class vgg(_fasterRCNN):
  def __init__(self, classes, num_layers=16, pretrained=False, class_agnostic=False):
    self.num_layers = num_layers
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    if self.num_layers == 11:
      vgg = models.vgg11()
      self.model_path = model_urls['vgg11']
      self.fix_layer = 6
    elif self.num_layers == 13:
      vgg = models.vgg13()
      self.model_path = model_urls['vgg13']
      self.fix_layer = 10
    elif self.num_layers == 16:
      vgg = models.vgg16()
      self.model_path = model_urls['vgg16']
      self.fix_layer = 10
    elif self.num_layers == 19:
      vgg = models.vgg19()
      self.model_path = model_urls['vgg19']
      self.fix_layer = 10
    else:
        raise Exception('No such vgg model')

    if self.pretrained:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = model_zoo.load_url(self.model_path)
      vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(self.fix_layer):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

