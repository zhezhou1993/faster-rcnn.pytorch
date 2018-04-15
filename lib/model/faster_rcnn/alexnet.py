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
	'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class alexnet(_fasterRCNN):
	def __init__(self, classes, pretrained=False, class_agnostic=False):
		self.dout_base_model = 256
		self.pretrained = pretrained
		self.class_agnostic = class_agnostic

		_fasterRCNN.__init__(self, classes, class_agnostic)

	def _init_modules(self):
		alexnet = models.alexnet()
		if self.pretrained:
			state_dict = model_zoo.load_url(model_urls['alexnet'])
			alexnet.load_state_dict({k:v for k,v in state_dict.items() if k in alexnet.state_dict()})

		alexnet.classifier = nn.Sequential(*list(alexnet.classifier._modules.values())[:-1])

		# not using the last maxpool layer
		self.RCNN_base = nn.Sequential(*list(alexnet.features._modules.values())[:-1])

		# Fix the layers before conv2:
		for layer in range(3):
			for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

		self.RCNN_top = alexnet.classifier

		self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

		if self.class_agnostic:
	  		self.RCNN_bbox_pred = nn.Linear(4096, 4)
		else:
	  		self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

	def _head_to_tail(self, pool5):
		
		pool5_flat = pool5.view(pool5.size(0), -1)
		fc7 = self.RCNN_top(pool5_flat)

		return fc7
