#!/bin/bash
python trainval_net.py --dataset pascal_voc --net vgg11 --bs 1 --cuda --epochs 6 --use_tfboard
python trainval_net.py --dataset pascal_voc --net vgg19 --bs 1 --cuda --epochs 6 --use_tfboard
exit 0