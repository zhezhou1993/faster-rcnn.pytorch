#!/bin/bash
python trainval_net.py --dataset pascal_voc --net res18 --bs 1 --cuda --epochs 7 --use_tfboard
python trainval_net.py --dataset pascal_voc --net res34 --bs 1 --cuda --epochs 7 --use_tfboard
exit 0


