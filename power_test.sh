#!/bin/bash
for number in 1 2 3 4 5
do
# nvprof --csv --system-profiling on --print-gpu-trace --devices 0 --log-file human-readable-vgg11-$number.log ./profile.py --net vgg11 --checksession 1 --checkepoch 6 --checkpoint 10021 --cuda --dataset pascal_voc --cls_thresh 0.1
# nvprof --csv --system-profiling on --print-gpu-trace --devices 0 --log-file human-readable-vgg16-$number.log ./profile.py --net vgg16 --checksession 1 --checkepoch 6 --checkpoint 10021 --cuda --dataset pascal_voc --cls_thresh 0.1
# nvprof --csv --system-profiling on --print-gpu-trace --devices 0 --log-file human-readable-vgg19-$number.log ./profile.py --net vgg19 --checksession 1 --checkepoch 20 --checkpoint 1251 --cuda --dataset pascal_voc --cls_thresh 0.1
nvprof --csv --system-profiling on --print-gpu-trace --devices 0 --log-file human-readable-res101-$number.log ./profile.py --net res101 --checksession 1 --checkepoch 7 --checkpoint 10021 --cuda --dataset pascal_voc --cls_thresh 0.1
done
exit 0