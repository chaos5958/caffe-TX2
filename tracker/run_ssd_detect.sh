#!/bin/sh

#sudo sh /home/nvidia/jetson_clocks.sh
./track ../models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt ../models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel ../examples/videos/test_multiple.avi

