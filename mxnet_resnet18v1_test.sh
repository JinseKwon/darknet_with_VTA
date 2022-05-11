#!/bin/bash
CFG=mxnet_resnet18v1.cfg
WEI=mxnet_resnet18v1.weights

if [ ! -f "$CFG" ]; then
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1azYos4Q-O3kfYS9vkuy2w8eBLA0R6pKN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1azYos4Q-O3kfYS9vkuy2w8eBLA0R6pKN" -O $CFG && rm -rf ~/cookies.txt
fi

if [ ! -f "$WEI" ]; then
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vhgeDVhh3QBJK-xgo_6RI0Aw1aHIruU4' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vhgeDVhh3QBJK-xgo_6RI0Aw1aHIruU4" -O mxnet_resnet18v1.weights && rm -rf ~/cookies.txt
fi

./darknet classifier predict cfg/caffe_image1k.data $CFG $WEI $1 -mean 0.485 0.456 0.406 -var 0.229 0.224 0.225

