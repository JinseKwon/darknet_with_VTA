#!/bin/bash
CFG=caffe_resnet18.cfg
WEI=caffe_resnet18.weights

if [ ! -f "$CFG" ]; then
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12Lv-VOjWBDLS9K1LWz16Eu5MD4qHVlHa' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12Lv-VOjWBDLS9K1LWz16Eu5MD4qHVlHa" -O $CFG && rm -rf ~/cookies.txt
fi

if [ ! -f "$WEI" ]; then
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nkAKXrNCGYv5aMTqNnMRsoZCxJ_Segu6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nkAKXrNCGYv5aMTqNnMRsoZCxJ_Segu6" -O $WEI && rm -rf ~/cookies.txt
fi 
./darknet classifier predict cfg/caffe_image1k.data $CFG $WEI $1 -img_range 255 -mean 104 116.7 122.7
