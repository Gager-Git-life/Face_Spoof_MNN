#!/bin/sh
###
 # @Descripttion: 
 # @version: 
 # @Author: Gager
 # @Date: 2020-11-26 15:50:00
 # @LastEditors: sueRimn
 # @LastEditTime: 2020-12-08 16:19:41
### 

if [ ! -d "./build/" ];then
  echo "[INFO]>>> 创建build文件夹"
  mkdir ./build
else
  echo "[INFO]>>> 清空build下内容"
  rm -rf build/*
fi

cd build
cmake ..
make -j8
mv face_spoof ..
cd ..

if [$1="pic"]
then
	./face_spoof
else
	./face_spoof
fi