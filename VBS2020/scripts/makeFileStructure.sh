#!/bin/bash

cp /home/bjonsson/Journal/vbs/*.sh .
cp /home/bjonsson/Journal/vbs/vbs_il.py .
cp /home/bjonsson/Journal/vbs/actors.json .
cp /home/bjonsson/Journal/vbs/vidinfo.json .
cp -r /home/bjonsson/ICMR21/vbs_il/filters .

mkdir comp
cp /var/scratch/bjonsson/Journal/2021/vbs/vis* comp/.

mkdir logs
mkdir logs/expansion
mkdir logs/fixed
mkdir results
mkdir results/measurements
mkdir results/measurements/fixed
mkdir results/measurements/expansion

