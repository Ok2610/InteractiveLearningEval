#!/bin/bash

MEAS=$1
EXP_NAME=$2
RES_FILE=$3
EXTRA=$4

if [ $MEAS == "y" ]
then
    ./run_il.sh results/measurements/$EXP_NAME/ $RES_FILE comp comp/vis_index_full.cnfg filters/filters.json filters/dist_cats.json filters/dist_tags.json "--measurements ${EXTRA}"
    mkdir logs/$EXP_NAME/$RES_FILE
    mv *.log logs/$EXP_NAME/$RES_FILE/
else
    ./run_il.sh results/no_measurements/$EXP_NAME/ $RES_FILE comp comp/vis_index_full.cnfg filters/dist_cats.json filters/dist_tags.json $EXTRA
fi
