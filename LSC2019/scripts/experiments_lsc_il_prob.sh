#!/bin/bash

MEAS=$1
EXP_NAME=$2
RES_FILE=$3
EXTRA=$4

if [ $MEAS == "y" ]
then
    ./run_lsc_il_prob.sh results/measurements/$EXP_NAME/ $RES_FILE comp_prob comp_prob/vis_index_full.cnfg filters/filters.json filters/locations.json "--measurements ${EXTRA}"
    mkdir logs/$EXP_NAME/$RES_FILE
    mv *.log logs/$EXP_NAME/$RES_FILE/
else
    ./run_lsc_il_prob.sh results/no_measurements/$EXP_NAME/ $RES_FILE comp_prob comp_prob/vis_index_full.cnfg filters/filters.json $EXTRA
fi
