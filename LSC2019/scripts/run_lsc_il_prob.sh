#!/bin/bash

RES_DIR=$1
RES_FILE=$2
COMP_DIR=$3
INDEX_FILE=$4
FILTERS_FILE=$5
LOCATIONS_FILE=$6
EXTRA=$7

echo "Running Exquisitor"
python lsc_il.py actors_prob.json $RES_DIR $RES_FILE $COMP_DIR/vis_init_feat.h5 $COMP_DIR/vis_ids.h5 $COMP_DIR/vis_ratios.h5 $INDEX_FILE $FILTERS_FILE $LOCATIONS_FILE $EXTRA
date
