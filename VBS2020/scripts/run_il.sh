#!/bin/bash

RES_DIR=$1
RES_FILE=$2
COMP_DIR=$3
INDEX_FILE=$4
FILTERS_FILE=$5
CATEGORIES_FILE=$6
TAGS_FILE=$7
EXTRA=$8

echo "Running Exquisitor"
python vbs_il.py actors.json $RES_DIR $RES_FILE $COMP_DIR/vis_init_feat.h5 $COMP_DIR/vis_feat_ids.h5 $COMP_DIR/vis_ratios.h5 $INDEX_FILE $FILTERS_FILE vidinfo.json $CATEGORIES_FILE $TAGS_FILE $EXTRA
date
