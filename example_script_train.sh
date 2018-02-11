#!/bin/bash

mkdir model

./train -infile:data/stackexchangedata_sample.csv -outfolder:model -lambda:0.001 -ycol:1 -n_rows:500000
