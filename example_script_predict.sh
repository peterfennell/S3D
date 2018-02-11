#!/bin/bash

mkdir predictions

./predict_expectations -datafile:data/stackexchangedata_sample.csv -infolder:model -outfolder:predictions -max_features:2
