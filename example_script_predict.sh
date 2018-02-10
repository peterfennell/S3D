#!/bin/bash

mkdir predictions

./predict_expectations -datafile:data/StackExchange.csv -infolder:model -outfolder:predictions -max_features:2
