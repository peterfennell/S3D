#!/bin/bash

## SPLIT CLS DATASETS
declare -a data_arr=(
                     "breastcancer" "spambase" "spectf" "parkinsons"
                     "stackoverflow_cleaned_subset" "khan_cleaned"
                     "digg" "twitter" "duolingo_cleaned"
                    )

for data in "${data_arr[@]}"
do
   #echo "$i"
   python split_data.py "$data" 5 -cf 1
done

## SPLIT REG DATASETS
declare -a data_arr=(
                     "appenergy" "building_sales" "building_costs"
                     "pol" "breastcancer_reg"  
                     "boston_housing" "triazines"
                     "parkinsons_motor" "parkinsons_total"
                    )

for data in "${data_arr[@]}"
do
   #echo "$i"
   python split_data.py "$data" 5 -cf 0
done
