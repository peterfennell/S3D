#!/bin/bash
red=`tput setaf 1`
green=`tput setaf 2`
magenta=`tput setaf 5`
cyan=`tput setaf 6`

RED=`tput setab 1`
GREEN=`tput setab 2`
MAGENTA=`tput setab 5`
CYAN=`tput setab 6`

reset=`tput sgr0`

usage="
$(basename "$0") [-h] -- program to split a predefined dataset.

checkout ${cyan}split_data.py${reset} for more information by ${cyan}python split_data.py -h${reset}

where:
    -h  show this help text

example usage:
    ${cyan}./split.sh${reset}
"

if [ "$1" == "-h" ]; then
    echo -e "$usage"
    exit 0
fi

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
