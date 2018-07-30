#1/bin/bash
red=`tput setaf 1`
green=`tput setaf 2`
magenta=`tput setaf 5`
cyan=`tput setaf 6`

RED=`tput setab 1`
GREEN=`tput setab 2`
MAGENTA=`tput setab 5`
CYAN=`tput setab 6`

reset=`tput sgr0`

## obtain runtime for s3d
usage="
$(basename "$0") [-h] DATANAME LAMBDA_ARR -- program to obtain runtime of s3d over a dataset given an array of lambda values

where:
    ${cyan}-h${reset}  show this help text
    ${cyan}DATANAME${reset}  data name
    ${cyan}LAMBDA_ARR${reset} array of lambda values, separated by space

example usage:
    ${cyan}./get_s3d_runtime.sh appenergy 0.003 0.003 0.003 0.003 0.001${reset}
"

if [ "$1" == "-h" ]; then
    echo "$usage"
    exit 0
fi

## model parameter
DATANAME=$1
LAMBDA_ARR=("${@:2}")
echo "lambda list: $LAMBDA_ARR"

## I/O
OUTFOLDER="tmp-models/$DATANAME/"

OUTPUTFILE="s3d-runtime/$DATANAME.csv"
> $OUTPUTFILE

if [ ! -d "$OUTFOLDER" ]; then
    mkdir $OUTFOLDER
    echo "create output folder for $DATANAME"
fi

## array to keep an record of elapsed time
#declare -a ELASPED_TIME_ARR
#ELASPED_TIME_ARR=()


for FOLD in {0..4}; do
    INPUTFILE="../splitted_data/$DATANAME/$FOLD/train.csv"
    LAMBDA="${LAMBDA_ARR[$FOLD]}"
    echo "working on fold $FOLD.. with lambda $LAMBDA"
    for i in {1..10}; do
        START=$(date +%s.%N)
        ./train -infile:$INPUTFILE -outfolder:$OUTFOLDER -lambda:$LAMBDA -ycol:0
        END=$(date +%s.%N)
        DIFF=$(python -c "print(${END} - ${START})")
        #ELASPED_TIME_ARR+=($DIFF)
        echo $DIFF >> $OUTPUTFILE
    done
done
