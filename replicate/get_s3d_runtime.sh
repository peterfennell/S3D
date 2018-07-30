#1/bin/bash
## obtain runtime for s3d

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
