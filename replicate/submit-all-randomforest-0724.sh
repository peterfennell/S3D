#1/bin/bash
## TODAY I UPDATED THE MAX_FEATURES IN RANDOM FOREST STRICTLY FOLLOWING THE HEURISTICS
## AND REMOVE THE RESTRICTION OF 20 FEATURES
#for f in $(eval ls run-*-bm-pure-*-randomforest.sh);
for f in $(eval ls run-regression-bm-pure-*-randomforest.sh);
#for f in $(eval ls run-classification-bm-pure-duolingo*.sh);
do
    #echo $f
    qsub $f
done

