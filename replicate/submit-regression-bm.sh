#1/bin/bash
#for f in $(eval ls run-regression-bm-pure-*.sh);
for f in $(eval ls run-regression-bm-pure-*-linearsvr.sh);
#for f in $(eval ls run-regression-bm-pure-*-randomforest.sh);
#for f in $(eval ls run-classification-bm-pure-duolingo*.sh);
do
    #echo $f
    qsub $f
done
