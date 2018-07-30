#1/bin/bash
for f in $(eval ls run-regression-bm-trim-*.sh);
#for f in $(eval ls run-classification-bm-pure-duolingo*.sh);
do
    qsub $f
done
