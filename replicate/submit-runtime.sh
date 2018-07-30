#1/bin/bash
for f in run-runtime-*.sh;
#for f in $(eval ls run-classification-bm-pure-duolingo*.sh);
do
    #echo $f
    qsub $f
done
