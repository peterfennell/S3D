#1/bin/bash
for f in run-s3d-runtime-*.sh;
#for f in $(eval ls run-classification-bm-pure-duolingo*.sh);
do
    #echo $f
    qsub $f
done

