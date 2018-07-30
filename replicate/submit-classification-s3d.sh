#1/bin/bash
for f in $(eval ls run-classification-s3d-*.sh);
do
    #echo $f
    qsub $f
done
