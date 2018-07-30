#1/bin/bash
for f in $(eval ls run-regression-s3d-*.sh);
do
    #echo $f
    qsub $f
done
