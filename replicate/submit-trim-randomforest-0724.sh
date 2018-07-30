#1/bin/bash
## I UPDATED THE `MAX_FEATURES` TO BE NONE IN ALL TRIMMED VERSION
## SO THAT THEY WILL CONSIDER EVERY TOP FEATURE
#for f in $(eval ls run-*-bm-pure-*-randomforest.sh);
#for f in $(eval ls run-*-bm-trim-*-randomforest_*.sh);
for f in $(eval ls run-classification-bm-trim-*-randomforest_*.sh);
#for f in $(eval ls run-classification-bm-pure-duolingo*.sh);
do
    #echo $f
    qsub $f
done

