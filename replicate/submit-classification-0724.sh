#1/bin/bash
## NOTE THAT THIS FILE IS TO SUBMIT JOBS FOR STACKOVERFLOW, TWITTER, AND KHAN (CLEANED) ONLY
## AFTER DISCUSSION WITH DR. LERMAN AND KEITH

declare -a arr=("twitter" "stackoverflow_cleaned_subset" "khan_cleaned")
               

for data_name in "${arr[@]}"
do
    for f in run-classification-*-$data_name*.sh
    do
        #echo $f
        qsub $f
    done
done
