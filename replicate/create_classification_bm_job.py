import sys
import pandas as pd

#s3d_metric = input("select a metric to select the best s3d param for random forest")
s3d_metric = 'auc_micro'

data_list = [
             #"breastcancer", "magic", "hillvalley", "gisette", "madelon",
             #"default", "spambase", "spectf", "parkinsons", "arcene",
             #"khan", "duolingo", "digg", "duolingo_cleaned"
             "khan_cleaned", 'stackoverflow_cleaned_subset', "twitter",
            ]

for i, data_name in enumerate(data_list):
    for clf_name in ['lasso', 'elasticnet', 'linearsvc', 'randomforest']:
        with open('run-classification-bm.sh', 'r') as f:
            job_file = f.read()
            job_file = job_file.replace('DATANAME', data_name)
            job_file = job_file.replace('CLF', clf_name)
            job_file = job_file.replace('NUM_F', '0')
            job_file = job_file.replace('TRIM', 'False')

        with open('run-classification-bm-pure-{}-{}.sh'.format(data_name, clf_name), 'w') as f:
            f.write(job_file)
