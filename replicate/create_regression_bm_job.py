#import sys, json
import sys
import pandas as pd

data_list = [
             "appenergy", "building_sales", "building_costs",
             "tomshardware", "ailerons", "elevators", "pol",
             "boston_housing", "pyrim", "triazines", "breastcancer_reg",
             "parkinsons_motor", "parkinsons_total"
            ]

for i, data_name in enumerate(data_list):
    for clf_name in ['lasso', 'elasticnet', 'linearsvr', 'randomforest']:
        with open('run-regression-bm.sh', 'r') as f:
            job_file = f.read()
            job_file = job_file.replace('DATANAME', data_name)
            job_file = job_file.replace('CLF', clf_name)
            job_file = job_file.replace('NUM_F', '0')
            job_file = job_file.replace('TRIM', 'False')

        with open('run-regression-bm-pure-{}-{}.sh'.format(data_name, clf_name), 'w') as f:
            f.write(job_file)
