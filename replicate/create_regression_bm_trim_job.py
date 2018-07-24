#import sys, json
import sys
import pandas as pd

#s3d_metric = input("select a metric to select the best s3d param for random forest")
#s3d_metric = 'f1_micro'
s3d_metric = 'mse'

data_list = [
             "appenergy", "building_sales", "building_costs",
             "tomshardware", "ailerons", "elevators", "pol",
             "boston_housing", "pyrim", "triazines", "breastcancer_reg",
             "parkinsons_motor", "parkinsons_total"
            ]

for i, data_name in enumerate(data_list):
    ## for each data, find the number of features selected by S3D
    #num_f_s3d = json.load(open('../S3D/hyperparams/{}.json'.format(data_name)))['num_features']
    fp = 'regression/s3d-hyperparams/{}/{}.csv'.format(s3d_metric, data_name)
    num_f_s3d = pd.read_csv(fp)['num_features']
    num_f_s3d = int(num_f_s3d.mean())
    print('s3d find {} features in {} data'.format(num_f_s3d, data_name))

    for num_f_clf in [num_f_s3d, num_f_s3d*2]:
        with open('run-regression-bm.sh', 'r') as f:
            job_file = f.read()
            job_file = job_file.replace('bm-DATANAME-CLF.log',
                                        'bm-DATANAME-CLF_{}.log'.format(num_f_clf))
            job_file = job_file.replace('DATANAME', data_name)
            job_file = job_file.replace('CLF', 'randomforest')
            job_file = job_file.replace('TRIM', 'True')
            job_file = job_file.replace('NUM_F', str(num_f_clf))

        ofile = 'run-regression-bm-trim-{}-{}_{}.sh'.format(data_name, 'randomforest', num_f_clf)
        with open(ofile, 'w') as f:
            f.write(job_file)
