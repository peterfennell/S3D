import os
import pandas as pd

for type_, metric in [['classification', 'auc_micro'],
                      ['regression', 'mse']]:
    dir_ = '{}/s3d-hyperparams/{}/'.format(type_, metric)
    file_list = os.listdir(dir_)
    for f in file_list:
        data_name = f[:-4]
        df = pd.read_csv(dir_+f)
        lambda_arr = df['lambda_'].values
        job_file = open('run-s3d-runtime.sh').read()
        job_file = job_file.replace('DATANAME', data_name)
        job_file = job_file.replace('LAMBDA_ARR', ' '.join(lambda_arr.astype(str)))
        with open('run-s3d-runtime-{}.sh'.format(data_name), 'w') as f:
            f.write(job_file)
