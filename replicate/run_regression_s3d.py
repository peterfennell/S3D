import sys
import pandas as pd
from pys3d import PYS3D

data_name = sys.argv[1]

dim_df = pd.read_csv('../data/regression-data-dim.csv', index_col=0, squeeze=True)
lambda_list = [0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]


max_features = dim_df.loc[data_name]
#max_features = min(max_features, 20)
num_cores = 30

s3d = PYS3D(data_name, classification_flag=False)
s3d.cross_val_multicore(lambda_list, max_features,
                        num_cores=num_cores)

## evaluate
s3d = PYS3D(data_name, classification_flag=False)
df = s3d.evaluate(num_jobs=num_cores, cv_metric='mse')
df.to_csv('regression/performance/{}-test-performance.csv'.format(data_name))
