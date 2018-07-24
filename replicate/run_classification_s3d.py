import sys
import pandas as pd
from pys3d import PYS3D

data_name = sys.argv[1]

dim_df = pd.read_csv('../data/classification-data-dim.csv', index_col=0, squeeze=True)

if data_name == 'twitter':
    #lambda_list = [0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
    lambda_list = [0.001, 0.0001, 0.00003, 0.00001]
else:
    lambda_list = [0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]

print(lambda_list)


max_features = dim_df.loc[data_name]
#max_features = min(max_features, 20)
num_cores = 30

s3d = PYS3D(data_name, classification_flag=True)
s3d.cross_val_multicore(lambda_list, max_features,
                        num_cores=num_cores)

## evaluate
s3d = PYS3D(data_name)
df = s3d.evaluate(num_jobs=num_cores)
df.to_csv('classification/performance/{}-test-performance.csv'.format(data_name))
