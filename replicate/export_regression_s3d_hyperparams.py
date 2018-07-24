import os, utils

data_list = [
             "appenergy", "building_sales", "building_costs",
             "tomshardware", "ailerons", "elevators", "pol",
             "boston_housing", "pyrim", "triazines", "breastcancer_reg",
             "parkinsons_motor", "parkinsons_total"
            ]

#metric_list = ['accuracy','auc_macro','auc_micro','f1_binary','f1_macro','f1_micro']
metric_list = ['mse','mae','mae_median','r2']
for metric in metric_list:
    dir_name = 'regression/s3d-hyperparams/'+metric
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

for metric in metric_list:
    for data_name in data_list:
        try:
            fn = 'cv/{}/performance.csv'.format(data_name)
            data_df = utils.find_best_param(fn, metric)
            data_df.to_csv('regression/s3d-hyperparams/{}/{}.csv'.format(metric, data_name),
                           index=False)
        except:
            print('not run for', data_name)
