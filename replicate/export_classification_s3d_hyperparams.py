import os, utils

data_list = [
             #"breastcancer", "magic", "hillvalley", "gisette", "madelon",
             #"default", "spambase", "spectf", "parkinsons", "arcene",
             #"khan", "duolingo", "digg", "duolingo_cleaned"
             "khan_cleaned", 'stackoverflow_cleaned_subset', "twitter",
            ]

data_list = [
             "breastcancer", "spambase", "spectf", "parkinsons",
             "stackoverflow_cleaned_subset", "khan_cleaned",
             "digg", "twitter", "duolingo_cleaned"
            ]

metric_list = ['accuracy','auc_macro','auc_micro','f1_binary','f1_macro','f1_micro']
for metric in metric_list:
    dir_name = 'classification/s3d-hyperparams/'+metric
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

for metric in metric_list:
    for data_name in data_list:
        try:
            fn = 'cv/{}/performance.csv'.format(data_name)
            data_df = utils.find_best_param(fn, metric)
            data_df.to_csv('classification/s3d-hyperparams/{}/{}.csv'.format(metric, data_name),
                           index=False)
        except:
            print('not run for', data_name)
