## classification first
type_ = 'classification'
data_list = [
             "breastcancer", "spambase", "spectf", "parkinsons",
             "stackoverflow_cleaned_subset", "khan_cleaned",
             "digg", "twitter", "duolingo_cleaned"
            ]

for data_name in data_list:
    with open('run-runtime.sh') as jobfile:
        jobfile = jobfile.read()
        jobfile = jobfile.replace('DATANAME', data_name)
        jobfile = jobfile.replace('TYPE', type_)
    with open('run-runtime-{}.sh'.format(data_name), 'w') as f:
        f.write(jobfile)

## regression
type_ = 'regression'
data_list = ["appenergy", "building_sales", "building_costs",
             "pol", "breastcancer_reg",
             "boston_housing", "triazines",
             "parkinsons_motor", "parkinsons_total",
            ]

for data_name in data_list:
    with open('run-runtime.sh') as jobfile:
        jobfile = jobfile.read()
        jobfile = jobfile.replace('DATANAME', data_name)
        jobfile = jobfile.replace('TYPE', type_)
    with open('run-runtime-{}.sh'.format(data_name), 'w') as f:
        f.write(jobfile)
