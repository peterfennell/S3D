data_list = [
             "appenergy", "building_sales", "building_costs",
             "tomshardware", "ailerons", "elevators", "pol",
             "boston_housing", "pyrim", "triazines", "breastcancer_reg",
             "parkinsons_motor", "parkinsons_total"
            ]

for i, data_name in enumerate(data_list):
    print(data_name)

    with open('run-regression-s3d.sh', 'r') as f:
        job_file = f.read()
        job_file = job_file.replace('DATANAME', data_name)
        #job_file = job_file.replace('MAX_FEATURES', str(max_features))
        job_file = job_file.replace('NUM_JOBS', '30')

        if i < 2:
            pass
        elif i < 4:
            job_file = job_file.replace('all.q', 'UI')
        else:
            job_file = job_file.replace('all.q', 'INFORMATICS')

    with open('run-regression-s3d-{}.sh'.format(data_name), 'w') as f:
        f.write(job_file)
