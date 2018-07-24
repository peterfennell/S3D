data_list = [
             #"breastcancer", "magic", "hillvalley", "gisette", "madelon",
             #"default", "spambase", "spectf", "parkinsons", "arcene",
             #"khan", "duolingo", "digg", "duolingo_cleaned",
             "khan_cleaned", 'stackoverflow_cleaned_subset', "twitter",
            ]

for i, data_name in enumerate(data_list):
    print(data_name)

    with open('run-classification-s3d.sh', 'r') as f:
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

    with open('run-classification-s3d-{}.sh'.format(data_name), 'w') as f:
        f.write(job_file)
