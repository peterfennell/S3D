import pandas as pd
from contextlib import redirect_stdout
from joblib import Parallel, delayed, cpu_count
import subprocess, os, shutil, time, utils, io, sys, warnings

class PYS3D(object):
    ''' a wrapper function to run s3d in python
        make it similar to sklearn interfaces
    '''
    def __init__(self,
                 data_name,
                 data_path='../splitted_data/',
                 model_path='models/',
                 prediction_path='predictions/',
                 classification_flag=True,
                 ):
        ''' initializer

            parameters
            ----------
            data_name : str
                intput data name
            data_path : str
                input data base path
            model_path : str
                base path for output of built models
            prediction_path : str
                base path for predicted expectations
            classification_flag : bool
                whether this is classification or regression. tihis is used for determining evaluation metrics

            for each path, we assume that there are sub folders for each test fold
        '''

        print('...s3d initializing...')
        self.data_name = data_name
        self.data_path = data_path + self.data_name + '/'
        self.model_path = model_path + self.data_name + '/'
        self.prediction_path = prediction_path + self.data_name + '/'
        self.classification_flag = classification_flag
        ## check path validity
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.prediction_path):
            os.makedirs(self.prediction_path)

        ## find the number of folds by counting the number of folders in self.data_path
        self.num_folds = len(os.listdir(self.data_path))
        ## inner_num_folds is the number of folds for parameter grid search
        self.inner_num_folds = self.num_folds - 1

        ## create a temporary folder for inner cross validation
        self.tmp_path = './tmp/{}/'.format(self.data_name)
        if not os.path.exists(self.tmp_path):
            #shutil.rmtree(self.tmp_path)
            os.mkdir(self.tmp_path)

        ## create a folder with similar structure for hyperparameter searching
        self.cv_path = './cv/{}/'.format(self.data_name)
        if not os.path.exists(self.cv_path):
            #os.rmdir(self.cv_path)
            #shutil.rmtree(self.cv_path)
            os.mkdir(self.cv_path)

        ## for both cv and tmp, create subfolders for individual folds
        for fold_index in range(self.num_folds):
            tmp_path = self.tmp_path + str(fold_index) + '/'
            ## remove it, if exist
            if not os.path.exists(tmp_path):
                ## create it
                os.mkdir(tmp_path)
            ## create a cv folder for each fold
            #cv_path = self.cv_path + str(fold_index) + '/'
            #if not os.path.exists(cv_path):
            #    os.mkdir(cv_path)

        print('s3d with {} data, splitted into {} folds'.format(self.data_name, self.num_folds))
        print('data will be loaded from {}'.format(self.data_path))
        print('built models will be saved to {}'.format(self.model_path))
        print('predictions will be saved to {}'.format(self.prediction_path))
        print('temporary subfolders in ', './tmp/{}'.format(self.data_name))
        print('...done initializing...\n')


    def fit(self, train_data_path, train_model_path,
            lambda_=0.01, max_features=None,
            start_skip_rows=-1, end_skip_rows=-1):
        ''' fit s3d with the given lambda value
            notify user if `max_features` cannot be attained

            parameters
            ----------
            train_data_path : str
                training data file
            train_model_path : str
                training data file
            lambda_ : float
                regularization parameter
            max_features : int
                maximum number of features to choose (default 20)
            start_skip_rows : int
                the index of the begining rows to skip (inclusive)
            end_skip_rows : int
                the index of the ending rows to skip (exclusive)
        '''

        if not os.path.exists(train_model_path):
            os.makedirs(train_model_path)

        c = './train -infile:{0} -outfolder:{1} -lambda:{2} -ycol:0'.format(train_data_path,
                                                                            train_model_path,
                                                                            lambda_)
        c += ' -start_skip_rows:{} -end_skip_rows:{}'.format(start_skip_rows, end_skip_rows)
        if max_features is not None:
            c += ' -max_features:{}'.format(max_features)

        #print('fitting s3d with', train_data_path)
        #print('command:', c)

        ## catch the output and save to a log file in the `outfolder`
        process = subprocess.Popen(c.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        with open(train_model_path+'fit.log', 'w') as logfile:
            logfile.write(c)
            logfile.write(output.decode('utf8'))
            logfile.write('---errors below (if any)---\n')
            if error is not None:
                logfile.write(error)

        ## read how many features are selected
        real_max_features = pd.read_csv(train_model_path+'levels.csv').shape[0]
        if real_max_features < max_features:
            warnings.warn("{} features requested by only {} selected".format(max_features,
                                                                             real_max_features),
                          UserWarning)

    def predict(self, test_data_path,
                train_model_path, prediction_path,
                max_features=None, min_samples=1,
                start_use_rows=0, end_use_rows=-1):
        ''' predict for the held-out set
            perform prediction for each individual number of features; from 1 to `max_features`

            parameters
            ----------
            test_data_path : str
                test data file for prediction
            train_model_path : str
                pre-trained model for prediction
            prediction_path : str
                prediction output path
            max_features : int
                maximum number of features used for prediction (default use the number of s3d chosen features)
            min_samples : int
                minimum number of samples required to make a prediction (default 1)
            start_use_rows : int
                the first index of row to use for prediction (inclusive)
            end_use_rows : int
                the last index of row to use for prediction (exclusive)
                rows between start_use_rows - end_use_rows are to be used for prediction
        '''

        real_max_features = pd.read_csv(train_model_path+'levels.csv').shape[0]
        if max_features is None:
            #print('max fea: ', max_features)
            max_features = real_max_features
        elif real_max_features < max_features:
            warnings.warn("{} features requested by only {} selected".format(max_features,
                                                                             real_max_features),
                          UserWarning)
            max_features = real_max_features

        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)

        ## perform prediction based on each number of features
        for n_f in range(1, max_features+1):
            c = './predict_expectations -datafile:{} -infolder:{} -outfolder:{}'.format(test_data_path,
                                                                                        train_model_path,
                                                                                        prediction_path
                                                                                       )
            c += ' -max_features:{}'.format(n_f)
            c += ' -start_use_rows:{} -end_use_rows:{}'.format(start_use_rows, end_use_rows)

            if min_samples > 1:
                c += ' -min_samples:{}'.format(min_samples)

            #print('predicting data in', test_data_path)
            #print('command:', c)

            process = subprocess.Popen(c.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            with open(prediction_path+'predict_MF_{}.log'.format(n_f), 'w') as logfile:
                logfile.write(c)
                logfile.write(output.decode('utf8'))
                logfile.write('---errors below (if any) ---\n')
                if error is not None:
                    logfile.write(error)


    def score(self, test_data_path,
              train_model_path, prediction_path,
              max_features=None, min_samples=1,
              start_use_rows=0, end_use_rows=-1,
              thres=0.5,
              train_data_path=None,
              calc_threshold=False):
        ''' similar to predict() function but to obtain scores
            the additional `train_data_path` was for calculating thresholds
            this will be done on each number of features: from 1 to `max_features`
        '''

        real_max_features = pd.read_csv(train_model_path+'levels.csv').shape[0]
        if max_features is None:
            #print('max fea: ', max_features)
            max_features = real_max_features
        elif real_max_features < max_features:
            warnings.warn("{} features requested by only {} selected".format(max_features,
                                                                             real_max_features),
                          UserWarning)
            max_features = real_max_features


        ## predict first
        self.predict(test_data_path, train_model_path,
                     prediction_path, max_features,
                     min_samples, start_use_rows, end_use_rows)

        ## if no `max_features` provided, go find it
        if max_features is None:
            try:
                max_features = sum(1 for line in open(train_model_path+'splits.csv'))
            except:
                max_features = None

        if calc_threshold and (train_data_path is None or max_features is None):
            raise Exception('calc_threshold is set to True but no training data or max_features is provided...')

        ## obtain prediction performance
        if end_use_rows < 0:
            y_true = pd.read_csv(test_data_path, usecols=[0], squeeze=True,
                                 skiprows=start_use_rows,
                                ).values
        else:
            y_true = pd.read_csv(test_data_path, usecols=[0], squeeze=True,
                                 skiprows=start_use_rows,
                                 nrows=end_use_rows-start_use_rows
                                ).values


        ## obtain accuracy for each `n_f`
        result_df = list()
        for n_f in range(1, max_features+1):
            ## probability scores
            y_score = pd.np.loadtxt(prediction_path+'predicted_expectations_MF_{}.csv'.format(n_f))

            ## prediction values based on probability scores
            if self.classification_flag:
                if calc_threshold and train_data_path is not None:
                    thres = self.calculate_disc_threshold(train_model_path, n_f)
                    print('threshold based on trianing set:', thres, 'for', n_f, 'features')
                y_pred = (y_score >= thres).astype(int)
                series = utils.obtain_metric_classification(y_true, y_pred, y_score)
                series.loc['threshold'] = thres
            else:
                series = utils.obtain_metric_regression(y_true, y_score)

            series.loc['num_features'] = n_f
            result_df.append(series)
        #print('y pred size:', y_pred.size)
        #print(utils.obtain_metric(y_true, y_pred))
        result_df = pd.DataFrame(result_df)
        result_df['num_features'] = result_df['num_features'].astype(int)
        return result_df


    def _inner_cross_validation_fold(self, fold_index,
                                     lambda_, max_features,
                                     thres=0.5,
                                     calc_threshold=True):
        '''
            do a k-fold inner cross validation for a given _outer_ fold: 1 fold for validation; others train
            all data files are prepared beforehand using `split_data` function
            this is _inner_ cv for finding the best hyperparameters, given one outer fold and one set of parameter

            parameters
            ----------
            lambda_ : float
                hyperparam 1: lambda_ parameter
            max_features : int
                hyperparam 2: max numbers of features
            thres : float
                threshold of prediction (default 0.5)
            calc_threshold : bool
                whether or not to calculate threshold based on the training data. Default: True
        '''

        print('--- inner cv for ', fold_index, 'th outer fold using lambda={0} and n_f={1} ---'.format(lambda_, max_features))
        start = time.time()

        ## use a dataframe to save the performance across different folds
        result_df = list()

        ## save the temporary model and predictions into the temporary files
        ## the contents (i.e., models in this case) will be constantly overwritten w.r.t. different validation folds
        subfolder = self.tmp_path+'{}/'.format(fold_index)
        #print(subfolder)
        ## train data of the `fold_index`-th fold
        train_data_path = self.data_path + '{}/train.csv'.format(fold_index)
        ## the number of rows for training is recorded in a separate file
        num_train_rows, _ = open(self.data_path + '{}/num_rows.csv'.format(fold_index)).readlines()
        num_train_rows = int(num_train_rows)
        #print(num_train_rows)
        ## scan through each fold
        for fold_i in range(self.inner_num_folds):
            ## for each fold, use start_skip_rows to split train/validation sets
            start_skip_rows = int((fold_i*num_train_rows) / self.inner_num_folds)
            end_skip_rows = int(((fold_i+1)*num_train_rows) / self.inner_num_folds)
            print(fold_i, '-th fold: start -', start_skip_rows, 'end -', end_skip_rows)
            ## train: model saved into `subfolder`
            self.fit(train_data_path, subfolder,
                     lambda_, max_features,
                     start_skip_rows=start_skip_rows,
                     end_skip_rows=end_skip_rows)

            ## read training score
            ## note that we can find the r^2 based on `levels` file.
            train_r2 = pd.read_csv(subfolder+'levels.csv', usecols=[1],
                                   squeeze=True).values

            ## use self.score() function for easier evaluation
            #3 recall that the prediction will be done for 1,2,...,max_feautres of features selected
            cv_score = self.score(train_data_path,
                                  subfolder, subfolder,
                                  max_features=max_features,
                                  start_use_rows=start_skip_rows,
                                  end_use_rows=end_skip_rows,
                                  thres=thres,
                                  train_data_path=train_data_path,
                                  calc_threshold=calc_threshold)

            cv_score['test_fold_i'] = fold_i
            cv_score['train_r2'] = train_r2
            result_df.append(cv_score)
            #break
        ## convert to dataframe
        performance_df = pd.concat(result_df, ignore_index=True)
        performance_df['lambda_'] = lambda_
        performance_df['max_features'] = max_features
        performance_df['split_version'] = fold_index
        performance_df['test_fold_i'] = performance_df['test_fold_i'].astype(int)
        ## include parameters lambda_ and max_features

        print('--- done inner cv for ', fold_index, 'th outer fold elapsed time: {0:.2f} seconds ---'.format(time.time()-start))

        return performance_df


    def _inner_cross_validation(self, fold_index,
                                lambda_list, max_features,
                                #eval_metric='auc_micro',
                                thres=0.5,
                                calc_threshold=True):
        '''
            same as _inner_cross_validation_fold, but a grid of parameters are given (param_grid)
            does cross validation on all of them
            the final outcome is the validation performance of all paramter settings
        '''

        stringio = io.StringIO()
        with redirect_stdout(stringio):
            ## capture everything and redirect into a file
            performance_df = pd.DataFrame()
            print('--- start hyperparameter search for fold', fold_index, '---')
            start = time.time()
            for lambda_ in lambda_list:
                ## average performance for this fold
                performance = self._inner_cross_validation_fold(fold_index,
                                                                lambda_, max_features,
                                                                thres,
                                                                calc_threshold=True)
                #print(performance)
                performance_df = performance_df.append(performance,
                                                       ignore_index=True)
            print('--- done hyperparam search (elapsed time {0:.2f} seconds)---'.format(time.time()-start))
            ## pick the best param for this test fold `fold_index`
        return performance_df


    def cross_val(self, lambda_list, max_features,
                  thres=0.5,
                  calc_threshold=True):
        ''' grid search for hyperparameters for each fold
        '''

        performance_df = pd.DataFrame()
        print('--- cross validation on', self.data_name, 'data ---')
        start = time.time()
        for fold_index in range(self.num_folds):
            print('starting on fold', fold_index)
            fold_start = time.time()
            performance = self._inner_cross_validation(fold_index, lambda_list,
                                                       max_features, thres,
                                                       calc_threshold)
            print('finish after {0:.2f} seconds'.format(time.time()-fold_start))
            performance_df = performance_df.append(performance, ignore_index=True)

        print('--- done cv; total elapsed time {0:.2f} seconds'.format(time.time()-start))

        #print(performance_df)
        performance_df.to_csv(self.cv_path+'performance.csv', index=False)

    def cross_val_multicore(self, lambda_list, max_features,
                            calc_threshold=True, thres=0.5,
                            num_cores=None):
        ''' grid search for hyperparameters for each fold
            this function will utilizes multiple cores available and
            parallelize each fold
        '''
        if num_cores is None:
            num_cores = self.num_folds
        if num_cores < 2:
            print('num_cores less than 2, still using the for-loop version of cross_val')
            self.cross_val(lambda_list, max_features, thres, calc_threshold)
        else:
            print('--- cross validation ({} cores) on'.format(num_cores), self.data_name, 'data ---')
            start = time.time()
            l = Parallel(n_jobs=num_cores)(delayed(self._inner_cross_validation)(fold_index,
                                                                                 lambda_list, max_features,
                                                                                 thres,
                                                                                 calc_threshold)\
                                           for fold_index in range(self.num_folds))
            print('--- done multi-core cv; total elapsed time {0:.2f} seconds'.format(time.time()-start))

            performance_df = pd.concat(l, axis=0)
            performance_df.to_csv(self.cv_path+'performance.csv', index=False)

    def _evaluate(self, fold_index, cv_metric):
        train_data_path = self.data_path + str(fold_index) + '/train.csv'
        test_data_path = self.data_path + str(fold_index) + '/test.csv'
        train_model_path = self.model_path + str(fold_index) + '/'

        lambda_, num_features, _, __ = self.cv_param_df.loc[fold_index]

        self.fit(train_data_path, train_model_path, lambda_, num_features)
        ## do not give it the number of features
        prediction_path = '{}/{}/'.format(self.prediction_path, fold_index)
        df = self.score(test_data_path, train_model_path,
        prediction_path, train_data_path=train_data_path,
        calc_threshold=True)
        #df.set_index('num_features').loc[num_features]

        ## since we've already updated `max_features` in `score` functions
        ## skip this step
        #if df['num_features'].max() < num_features:
        #    return pd.DataFrame()

        series = df.set_index('num_features').loc[df.num_features.max()]
        series['lambda_'] = lambda_
        series['split_version'] = fold_index
        return series

    def evaluate(self, cv_metric=None, num_jobs=1):
        ''' evaluate s3d on the held out test set using the best parameters based on '''

        if cv_metric is None and self.classification_flag:
            cv_metric = 'auc_micro'
        elif cv_metric is None and not self.classification_flag:
            cv_metric = 'r2'

        fold_list = pd.np.arange(self.num_folds)

        num_jobs = min(num_jobs, cpu_count())
        print('evaluating s3d model using {} cores...'.format(num_jobs))
        self.cv_param_df = utils.find_best_param(self.cv_path+'performance.csv',
                                                 validation_metric=cv_metric)
        self.cv_param_df.set_index('split_version', inplace=True)
        l = Parallel(n_jobs=num_jobs)(delayed(self._evaluate)(fold_index, cv_metric)\
                                      for fold_index in fold_list)
        df = pd.DataFrame(l)
        df.index.name = 'num_features'
        return df.reset_index()

    def calculate_disc_threshold(self, subfolder, max_features):
        ''' this function from peter's code on dropbox
            pick a threshold such as the number of predicted one's in the training set
            will be no less than the actual number of ones in the training set
        '''

        # (i) read in data
        # read in the number of datapoints in each group
        with open(subfolder+'N_tree.csv', 'r') as f:
            for row, line in enumerate(f):
                if row == max_features:
                    ## number of datapoints in each group
                    num_s = pd.np.array([int(x) for x in line.split()[0].split(',')])
                    break
        # total number of datapoints
        num_tot = num_s.sum()
        #print('num_tot:', num_tot)
        # read in the average value of y per group
        with open(subfolder+'ybar_tree.csv', 'r') as f:
            for row, line in enumerate(f):
                if row == 0:
                    ## global y-bar
                    ybar = float(line.split()[0])
                if row == max_features:
                    ## average y (y-bar's) in each group
                    ys = pd.np.array([float(x) for x in line.split()[0].split(',')])
                    break
        # total number of 1's
        num_ones = num_tot*ybar
        #print('num_ones:', num_ones)

        # (ii)  sort the ybars and N's
        #sort_indices_1 = pd.np.flip(pd.np.argsort(ys), axis=0)
        sort_indices = pd.np.argsort(-ys)
        ybar_sorted = ys[sort_indices]

        # (iii) get the cumsum of the ns
        ## sort the number of datapoints by their y bar values (large -> small)
        ## then get the cumulative number of datapoints
        num_sorted_cum = pd.np.cumsum(num_s[sort_indices])

        # (iv) find the first cum num value which is greater than num_ones
        num_ones_thres = pd.np.argmax(num_sorted_cum >= num_ones)

        # (v) pick the next element as the threshold, now all elememts greater than that threshold will be chosen
        disc_thresh = ybar_sorted[num_ones_thres]

        return disc_thresh
