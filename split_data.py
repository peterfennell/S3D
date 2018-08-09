import warnings
import pandas as pd
import os, sys, time, argparse
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold

#warnings.simplefilter("once")

random_state=10

class DataSplitter(object):
    ''' split data;
        stratefied for classification; regulalr k-fold for regression
        for regression data, standardize data by default but can be turned off by setting standardize_flag to 0 (both X and y; using scaler fit by training set on test set)
    '''
    def __init__(self, data_path, data_name,
                 classification_flag=True,
                 standardize_flag=True,
                 outfolder=None):

        ''' initializer
            parameters
            ----------
            data_path : str
                input data
            data_name : str
                name of data; for saving
            classification_flag : bool
                whether it's for regression or classification (default True)
            standardize_flag : bool
                whether features will be standardized (default True)
            outfolder : string
                output folder directory
        '''

        if outfolder is None:
            self.outfolder = 'splitted_data/{}'.format(data_name)
        else:
            self.outfolder = outfolder

        self.data_name = data_name
        self.data = pd.read_csv(data_path)
        #class_dist_d = self.data['target'].value_counts().to_dict()
        #print('class distribution -', class_dist_d)
        self.nrows, _ = self.data.shape
        self.classification_flag = classification_flag
        self.standardize_flag = standardize_flag
        if not self.classification_flag and self.standardize_flag:
            print('standardize values')

    def split_data(self, num_folds=5, num_jobs=1):
        ''' generate equal folds of data. this is mainly for s3d
            apply stratification to account for class imbalance
            make this parallelizable
            parameters
            ----------
            num_folds : int
                the number of folds to use for cross validation
            outfolder : str
                where to output the splitted datasets
            the function will export each fold into `outfolder`
            with names formatted as `data_name_i.csv` where `i` is the fold index
        '''

        self.num_folds = num_folds

        X = self.data[self.data.columns[self.data.columns!='target']].values
        if self.classification_flag:
            #print('classification data')
            y = self.data['target'].values.astype(int)
            kf = StratifiedKFold(n_splits=self.num_folds, shuffle=True,
                                 random_state=random_state)
        else:
            #print('regression data')
            y = self.data['target'].values
            kf = KFold(n_splits=self.num_folds, shuffle=True,
                       random_state=random_state)
        ## split
        print('splitting {} data ({} rows) into {} folds'.format(self.data_name,
                                                                 self.nrows, self.num_folds))
        ## export different folds


        print('using {} cores'.format(num_jobs))
        #for i, indices_subarr in enumerate(indices_split):
        num_jobs = min([num_jobs, self.num_folds])
        Parallel(n_jobs=num_jobs)(delayed(self.save_folds)(i, train_index, test_index)\
                                  for i, (train_index, test_index) in enumerate(kf.split(X, y)))


    def save_folds(self, i, train_index, test_index):
        start = time.time()
        print('working on fold {}'.format(i), end=' ')
        ## create the corresponding fold-folder to save test and train/tune datasets
        out = '{}/{}/'.format(self.outfolder, i)
        if not os.path.exists(out):
            os.makedirs(out)

        if self.classification_flag:
            train_index = self.adjust_rows(train_index)

        ## export train/tune dataset: use stratified row indices to rearrange the training set and make it stratified
        train_values = self.data.values[train_index]
        test_values = self.data.values[test_index]
        ## fit a scaler using training data
        ## transofrm is for regression if standardize_flag==True
        if not self.classification_flag and self.standardize_flag:
            scaler = StandardScaler()
            train_values = scaler.fit_transform(train_values)
            test_values = scaler.transform(test_values)
        ## save
        pd.DataFrame(train_values, columns=self.data.columns.values).to_csv(out+'train.csv', index=False)
        pd.DataFrame(test_values, columns=self.data.columns.values).to_csv(out+'test.csv', index=False)

        ## also export the number of rows for train/test into a text file
        with open(out+'num_rows.csv', 'w') as f:
            ## first train, then test
            f.write(str(train_index.size)+'\n')
            f.write(str(test_index.size)+'\n')
        assert train_index.size+test_index.size == self.nrows
        print('fold {0} (elapsed time: {1:.2f} seconds)'.format(i, time.time()-start))


    def adjust_rows(self, train_index):
        ''' adjust the rows in the training sets so that
            every interval of subsets in training will have the same ratio of positive/negative
            where intervals are determined by `num_folds-1`
        '''
        train_values = self.data.values[train_index]
        train_data = pd.DataFrame(train_values, columns=self.data.columns.values)

        X = train_data[train_data.columns[train_data.columns!='target']].values
        y = train_data['target'].values
        #print(pd.np.bincount(y))

        skf = StratifiedKFold(n_splits=self.num_folds-1, shuffle=True, random_state=random_state)

        stratified_row_indices = list()
        for _, tst_idx in skf.split(X, y):
            ## for every test fold, i will append it to make a "stratified training set"
            stratified_row_indices.extend(tst_idx)
        return pd.np.array(stratified_row_indices)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("data_name", type=str, help="data to be splitted")
    parser.add_argument("num_folds", type=int, help="number of folds")

    ## optional
    parser.add_argument("-cf", "--classification-flag", type=int,
                        choices=[0, 1], default=1,
                        help="whether the dataset is for classification or not (default 1 - yes); 0 for regression")
    parser.add_argument("-s", "--standardize-flag", type=int,
                        choices=[0, 1], default=1,
                        help="whether features will be standardized (based on training data); only for regression (when -cf set to 0)")
    parser.add_argument("-j", "--num-jobs", type=int, default=1,
                        help="the number of parallel jobs (default 1)")

    args = parser.parse_args()

    data_path = 'data/{}.csv'.format(args.data_name)

    ds = DataSplitter(data_path, args.data_name,
                      bool(args.classification_flag),
                      bool(args.standardize_flag))

    ds.split_data(args.num_folds, args.num_jobs)
