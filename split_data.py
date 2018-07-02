import os, sys, time
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

random_state=10

class DataSplitter(object):
    def __init__(self, data_path, data_name,
                 outfolder=None):

        ''' initializer

            parameters
            ----------
            data_path : str
                input data
            data_name : str
                name of data; for saving
        '''

        if outfolder is None:
            self.outfolder = 'splitted_data/{}'.format(data_name)
        else:
            self.outfolder = outfolder

        self.data_name = data_name
        self.data = pd.read_csv(data_path)
        class_dist_d = self.data['target'].value_counts().to_dict()
        #print('class distribution -', class_dist_d)
        self.nrows, _ = self.data.shape

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
        y = self.data['target'].values
        ## split
        print('splitting {} data ({} rows) into {} folds'.format(self.data_name,
                                                                 self.nrows, self.num_folds))
        ## export different folds
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=random_state)


        print('using {} cores'.format(num_jobs))
        #for i, indices_subarr in enumerate(indices_split):
        num_jobs = min([num_jobs, self.num_folds])
        Parallel(n_jobs=num_jobs)(delayed(self.save_folds)(i, train_index, test_index)\
                                  for i, (train_index, test_index) in enumerate(skf.split(X, y)))


    def save_folds(self, i, train_index, test_index):
        start = time.time()
        print('working on fold {}'.format(i), end=' ')
        ## create the corresponding fold-folder to save test and train/tune datasets
        out = '{}/{}/'.format(self.outfolder, i)
        if not os.path.exists(out):
            os.makedirs(out)
        ## export test dataset
        test_values = self.data.values[test_index]
        pd.DataFrame(test_values, columns=self.data.columns.values).to_csv(out+'test.csv', index=False)
        ## export train/tune dataset: use stratified row indices to rearrange the training set and make it stratified
        train_index = self.adjust_rows(train_index)
        train_values = self.data.values[train_index]
        pd.DataFrame(train_values, columns=self.data.columns.values).to_csv(out+'train.csv', index=False)


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

    data_name, num_folds = sys.argv[1:3]
    data_path = 'data/{}.csv'.format(data_name)
    try:
        num_jobs = int(sys.argv[3])
    except:
        num_jobs = 1

    ds = DataSplitter(data_path, data_name)
    ds.split_data(int(num_folds), num_jobs)
