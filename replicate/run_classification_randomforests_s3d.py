## run random forests by using s3d as a way of feature selection

import sys, os, json, time
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score

def obtain_metric(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)

    f1_binary = f1_score(y_true, y_pred, average='binary')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    r2 = r2_score(y_true, y_pred)

    ## use probability scores not predicted values for auc
    auc_macro = roc_auc_score(y_true, y_score)
    auc_micro = roc_auc_score(y_true, y_score, 'micro')

    d = {'accuracy': acc, 'auc_macro': auc_macro,
         'auc_micro': auc_micro, 'f1_binary':f1_binary,
         'f1_macro': f1_macro, 'f1_micro': f1_micro, 'r2': r2}
    return pd.Series(d)

def get_top_features(clf, feature_names):
    feature_rankings = pd.Series(dict(zip(feature_names,
                                          clf.feature_importances_)))
    return feature_rankings

def get_randomforest(random_state=100, **kwargs):
    return RandomForestClassifier(random_state=random_state,
                                  max_features=None,
                                  class_weight='balanced',
                                  n_estimators=50, **kwargs)

def get_feature_indices(data_name, fold_index):
    levels_file = 'models/{}/{}/levels.csv'.format(data_name, fold_index)
    selected_features = pd.read_csv(levels_file, usecols=[0], squeeze=True)
    ## locate these features
    fn = '../splitted_data/{}/0/test.csv'.format(data_name)
    feature_columns = pd.read_csv(fn, header=None, nrows=1).loc[0].values[1:].tolist()
    return [feature_columns.index(feat) for feat in selected_features]

def run_rf_s3d(data_name, params, n_jobs=30,
               scoring='roc_auc',
               calc_threshold=True):
    result_l = list()
    feature_rankings_l = list()
    start = time.time()
    for fold_index in range(5):
        print('working on fold', fold_index, end='; ')
        ## first get feature index
        feature_indices = get_feature_indices(data_name, fold_index)
        df = pd.read_csv('../splitted_data/%s/%s/train.csv'%(data_name, fold_index))
        X = df[df.columns[1:]].values
        ## subset input matrix
        X = X[:, feature_indices]
        y = df['target'].values.astype(int)
        clf_dist = pd.Series(y).value_counts()
        print('class distribution: ', clf_dist.to_dict(), end='; ')
        feature_names = df.columns[1:].values

        df_test = pd.read_csv('../splitted_data/%s/%s/test.csv'%(data_name, fold_index))
        X_test = df_test[df_test.columns[1:]].values
        ## subset test set as well
        X_test = X_test[:, feature_indices]
        y_test = df_test['target'].values.astype(int)

        ## 5 fold stratefied cross validation
        skf = StratifiedKFold(n_splits=4, random_state=100)
        grid = GridSearchCV(get_randomforest(), params,
                            n_jobs=n_jobs,
                            scoring=scoring, cv=skf)
        grid.fit(X, y)

        ## retrain on the full training data
        my_clf = get_randomforest(**grid.best_params_)
        my_clf.fit(X, y)
        ## obtain top feature importance after retraining with the full training set
        feature_rankings = get_top_features(grid.best_estimator_, feature_names)
        feature_rankings.loc['fold_index'] = fold_index
        feature_rankings_l.append(feature_rankings.copy())
        feature_rankings.drop('fold_index', inplace=True)

        y_score = my_clf.predict_proba(X_test)
        y_pos_score = y_score[:, my_clf.classes_.argmax()]

        ## calculate threshold
        if calc_threshold:
            pos_label_prop = (y==1).sum() / y.size
            thres = pd.np.percentile(sorted(y_pos_score), 100-pos_label_prop*100)
            y_pred = (y_pos_score>=thres).astype(int)
        else:
            thres = None
            y_pred = my_clf.predict(X_test)

        metric_series = obtain_metric(y_test, y_pred, y_pos_score).to_dict()
        metric_series['best_params'] = grid.best_params_
        metric_series['fold_index'] = fold_index
        metric_series['threshold'] = thres
        #print(metric_series)
        #break

        result_l.append(metric_series)
        print('elapsed time:%.4f'%(time.time()-start))
    odir = "classification/bm-performance/"
    this_name = 'randomforest_with_s3d'
    pd.DataFrame(feature_rankings_l).to_csv(odir+'{}-{}-feature_rankings.csv'.format(data_name, this_name))
    pd.DataFrame(result_l).to_csv(odir+'{}-{}.csv'.format(data_name, this_name))

if __name__ == '__main__':
    data_name = sys.argv[1]
    params = json.load(open('classification_param_grid/randomforest.grid'))
    run_rf_s3d(data_name, params)
