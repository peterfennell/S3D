import pandas as pd
import sys, json, time
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_absolute_error, r2_score

def get_linearsvr(random_state=100, **kwargs):
    return LinearSVR(random_state=random_state)

def get_lasso(max_iter=1000, random_state=100, **kwargs):
    '''
        defautlt loss: squared loss
    '''
    return SGDRegressor(penalty='l1', max_iter=max_iter,
                        random_state=random_state, **kwargs)

def get_elasticnet(max_iter=1000, random_state=100, **kwargs):
    return SGDRegressor(penalty='elasticnet', max_iter=max_iter,
                        random_state=random_state, **kwargs)

def get_randomforest(random_state=100, **kwargs):
    return RandomForestRegressor(random_state=random_state,
                                 n_estimators=50, **kwargs)

def get_top_features(reg, reg_name, feature_names):
    if 'forest' in reg_name:
        feature_rankings = pd.Series(dict(zip(feature_names,
                                              reg.feature_importances_)))
    else:
        feature_rankings = pd.Series(dict(zip(feature_names, reg.coef_)))
    return feature_rankings

def obtain_metric(y_true, y_pred):
    ''' for regression metrics, set errors to be negative (to pick the largest ones)
        a copy from utils.py in pys3d
    '''
    r2 = r2_score(y_true, y_pred)
    mae_median = -median_absolute_error(y_true, y_pred)
    mae = -mean_absolute_error(y_true, y_pred)
    mse = -mean_squared_error(y_true, y_pred)

    d = {'r2': r2, 'mae_median': mae_median,
         'mae': mae, 'mse': mse}

    return pd.Series(d)

def cv_wrapper(data_name, reg_name,
               params, n_jobs=30,
               scoring='r2', trim_features=False,
               num_f=None):

    ''' wrapper function to do cross val and test given a regressor name '''

    result_l = list()
    feature_rankings_l = list()
    start = time.time()
    for fold_index in range(5):
        print('working on fold', fold_index, end='; ')
        df = pd.read_csv('../splitted_data/%s/%s/train.csv'%(data_name, fold_index))
        X = df[df.columns[df.columns!='target']].values
        #y = df['target'].values.astype(int)
        y = df['target'].values
        feature_names = df.columns[1:].values

        df_test = pd.read_csv('../splitted_data/%s/%s/test.csv'%(data_name, fold_index))
        X_test = df_test[df_test.columns[df_test.columns!='target']].values
        #y_test = df_test['target'].values.astype(int)
        y_test = df_test['target'].values

        if 'forest' not in reg_name:
            # if not random forest, standardize data
            print('standardize data for', reg_name)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_test = scaler.transform(X_test)

        ## 5 fold stratefied cross validation
        #skf = StratifiedKFold(n_splits=5, random_state=100)
        ## if `max_features` is too much, update this:
        if 'max_features' in params and\
           len(params['max_features']) > 1 and\
           max(params['max_features']) > X.shape[1]:
            params['max_features'] = pd.np.arange(1, X.shape[1]+1)

        grid = GridSearchCV(eval('get_%s()'%reg_name), params,
                            n_jobs=n_jobs,
                            scoring=scoring)
                            #cv=skf)
        grid.fit(X, y)

        ## retrain on the full training data
        my_reg = eval('get_%s(**grid.best_params_)'%reg_name)
        my_reg.fit(X, y)
        ## obtain top feature importance after retraining with the full training set
        feature_rankings = get_top_features(grid.best_estimator_, reg_name,
                                            feature_names)
        feature_rankings.loc['fold_index'] = fold_index
        feature_rankings_l.append(feature_rankings.copy())
        feature_rankings.drop('fold_index', inplace=True)
        ## retrain random forest with the top features
        if trim_features and 'forest' in reg_name and num_f > 0:
            ## only trim features for the top k features
            features_to_keep = feature_rankings.index.values[:num_f]
            X = df[features_to_keep].values
            X_test = df_test[features_to_keep].values
            ''' important: overwrite max_features in best parameters to None '''
            param_dict = {k: v for k, v in grid.best_params_.items()}
            param_dict['max_features'] = None
            my_reg = eval('get_%s(**param_dict)'%reg_name)
            #my_reg = eval('get_%s(**grid.best_params_)'%reg_name)
            my_reg.fit(X, y)

        y_pred = my_reg.predict(X_test)

        metric_series = obtain_metric(y_test, y_pred).to_dict()
        metric_series['best_params'] = grid.best_params_
        metric_series['fold_index'] = fold_index

        result_l.append(metric_series)
        print('elapsed time:%.4f'%(time.time()-start))

    if trim_features:
        reg_name = reg_name + '_trimmed' + '_{}'.format(num_f)

    odir = 'regression/bm-performance/'
    pd.DataFrame(feature_rankings_l).to_csv(odir+'{}-{}-feature_rankings.csv'.format(data_name, reg_name))
    pd.DataFrame(result_l).to_csv(odir+'{}-{}.csv'.format(data_name, reg_name))

if __name__ == '__main__':
    data_name, reg_name = sys.argv[1:3]
    n_jobs = int(sys.argv[3])

    trim, num_f = sys.argv[4:6]
    trim = eval(trim)
    #print(trim)
    num_f = int(num_f)

    params = json.load(open('regression_param_grid/'+reg_name+'.grid'))

    if 'forest' in reg_name and not trim:
        #dim_df = pd.read_csv('../data/classification-data-dim.csv', index_col=0, squeeze=True)
        dim_df = pd.read_csv('../data/regression-data-dim.csv', index_col=0, squeeze=True)
        num_features = dim_df.loc[data_name]
        #max_features = int(min(20, num_features//3))
        max_features = num_features//3
        params['max_features'] = pd.np.arange(1, max_features+1)

        cv_wrapper(data_name, reg_name,
                   params, n_jobs=n_jobs,
                   trim_features=trim, num_f=num_f
                  )

    elif 'forest' in reg_name and trim:
        dim_df = pd.read_csv('../data/regression-data-dim.csv', index_col=0, squeeze=True)
        num_features = dim_df.loc[data_name]
        num_f = min(num_f, num_features)
        params['max_features'] = [num_f]

        cv_wrapper(data_name, reg_name,
                   params, n_jobs=n_jobs,
                   trim_features=trim, num_f=num_f,
                  )
    else:
        cv_wrapper(data_name, reg_name,
                   params, n_jobs=n_jobs
                  )
