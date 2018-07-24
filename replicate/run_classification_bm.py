import pandas as pd
import sys, json, time
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score

def get_linearsvc(random_state=100, **kwargs):
    return LinearSVC(random_state=random_state)

def get_lasso(max_iter=1000, random_state=100, **kwargs):
    return SGDClassifier(loss='log', penalty='l1',
                         max_iter=max_iter, class_weight='balanced',
                         random_state=random_state, **kwargs)

def get_elasticnet(max_iter=1000, random_state=100, **kwargs):
    return SGDClassifier(loss='log', penalty='elasticnet',
                         max_iter=max_iter, class_weight='balanced',
                         random_state=random_state, **kwargs)

def get_randomforest(random_state=100, **kwargs):
    return RandomForestClassifier(random_state=random_state,
                                  class_weight='balanced',
                                  n_estimators=50, **kwargs)

def get_top_features(clf, clf_name, feature_names):
    if 'forest' in clf_name:
        feature_rankings = pd.Series(dict(zip(feature_names,
                                              clf.feature_importances_)))
    else:
        feature_rankings = pd.Series(dict(zip(feature_names, clf.coef_[0])))
    return feature_rankings

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

def cv_wrapper(data_name, clf_name,
               params, n_jobs=30,
               scoring='roc_auc',
               trim_features=False, num_f=None,
               calc_threshold=True):
    ''' wrapper function to do cross val and test given a classifier name '''

    result_l = list()
    feature_rankings_l = list()
    start = time.time()
    for fold_index in range(5):
        print('working on fold', fold_index, end='; ')
        df = pd.read_csv('../splitted_data/%s/%s/train.csv'%(data_name, fold_index))
        X = df[df.columns[df.columns!='target']].values
        y = df['target'].values.astype(int)
        clf_dist = pd.Series(y).value_counts()
        print('class distribution: ', clf_dist.to_dict(), end='; ')
        feature_names = df.columns[1:].values

        df_test = pd.read_csv('../splitted_data/%s/%s/test.csv'%(data_name, fold_index))
        X_test = df_test[df_test.columns[df_test.columns!='target']].values
        y_test = df_test['target'].values.astype(int)

        if 'forest' not in clf_name:
            # if not random forest, standardize data
            print('standardize data for', clf_name)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_test = scaler.transform(X_test)

        ## 5 fold stratefied cross validation
        skf = StratifiedKFold(n_splits=5, random_state=100)
        ## if `max_features` is too much, update this:
        if 'max_features' in params and\
           len(params['max_features']) > 1 and\
           max(params['max_features']) > X.shape[1]:
            params['max_features'] = pd.np.arange(1, X.shape[1]+1)

        grid = GridSearchCV(eval('get_%s()'%clf_name), params,
                            n_jobs=n_jobs,
                            scoring=scoring, cv=skf)
        grid.fit(X, y)

        ## retrain on the full training data
        my_clf = eval('get_%s(**grid.best_params_)'%clf_name)
        my_clf.fit(X, y)
        ## obtain top feature importance after retraining with the full training set
        feature_rankings = get_top_features(grid.best_estimator_, clf_name,
                                            feature_names)
        feature_rankings.loc['fold_index'] = fold_index
        feature_rankings_l.append(feature_rankings.copy())
        feature_rankings.drop('fold_index', inplace=True)
        ## retrain random forest with the top features
        if trim_features and 'forest' in clf_name and num_f is not None:
            ## only trim features for the top k features
            print('my num_f', num_f)
            features_to_keep = feature_rankings.index.values[:num_f]
            X = df[features_to_keep].values
            X_test = df_test[features_to_keep].values
            ''' important: overwrite max_features in best parameters to None '''
            param_dict = {k: v for k, v in grid.best_params_.items()}
            param_dict['max_features'] = None
            my_clf = eval('get_%s(**param_dict)'%clf_name)
            #my_clf = eval('get_%s(**grid.best_params_)'%clf_name)
            my_clf.fit(X, y)

        #y_pred = my_clf.predict(X_test)
        ## instead of using `predict`, use a customized threshold for prediction
        ## note that for svm, we need to use decision function
        if 'svc' in clf_name:
            y_pos_score = my_clf.decision_function(X_test)
        else:
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

        result_l.append(metric_series)
        print('elapsed time:%.4f'%(time.time()-start))

    if trim_features:
        clf_name = clf_name + '_trimmed' + '_{}'.format(num_f)
    odir = "classification/bm-performance/"
    pd.DataFrame(feature_rankings_l).to_csv(odir+'{}-{}-feature_rankings.csv'.format(data_name, clf_name))
    pd.DataFrame(result_l).to_csv(odir+'{}-{}.csv'.format(data_name, clf_name))

if __name__ == '__main__':
    data_name, clf_name = sys.argv[1:3]
    n_jobs = int(sys.argv[3])

    trim, num_f = sys.argv[4:6]
    trim = eval(trim)
    #print(trim)
    num_f = int(num_f)

    #params = json.load(open('classification_parameter_grid/'+clf_name+'.grid'))
    params = json.load(open('classification_param_grid/'+clf_name+'.grid'))

    if 'forest' in clf_name and not trim:
        dim_df = pd.read_csv('../data/classification-data-dim.csv', index_col=0, squeeze=True)
        num_features = dim_df.loc[data_name]
        #max_features = int(min(20, pd.np.sqrt(num_features)))
        max_features = int(pd.np.sqrt(num_features))
        params['max_features'] = pd.np.arange(1, max_features+1)

        cv_wrapper(data_name, clf_name,
                   params, n_jobs=n_jobs,
                  )

    elif 'forest' in clf_name and trim:
        dim_df = pd.read_csv('../data/classification-data-dim.csv', index_col=0, squeeze=True)
        num_features = dim_df.loc[data_name]
        num_f = min(num_f, num_features)
        #params['max_features'] = [num_f]
        cv_wrapper(data_name, clf_name,
                   params, n_jobs=n_jobs,
                   trim_features=trim, num_f=num_f,
                  )

    else:
        cv_wrapper(data_name, clf_name,
                   params, n_jobs=n_jobs
                  )
