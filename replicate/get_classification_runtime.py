from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
##
import os, time, sys
import pandas as pd
from pys3d import PYS3D
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

def get_runtime(data_name, clf_name, type_):
    performance_df = pd.read_csv('{}/bm-performance/{}-{}.csv'.format(type_, data_name, clf_name),
                                 index_col=0)
    runtime_df = list()
    for i, series in performance_df.iterrows():
        fold_index, best_params_str = series[['fold_index', 'best_params']]
        best_params = eval(best_params_str)
        ## obtain data
        df = pd.read_csv('../splitted_data/%s/%s/train.csv'%(data_name, fold_index))
        X = df[df.columns[df.columns!='target']].values
        y = df['target'].values.astype(int)

        df_test = pd.read_csv('../splitted_data/%s/%s/test.csv'%(data_name, fold_index))
        X_test = df_test[df_test.columns[df_test.columns!='target']].values
        y_test = df_test['target'].values.astype(int)

        if 'forest' not in clf_name:
            # if not random forest, standardize data
            #print('standardize data for', clf_name)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_test = scaler.transform(X_test)

        for _ in range(10):
            clf = eval("get_{}(**best_params)".format(clf_name))
            start_time = time.time()
            clf.fit(X, y)
            #train_end_time = time.time()
            #clf.predict(X_test)
            end_time = time.time()
            runtime_df.append({'data_name': data_name, 'clf_name': clf_name,
                               'fold_index': fold_index,
                               #'train_time': train_end_time-start_time,
                               'train_time': end_time-start_time,
                              })
                               #'total_time': end_time-start_time})
    return pd.DataFrame(runtime_df)

def get_runtime_s3d(data_name, type_):
    model_base_path = 'tmp-models/'
    pred_base_path = 'tmp-predictions/'
    performance_df = pd.read_csv('{}/performance/{}-test-performance.csv'.format(type_, data_name),
                                 index_col=0)
    runtime_df = list()
    for i, series in performance_df.iterrows():
        fold_index, n_f, lambda_ = series[['split_version', 'num_features', 'lambda_']]
        fold_index = int(fold_index)
        ## obtain data
        train_data = '../splitted_data/{}/{}/train.csv'.format(data_name, fold_index)
        test_data = '../splitted_data/{}/{}/test.csv'.format(data_name, fold_index)
        ## model and prediction paths
        train_model = model_base_path+data_name+'/'
        pred_path = pred_base_path+data_name+'/'
        for _ in range(10):
            s3d = PYS3D(data_name, model_path=model_base_path,
                        prediction_path=pred_base_path)
            start_time = time.time()
            s3d.fit(train_data, train_model, lambda_=lambda_, max_features=100)
            #train_end_time = time.time()
            #s3d.predict(test_data, train_model, pred_path)
            end_time = time.time()
            runtime_df.append({'data_name': data_name, 'clf_name': 's3d', 
                               'fold_index': fold_index,
                               #'train_time': train_end_time-start_time,
                               'train_time': end_time-start_time,
                              })
    return pd.DataFrame(runtime_df)

def get_runtime_wrapper(data_name, clf_name_list, type_):
    ## s3d
    #s3d_rt_df = get_runtime_s3d(data_name, type_)
    ## benchmark
    l = Parallel(n_jobs=len(clf_name_list))(delayed(get_runtime)(data_name, clf_name, type_) for clf_name in clf_name_list)
    bm_rt_df = pd.concat(l, ignore_index=True)
    #return s3d_rt_df.append(bm_rt_df, ignore_index=True)
    return bm_rt_df

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

if __name__ == '__main__':
    clf_name_list = ['linearsvc', 'lasso', 'elasticnet', 'randomforest']
    data_name, type_ = sys.argv[1:3]
    df = get_runtime_wrapper(data_name, clf_name_list, type_)
    df.to_csv('runtime/{}.csv'.format(data_name), index=False)
