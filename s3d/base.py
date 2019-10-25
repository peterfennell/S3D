import subprocess
import os
import warnings
import numpy as np
import pandas as pd

DIR_NAME = os.path.dirname(os.path.abspath(__file__))

class S3D(object):
  ''' a wrapper function to run s3d in python
    make it similar to sklearn interfaces
  '''
  def __init__(self, classification_flag=True):
    ''' initializer

    Parameters
    ----------
    classification_flag : bool
      whether this is classification or regression. this is used for determining evaluation metrics
    '''
    self.classification_flag = classification_flag
    self.all_features = None
    self.selected_features = None
    self.N_selected_features = None
    self.r2_by_level = None
    self.bins = None
    self.r2_improvements = None
    self.ybar_tree = None
    self.N_tree = None
    self.ybar_mat = None
    self.N_mat = None

  def fit(self, X, y, lambda_=0.01, max_features=None, debug=False):
    '''Fit an S3D model

    Parameters
    ----------
    X : dataframe
      matrix of features
    y : series
      outcome variable vector
    lambda_ : float
      regularization parameter
    max_features : int
      maximum number of features to choose (default 20)
    '''
    assert len(X) == len(y)
    # Save the data to a file and use the fit_from_file function
    train_filename = os.path.join(DIR_NAME, 'temp_training_data.csv')
    pd.concat([pd.Series(y), pd.DataFrame(X)], axis=1).to_csv(train_filename, index=False)
    self.fit_log = self.fit_from_file(train_filename, lambda_=lambda_, max_features=max_features, debug=debug)

    # delete temporary file
    os.remove(train_filename)


  def fit_from_file(self, train_filename,
      lambda_=0.01, max_features=None, ycol=0,
      start_skip_rows=None, end_skip_rows=None, n_rows=None, remove_model_dir=True,
      debug=False):
    ''' Fit an S3D model with the given lambda

    Parameters
    ----------
    train_filename : str
      Path of the filename where the data lives
    lambda_ : float
      regularization parameter
    max_features : int
      maximum number of features to choose (default 20)
    ycol : int
      Index of the column to use as the target
    start_skip_rows : int
      the index of the begining rows to skip (inclusive)
    end_skip_rows : int
      the index of the ending rows to skip (exclusive)
    n_rows : int
      the number of rows to use for training
      the first n_rows rows between start_use_rows and end_use_rows are to be used for prediction
    remove_model_path : bool
      whether or not to remove the temporary model directory after training
    '''
    assert os.path.exists(os.path.join(DIR_NAME, train_filename)), "{} is not a valid path".format(train_filename)

    # Execute the C++ program
    model_path = os.path.join(DIR_NAME, 'temp_folder')
    if not os.path.exists(model_path):
      os.mkdir(model_path)
    self._run_train_cpp(train_filename, lambda_, ycol, max_features, model_path,
      start_skip_rows, end_skip_rows, n_rows, debug)

    # READ IN MODEL COMPONENTS
    # All features
    self.all_features = pd.read_csv(train_filename, nrows=0).columns.tolist()
    self.all_features.pop(ycol)
    # Features selected by S3D
    levels = pd.read_csv(os.path.join(model_path, 'levels.csv'))
    self.selected_features = levels['best_feature'].values
    self.N_selected_features = levels['best_feature'].size
    if self.N_selected_features < max_features:
      warnings.warn(
        "{} features requested but only {} selected by algorithm".format(max_features, N_selected_features),
        UserWarning
      )
    # R2 (variance explained) by level
    self.r2_by_level = levels['R2'].values
    # Bins
    self.bins = {}
    with open(os.path.join(model_path, 'splits.csv'), 'r') as f:
      for line in f.readlines():
        if line:
          self.bins[line.split(',')[0]] = map(float, line.split(',')[1:])
    # Splits
    self.splits = {}
    for i, bins in self.bins.iteritems():
      self.splits[i] = bins[1:-1]
    # Number of bins per level
    self.count_bins = [len(self.bins[feat]) - 1 for feat in self.selected_features]
    if debug:
      print('bins', self.bins)
      print('splits', self.splits)
      print('count_bins', self.count_bins)
    # R2 improvements by each feature at each level
    self.r2_improvements = pd.read_csv(os.path.join(model_path, 'R2improvements.csv'))
    # Predicted value for each bin for each level
    self.ybar_tree = {}
    with open(os.path.join(model_path, 'ybar_tree.csv'), 'r') as f:
      for i, line in enumerate(f.readlines()):
        if line:
          if i == 0:
            self.ybar_tree[i] = float(line)
          else:
            self.ybar_tree[i] = map(float, line.split(','))
    self.ybar_mat = {}
    with open(os.path.join(model_path, 'ybar_tree.csv'), 'r') as f:
      for i, line in enumerate(f.readlines()):
        if line:
          if i == 0:
            self.ybar_mat[i] = float(line)
          else:
            self.ybar_mat[i] = np.reshape(map(float, line.split(',')), self.count_bins[:i])
    # N data points for each bin for each level
    # self.N_tree = {}
    # with open(os.path.join(model_path, 'N_tree.csv'), 'r') as f:
    #   for i, line in enumerate(f.readlines()):
    #     if line:
    #       if i == 0:
    #         self.N_tree[i] = int(line)
    #       else:
    #         self.N_tree[i] = map(int, line.split(','))
    self.N_mat = {}
    with open(os.path.join(model_path, 'N_tree.csv'), 'r') as f:
      for i, line in enumerate(f.readlines()):
        if line:
          if i == 0:
            self.N_mat[i] = int(line)
          else:
            self.N_mat[i] = np.reshape(map(int, line.split(',')), self.count_bins[:i])

    if debug:
      print('all_features:', self.all_features)
      print('selected_features:', self.selected_features)
      print('N_selected_features:', self.N_selected_features)
      print('r2_by_level:', self.r2_by_level)
      print('r2_improvements:', self.r2_improvements)
      print('ybar_tree:', self.ybar_tree)
      print('N_tree:', self.N_tree)
      print('ybar_mat:', self.ybar_mat)
      print('N_mat:', self.N_mat)


    # if remove_model_dir:
    #   self._remove_model_directory(model_path)


  def get_expectation(self, X, max_features=None, min_samples=1, debug=False):
    '''Get the expected value of Y (from the training data) for inputs X.

    Parameters
    ----------
    X : data frame or dictionary
      if data frame, columns must be equal to self.columns
    max_features : int
      maximum number of features used for prediction
      similar to max_depth parameter in decision trees
      default: use the number of s3d chosen features
    min_samples : int
      minimum number of samples required to make a prediction
      default: 1

    Returns
    -------
    y_hat : list of floats
      Prediction for each row of X
    '''
    assert min_samples <= self.N_mat[0], "min_samples greater than size of training data"

    if max_features is None:
      max_features = self.N_selected_features
    elif max_features > self.N_selected_features:
      warnings.warn(
        "{0} features requested but only {1} selected in training; using {1}".format(max_features, self.N_selected_features),
        UserWarning
      )
      max_features = self.N_selected_features

    bin_indices = np.transpose([
      np.searchsorted(self.splits[feat], X[feat])
      for feat in self.selected_features[:max_features]
    ])
    if debug:
      print('bin_indices:', bin_indices)

    predictions = []
    for bi in bin_indices:
      # Iterate though smaller number of features until min_samples is hit
      for n_features in range(max_features, 0, -1):
        if self.N_mat[n_features].item(tuple(bi[:n_features])) >= min_samples:
          predictions.append(self.ybar_mat[n_features].item(tuple(bi[:n_features])))
          break
        if n_features == 0:
          # No min samples hit so just return the prediction with no features
          predictions.append(self.ybar_mat[0])

    if debug:
      print('predictions:', predictions)

    return predictions

  def _remove_model_directory(self, model_directory_path):

    # os.removedir(model_directory_path)
    for f in os.listdir(model_directory_path):
      os.remove(model_directory_path + '/' + f)
    os.rmdir(model_directory_path)

  def _run_train_cpp(self, train_filename, lambda_, ycol, max_features, model_path,
      start_skip_rows, end_skip_rows, n_rows, debug=False):

    c = '{4} -infile:{0} -outfolder:{1} -lambda:{2} -ycol:{3}'.format(
      os.path.join(DIR_NAME, train_filename), model_path, lambda_, ycol, os.path.join(DIR_NAME, 'train')
    )
    if start_skip_rows is not None:
      c += ' -start_skip_rows:{}'.format(start_skip_rows)
    if end_skip_rows is not None:
      c += ' -end_skip_rows:{}'.format(end_skip_rows)
    if n_rows is not None:
      c += ' -n_rows:{}'.format(n_rows)
    if max_features is not None:
      c += ' -max_features:{}'.format(max_features)

    if debug:
      print('fitting s3d with', train_filename)
      print('command:', c)

    ## catch the output and save to a log file in the `outfolder`
    process = subprocess.Popen(c.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    fit_log = '\n\n\n'.join([c, output.decode('utf8')])
    if error is not None:
      '\n\n\n'.join([fit_log, '\n'.join(['ERRORS:', error])])

    return fit_log

    # with open(os.path.join(DIR_NAME, os.path.join(model_path, 'fit.log')), 'w') as logfile:
    #   logfile.write(c)
    #   logfile.write(output.decode('utf8'))
    #   logfile.write('---errors below (if any)---\n')
    #   if error is not None:
    #     logfile.write(error)


def test():

  print ""
  print "TEST 1"
  print ""

  N = 10
  y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  x1 = range(N)
  x2 = np.zeros(N)
  X = pd.DataFrame(zip(x1, x2), columns=['x1', 'x2'])

  model = S3D()
  model.fit(X, y, debug=True)

  preds = model.get_expectation(X, debug=True)
  assert np.mean(y) == np.mean(preds)

  print ""
  print "TEST 2"
  print ""

  N = 6
  y = [0, 0, 0, 1, 1, 1]
  x1 = [0, 0, 1, 0, 1, 1]
  x2 = [1, 1, 0, 0, 1, 1]
  x3 = np.zeros(N)
  X = pd.DataFrame(zip(x1, x2, x3), columns=['x1', 'x2', 'x3'])

  model = S3D()
  model.fit(X, y, debug=True)

  preds = model.get_expectation(X, debug=True)
  assert np.mean(y) == np.mean(preds)


if __name__ == '__main__':
  test()
