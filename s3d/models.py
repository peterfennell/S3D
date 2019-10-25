import numpy as np
import pandas as pd
from s3d.base import S3D

class S3DClassifier(S3D):
  '''a wrapper function to run s3d in python
    make it similar to sklearn interfaces
  '''
  def __init__(self):
    '''initializer
    '''
    S3D.__init__(self, classification_flag=True)

  def predict_proba(self, X, max_features=None, min_samples=1, debug=False):
    expectations = self.get_expectation(X, max_features, min_samples, debug)
    class_probas = [[1 - p, p] for p in expectations]
    return class_probas

  def predict(self, X, max_features=None, min_samples=1, debug=False):
    expectations = self.get_expectation(X, max_features, min_samples, debug)
    decisions = [1 if p > 0.5 else 0 for p in expectations]
    return decisions


class S3DRegressor(S3D):
  '''a wrapper function to run s3d in python
    make it similar to sklearn interfaces
  '''
  def __init__(self):
    '''initializer
    '''
    S3D.__init__(self, classification_flag=False)

  def predict(self, X, max_features=None, min_samples=1, debug=False):
    expectations = self.get_expectation(X, max_features, min_samples, debug)
    return expectations


def test():

  print ""
  print "TEST 1"
  print ""

  N = 10
  y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  x1 = range(N)
  x2 = np.zeros(N)
  X = pd.DataFrame(zip(x1, x2), columns=['x1', 'x2'])

  model = S3DClassifier()
  model.fit(X, y, debug=True)

  preds = model.predict_proba(X, debug=True)
  prod_positive_class = [p[1] for p in preds]
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

  model = S3DClassifier()
  model.fit(X, y, debug=True)

  preds = model.predict_proba(X, debug=True)
  prod_positive_class = [p[1] for p in preds]
  assert np.mean(y) == np.mean(preds)


if __name__ == '__main__':
  test()
