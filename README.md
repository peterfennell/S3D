# Structured Sum of Squares Decomposition (S3D) algorithm

C++ code for fast implementation of the Structured Sum of Squares Decomposition (S3D) algorithm

---

### Installation

```
$ make all
```

---

### Python Wrapper

see [`code/quickstart.ipynb`](code/quickstart.ipynb)

---

### Splitting Data
todo

---

### Inputs and Outputs

#### Traininig
##### Inputs
 Required
 - infile:           Data file
 - outfolder:        Directory to save the model
 - lambda:           Lambda binning parameter

Optional
 - n_rows:           Number of rows of infile to use
 - ycol:             Column number of response variable y (0-indexed) (default 0)
 - start_skip_rows:  First row of a continuous block of rows to skip
 - end_skip_rows:    Row after the last row of a contiguous block of rows to skip
 - max_features:     Maximum number of features to choose (default 20)

##### Outputs
 - levels.csv
   - L rows, 2 columns
   - each row l is of the form `feature, R2`
     - feature: the chosen feature at level l of the model
     - R2: the total R-squared of the model at level l
 - splits.csv
   - L rows, variable columns
   - row l has the list of splits for the bins of the chosen variable at level l
   - manually check the matching of bins
 - R2improvements.csv
   - L rows, M columns
   - entry `(l, m)` is the R2 improvement of the model by the addition of feature m at level l
 - ybartree.csv
   - the first row is the global y bar (averagee y values)
   - L rows, variable columns
   - row l has the ybar values for each partition element of level l
 - Ntree.csv
   - L rows, variable columns
   - row l has the number of elements N in each partition element of level l

#### Prediction
##### Inputs
 Required

 - datafile:         Datafile for which to make the predictions
 - infolder:         Directory where S3D is located (this is "outfile" of train.cpp program)
 - outfolder:        Directory where the predictions will be saved

Optional
 - max_features:     Maximum number of features to use for prediction (int >=0, default use all S3D chosen features)
 - min_samples:      Minimum number of samples required to make a prediction (default 1)
 - start_use_rows:   First row of a continuous block of rows to use for prediction (default 0)
 - end_use_rows:     Row after the last row of a contiguous block of rows to use for prediction (default n_predictions)

##### Outputs
 - predicted_expectations.csv
   - vector with predicted expectations for each row of the input datafile

---

### Datasets

- Stack Exchange: https://drive.google.com/open?id=19321sUtWQhyJHNa20ctk6gm7C8uR9kWk

- Twitter: https://drive.google.com/open?id=1mJ8G5YYymF1cVgldhgcBRHYqOAACt0bs

- Digg: https://drive.google.com/open?id=1e1OFqNH7ZlvgP_UjOVkkEJJsOINcCY8A
