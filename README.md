## structured sum of squares decomposition (s3d) algorithm

c++ code for fast implementation of the structured sum of squares decomposition (s3d) algorithm

note: this is a python wrapper. see original repo: https://github.com/peterfennell/S3D for pure c++ implementation

---

### clarification

the design of the constructor is meant to be easy for hyperparameter tuning by cross validation. therefore, the methods `fit` and `predict` may be a little weird to use as standalone functions.

see a [quickstart notebook](s3d/quickstart.ipynb) for more

#### python dependencies

- `pandas`
- `joblib`
- `networkx` (the latest version has the edge arrow fixed!)
- `matplotlib`
- `scikit-learn`
- `seaborn`
- `palettable`

these can be installed by:
```bash
pip install -r requirements.txt
```

#### compiling `s3d`

compiling is simple and straight forward, do:
```bash
make all
```
or 
```bash
make clean && make all
```
to remove previous compiled files.

#### data for cross val

for cross validation processes, datasets should be splitted first. the script `split_data.py` can be used to do so. the script can be run as the following:
```python
python split_data.py data_name num_folds
```

this will read in `data_name.csv` from `data/` folder; split data and store the train/test sets into `splitted_data/data_name/` folder. for example, if there is a dataset called `breastcancer.csv` in `data/`, we can do:
```bash
python split_data.py breastcancer 5
```

if we do `ls splitted_data/breastcancer/`, we will see:
```bash
0  1  2  3  4
```
which are fold indices named folder, each of which there will be:
```bash
num_rows.csv  test.csv  train.csv
```

`train.csv` is the training set; `test.csv` is the testing set; `num_rows.csv` store the number of rows in train/test respectively.

finally, you can parallelize the data partitioning with more cores:
```bash
ptyhon split_data.py data_name num_folds num_jobs
```

where `num_jobs` is 1 by default.

#### data format

data format: in `data_name.csv`, the first column is the target column named `target`, followed by features:
> `target,feature_1,feature_2,...,feature_p`

#### file structure

file directory setup: `PYS3D` class will by default create subfodlers `data_name` in:
```bash
tmp/ predictions/ models/ cv/
```

where:
- `tmp` will store the "inner cross validation" temporary files
- `predictions` will store the prediction results
- `models` will store the trained models
- `cv` will store the cross validation performance (on the validation set)

therefore, the first step is to create all these 4 for your convenience. you can do:
```bash
./init
```
which will cleanup and create these folders

reversely, run `./cleanup` to remove these folders
