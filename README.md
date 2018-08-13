this branch is for replicating the results in the manuscript

- all datasets can be found in downloaded from this link (___tbd___)

- experiments can be replicated by running the following python scripts:
    - `run_{classification,regression}_s3d.py` for s3d
    - `run_{classification,regression}_bm.py` for benchmark
    - `run_{classification,regression}_randomforests_s3d.py` for random forests on top of features selected by s3d

- hyperparameter candidates for benchmark are stored in `{classification,regression}_param_grid/` as json files for lasso, elatic net, linear SVM, and random forests.

- runtime analyis
    - s3d: `./get_s3d_runtime.sh -h`
    - benchmark algorithms: `get_{classification,regression}_runtime.py`
        - example: `python get_classification_runtime.py digg`
    - the corresponding run time for each dataset will be stored in `s3d-runtime/data_name.csv` (s3d) and `runtime/data_name.csv` (benchmark).

- the accompanying notebooks are used for visualizing and organizing results reported in the paper
