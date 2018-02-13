## C++ code for fast implementation of the Structured Sum of Squares Decomposition (S3D) algorithm

### Installation

Download folder, then run the commands 
```
$ make train
$ make predict
```
from the terminal/command-line (in the folder directory) to create the excecutable files `train` and `predict`. `train` takes as input a .csv file and trains an S3D model on the data, while `predict` uses the model to make predictions. 

### Inputs and Outputs

Details of both the required and optional inputs to `train` and `predict` are found in the headers of the train.cpp and predict_expectations.cpp files respectively.

### Examples

An example for implementation is given by the scripts example_script_train.sh and example_script_predict.sh



Inputs: 
- .csv datafile (with first row column names) 

Outputs: 
- R2improvements.csv
- L rows, M columns 
- entry (l,m) is the R2 improvement of the model by the addition of feature m at level l

- splits.csv
- L rows, variable columns
- the list of splits for the chosen variable at each level

- levels.csv
- L rows, 2 + variable columns
- of the form
- feature,R2,s1,s2,...,sn
- the chosen feature, the R2 of the model, the splits {si} of the chosen feature

- ybartree.csv
- L rows, variable columns
- each row contains the ybar value for each partition element of the level 


- yvartree.csv
- L rows, variable columns
- each row contains the yvar value for each partition element of the level 


- Ntree.csv
- L rows, variable columns
- each row contains the number of elements N in each partition element of the level 



