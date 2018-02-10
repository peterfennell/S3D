SSD algorithm

Peter Fennell, July 2017


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



