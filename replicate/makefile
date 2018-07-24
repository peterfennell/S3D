CPP=c++
CPP_OPTS=-std=c++11 -g

all:			train predict_expectations

clean:
			rm train predict_expectations

train:			train.cpp train.h
			$(CPP) $(CPP_OPTS) -o train train.cpp

predict_expectations:	predict_expectations.cpp
			$(CPP) $(CPP_OPTS) -o predict_expectations predict_expectations.cpp
