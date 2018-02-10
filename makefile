train:			train.cpp train.h
			c++ -o train train.cpp

predict_expectations:	predict_expectations.cpp
			g++ -o predict_expectations predict_expectations.cpp