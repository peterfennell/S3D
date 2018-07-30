// S3D ALGORITHM
// 
// Inputs:
//  (Required)
//  - infile:           Datafile
//  - outfolder:        Directory to save the model
//  - lambda:           Lambda binning parameter
//  (Optional)
//  - n_rows:           Number of rows of infile to use
//  - ycol:             Column number of response variable y (0-indexed) (default 0)
//  - start_skip_rows:  First row of a continuous block of rows to skip
//  - end_skip_rows:    Row after the last row of a contiguous block of rows to skip
//  - max_features:     Maximum number of features to choose (default 20)
//
// Outputs:
//  - levels.csv
//    - L rows, 2 columns
//    - each row l is of the form "feature,R2"
//      - feature: the chosen feature at level l of the model
//      - R2: the total R-squared of the model at level l
//  - splits.csv
//    - L rows, variable columns
//    - row l has the list of splits for the bins of the chosen variable at level l
//  - R2improvements.csv
//    - L rows, M columns
//    - entry (l,m) is the R2 improvement of the model by the addition of feature m at level l
//  - ybartree.csv
//    - L rows, variable columns
//    - row l has the ybar values for each partition element of level l
//  - Ntree.csv
//    - L rows, variable columns
//    - row l has the number of elements N in each partition element of level l


#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <string>
#include <algorithm>
#include <cstring>
#include <limits>

using namespace std;

const int DL_thresh = 5; // when the number of suceesive times that the loss function is increasing > DL_thresh, break
const int G_THRESH = 10000000; // Maximum number of bins: when G > G_THRESH, break

#include "train.h"

int main(int argc, char* argv[])
{

    //cout << "\n* STRUCTURED SUM OF SQUARES DECOMPOSITION ALGORITHM *\n" << endl;
    clock_t start = clock();
    
    // What is the first thing that we do?
    
    // ------------------------------
    // 1: Declare our data in a class
    //    - Data is often going to be accessed, so declare it in a class
    
    ifstream datafile;
    string outfolder;
    int N;
    int start_skip_rows=-1, end_skip_rows=-1;
    int n_rows=-1;
    double lambda = -1;
    int max_features = 20;
    string lambda_val;
    int y_col = 0;
    for(int i=1; i<argc; i++)
    {
        // Required
        if(strncmp(argv[i],"-infile:",8)==0){
            datafile.open(string(argv[i]).erase(0,8).c_str());
            //cout << " Reading data from " << string(argv[i]).erase(0,8) << endl;
            N = calculate_N(datafile);
            continue;
        }
        if(strncmp(argv[i],"-outfolder:",11)==0){
            outfolder = string(argv[i]).erase(0,11);
            //cout << " Writing data to folder " << outfolder << endl;
            continue;
        }
        if(strncmp(argv[i],"-lambda:",8)==0)
        {
            lambda = atof(string(argv[i]).erase(0,8).c_str());
            lambda_val = to_string(lambda);
            continue;
        }
        // Optionals
        if(strncmp(argv[i],"-ycol:",6)==0){
            y_col = atol(string(argv[i]).erase(0,6).c_str());
            continue;
        }
        if(strncmp(argv[i],"-n_rows:",8)==0){
            n_rows = atol(string(argv[i]).erase(0,8).c_str());
            continue;
        }
        if(strncmp(argv[i],"-start_skip_rows:",17)==0){
            start_skip_rows = atoi(string(argv[i]).erase(0,17).c_str());
            continue;
        }
        if(strncmp(argv[i],"-end_skip_rows:",15)==0){
            end_skip_rows = atoi(string(argv[i]).erase(0,15).c_str());
            continue;
        }
        if(strncmp(argv[i],"-max_features:",14)==0){
            max_features = atoi(string(argv[i]).erase(0,14).c_str());
            continue;
        }
        // Invalid argument, argument doesnt match anything
        //cout << "ERROR: Unknown argument " << argv[i] << endl;
        return 0;

    }
    //cout << " - lambda = " << lambda << endl;
    
    // check if infile is open
    if(!datafile.is_open())
    {
        //cout << "ERROR: datafile not found" << endl;
        return 1;
    }
    // check if lambda is given
    if(lambda < 0)
    {
        //cout << "ERROR: lambda not specified" << endl;
        return 1;
    }
    // check to see that end_skip_rows is not behind start skip rows
    if(end_skip_rows < start_skip_rows)
    {
        //cout << "ERROR: end_skip_rows must be declared greater than start_skip_rows" << endl;
        return 1;
    }
    // calculate actual number of rows that we will use
    if(n_rows == -1) // n_rows has been not been   declared
        N = N - (end_skip_rows - start_skip_rows);
    else
        N = min(n_rows,N) - (end_skip_rows - start_skip_rows); // min here, n_rows should not be greater than N
    // output which rows are being skipped
    if(start_skip_rows > -1)
        ;
        //cout << " - skipping rows " << start_skip_rows << " to " << end_skip_rows-1 << " inclusive." << endl;
    
    // -------------------------------
    // 2: Read in features and y data
    //    - Get ybar and N
    
    // Features:
    vector<string> features;
    get_column_names(datafile, features);
    int N_columns = features.size();
    // output the features if there are not too many
    if(N_columns > 100)
        //cout << '\n' << N_columns << " FEATURES" << endl;
        ;
    else
    {
        //cout << "\nFEATURES:" << endl;
        for(auto it =features.begin(); it!=features.end(); ++it)
            if(*it != features[y_col])
                //cout << *it << endl;
                ;
    }
    //cout << endl;
    
    // Y data:
    //cout << "TARGET VARIABLE:\n" << features[y_col] << endl << endl;
    //cout << "Reading in Y data..." << endl;
    double y, ybar=0, SST=0, R2=0;

    for(int r=0; r<N; r++)
    {
        // If we are skipping rows
        if(r == start_skip_rows)
        {
            // skip the next few rows
            for(int j=r; j<end_skip_rows; j++)
                datafile.ignore(numeric_limits<streamsize>::max(),'\n');
        }
        // read the y value
        read_single_column(y_col, datafile, y);
        ybar += y;
        SST += y*y;
    }
    ybar /= N;
    SST -= N*ybar*ybar; // SST = \sum y_i^2 - N*ybar^2
    
    //cout << 1.0*(clock() - start)/CLOCKS_PER_SEC << " seconds to read in single column." <<  endl;
    
    //cout << " - " << N << " rows of data\n - SampleMean(Y) = " << ybar << "\n - SampleSTD(Y) = " << sqrt(1.0*SST/(N-1)) << "\n - SST(Y) = " << SST<< endl;
    
    // ----------------------------
    // 3: Declare variables
    
    // Groupings
    int G=1; // number of groups
    vector<unsigned int> groups(N,0); // group number of each row of data
    // alternative approach of the above is to save it to the file
    
    
    // Cumulative
    vector<map<double, unsigned int> > cumcount(G,map<double, unsigned int>());
    vector<map<double, double> > cumsum(G,map<double, double>());;
    
    // Best feature variables (chosen at each level after looping through each variable
    double R2_improvement_max;
    int best_feature;
    vector<double> R2improvements_vec;
    vector<double> best_splits;
    vector<unsigned int> best_N_vec(1,N);
    vector<double> best_ybar_vec(1,ybar);
    
    // Feature splitting variables
    double R2_improvement, best_R2_improvement, R2_improvement_s;
    vector<double> R2_improvements, DL;
    set<double> range;
    vector<double> splits;
    map<double, double> deltaR2_p, split_p;
    double pl, pu, pl1, pu1, pl2, pu2, split;
    
    
    // Variables
    vector<short> chosen_feature(N_columns,0);
    vector<short> bad_feature(N_columns,0);
    //bad_feature[0] = 1; // meme
    //bad_feature[1] = 1; // user_id
    //bad_feature[2] = 1; // neigh_id
    //bad_feature[3] = 1; // global_time
    
    // ------------------------------
    // 4: Open outfiles & add headers
    
    ofstream R2improvements_file(string(outfolder).append("/R2improvements.csv").c_str());
    if(!R2improvements_file.is_open())
    {
        //cout << "ERROR: Outfolder files are not open." << endl;
        return 1;
    }
    for(int i=0; i<features.size()-1; i++)
    {
        if((i==y_col)||(bad_feature[i]==1))
            continue;
        R2improvements_file << features[i] << ",";
    }
    R2improvements_file << features.back() << endl;
    ofstream splits_file(string(outfolder).append("/splits.csv").c_str());
    if(!splits_file.is_open())
    {
        //cout << "ERROR: Outfolder files are not open." << endl;
        return 1;
    }
    ofstream levels_file(string(outfolder).append("/levels.csv").c_str());
    if(!levels_file.is_open())
    {
        //cout << "ERROR: Outfolder files are not open." << endl;
        return 1;
    }
    levels_file << "best_feature,R2" << endl;
    ofstream ybar_file(string(outfolder).append("/ybar_tree.csv").c_str());
    if(!ybar_file.is_open())
    {
        //cout << "ERROR: Outfolder files are not open." << endl;
        return 1;
    }
    ybar_file << ybar << endl;
    ofstream N_file(string(outfolder).append("/N_tree.csv").c_str());
    if(!N_file.is_open())
    {
        //cout << "ERROR: Outfolder files are not open." << endl;
        return 1;
    }
    N_file << N << endl;
    
    
    // --------------------------
    //  : CREATE THE TREE
    // --------------------------
    
    for(int l=0;l<max_features;l++) // Loop over the levels of the tree
    {
        //cout << "\n--------------------------------------" << endl;
        //cout << "   CREATING LEVEL " << l+1 << " OF THE TREE."  << endl;
        //cout << "--------------------------------------" << endl << endl;
        
        // ------------------------------------------------
        // 1: CALCULATE THE R2_improvement FOR EACH FEATURE
        //    AND PICK THE BEST ONE
        // ------------------------------------------------
        
        R2_improvement_max = 0;
        
        for(int j=0; j<N_columns; j++) // for each feature
        {
            
            // if(j!=4)
            //    continue;
            
            // 0: check if column is good
            if(j == y_col) // if this it the y_columns
                continue;
            if(chosen_feature[j]==1) // if the feature has already been chosen
            {
                //cout << " - skipping chosen feature " << features[j] << endl << endl;
                R2improvements_vec.push_back(0);
                continue;
            }
            if(bad_feature[j]==1) // if its not a wanted feature
            {
                //cout << " - skipping bad feature " << features[j] << endl << endl;
                continue;
            }
            
            //cout << " - processing " << features[j] << endl;
            //cout << "   - reading data..." << endl;
            
            // i: read in the feature column from file to create cumcount and cumsum vectors
            create_Xj_info(j, cumcount, cumsum, range, datafile, groups, G, y_col, N, start_skip_rows, end_skip_rows);
            // error check on the ybars (deleted as comparing floating point numbers (not good)
            
            //cout << "   - range: [" << *(range.begin()) << "," << *(std::prev(range.end())) << "]" << endl;
            
            // ii: initialize the  R2 data and the deltaR2_p map
            R2_improvement = 0;
            R2_improvements.clear();
            deltaR2_p.clear();
            split_p.clear();
            splits.clear();
            splits.push_back(*(range.begin())); // first and last values
            splits.push_back(*(std::prev(range.end())));
            DL.clear();
            
            // iii: find the best first split
            pl = *(range.begin());
            pu = *(std::prev(range.end())); // pl and pu are lower and upper bounds of the partition
            calculate_best_split(cumcount, cumsum, pl, pu, range, G, split, R2_improvement_s);
            best_R2_improvement = R2_improvement_s/SST;
            if(best_R2_improvement == 0)
            {
                R2improvements_vec.push_back(0);
                continue;
            }
            R2_improvement = best_R2_improvement;
            update_R2_DL(best_R2_improvement, R2_improvements, R2, DL, lambda);
            
            // iv: create the two subpartitions from the best split, [Xl,split] and (split,Xu)
            splits.push_back(split);
            pl2 = *(std::next(range.find(split))); // next element after split
            pu2 = pu;
            pl1 = pl;
            pu1 = split;
            
            //for(auto it=cumcount[0].begin(); it != cumcount[0].end(); ++it)
            //    //cout << it->first << "\t" << it->second << endl;
            
            
            //cout << "   - splits:" << endl;
            //cout << "     - (interval, splits, ybars, Ns, R2, delta LF)" << endl;
            
            
            // iii: calculate the rest of the splits until the stopping condition has been reached
            while(!stopping_condition(DL, best_R2_improvement, R2_improvements, splits, R2_improvement))
            {
                // a: change cumcount and cumsum for the split (only have the change the second partition
                recalculate_partitioned_cums(cumcount, cumsum, pl1, pl2, pu2, G);
                
                
                // output the information
                //cout << "     - [" << pl << "," << pu << "], splits = [" << pl << "," << split << "] and [" << pl2 << "," << pu << "], ";
                //cout << " total R2 improvement = " << R2_improvement << ", DL = " << DL.back() << endl;
                
                // b: calculate the best splits and R2 improvements for the splitted partition
                calculate_best_split(cumcount, cumsum, pl1, pu1, range, G, split, R2_improvement_s);
                deltaR2_p[pu1] = R2_improvement_s/SST;
                split_p[pu1] = split;
                
                calculate_best_split(cumcount, cumsum, pl2, pu2, range, G, split, R2_improvement_s);
                deltaR2_p[pu2] = R2_improvement_s/SST;
                split_p[pu2] = split;
                
                // c: calculate the best partition and split overall (start from second element, first is placeholder);
                best_R2_improvement = deltaR2_p.begin()->second;
                split = split_p.begin()->second;
                pl = *(range.begin());
                pu = deltaR2_p.begin()->first;
                for(auto it = std::next(deltaR2_p.begin()); it!=deltaR2_p.end(); ++it)
                {
                    if(it->second > best_R2_improvement)
                    {
                        best_R2_improvement = it->second;
                        split = split_p[it->first];
                        pl = *std::next(range.find(std::prev(it)->first)); //  the bottom of the this interval is the next element after the end of the last interval
                        pu = it->first;
                    }
                }
                
                
                // d: create the two subpartitions from the best split
                splits.push_back(split);
                pl2 = *(std::next(range.find(split))); // next element after split
                pu2 = pu;
                pl1 = pl;
                pu1 = split;
                
                
                
                // e: update the R2 improvement and DL
                update_R2_DL(best_R2_improvement, R2_improvements, R2, DL, lambda);
                R2_improvement += best_R2_improvement;
                
            } // splits calculated
            
            
            //cout << "     - [" << pl << "," << pu << "] split into [" << pl << "," << split << "] and [" << pl2 << "," << pu << "], total R2 improvement = " << R2_improvement << ", DL = " << DL.back() << endl;
            //cout << "   - Chosen partitions: ";
            std::sort(splits.begin(), splits.end());
            //cout << "[" << *splits.begin() << "," << *std::next(splits.begin()) << "], ";
            for(auto it = std::next(std::next(splits.begin())); it!=splits.end(); ++it)
                //cout << "(" << *std::prev(it) << "," << *it << "], ";
            //cout << endl;
            //cout << "   - R2 improvement: " << R2_improvement << endl << endl;
            
            // iv: Record the R2_improvement
            R2improvements_vec.push_back(R2_improvement);
            
            // v: check to see if the feature is the best one
            if(R2_improvement > R2_improvement_max)
            {
                best_feature = j;
                R2_improvement_max = R2_improvement;
                best_splits = splits;
                std::sort(best_splits.begin(), best_splits.end());
            }
            
        }
        
        
        
        // -------------------------------------------
        // 2: BEST FEATURE CHOSEN, UPDATE & SAVE THE OUTPUT
        // -------------------------------------------
        
        if(R2_improvement_max == 0)
        {
            //cout << " No further improvements in R2; Breaking" << endl << endl;
            break;
        }
        
        // Update
        R2 += R2_improvement_max;
        chosen_feature[best_feature] = 1;
        
        // Save output
        
        //cout << "SUMMARY:\n - Best feature is " << features[best_feature] << " with R2 improvement of " << R2_improvement_max << "." << endl;
        //cout << " - Total R2 is now " << R2 << endl;
        
        // // remove first and last placeholders from best_splts
        // best_splits.pop_back();
        // best_splits.erase(best_splits.begin());
        
        // Update the groupings
        //cout << " - Updating groups...";
        update_groupings(groups, G, best_feature, best_splits, datafile, N, best_ybar_vec, best_N_vec, y_col, start_skip_rows, end_skip_rows);
        //cout << G << " groups now." << endl << endl << endl;
        
        // i) save the level info
        levels_file << features[best_feature] << ',' << R2 << endl;
        // ii) save the splits
        splits_file << features[best_feature] << ',';
        for(auto it = best_splits.begin(); it!= std::prev(best_splits.end()); ++it)
            splits_file << *it << ',';
        splits_file << *(std::prev(best_splits.end())) << endl;
        // iii) save the nodal elements for the first three elements (otherwise it becomes too big)
        
        for(int g=0; g<G-1; g++)
        {
            ybar_file << best_ybar_vec[g] << ',';
            N_file << best_N_vec[g] << ',';
        }
        ybar_file << best_ybar_vec.back() << endl;
        N_file << best_N_vec.back() << endl;
        
        
        // iv) Save the R2_improvements_info
        for(int i=0; i<R2improvements_vec.size()-1; i++)
            R2improvements_file << R2improvements_vec[i] << ",";
        R2improvements_file << R2improvements_vec.back() << endl;
        R2improvements_vec.clear();
        
        // check to see if G is greater than the breaking threshold
        if(G > G_THRESH)
        {
            //cout << "Number of groups G = " << G << " is greater than G_THRESH = " << G_THRESH << "; Breaking" << endl << endl;
            break;
        }
        
    }
    
    
    
    clock_t end = clock();
    //cout << "Time = " << 1.0*(end - start)/CLOCKS_PER_SEC << " seconds." <<  endl;
    
    
}
