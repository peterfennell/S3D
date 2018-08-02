// PREDICTING EXPECTED OUTCOMES WITH S3D
//
// Inputs:
//  (Required)
//  - datafile:         Datafile for which to make the predictions
//  - infolder:         Directory where S3D is located (this is "outfile" of train.cpp program)
//  - outfolder:        Directory where the predictions will be saved
//  (Optional)
//  - max_features:     Maximum number of features to use for prediction (int >=0, default use all S3D chosen features)
//  - min_samples:      Minimum number of samples required to make a prediction (default 1)
//  - start_use_rows:   First row of a continuous block of rows to use for prediction (default 0)
//  - end_use_rows:     Row after the last row of a contiguous block of rows to use for prediction (default n_predictions)
//
// Outputs:
//  - predicted_expectations.cpp
//    - vector with predicted expectations for each row of the input datafile

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <ctime>

using namespace std;

void get_column_names(ifstream & datafile, vector<string> & columns);
void create_splits_vec(map<string, vector<double> > &splits, vector<string> &best_feats, ifstream &splits_file, vector<int> & n_intervals);
int get_row(map<string,double> & Xvals, ifstream & datafile, vector<string> & features);
void get_index_from_splits(double xval, vector<double> & feature_splits, int & interval_index);
void get_N_from_row(ifstream & N_file, int column_number, int & N);
void get_ybar_from_file(ifstream & ybar_file, int level, int column_number, double & ybar);


int main(int argc, char * argv[]){
    
    cout << "\n* PREDICT EXPECTATIONS ALGORITHM *\n" << endl;
    clock_t t_start = clock();
    
    // -----------------------------------------
    // 0: DECLARE AND READ IN PROGRAM PARAMETERS
    // -----------------------------------------
    
    ifstream datafile;
    string infolder = "", outfolder = "", max_features_tag = "", min_samples_tag = ""; // locations to read inputs and write outputs
    int max_features = 1000000; // maximum number of features to use for prediction
    int min_samples = 1; // minimum number of samples required to make a prediction
    int start_use_rows=0, end_use_rows=-1;

    for(int i=1; i<argc; i++) // go from 1st as 0 is programme name
    {
        // Required
        if(strncmp(argv[i],"-infolder:",10)==0){
            infolder = string(argv[i]).erase(0,10);
            cout << " Reading SSML decomposition output from " << infolder << endl;
            continue;
        }
        if(strncmp(argv[i],"-datafile:",10)==0){
            datafile.open(string(argv[i]).erase(0,10).c_str());
            cout << " Reading data from " << string(argv[i]).erase(0,10) << endl;
            continue;
        }
        if(strncmp(argv[i],"-outfolder:",11)==0){
            outfolder = string(argv[i]).erase(0,11);
            continue;
        }
        // Optional
        if(strncmp(argv[i],"-max_features:",14)==0){
            max_features = atoi(string(argv[i]).erase(0,14).c_str());
            max_features_tag.append("_MF_").append(string(argv[i]).erase(0,14));
            continue;
        }
        if(strncmp(argv[i],"-min_samples:",13)==0)
        {
            min_samples = atoi(string(argv[i]).erase(0,13).c_str());
            min_samples_tag.append("_MSL_").append(string(argv[i]).erase(0,13));
            continue;
        }
        if(strncmp(argv[i],"-start_use_rows:",16)==0){
            start_use_rows = atoi(string(argv[i]).erase(0,16).c_str());
            continue;
        }
        if(strncmp(argv[i],"-end_use_rows:",14)==0){
            end_use_rows = atoi(string(argv[i]).erase(0,14).c_str());
            continue;
        }
        // Invalid argument, argument doesnt match anything
        cout << "ERROR: Unknown argument " << argv[i] << endl;
        return 0;
    }
    
    // check to see if datafile is open
    if(!datafile.is_open())
    {
        cout << "ERROR: datafile not found" << endl;
        return 1;
    }
    // open model files
    ifstream splits_file(string(infolder).append("/splits.csv").c_str());
    if(!splits_file.is_open())
    {
        cout << "ERROR: Infolder files are not open." << endl;
        return 1;
    }
    ifstream ybar_file(string(infolder).append("/ybar_tree.csv").c_str());
    if(!ybar_file.is_open())
    {
        cout << "ERROR: Infolder files are not open." << endl;
        return 1;
    }
    double ybar_global;
    ybar_file >> ybar_global;
    ybar_file.clear();
    ybar_file.seekg(0, ios::beg);
    ybar_file.ignore(numeric_limits<streamsize>::max(),'\n'); // skip the first row of the file as this is the overall probability

    ifstream N_file(string(infolder).append("/N_tree.csv").c_str());
    if(!N_file.is_open())
    {
        cout << "ERROR: Infolder files are not open." << endl;
        return 1;
    }
    // error check to see if N < min_samples
    int N;
    N_file >> N;
    if(min_samples > N)
    {
        cout << "ERROR: min_samples is greater than total number of datapoints on training data." << endl;
        return 1;
    }
    // outfolder
    string outfile_name = string(outfolder).append("/predicted_expectations").append(max_features_tag).append(min_samples_tag).append(".csv");
    ofstream prediction_file(outfile_name.c_str());
    if(!prediction_file.is_open())
    {
        cout << "ERROR: Outfolder files are not open." << endl;
        return 1;
    }
    cout << " Writing predictions to file " << outfile_name << endl;

    // Calculate number of predictions
    // i) get number of lines from file
    int len_datafile=0;
    datafile.ignore(numeric_limits<streamsize>::max(),'\n'); // skip the first line, just column names
    char c;
    while(datafile.get(c)) // while there is not the end of file marker
    {
        // go to end of line; if its bad then break
        datafile.ignore(numeric_limits<streamsize>::max(),'\n');
        len_datafile++;
    }
    // ii) use this along with start_use_rows and end_use_rows for to calculate number of rows
    int n_predictions;
    if(end_use_rows == -1) // end_use_rows undeclared
        n_predictions = len_datafile - start_use_rows;
    else // end_use_rows declared
        n_predictions = end_use_rows - start_use_rows;
    cout << "\n  - " << n_predictions << " rows for prediction" << endl;
    // reset the datafile to the first line
    datafile.clear();
    datafile.seekg(0, ios::beg);

    // If max_features = 0, just output the global average and finish
    if(max_features == 0)
    {
        cout << " max_features = 0, predicting global average..." << endl;
        
        for(int i=0; i<n_predictions; i++)
        {
            prediction_file << ybar_global << "\n";
        }

        cout << "\n DONE" << endl;
        cout << "  - program took " << 1.0*(clock() - t_start)/CLOCKS_PER_SEC << " seconds to execute." << endl << endl;
        return 0;
    }
    
    // max_features is positive, so must process each prediction of y according to values of the features
    
    // ---------------------------------
    // 1: PROCESS SPLITS
    // ---------------------------------
    
    map<string, vector<double> > splits;
    vector<string> best_feats;
    vector<int> n_intervals; // number of intervals in each split
    create_splits_vec(splits, best_feats, splits_file, n_intervals);
    int n_chosen_features = best_feats.size();
    int n_predictive_features = min(n_chosen_features, max_features);

    cout << "\n FEATURES for prediction are:" << endl;
    for(int i=0; i<n_predictive_features; i++)
        cout << "  - " << best_feats[i] << endl;
    
    
    // --------------------------------------
    // 2: READ DATA AND PREDICT PROBABILITIES
    // --------------------------------------
    
    string feature;
    vector<string> features;
    get_column_names(datafile, features);
    map<string,double> Xvals;   // the values of the best_feats (when we loop through each row of data)
    int interval_index;
    vector<int> position(n_predictive_features,0); // the split position of the row for each best_feature
    double ybar; //  the predicted expectation
    
    // posititon_vectors
    
    // iii) initialize the position vector and inverse map
    vector<vector<int> > all_positions_N(n_predictions, vector<int>(n_predictive_features,0));
    vector<map<int, vector<int> > > predictions_at_position_N(n_predictive_features, map<int, vector<int> >());
    vector<map<int, vector<int> > > predictions_at_position_ybar(n_predictive_features, map<int, vector<int> >());
    vector<bool> min_samples_reached(n_predictions, false);
    
    cout << "\n PERFORMING PREDICTIONS \n" << endl;
    
    // -------------------------------------------------
    // A: Get position of every row prediction in N file

    cout << " A: Calculating positions..." << endl;
    
    // skip the rows up to start_use_rows (if start_use_rows undeclared then nothing skips)
    for(int i=0; i<start_use_rows; i++)
        datafile.ignore(numeric_limits<streamsize>::max(),'\n');
    // now make predictions on the rest
    for(int i=0; i<n_predictions; i++) //  for each row of data in the file in the file
    {
        
        // a: read in the row
        //if(get_row(Xvals, datafile, features)==0)
        //    break;
        get_row(Xvals, datafile, features);
        
        // b: get positions from elements
        for(int level=0; level<n_predictive_features; level++)
        {
            feature = best_feats[level];
            get_index_from_splits(Xvals[feature], splits[feature], interval_index);
            if(level == 0)
            {
                all_positions_N[i][level] = interval_index;
                predictions_at_position_N[level][interval_index].push_back(i);
            }
            else
            {
                all_positions_N[i][level] = all_positions_N[i][level-1]*n_intervals[level] + interval_index;
                predictions_at_position_N[level][all_positions_N[i][level]].push_back(i);
            }
            
        }
        
    }
    
    
    // ------------------------------------------------------
    // B: loop through N_file to find when to break for y_bar
    
    cout << " B: Scanning N_file for to check for min_samples..." << endl;
    
    // introduce the final predictions vector
    vector<double> predicted_expectations(n_predictions,0);
    
    // reset the N_file to the first line
    N_file.clear();
    N_file.seekg(0, ios::beg);
    N_file.ignore(numeric_limits<streamsize>::max(),'\n');
    
    int col_index, col_skip, n_predictions_remaining = n_predictions, previous_level_position;
    for(int level=0; level<n_predictive_features; level++)
    {
        col_index = 0;
        
        // for each N entry in the level
        for(map<int, vector<int> >::iterator it = predictions_at_position_N[level].begin(); it != predictions_at_position_N[level].end(); ++it)
        {
            // 0: check to see if the elements in this group have already reached the min_samples; if so continue
            //    (each element will have come here along the same branches to all are equivalent.
            if(min_samples_reached[(it->second)[0]]==true)
                continue;
            
            // i: skip to the relevant row
            col_skip = it->first - col_index; // number of columns to skip
            for(int Dcol = 0; Dcol < col_skip; Dcol ++)
                N_file.ignore(numeric_limits<streamsize>::max(),',');
            // update the column index
            col_index = it->first;
            
            // ii: read the entry
            N_file >> N;
            
            // iii: if N is less than the minimum_sample_leaf then we have found the position for the ybar for each of the elements
            //      - NOTE: that this position is the last position.
            if(N < min_samples)
            {
                if(level == 0) // if its the first level of the tree just set the probability to be the global one
                {
                    // loop through each element, mark them as sub and assign the probability to them
                    for(auto it2 = (it->second).begin(); it2 != (it->second).end(); ++ it2)
                    {
                        min_samples_reached[*it2] = true;
                        n_predictions_remaining --;
                        predicted_expectations[*it2] = ybar_global;
                    }
                }
                else
                {
                    // this level is below the min_samples so take position for the last level
                    previous_level_position = all_positions_N[(it->second)[0]][level-1];
                    
                    // all of the elements here will have the same position for the ybars.
                    // Note that other elements could also have the same previous position so append
                    predictions_at_position_ybar[level-1][previous_level_position].insert(predictions_at_position_ybar[level-1][previous_level_position].end(), (it->second).begin(), (it->second).end());
                    
                    // these elements are now sub min_elements_reached to mark them as true
                    for(auto it2 = (it->second).begin(); it2 != (it->second).end(); ++ it2)
                    {
                        min_samples_reached[*it2] = true;
                        n_predictions_remaining --;
                    }
                }

            }
            else{
                // if we have reached the end of the tree and N is not less than min_samples then take the last element
                if(level == n_predictive_features-1)
                {
                    predictions_at_position_ybar[level][col_index] = it->second;
                    // these elements are now sub min_elements_reached to mark them as true
                    for(auto it2 = (it->second).begin(); it2 != (it->second).end(); ++ it2)
                    {
                        n_predictions_remaining --;
                    }
                }
            }
            
        }
        
        // have processed each entry at this level, now skip to the end of the line and go to next level
        N_file.ignore(numeric_limits<streamsize>::max(),'\n');

        
    }

    // Clear some memory
    vector<map<int, vector<int> > >().swap(predictions_at_position_N);
    vector<vector<int> >().swap(all_positions_N);
    
    for(int i=0; i<n_predictions; i++)
    {
        if(predicted_expectations[i] > 0)
            n_predictions_remaining ++;
    }
    // cout << n_predictions_remaining << " ybar_globals " << endl;
    
    int g=0;
    for(int level=0; level<n_predictive_features; level++)
        for(auto it = predictions_at_position_ybar[level].begin(); it != predictions_at_position_ybar[level].end(); ++it)
        {
            g += it->second.size();
        }
    //cout << g << " elements accounted for in ppybar" << endl;
            

    // --------------------------------------------------------
    // C: loop through yfile and attain appropriate predictions
    
    cout << " C: Extracting predicted expectations from ybar_file..." << endl;
    
    for(int level=0; level<n_predictive_features; level++)
    {
        
        col_index = 0;
        
        // for each N entry in the level
        for(auto it = predictions_at_position_ybar[level].begin(); it != predictions_at_position_ybar[level].end(); ++it)
        {

            // i: skip to the relevant row
            col_skip = it->first - col_index; // number of columns to skip
            for(int Dcol = 0; Dcol < col_skip; Dcol ++)
                ybar_file.ignore(numeric_limits<streamsize>::max(),',');
            // update the column index
            col_index = it->first;
            
            // ii: read the entry
            ybar_file >> ybar;
            
            // iii: for each appropriate element record the probability

            for(auto it2 = (it->second).begin(); it2 != (it->second).end(); ++ it2)
            {
                predicted_expectations[*it2] = ybar;
                n_predictions_remaining ++;
            }
            
        }
        
        // have processed each entry at this level, now skip to the end of the line and go to next level
        ybar_file.ignore(numeric_limits<streamsize>::max(),'\n');
        
        
    }

    // cout << "    - " << n_predictions - n_predictions_remaining << " rows have not been clasified" << endl;
    
    // ---------------------
    // D: Save the output
    
    cout << " D: Saving..." << endl;
    
    for(int i=0; i<n_predictions; i++)
    {
        prediction_file << predicted_expectations[i] << "\n";
    }
    
    // Have created a predicted probability for each row, DONE
    
    cout << "\n DONE" << endl;
    cout << "  - program took " << 1.0*(clock() - t_start)/CLOCKS_PER_SEC << " seconds to execute." << endl << endl;
    return 0;
    
}

// FUNCTIONS

void get_N_from_row(ifstream & N_file, int column_number, int & N)
{
    // skip first column_number entries
    for(int i=0; i<column_number; i++)
    N_file.ignore(numeric_limits<streamsize>::max(),',');
    
    // read next entry
    N_file >> N;
    
    // skip rest of the file
    N_file.ignore(numeric_limits<streamsize>::max(),'\n');
    
}

void get_ybar_from_file(ifstream & ybar_file, int level, int column_number, double & ybar)
{
    // skip first level number of rows
    for(int l=0; l<level; l++)
    ybar_file.ignore(numeric_limits<streamsize>::max(),'\n');
    
    // then skip the first column_number of columns to the desired entry
    for(int i=0; i<column_number; i++)
    ybar_file.ignore(numeric_limits<streamsize>::max(),',');
    
    // read next entry
    ybar_file >> ybar;
    
    // done
    
}

void get_index_from_splits(double xval, vector<double> & feature_splits, int & interval_index)
{
    interval_index = 0;
    for(auto it = feature_splits.begin(); it != feature_splits.end(); ++it)
    {
        if(xval <= *it)
        break;
        interval_index ++;
    }
    
}

int get_row(map<string,double> & Xvals, ifstream & datafile, vector<string> & features)
{
    double xval;
    char c;
    int col=0;
    
    datafile >> xval;
    Xvals[features[col]] = xval;
    if(!datafile.get(c))
    return 0;
    
    while(c!='\n') // while we have not reached the end of line
    {
        col ++;
        datafile >> xval;
        Xvals[features[col]] = xval;
        datafile.get(c);
    }
    
    return 1;
    
}

void create_splits_vec(map<string, vector<double> > &splits, vector<string> &best_feats, ifstream &splits_file, vector<int> & n_intervals)
{
    string feat_name;
    char c;
    double split_ind;
    
    while(!splits_file.eof())
    {
        // read in the name of the feature;
        if(!splits_file.get(c)) // get next character in stream, and if theres none there (i.e., EOF) then break
            break;
        while(c!=',')
        {
            feat_name.push_back(c);
            splits_file.get(c);
        }
        splits[feat_name] = vector<double>();
        best_feats.push_back(feat_name);
        
        // read in the values of the splits until we come to the next line
        while(c!='\n')
        {
            splits_file >> split_ind;
            splits[feat_name].push_back(split_ind);
            if(!splits_file.get(c)) // get next character in stream, and if theres none there (i.e., EOF) then break
                break;
        }
        
        // reached the end of the line, record the number of splits and then clear feat_name and read next splits
        n_intervals.push_back(splits[feat_name].size()-1);
        feat_name.clear();
        
    }
    
    // read in all of the splits, no prune the first and the last element of each vector as they are redundant (will be replaces by < 2nd element and > n-1'st element)
    for(auto it=splits.begin(); it!=splits.end(); ++it)
    {
        
        (it->second).pop_back();  // remove the last element
        (it->second).erase((it->second).begin());   // remove the first element
    }
    
    // Done
    
}


void get_column_names(ifstream & datafile, vector<string> & columns)
{
    string s;
    char c;
    
    datafile.get(c);
    // read columns until get to end of the line
    while(true)
    {
        
        // read characters into s until get to a ,
        while((c != ',')&&(c != '\n'))
        {
            s.push_back(c);
            datafile.get(c);
        }
        
        // add string to columns and clear s (and advance to next symbol)
        columns.push_back(s);
        s.clear();
        
        // check to see if reached end of line
        if(c == '\n')
        break;
        else
        datafile.get(c);
        
    }
    
}

