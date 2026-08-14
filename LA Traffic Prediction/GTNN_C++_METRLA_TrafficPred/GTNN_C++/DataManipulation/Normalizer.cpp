

//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS, College of Engineering
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++


#include "Normalizer.h"

#include<random>
#include<iostream>
#include<string>
#include<sstream>
#include<fstream>

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// NORMALIZER INITIALIZATION

Normalizer::Normalizer(int numfeats, int numOutputNeurons, float normstart, float normrange) {
    this->numfeats = numfeats, this->numOutputNeurons = numOutputNeurons;
    this->normstart = normstart; this->normrange = normrange;

    resizeVector(mins_x_cols, numfeats);
    resizeVector(x_col_ranges, numfeats);
    resizeVector(y_mins,numOutputNeurons);
    resizeVector(y_ranges, numOutputNeurons);

    std::fill(mins_x_cols, mins_x_cols + numfeats, 999999.0);
    std::fill(x_col_ranges, x_col_ranges + numfeats, -999999.0);
    std::fill(y_mins, y_mins + numOutputNeurons, 999999.0);
    std::fill(y_ranges, y_ranges + numOutputNeurons, -999999.0);
}

// initialize normalization features
int Normalizer::init_normalization_x_y(std::string filename) {
    // THIS FUNCTION initializes:
    // x_col_ranges, mins_x_cols
    // y_range, y_min
    // based off of a file read

    // open file
    std::string line, word; int numlines = 0;
    std::ifstream rfile(filename);
    std::getline(rfile, line);        // get the headers

    int col = 0; float num;
    while (getline(rfile, line)) {
        std::istringstream s(line);
        // get x features
        for (col = 0; col < numfeats; col++) {
            getline(s, word, ',');
            num = std::stof(word);

            mins_x_cols[col] = std::min(mins_x_cols[col], num);
            x_col_ranges[col] = std::max(x_col_ranges[col], num);
        }
        // get y
        for (int i = 0; i < numOutputNeurons; i++) {
            getline(s, word, ',');
            num = std::stof(word);

            y_mins[i] = std::min(y_mins[i], num);
            y_ranges[i] = std::max(y_ranges[i], num);
        }

        numlines++;     // increment num lines
    }
    // set the x, y ranges
    for (col = 0; col < numfeats; col++) {
        x_col_ranges[col] -= mins_x_cols[col];
    }
    for (int i = 0; i < numOutputNeurons; i++) {
        y_ranges[i] -= y_mins[i];
    }

    // Close the file
    rfile.close();
    return numlines;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// NORMALIZER ACTION

// denormalize y (apply to 'a' vector of output layer AND 'y_true' vector of layer)
void Normalizer::denormalize_ybatch(float**& y, int batchsize, int numCols) {
    // DENORMALIZATION
    for (int i = 0; i < batchsize; i++) {
        for (int j = 0; j < numCols; j++) {
            denorm_func(y[i][j], y_mins[j], y_ranges[j]);
        }
    }
}
void Normalizer::renormalize_ybatch(float**& y, int batchsize, int numCols) {
    // NORMALIZATION
    for (int i = 0; i < batchsize; i++) {
        for (int j = 0; j < numCols; j++) {
            norm_func(y[i][j], y_mins[j], y_ranges[j]);
        }
    }
}

//void Normalizer::renormalize_ydata(float**& y, int numsamples, int cols[]) {
//    // NORMALIZATION
//    int k;
//    for (int i = 0; i < numsamples; i++) {
//        k = 0;
//        for (int j = 0; j < numOutputNeurons; j++) {
//            try {
//                if (j == cols[k]) {
//                    norm_func(y[i][j], y_mins[j], y_ranges[j]);
//                    k++;
//                }
//            }
//            catch (std::out_of_range(const std::string& what_arg)) {
//                continue;
//            }
//        }
//    }
//}


// NORMALIZE ENTIRE DATAFILE, WRITE TO NEW FILE
void Normalizer::denorm_func(float& i, float& min, float& range) {
    i = (((i - normstart) / normrange) * range) + min;
}
void Normalizer::norm_func(float& i, float& min, float& range) {
    i = (((i - min) / range) * this->normrange) + this->normstart;
}

void Normalizer::normalize_datafile(std::string filename, std::string newfilename) {
    // open files
    std::string line, word;
    std::ifstream rfile(filename);
    std::ofstream wfile(newfilename);

    // copy the headers into wfile
    std::getline(rfile, line);
    wfile << line << "\n";

    // xdata, ydata matrices to hold values
    float* xdata; resizeVector(xdata, numfeats);
    float* ydata; resizeVector(ydata, numOutputNeurons);

    // read thru entire file
    while (getline(rfile, line)) {
        std::istringstream s(line);

        // get x features & normalize each
        for (int col = 0; col < numfeats; col++) {
            getline(s, word, ',');
            xdata[col] = std::stof(word);
            // apply normalization
            norm_func(xdata[col], mins_x_cols[col], x_col_ranges[col]);
        }
        for (int i = 0; i < numOutputNeurons; i++) {
            // get y & normalize it
            getline(s, word, ',');
            ydata[i] = std::stof(word);
            norm_func(ydata[i], y_mins[i], y_ranges[i]);       // apply normalization
        }
        // generate ostream from xdata,ydata values
        std::ostringstream ostream;
        for (int j = 0; j < numfeats; j++) { ostream << xdata[j] << ","; }
        for (int j = 0; j < numOutputNeurons; j++) { ostream << ydata[j] << ","; }
        // generate line (str) from ostream
        line = (ostream.str());
        line[line.size() - 1] = '\n';
        // write line to new data file
        wfile << line;
    }
    // Close the file
    rfile.close();
    wfile.close();
}


//// UNUSED FUNCTION
//// send matrix to file
//// matrix contains: x1,x2,...y
//// matrix size: [batchsize][numfeats+1]
//void Normalizer::mat_to_file(float newdata[batchsize][numfeats + 1], std::ofstream& wfile) {
//    // variables for writing the str
//    std::string rowline;
//    int rowlen = numfeats + 1;
//    std::string line;
//
//    // write each line to wfile
//    for (int i = 0; i < batchsize; i++) {
//        // convert arr row to string
//        std::ostringstream ostream;
//        for (int j = 0; j < rowlen; j++) { ostream << newdata[i][j] << ","; }
//        line = (ostream.str());
//        line[line.size() - 1] = '\n';
//        // write string to new data file
//        wfile << line;
//    }
//}

int Normalizer::countRows(std::string filename) {
    std::ifstream rfile(filename);
    std::string line;
    getline(rfile, line);       // get the headers

    int numRows = 0;
    while (getline(rfile, line)) {
        numRows++;
    }
    rfile.close();
    return numRows;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// normalizing graph data

int Normalizer::init_normalization_g(std::string filename) {
    // THIS FUNCTION initializes:
    // x_col_ranges, mins_x_cols
    // based off of a file read

    // open file
    std::string line, word; int numlines = 0;
    std::ifstream rfile(filename);
    std::getline(rfile, line);        // get the headers

    int col = 0; float num;
    while (getline(rfile, line)) {
        std::istringstream s(line);
        getline(s, word, ',');      // take out the time value

        // get x features
        for (col = 0; col < numfeats; col++) {
            getline(s, word, ',');
            num = std::stof(word);

            mins_x_cols[col] = std::min(mins_x_cols[col], num);
            x_col_ranges[col] = std::max(x_col_ranges[col], num);
        }

        numlines++;     // increment num lines
    }
    // set the x ranges
    for (col = 0; col < numfeats; col++) {
        x_col_ranges[col] -= mins_x_cols[col];
    }

    // Close the file
    rfile.close();
    return numlines;
}

void Normalizer::normalize_datafile_g(std::string filename, std::string newfilename) {
    // open files
    std::string line, word;
    std::ifstream rfile(filename);
    std::ofstream wfile(newfilename);

    // copy the headers into wfile
    std::getline(rfile, line);
    wfile << line << "\n";

    // xdata, ydata matrices to hold values
    float* xdata; resizeVector(xdata, numfeats);
    float* ydata; resizeVector(ydata, numOutputNeurons);
    int t = -1;

    // read thru entire file
    while (getline(rfile, line)) {
        std::istringstream s(line);
        getline(s, word, ','); t = std::stoi(word);     // get t

        // get x features & normalize each
        for (int col = 0; col < numfeats; col++) {
            getline(s, word, ',');
            xdata[col] = std::stof(word);
            // apply normalization
            norm_func(xdata[col], mins_x_cols[col], x_col_ranges[col]);
        }
        // generate ostream from xdata values
        std::ostringstream ostream; ostream << t << ",";
        for (int j = 0; j < numfeats; j++) { ostream << xdata[j] << ","; }
        // generate line (str) from ostream
        line = (ostream.str());
        line[line.size() - 1] = '\n';
        // write line to new data file
        wfile << line;
    }
    // Close the file
    rfile.close();
    wfile.close();
}

void Normalizer::denormalize_gstate(float**& s, int numnodes, int numfeats) {
    // NORMALIZATION
    for (int i = 0; i < numnodes; i++) {
        for (int j = 0; j < numfeats; j++) {
            denorm_func(s[i][j], mins_x_cols[j], x_col_ranges[j]);
        }
    }
}
void Normalizer::renormalize_gstate(float**& s, int numnodes, int numfeats) {
    // NORMALIZATION
    for (int i = 0; i < numnodes; i++) {
        for (int j = 0; j < numfeats; j++) {
            norm_func(s[i][j], mins_x_cols[j], x_col_ranges[j]);
        }
    }
}
void Normalizer::denormalize_gstate_col(float**& s, int numnodes, int numfeats, int col) {
    // NORMALIZATION
    for (int i = 0; i < numnodes; i++) {
        for (int j = 0; j < numfeats; j++) {
            denorm_func(s[i][col], mins_x_cols[col], x_col_ranges[col]);
        }
    }
}
void Normalizer::renormalize_gstate_col(float**& s, int numnodes, int numfeats, int col) {
    // NORMALIZATION
    for (int i = 0; i < numnodes; i++) {
        norm_func(s[i][col], mins_x_cols[col], x_col_ranges[col]);
    }
}