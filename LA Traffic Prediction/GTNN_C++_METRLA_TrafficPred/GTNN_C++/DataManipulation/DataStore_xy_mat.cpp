

//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS, College of Engineering
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++


#include<iostream>
#include<unordered_map>
#include<sstream>
#include<fstream>
#include "DataStore.h"

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// XY Matrix generation

// write appropriate column headers into csv (x1, x2, ... y)
void write_data_headers(std::ofstream& wfile, int numfeats, int numOutputNeurons) {
	// WRITE THE HEADERS
	std::string headers = "";
	for (int i = 0; i < numfeats; i++) {				// write the x features
		headers += "x" + std::to_string(i + 1) + ",";
	}

	for (int j = 0; j < numOutputNeurons; j++)		// write the y header(s)
		headers += "y" + std::to_string(j + 1) + ",";

	headers[headers.size() - 1] = '\n';
	wfile << headers;
}

void vect_times_matrix(float*& xdata, float**& w, float*& ydata, int num_wR, int num_wC) {
	for (int i = 0; i < num_wC; i++) {
		ydata[i] = 0;
		for (int j = 0; j < num_wR; j++) {
			ydata[i] += xdata[j] * w[j][i];
		}
	}
}

void write_data(std::ofstream& wfile, \
	std::uniform_real_distribution<float>& mydist, std::default_random_engine& mygen, \
	float**& w, int numsamples, int numfeats, int numOutputNeurons) {

	// create xdata, ydata storage
	float* xdata; resizeVector(xdata, numfeats);
	float* ydata; resizeVector(ydata, numOutputNeurons);

	// variables for writing the str
	std::string line;

	// RANDOMIZE X VALUES, CALCULATE CORRESPONDING Y, WRITE EACH LINE TO FILE
	for (int n = 0; n < numsamples; n++) {

		// generate data sample
		getRandomFloats(mydist, mygen, xdata, numfeats);			// x
		vect_times_matrix(xdata, w, ydata, numfeats, numOutputNeurons);		// y

		// convert sample to string
		std::ostringstream ostream;
		for (int f = 0; f < numfeats; f++) {
			ostream << xdata[f] << ",";
		} for (int o = 0; o < numOutputNeurons; o++) {
			ostream << ydata[o] << ",";
		}

		line = (ostream.str());		// format data sample str
		line[line.size() - 1] = '\n';

		wfile << line;		// write string to new data file
	}
}

// simulate data for train AND test datafiles
void DataStore::gen_xy_datafiles(std::string newfilename, std::string filename_test, \
	int numsamples_train, int numsamples_test, int numfeats, int numOutputNeurons) {

	std::ofstream wfile(newfilename);		// open train file

	// init randomization variables
	std::uniform_real_distribution<float> mydist(-25.0,50.0);
	std::default_random_engine mygen;
	mygen.seed(24);

	write_data_headers(wfile, numfeats, numOutputNeurons);		// WRITE THE HEADERS

	// GENERATE RANDOM PATTERN OF weights (matrix w)
	float** w; resizeMatrix(w, numfeats, numOutputNeurons);
	for (int i = 0; i < numfeats; i++) {
		getRandomFloats(mydist, mygen, w[i], numOutputNeurons);
	}
	//displayMat(w, numfeats, numOutputNeurons, "gen w");

	// generate train data
	write_data(wfile, mydist, mygen, w, numsamples_train, numfeats, numOutputNeurons);
	wfile.close();

	// generate test data
	std::ofstream wfile_test(filename_test);
	write_data_headers(wfile_test, numfeats, numOutputNeurons);
	write_data(wfile_test, mydist, mygen, w, numsamples_train, numfeats, numOutputNeurons);
	wfile_test.close();

}


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// XY Matrix data retrieval

void DataStore::retrieve_xy_data(std::string datafile, int xdataC, int ydataC) {
	this->xdataC = xdataC; this->ydataC = ydataC;

	std::ifstream rfile(datafile);
	this->numsamples = (-1) + std::count(std::istreambuf_iterator<char>(rfile),
		std::istreambuf_iterator<char>(), '\n');
	rfile.seekg(0);

	// resize xdataAll and ydataAll
	if (xdata == nullptr) {
		resizeMatrix(xdata, numsamples, xdataC);
		resizeMatrix(ydata, numsamples, ydataC);
		resizeMatrix(xbatch, batchsize, xdataC);
		resizeMatrix(ybatch, batchsize, ydataC);
		pre_initialized_matrices = true;
	}

	// file read variables
	std::string line, word;
	std::getline(rfile, line);        // get the headers

	// read each line
	int col = 0; int row = 0;
	while (getline(rfile, line)) {
		std::istringstream s(line);
		// get x features
		for (col = 0; col < xdataC; col++) {
			getline(s, word, ',');
			xdata[row][col] = std::stof(word);
		}
		// get y
		for (int i = 0; i < ydataC; i++) {
			getline(s, word, ',');
			ydata[row][i] = std::stof(word);
		}
		row++;
	}
	rfile.close();

	//displayMat(xdata, numsamples, xdataC, "xdata");
	//displayMat(ydata, numsamples, ydataC, "ydata");
}

void DataStore::mat_to_file(float**& mat, std::ofstream& wfile, int numsamples, int numcols) {
	std::string line;
	for (int i = 0; i < numsamples; i++) {
		std::ostringstream ostream;		// ostream for each node
		for (int j = 0; j < numcols; j++) {
			ostream<< mat[i][j] << ',';
		}
		line = ostream.str();
		line[line.size() - 1] = '\n';
		wfile << line;
	}
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// XY Matrix DATA SHUFFLING

// swap two arrays
void swap_arrays(float*& v1, float*& v2, float*& temp, int numElems) {
	std::memcpy(temp, v1, sizeof(float) * numElems);
	std::memcpy(v1, v2, sizeof(float) * numElems);
	std::memcpy(v2, temp, sizeof(float) * numElems);
}
void DataStore::shuffle_xy_data() {
	// initialize temp storage vectors
	float* xtemp; resizeVector(xtemp, xdataC);
	float* ytemp; resizeVector(ytemp, ydataC);

	// initialize the random generator
	std::uniform_int_distribution<> mydist(0, numsamples - 1);
	std::default_random_engine mygen;
	mygen.seed(37);

	// loop thru xdataAll & ydataAll, swapping elems
	for (int j = 0; j < numsamples; j++) {
		int swapidx = mydist(mygen);
		swap_arrays(xdata[j], xdata[swapidx], xtemp, xdataC);
		swap_arrays(ydata[j], ydata[swapidx], ytemp, ydataC);
	}
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// XY matrix data sending

void DataStore::display_xy() {
	displayMat(xdata, numsamples, xdataC, "xdata");
	displayMat(ydata, numsamples, ydataC, "ydata");
}

float**& DataStore::get_x_batch(int batchidx) {
	int idx = batchidx * batchsize;
	for (int i = 0; i < batchsize; i++)
		std::memcpy(xbatch[i], xdata[idx + i], xdataC * sizeof(float));
	// displayMat(xbatch, batchsize, xdataC, "x batch");

	return xbatch;
}
float**& DataStore::get_y_batch(int batchidx) {
	int idx = batchidx * batchsize;
	for (int i = 0; i < batchsize; i++)
		std::memcpy(ybatch[i], ydata[idx + i], ydataC * sizeof(float));
	// displayMat(ybatch, batchsize, ydataC, "y batch");
	return ybatch;
}

