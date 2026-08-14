#pragma once

//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++

#include<unordered_map>
#include <random>
#include<string>

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// general funcs

void resizeVector(float*& mat, int numElems);
void resizeMatrix(float**& mat, int numRows, int numCols);
void freeMatrix(float**& mat, int numRows);
void getRandomFloats(std::uniform_real_distribution<float>& dist,
	std::default_random_engine& gen, float*& storage, int numElems);
void randomizeMat(std::uniform_real_distribution<float>& dist,
	std::default_random_engine& gen, float*& storage, int numElems);
void fillMat(float**& mat, int numR, int numC, int num);
void randomizeMat(float**& mat, int numR, int numC, float randmax=1, float randmin=0);
void displayMat(float**& dataAll, int numRows, int numCols, std::string title = "", bool pause = false);
void append_mat_to_str(float**& dataAll, int numRows, int numCols, std::string title, std::string& storage);
void displayVect(float*& vec, int numElems, std::string title = "");
void copyMat(float**& dest, float**& src, int numRows, int numCols);
void copyVect(float*& dest, float*& src, int numElems);
void wait();
#include<iostream>
#include<string>

// store a float** matrix into a file
void mat_to_file(float**& mat, std::string filename, int numsamples, int numcols, std::string headers="");

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Data Storage class (file reading/ storing)

class DataStore {
public:
	// vector
	float* vdata; float* vecbatch;
	int num_vtimesteps;

	// graph (consists many state matrices)
	std::unordered_map<int, float**> gdata;
	float** gstate;
	int num_gnodes; int num_gfeats; int num_gtimesteps;

	// x, y matrices
	float** xdata; float** ydata;
	float** xbatch; float** ybatch;
	int xdataC; int ydataC;		// num cols in x, y matrices
	int batchsize;				// batchsize (num rows of batch)
	int numsamples;
	bool pre_initialized_matrices = false;

public:
	//static std::uniform_real_distribution<float> dist;
	//static std::default_random_engine gen;


	// matrix
	DataStore(int batchsize=1) : batchsize(batchsize), xdata(nullptr), ydata(nullptr) {}
	void retrieve_xy_data(std::string datafile, int xdataC, int ydataC);
	void shuffle_xy_data();
	float**& get_x_batch(int batchidx);
	float**& get_y_batch(int batchidx);
	int count_xy_batches() { return numsamples / batchsize; }
	void gen_xy_datafiles(std::string newfilename, std::string filename_test, \
		int numsamples_train, int numsamples_test, int num_gfeats, int numOutputNeurons);	// data generation
	void display_xy();
	void mat_to_file(float**& mat, std::ofstream& wfile, int numsamples, int numcols);

	// graph
	void retrieve_g_data(std::string alldatafile, int num_gnodes, int num_gfeats, int numrows=999999999);
	float**& get_g_state(int t);
	void gen_g_datafiles(std::string newfilename, std::string filename_test, \
		int ntimesteps_train, int ntimesteps_test, int num_gnodes, int num_gfeats);	// data generation
	void display_g();
	int count_g_states();

	// vector
	void retrieve_v_data(std::string datafile, int col);
	float& get_v_val(int idx);
	void gen_v_datafiles(std::string newfilename, std::string filename_test, \
		int ntimesteps_train, int ntimesteps_test);	// data generation
	void display_vdata() { displayVect(vdata, num_vtimesteps); }

};


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// DataStore for motion data (pos/vel/acc data)

class AVX_Data : public DataStore {
public:
	int Fchange_mode;
	float F = 0; float Fprev = 0; float Fgaslim; float Fbrakelim;
	float a = 0; float aprev = 0; float v = 0; float vprev = 0; float x = 0; float xprev = 0; float m;
	float v_max = 50; float v_min = 0;
	float dt; int ntimesteps;

	// resizeMatrix(mat, numr, numc);
	float** data; float** data_denoised;
	float** datawindow; int windowsize;
	float** datasegment;
public:
	AVX_Data(int ntimesteps, int windowsize, float Fgaslim_ = 5.0, float Fbrakelim_ = 5.0, float mass = 1.0, \
		float v_max_ = 50, float v_min_ = 0, float dt_ = 0.05);

	void simforce(std::uniform_real_distribution<float>& dist, std::default_random_engine& gen);
	void sim_vehicle_motion_forward(std::string filename, int ntimesteps, std::string denoised_file="");
	void retrieve_denoised_data(std::string filename);
	void retrieve_noised_data(std::string filename);
};

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// DataStore for noise sim data

// BRYAN TODO
// sliding window that returns a float*& (array) of the vals from t-n to t
class AVXwindow : public AVX_Data {
protected:
	//float* v_window;
	//float* a_window;
	//float* x_window;
	
	//LinkList* v_window;
	//LinkList* a_window;
	//LinkList* x_window;
	float** normalized_window; float normalize_var;
public:
	AVXwindow(int ntimesteps, int windowsize, float Fgaslim_=5.0, float Fbrakelim_=5.0, float mass=1.0, \
		float v_max_ = 50, float v_min_ = 0, float dt_ = 0.05) :
		AVX_Data(ntimesteps, windowsize, Fgaslim_, Fbrakelim_, mass, v_max_, v_min_, dt_) {}

	float**& get_input_normalized_window(int tstart, int tend, int dataCol);
	float**& get_true_normalized_window(int tstart, int tend, int dataCol);
	void normalize_window(); void denormalize_window(float**& preds);

	float**& get_input_window(int tstart, int tend, int dataCol);	// store window into [[_,_,_]]
	float**& get_true_window(int tstart, int tend, int dataCol);	// store window into [[_,_,_]]
	float**& get_input_segment(int tstart, int tend);	// x,v,a window
	float**& get_true_segment(int tstart, int tend);	// x,v,a window
	// void retrieve_denoised_data(std::string filename) {}
};
