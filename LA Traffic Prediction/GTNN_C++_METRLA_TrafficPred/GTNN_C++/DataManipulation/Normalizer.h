#pragma once

//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS, College of Engineering
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++

#include "string"
#include "DataStore.h"


// Normalizer

class Normalizer {
private:
	float* mins_x_cols;
	float* x_col_ranges;
	float* y_mins;
	float* y_ranges;
	float normstart, normrange;

	int numfeats; int numOutputNeurons;
public:
	// xy matrix data
	Normalizer(int numfeats, int numOutputNeurons=0, float normstart=0.0, float normrange=1.0);
	int init_normalization_x_y(std::string filename);
	void denormalize_ybatch(float**& y, int batchsize, int numCols);
	void renormalize_ybatch(float**& y, int batchsize, int numCols);

	void normalize_datafile(std::string filename, std::string newfilename);
	void denorm_func(float& i, float& min, float& range);
	void norm_func(float& i, float& min, float& range);

	int countRows(std::string filename);

	// graph data
	int init_normalization_g(std::string filename);
	void normalize_datafile_g(std::string filename, std::string newfilename);
	void denormalize_gstate(float**& s, int numnodes, int numfeats);
	void renormalize_gstate(float**& s, int numnodes, int numfeats);
	void denormalize_gstate_col(float**& s, int numnodes, int numfeats, int col=0);
	void renormalize_gstate_col(float**& s, int numnodes, int numfeats, int col = 0);

};


