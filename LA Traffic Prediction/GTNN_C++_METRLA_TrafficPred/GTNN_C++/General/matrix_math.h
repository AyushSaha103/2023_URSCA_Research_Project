#pragma once

//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// matrix math

// standard NN
void mat_times_mat_T(float**& x, float**& y, float**& storage, \
	int numRowsA, int numColsA, int numRowsB);
void mat_T_times_mat(float**& x, float**& y, float**& storage, \
	int numRowsA, int numColsA, int numColsB);
void mat_times_scaler(float**& x, float y, float**& storage, int numR, int numC);
void matmul_elemwise(float**& x, float**& y, float**& storage, int numR, int numC);

void relu(float& a);
void sigmoid(float& a);
void tanh_(float& a);
void leaky_relu(float& a);
void deriv_relu(float**& a, float**& storage, int numR, int numC);
void deriv_sigmoid(float**& a, float**& storage, int numR, int numC);
void deriv_tanh_(float**& a, float**& storage, int numR, int numC);
void deriv_leaky_relu(float**& a, float**& storage, int numR, int numC);

// GTNN
float get_avg_value(float**& mat, int numR, int numC);
float matsub(float**& x, float**& y, float**& storage, \
	int numRowsA, int numColsA);		// returns error metric (MAE, RSME, R2)
void matmul(float**& x, float**& y, float**& storage, \
	int numRowsA, int numColsA, int numColsB);

void vect_times_mat(float*& a, float**& b, float*& storage, int numColsA, int numColsB);
void apply_softmax(float*& vec, float*& storage, int numElems);
void de_softmax(float*& a, float**& dadz, int numElems);


// LSTM
float relu(float& a, bool returnval);
float sigmoid(float& a, bool returnval);
float deriv_relu(float& a, bool returnval);
float deriv_sigmoid(float& a, bool returnval);

void deriv_sigmoid(float& a);
float tanh_(float& a, bool returnval);
float deriv_tanh_(float& a, bool returnval);
void deriv_tanh_(float& a);
// get avg dif btwn 2 matrices
float get_avg_diff(float**& a, float**& b, int numR, int numC);

// GTNN_2 (using adj list)
void vect_sum_scaled_src(float*& dest, float*& src, float scale, int numElems);
void reset_vect(float*& vec, int numElems);
float get_RSME(float**& m1, float**& m2, int rows, int cols);


