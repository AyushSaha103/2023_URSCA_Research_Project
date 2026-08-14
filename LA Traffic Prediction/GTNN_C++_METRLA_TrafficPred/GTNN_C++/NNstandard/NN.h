#pragma once

//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++

#define learning_rate 0.001


#include<iostream>
#include<unordered_map>
#include<string>
#include<sstream>
#include<fstream>
#include<random>

#include "../General/matrix_math.h"
#include "../DataManipulation/Normalizer.h"

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class NNlayer; class NN;

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// NNlayer CLASS
class NNlayer {
public:
	int numNeurons, numNeuronsNext; int batchsize;
	NNlayer* next=nullptr; NNlayer* prev=nullptr;

	// matrices
	float** a;		// [batchsize][numNeurons]
	float** w;		// [numNeurons][numNeuronsNext]
	float** b;		// [batchsize][numNeuronsNext]

	float** dcdw;		// w size
	float** dcdb;		// b size
	float** dcda; float** dcdz; float** dadz;		// a size

	std::string activation;			// specifies which activation to apply in forward prop.

public:
	// INITIALIZATION
	friend class NN; friend class NNspecial; friend class NN_LSTM; friend class NN_GTNN;
	//friend class AutoEnc;
	friend void link_layers(NNlayer* ptr1, NNlayer* ptr2);

	NNlayer(int batchsize_, int numNeurons_, std::string activation_);

	void initMatrices();
	void randomizeWB(std::uniform_real_distribution<float>&dist, std::default_random_engine& gen);
	void changeWB(); void changeWB(float& scale_lrate);
};

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class NN {
protected:
	NNlayer** layers;
	std::string* activations;
	int* layersizes; int numLayers;
	int batchsize; int numfeats; int numOutputNeurons;

	Normalizer* norm; DataStore* dstore;
public:
	void show_metadata(); void store_metadata(std::string& storage);
	void cpy_normalizer(Normalizer& myNorm) { this->norm = &myNorm;  }
	void cpy_DataStore(DataStore& ds) { this->dstore = &ds;  }

	NN(int numfeats, int numOutputNeurons, int batchsize, int numLayers, int layer_sizes_[], std::string layer_activations[]);
	void init_layer_activations(); void init_network_matrices();
	void randomize_WBs(std::uniform_real_distribution<float>& dist, std::default_random_engine& gen);

	// FORWARD/ BACKWARD PROP.
	void forwardMath(float**& a, float**& w, float**& b, float**& nexta, \
		std::string& activation, int numRowsA, int numColsA, int numColsW);
	void forward(float**& input);
	void show_preds(float**& y_true, bool pause = false);
	float get_MAE(float**& y_true); float get_R2(float**& y_true); float get_RSME(float**& y_true);
	float backward(float**& y_true);
	float train(int num_epochs=80000);
	float test();
	float**& get_preds();

	// STATIC MEMBER VARIABLES
	static std::unordered_map<std::string, void(*)(float& a)> activmap;
	static std::unordered_map<std::string, void(*)(float**& a, float**& storage, \
		int numR, int numC)> deriv_activmap;
};


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

