#pragma once

//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++


#include "../NNstandard/NN.h"

class NN_LSTM : public NN {
public:
	NN_LSTM(int numfeats, int numOutputNeurons, int batchsize, int numLayers, int layer_sizes_[], std::string layer_activations[]) :
		NN(numfeats, numOutputNeurons, batchsize, numLayers, layer_sizes_, layer_activations) {
	}
	void predict_lstm(float**& h, float**& gstate, int gcol, float**& storage);
	float backward_lstm(float**& dcda_final);
	float**& get_layer0_dcda();
};

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// LSTM NN class

class LSTM {
private:
	int numnodes; int ntimesteps_train;  int g_datacol;
	NN_LSTM* newmem;
	DataStore* dstore; Normalizer* norm;
	//float** testmat;

	//float** xt;
	float** abcd; float** dcd_abcd;
	float** ht_1; float** ht; float** dcdht;
	float** ct_1; float** ct; float** dcdct;
	
public:
	LSTM(int numnodes);
	void init_feedinNN(int numLayers, int layer_sizes[], std::string layer_activations[]);
	void randomize_WBs(std::uniform_real_distribution<float>& dist, std::default_random_engine& gen);

	void cpy_DataStore(DataStore& dstore, int g_datacol=0);
	void cpy_normalizer(Normalizer& myNorm);

	void update_mem(); void reset_mem(); void show_preds(float**& ht_true, int t);
	void forward(float**& xt);
	float backward(float**& ht_true);
	void train(int numepochs);
	void test();

	// added functions to support GTNN
	void set_gdatacol(int gcol);
	void backward_using_derivative(float**& dcdht_given);
	void forward(float**& xt, float**& storage);
	void get_dcdxt(float**& storage);
	void update_mem(float**& dcdht_given);
};

