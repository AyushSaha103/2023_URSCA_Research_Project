#pragma once

//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++

#include "../LSTM/LSTM.h"
#include<map>

class NN_GTNN : public NN {
public:
	NN_GTNN(int numfeats, int numOutputNeurons, int batchsize, int numLayers, int layer_sizes_[], std::string layer_activations[]) :
		NN(numfeats, numOutputNeurons, batchsize, numLayers, layer_sizes_, layer_activations) {
	}

	float predict(float**& x1, int col, float**& storage);
	void predict(float**& h, float**& gstate, int gcol, float**& storage);
	void backward_using_derivative(float**& dcda_final);
	float**& get_layer0_dcda();
	float predict(float*& h1, float*& h2);
	void backward(float*& input1, float*& input2, float& dcda_final, float scale_lrate = 1.0);
	float backward(float**& y_true);

	friend class GTNN;
};

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// graph NN class

class adjlistrow {
private:
	//int n;		(index n is implied via index of adjlist row in adjlist)
	int consize;
	int* con;
	float* att;
public:
	friend class GTNN;
	void init_connections(int n) {
		con = (int*)malloc(n * sizeof(int));
		att = (float*)malloc(n * sizeof(float));
	}
	void set_connections(int* convec, float*& attvec, int consizee) {
		consize = consizee;
		init_connections(consize);
		std::memcpy(con, convec, sizeof(int) * consize);
		std::memcpy(att, attvec, sizeof(float) * consize);
	}
};


class GTNN {
private:
	// std::unordered_map<int, float**> alldata;
	int ntimesteps;
	int numnodes; int numfeats;

	float** dcd_adjs; float** dcd_adj; float** dcdadj_pre; float** dadj_dadjpre;
	float** adj; float** adj_pre; float** adjs; float** adjs_u;

	NN_GTNN* anet; NN_GTNN* gnet;
	DataStore* dstore; Normalizer* norm;
	LSTM** memupdaters;

public:

	GTNN(int numnodes, int numfeats);
	void init_matrices();
	void init_anet(int numLayers, int layer_sizes_[], std::string layer_activations[]);
	void init_gnet(int numLayers, int layer_sizes_[], std::string layer_activations[]);
	void init_memupdaters(int numLayers, int layer_sizes[], std::string layer_activations[], \
		std::uniform_real_distribution<float>& dist, std::default_random_engine& gen);
	void randomize_WBs(std::uniform_real_distribution<float>& dist, std::default_random_engine& gen);

	void cpy_DataStore(DataStore& ds) { dstore = &ds; ntimesteps = dstore->count_g_states(); }
	void cpy_Normalizer(Normalizer& nor) { norm = &nor; }

	void reset_all_mem(); void update_all_mem();
	void forward(float**& state_in);
	float backward(float**& state_in, float**& strue);
	void train(int num_epochs);

	void show_preds(float**& strue, int t);

	//~~~~~~~~~~ADDED for METRLA traffic pred~~~~~~~~~~~
private:
	adjlistrow* adjlist;
public:
	void init_adjlist(std::string fname);
	void disp_adjlist();
	void randomize_WBs2(std::uniform_real_distribution<float>& dist, std::default_random_engine& gen);
	void forward2(float**& state_in);
	float backward2(float**& state_in, float**& strue);
	void train2(int num_epochs);
	void show_preds2(float**& strue, int t, float upscale=1);
	void test2(int ntsteps_pred=1);

	// memory updates
	void update_all_mem(float**& dcdht);

};

