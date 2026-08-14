#pragma once

//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++

#include "../LSTM/LSTM.h"

class NN_GAT : public NN {
public:
	NN_GAT(int numfeats, int numOutputNeurons, int batchsize, int numLayers, int layer_sizes_[], std::string layer_activations[]) :
		NN(numfeats, numOutputNeurons, batchsize, numLayers, layer_sizes_, layer_activations) {
	}

	float predict(float**& x1, int col, float**& storage);
	void predict(float**& h, float**& gstate, int gcol, float**& storage);
	void backward_using_derivative(float**& dcda_final);
	float**& get_layer0_dcda();
	float predict(float*& h1, float*& h2);
	void backward(float*& input1, float*& input2, float& dcda_final, float scale_lrate = 1.0);
	float backward(float**& y_true);
};

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// graph NN class

class GAT {
private:
	// std::unordered_map<int, float**> alldata;
	int ntimesteps;
	int numnodes; int numfeats;

	float** dcd_adjs; float** dcd_adj; float** dcdadj_pre; float** dadj_dadjpre;
	float** adj; float** adj_pre; float** adjs;

	NN_GAT* anet; NN_GAT* gnet;
	DataStore* dstore; Normalizer* norm;

public:

	GAT(int numnodes, int numfeats);
	void init_matrices();
	void init_anet(int numLayers, int layer_sizes_[], std::string layer_activations[]);
	void init_gnet(int numLayers, int layer_sizes_[], std::string layer_activations[]);
	void randomize_WBs(std::uniform_real_distribution<float>& dist, std::default_random_engine& gen);

	void cpy_DataStore(DataStore& ds) { dstore = &ds; ntimesteps = dstore->count_g_states(); }
	void cpy_Normalizer(Normalizer& nor) { norm = &nor; }

	void forward(float**& state_in);
	float backward(float**& state_in, float**& strue);
	void train(int num_epochs);

	void show_preds(float**& strue);
};


// // working decent (todo: simulate better data)
// void run_GAT_example() {
	// // GENERATE train AND test datafiles
	// DataStore dstore;
	// std::string gdata_train = "../Data/gdata.csv"; std::string gdata_test = "../Data/gdata_test.csv";
	// int numnodes = 3, numfeats = 3; int ntimesteps = 19;
	// dstore.gen_g_datafiles(gdata_train, gdata_test, ntimesteps, ntimesteps, numnodes, numfeats);
	// dstore.retrieve_g_data(gdata_train, numnodes, numfeats);
	// //dstore.display_g();

	// //// NORMALIZE TRAIN DATA
	// //std::string norm_gfile = "../Data/gdata_norm.csv";
	// //std::string norm_gfile_test = "../Data/gdata_norm_test.csv";
	// //Normalizer myNorm(numfeats);
	// //int numsamples = myNorm.init_normalization_g(gdata_train);
	// //myNorm.normalize_datafile_g(gdata_train, norm_gfile);

	// //dstore.retrieve_g_data(norm_gfile, numnodes, numfeats);		// retrieve data from normalized file
	// ////dstore.display_g();

	// // INITIALIZE GAT's A-NET
	// int numLayersA = 2;
	// int layer_sizesA[] = { numfeats * 2, 1 };
	// std::string layer_activationsA[] = { "leaky_relu", "sigmoid" };

	// GAT GAT(numnodes, numfeats);
	// GAT.init_anet(numLayersA, layer_sizesA, layer_activationsA);

	// // INITIALIZE GAT's G-NET
	// int numLayersG = 2;
	// int layer_sizesG[] = { numfeats, numfeats };
	// std::string layer_activationsG[] = { "relu", "sigmoid" };
	// GAT.init_gnet(numLayersG, layer_sizesG, layer_activationsG);

	// // randomize GAT's A-NET & G-NET W's and B's
	// std::uniform_real_distribution<float> dist(-0.0, 0.5); std::default_random_engine gen; gen.seed(46);
	// GAT.randomize_WBs(dist, gen);

	// // GIVE NORMALIZER * DATASTORE COPIES TO GAT
	// GAT.cpy_DataStore(dstore);
	// //GAT.cpy_Normalizer(myNorm);

	// GAT.train(60000);
// }
