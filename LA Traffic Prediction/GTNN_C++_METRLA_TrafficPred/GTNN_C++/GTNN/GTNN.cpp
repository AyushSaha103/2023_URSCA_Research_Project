 
//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS, College of Engineering
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++


#include "GTNN.h"

#include<iostream>
#include<unordered_map>
#include<random>

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// INIT

// main init variables
GTNN::GTNN(int numnodes, int numfeats) {
	this->numnodes = numnodes;
	this->numfeats = numfeats;
	init_matrices();
}
// INITIALIZE matrix sizes
void GTNN::init_matrices() {
	resizeMatrix(adj_pre, numnodes, numnodes);
	resizeMatrix(adj, numnodes, numnodes);
	resizeMatrix(adjs, numnodes, numfeats);
	resizeMatrix(adjs_u, numnodes, numfeats);

	resizeMatrix(dcd_adj, numnodes, numnodes);
	resizeMatrix(dcd_adjs, numnodes, numfeats);
	resizeMatrix(dadj_dadjpre, numnodes, numnodes);
	resizeMatrix(dcdadj_pre, numnodes, numnodes);

}
// rangomize A-NET & G-NET W's and B's
void GTNN::randomize_WBs(std::uniform_real_distribution<float>& dist, std::default_random_engine& gen) {
	anet->randomize_WBs(dist, gen);
	gnet->randomize_WBs(dist, gen);
}
// initialize A-NET
void GTNN::init_anet(int numLayers, int layer_sizes_[], std::string layer_activations[]) {
	int numfeats_anet = numfeats * 2;
	anet = new NN_GTNN( numfeats_anet , 1, 1, numLayers, layer_sizes_, layer_activations);
}
// init G-NET
void GTNN::init_gnet(int numLayers, int layer_sizes_[], std::string layer_activations[]) {
	gnet = new NN_GTNN(numfeats, numfeats, numnodes, numLayers, layer_sizes_, layer_activations);
}
// initialize ALL memupdaters
void GTNN::init_memupdaters(int numLayers, int layer_sizes[], std::string layer_activations[], \
	std::uniform_real_distribution<float>& dist, std::default_random_engine& gen) {

	memupdaters = (LSTM**)malloc(numfeats * sizeof(LSTM*));
	for (int f = 0; f < numfeats; f++) {
		memupdaters[f] = new LSTM(numnodes);
		memupdaters[f]->init_feedinNN(numLayers, layer_sizes, layer_activations);
		memupdaters[f]->randomize_WBs(dist, gen);
		memupdaters[f]->set_gdatacol(f);
	}
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// FORWARD PROP

void GTNN::forward(float**& state_in) {
	for (int n = 0; n < numnodes; n++) {
		for (int c = 0; c < numnodes; c++) {
			// attention = A-NET ( state_in[c] | state_in[n] )
			adj_pre[n][c] = anet->predict(state_in[c], state_in[n]);
		}
		// attentions = softmax ( attentions )
		apply_softmax(adj_pre[n], adj[n], numnodes);
	}
	// adjs = adj * state_in
	matmul(adj, state_in, adjs, numnodes, numnodes, numfeats);

	// adjs_updated[:, feat] = memupdaters[f]->forward ( adjs[:,feat] )
	for (int f = 0; f < numfeats; f++) {
		memupdaters[f]->forward(adjs, adjs_u);
	}

	// G-NET forward props on adjs matrix
	gnet->forward(adjs_u);
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// BACK PROP

float GTNN::backward(float**& state_in, float**& strue) {
	float graphpred_cost = gnet->backward(strue);		// cost for graph-state predictions

	float**& dcd_adjsu = gnet->get_layer0_dcda();
	for (int f = 0; f < numfeats; f++) {
		// backprop memupdaters[f]
		memupdaters[f]->backward_using_derivative(dcd_adjsu);
		// dcd_adjs[:,feat] = memupdaters[f]->dcdxt[:,0]
		memupdaters[f]->get_dcdxt(dcd_adjs);
	}

	// dcd_adj = dcdadjs * state_in
	mat_times_mat_T(dcd_adjs, state_in, dcd_adj, numnodes, numfeats, numnodes);

	// float attention_cost = get_avg_value(dcd_adj, numnodes, numnodes);		// cost for attention predictions
	float learnrate_scale = (1 / std::pow(numnodes, 2));
	for (int n = 0; n < numnodes; n++) {

		de_softmax(adj[n], dadj_dadjpre, numnodes);
		vect_times_mat(dcd_adj[n], dadj_dadjpre, dcdadj_pre[n], numnodes, numnodes);

		for (int c = 0; c < numnodes; c++) {
			anet->backward(state_in[c], state_in[n], dcdadj_pre[n][c], learnrate_scale);
		}
	}
	return graphpred_cost;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// TRAIN/TEST ROUTINES

void GTNN::show_preds(float**& strue, int t) {
	float**& preds = gnet->get_preds();

	std::cout << "\nPREDICTIONS @ t=" << std::to_string(t) << "\n";
	for (int j = 0; j < numfeats; j++) {
		std::cout << "feat" + std::to_string(j) + "_in" << " " << \
			"feat" + std::to_string(j) + "_true" << "\t";
	} std::cout << "\n";
	for (int i = 0; i < numnodes; i++) {
		for (int j = 0; j < numfeats; j++) {
			std::cout << preds[i][j] << " " << strue[i][j] << "\t";
		}
		std::cout << "\n";
	}
	std::cout << "\n\n";
	std::cout << "MAE: " << get_avg_diff(preds, strue, numnodes, numfeats) << "\n";
}

void GTNN::reset_all_mem() {
	for (int f = 0; f < numfeats; f++) {
		memupdaters[f]->reset_mem();
	}
}
void GTNN::update_all_mem() {
	for (int f = 0; f < numfeats; f++) {
		memupdaters[f]->update_mem();
	}
}
void GTNN::update_all_mem(float**& dcdht) {
	for (int f = 0; f < numfeats; f++) {
		memupdaters[f]->update_mem(dcdht);
	}
}

void GTNN::train(int num_epochs) {
	for (int p = 0; p < num_epochs; p++) {
		reset_all_mem();
		for (int t = 0; t < ntimesteps-1; t++) {
			// get state_in and state_true matrices
			float**& sin = dstore->get_g_state(t);
			float**& strue = dstore->get_g_state(t+1);

			// forward prop./back prop
			forward(sin);		// forward prop.
			float MAE = backward(sin, strue);		// back prop.

			// update_all_mem();

			// display predictions
			if (rand() % 5000 == 0) show_preds(strue, t);
		}
	}
}
