 
//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS, College of Engineering
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++


#include "GAT.h"

#include<iostream>
#include<unordered_map>
#include<random>

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// INIT

// main init variables
GAT::GAT(int numnodes, int numfeats) {
	this->numnodes = numnodes;
	this->numfeats = numfeats;
	init_matrices();
}
// INITIALIZE matrix sizes
void GAT::init_matrices() {
	resizeMatrix(adj_pre, numnodes, numnodes);
	resizeMatrix(adj, numnodes, numnodes);
	resizeMatrix(adjs, numnodes, numfeats);

	resizeMatrix(dcd_adj, numnodes, numnodes);
	resizeMatrix(dadj_dadjpre, numnodes, numnodes);
	resizeMatrix(dcdadj_pre, numnodes, numnodes);

}
// rangomize A-NET & G-NET W's and B's
void GAT::randomize_WBs(std::uniform_real_distribution<float>& dist, std::default_random_engine& gen) {
	anet->randomize_WBs(dist, gen);
	gnet->randomize_WBs(dist, gen);
}
// initialize A-NET
void GAT::init_anet(int numLayers, int layer_sizes_[], std::string layer_activations[]) {
	int numfeats_anet = numfeats * 2;
	anet = new NN_GAT( numfeats_anet , 1, 1, numLayers, layer_sizes_, layer_activations);
}
// init G-NET
void GAT::init_gnet(int numLayers, int layer_sizes_[], std::string layer_activations[]) {
	gnet = new NN_GAT(numfeats, numfeats, numnodes, numLayers, layer_sizes_, layer_activations);
}


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// FORWARD PROP

void GAT::forward(float**& state_in) {
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
	// G-NET forward props on adjs matrix
	gnet->forward(adjs);
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// BACK PROP

float GAT::backward(float**& state_in, float**& strue) {
	float graphpred_cost = gnet->backward(strue);		// cost for graph-state predictions

	// dcd_adj = dcdadjs * state_in
	float**& dcdadjs = gnet->get_layer0_dcda();
	mat_times_mat_T(dcdadjs, state_in, dcd_adj, numnodes, numfeats, numnodes);

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

void GAT::show_preds(float**& strue) {
	float**& preds = gnet->get_preds();

	std::cout << "\nPREDICTIONS\n";
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

void GAT::train(int num_epochs) {
	for (int p = 0; p < num_epochs; p++) {
		for (int t = 0; t < ntimesteps-1; t++) {
			// get state_in and state_true matrices
			float**& sin = dstore->get_g_state(t);
			float**& strue = dstore->get_g_state(t+1);
			forward(sin);		// forward prop.

			float MAE = backward(sin, strue);		// back prop.
			if (rand() % 5000 == 1) show_preds(strue);
			
			// wait();
		}
	}
}
