
//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS, College of Engineering
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++

#include "NN.h"

#include<cstring>
#include<iostream>
#include<sstream>
#include<string>
#include<fstream>
#include<random>
#include<unordered_map>

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// STATIC MEMBERS (y_true, random generator)
std::unordered_map<std::string, void(*)(float& a)> NN::activmap;
std::unordered_map<std::string, void(*)(float**& a, float**& storage, \
	int numR, int numC)> NN::deriv_activmap;

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// returns the final cost
float NN::train(int num_epochs) {
	// displayMat(ydataAll, numsamples, numOutputNeurons, "YDATA_ALL");
	// displayMat(xdataAll, numsamples, numfeats, "XDATA_ALL");

	float prev_cost = 9999999;
	float cost = 0;
	// by increasing min_cost_decline, we can make training halt earlier
	float minimal_cost_decline = 0.0;//0.001;
	int batchcnt = 0;
	int numbatches = dstore->count_xy_batches();

	// forwardprop / backprop cycle
	// train entire dataset, #cycles = num_epochs
	for (int i = 0; i < num_epochs; i++) {
		cost = 0;
		// for each batch in dataset
		for (int curbatch = 0; curbatch < numbatches; curbatch++) {

			// forward prop
			// input: xbatch, output: stored in last layer a
			forward(dstore->get_x_batch(curbatch));

			// backward prop (modifies all w, b matrices in network)
			// input: ybatch = the supposed y_true matrix
			// output: mean residual
			cost += backward(dstore->get_y_batch(curbatch));

			// rarely show predictions
			if (rand() % 500 == 1) {
				show_preds(dstore->get_y_batch(curbatch), false);
				//// dstore.display_xy();
				//for (int i = 0; i < numbatches; i++) {
				//	displayMat(dstore->get_x_batch(i), batchsize, numfeats, "x batch " + std::to_string(i));
				//}
				//for (int i = 0; i < numbatches; i++) {
				//	displayMat(dstore->get_y_batch(i), batchsize, numOutputNeurons, "y batch " + std::to_string(i), true);
				//}
			}
		}
		cost /= numbatches;
		//std::cout << "avg cost: " << cost << "\n";		// display cost

		// break if dcost < min_cost_decline
		if (std::abs(prev_cost - cost) < minimal_cost_decline)
			break;
		prev_cost = cost;
	}

	return cost;
}


float NN::test() {

	float cost = 0;
	int numbatches = dstore->count_xy_batches();

	// for each batch in dataset
	for (int curbatch = 0; curbatch < numbatches; curbatch++) {
		// forward prop
		// input: xbatch, output: stored in last layer a
		forward(dstore->get_x_batch(curbatch));

		show_preds(dstore->get_y_batch(curbatch), false);
		cost += get_MAE(dstore->get_y_batch(curbatch));
	}
	cost /= numbatches;

	std::cout << "\n--------------------------\nAVG COST AFTER TESTING: " << cost << "\n";
	return cost;
}


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// FORWARD PROP.

// perform computations (nexta = relu(a * w + b))
void NN::forwardMath(float**& a, float**& w, float**& b, float**& nexta, \
	std::string& activation, int numRowsA, int numColsA, int numColsW) {

	for (int i = 0; i < numRowsA; i++) {
		for (int j = 0; j < numColsW; j++) {
			// nexta = a . w
			nexta[i][j] = 0;
			for (int k = 0; k < numColsA; k++) {
				nexta[i][j] += a[i][k] * w[k][j];
			}
			// nexta += b
			nexta[i][j] += b[i][j];
			// nexta = relu(nexta)
			NN::activmap[activation]\
				(nexta[i][j]);
		}
	}
}

void NN::forward(float**& input) {
	NNlayer* ptr = layers[0];

	// a = input (passed parameter)
	copyMat(ptr->a, input, batchsize, ptr->numNeurons);

	// iterate thru network, applying same procedure
	while (ptr->next != nullptr) {
		forwardMath(ptr->a, ptr->w, ptr->b, ptr->next->a, ptr->activation, \
			batchsize, ptr->numNeurons, ptr->numNeuronsNext);

		ptr = ptr->next;
	}
}


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// BACK PROP.

float**& NN::get_preds() {
	return layers[numLayers - 1]->a;
}

// display all predictions for current batch
// display: y_pred (cols)			y_true(cols)
void NN::show_preds(float**& y_true, bool pause) {

	// print elems in last layer (y_pred) | y_true values
	NNlayer* ptr = layers[numLayers - 1];       // ptr to last layer

	//// denormalize y
	norm->denormalize_ybatch(y_true, batchsize, numOutputNeurons);
	norm->denormalize_ybatch(ptr->a, batchsize, numOutputNeurons);

	// DISPLAY TITLE
	// ypred[:,col] <-vs-> ytrue[:,col]      "col+1 <-vs-> "col+1
	std::cout << "\n\nPredictions\n";
	for (int i = 0; i < numOutputNeurons; i++)
		std::cout << "y_pred\t\ty_true\t";
	std::cout << "\n";

	// DISPLAY VALUES
	for (int i = 0; i < batchsize; i++) {
		for (int j = 0; j < numOutputNeurons; j++)
			std::cout << ptr->a[i][j] << "\t" << y_true[i][j] << "\t\t";
		std::cout << "\n";
	}
	std::cout << "\n";

	// display MAE error
	std::cout << "MAE: " << get_MAE(y_true) << "\n";

	if (pause) {
		std::string buf; std::getline(std::cin, buf);
	}
}

// NOTE: RSME function acting strange
float NN::get_RSME(float**& y_true) {

	NNlayer* ptr = layers[numLayers - 1];

	float avg_cost = 0;
	for (int i = 0; i < batchsize; i++) {
		for (int j = 0; j < ptr->numNeurons; j++) {
			avg_cost += std::pow(y_true[i][j] - ptr->a[i][j], 2) / (ptr->numNeurons * batchsize);
		}
	}
	//avg_cost /= ( ptr->numNeurons * batchsize );
	return std::sqrt(avg_cost);
}

// NOTE: R2 function acting strange
float NN::get_R2(float**& y_true) {

	NNlayer* ptr = layers[numLayers - 1];

	float RSS = 0; float avg_ytrue = 0;
	for (int i = 0; i < batchsize; i++) {
		for (int j = 0; j < ptr->numNeurons; j++) {
			// ptr->dcda[i][j] = 2.0 * (ptr->a[i][j] - y_true[i][j]);
			RSS += std::pow(y_true[i][j] - ptr->a[i][j], 2) / (ptr->numNeurons * batchsize);
			avg_ytrue += std::abs(y_true[i][j]) / (ptr->numNeurons * batchsize);
		}
	}
	float TSS = 0;
	for (int i = 0; i < batchsize; i++) {
		for (int j = 0; j < ptr->numNeurons; j++) {
			TSS += std::abs(y_true[i][j] - avg_ytrue) / (ptr->numNeurons * batchsize);
		}
	}
	//avg_ytrue /= (ptr->numNeurons * batchsize);
	//RSS /= (ptr->numNeurons * batchsize);
	std::cout << RSS << " " << TSS << " ";
	return (1 - RSS / TSS);
}


float NN::get_MAE(float**& y_true) {
	NNlayer* ptr = layers[numLayers - 1];

	float avg_cost = 0;
	for (int i = 0; i < batchsize; i++) {
		for (int j = 0; j < ptr->numNeurons; j++) {
			// ptr->dcda[i][j] = 2.0 * (ptr->a[i][j] - y_true[i][j]);
			avg_cost += std::abs(ptr->a[i][j] - y_true[i][j]);
		}
	}
	avg_cost /= (ptr->numNeurons * batchsize);
	return avg_cost;
}

float NN::backward(float**& y_true) {
	NNlayer* ptr = layers[numLayers - 1];

	// get final layer's dcda
	// and calc. avg_cost
	float avg_cost = 0;
	for (int i = 0; i < batchsize; i++) {
		for (int j = 0; j < ptr->numNeurons; j++) {
			ptr->dcda[i][j] = 2.0 * (ptr->a[i][j] - y_true[i][j]);
			avg_cost += 0.5 * std::abs(ptr->dcda[i][j]);
		}
	}
	avg_cost /= (ptr->numNeurons * batchsize);
	ptr = ptr->prev;

	// iterate thru prev. layers
	while (ptr != nullptr) {
		// next dadz = de_relu(next a)
		NN::deriv_activmap[ptr->activation]\
			(ptr->next->a, ptr->next->dadz, batchsize, ptr->next->numNeurons);
		// next dcdz = next dcda .* next dadz
		matmul_elemwise(ptr->next->dcda, ptr->next->dadz, ptr->next->dcdz, batchsize, ptr->next->numNeurons);
		// dcdw = a_T * next dcdz
		mat_T_times_mat(ptr->a, ptr->next->dcdz, ptr->dcdw, batchsize, ptr->numNeurons, ptr->next->numNeurons);
		// dcdb = next dcdz
		copyMat(ptr->dcdb, ptr->next->dcdz, batchsize, ptr->next->numNeurons);
		// dcda = next dcdz * w_T
		mat_times_mat_T(ptr->next->dcdz, ptr->w, ptr->dcda, batchsize, ptr->next->numNeurons, ptr->numNeurons);
		// modify w, b matrices (for cur. layer)
		ptr->changeWB();

		ptr = ptr->prev;
	}
	return avg_cost;
}


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// INIT.

// init
NN::NN(int numfeats, int numOutputNeurons, int batchsize, int numLayers, int layer_sizes_[], std::string layer_activations[]) {
	this->batchsize = batchsize;
	this->numOutputNeurons = numOutputNeurons;
	this->numfeats = numfeats;

	this->layersizes = layer_sizes_;
	this->numLayers = numLayers;

	layers = (NNlayer**) malloc(numLayers * sizeof(NNlayer*));
	layers[0] = new NNlayer(batchsize, layer_sizes_[0], layer_activations[0]);
	for (int i = 1; i < numLayers; i++) {
		layers[i] = new NNlayer(batchsize, layer_sizes_[i], layer_activations[i]);
		link_layers(layers[i - 1], layers[i]);
	}
	init_network_matrices();
	init_layer_activations();
}

void NN::init_network_matrices() {
	for (int i = 0; i < numLayers; i++) {
		layers[i]->initMatrices();
	}
}
void NN::init_layer_activations() {
	NN::activmap["relu"] = relu;
	NN::activmap["sigmoid"] = sigmoid;
	NN::activmap["tanh_"] = tanh_;
	NN::activmap["leaky_relu"] = leaky_relu;

	NN::deriv_activmap["relu"] = deriv_relu;
	NN::deriv_activmap["sigmoid"] = deriv_sigmoid;
	NN::deriv_activmap["tanh_"] = deriv_tanh_;
	NN::deriv_activmap["leaky_relu"] = deriv_leaky_relu;
}

void NN::randomize_WBs(std::uniform_real_distribution<float>& dist, std::default_random_engine& gen) {
	for (int i = 0; i < numLayers; i++) {
		layers[i]->randomizeWB(dist, gen);
	}
}

// debug
void NN::show_metadata() {
	std::cout << "numLayers: " << numLayers << "\n";
	std::cout << "batchsize: " << batchsize << "\n";
	std::cout << "numOutputNeurons: " << numOutputNeurons << "\n";
	std::cout << "numfeats: " << numfeats << "\n";

	std::cout << "\n\nLAYER INFO: \n";
	NNlayer* ptr = layers[0];
	for (int i = 0; i < numLayers; i++) {
		std::cout << "layer[: " << i << "]:\n";
		std::cout << "size: " << layersizes[i] << "\n";
		std::cout << "neurons: " << ptr->numNeurons << "\n";
		std::cout << "neuronsNext: " << ptr->numNeuronsNext << "\n";
		std::cout << "activation: " << ptr->activation << "\n";
		displayMat(ptr->a, batchsize, ptr->numNeurons, "layer " + std::to_string(i) + "a");
		displayMat(ptr->b, batchsize, ptr->numNeuronsNext, "layer " + std::to_string(i) + "b");
		displayMat(ptr->w, ptr->numNeurons, ptr->numNeuronsNext, "layer " + std::to_string(i) + "w");
		std::cout << "\n\n";
		ptr = ptr->next;
	}

}



void NN::store_metadata(std::string& storage) {
	storage = "";
	storage += "std::to_string(numLayers): " + std::to_string(numLayers) + "\n";
	storage += "std::to_string(batchsize): " + std::to_string(batchsize) + "\n";
	storage += "numOutputNeurons: " + std::to_string(numOutputNeurons) + "\n";
	storage += "numfeats: " + std::to_string(numfeats) + "\n";

	storage += "\n\nLAYER INFO: \n";
	NNlayer* ptr = layers[0];
	for (int i = 0; i < numLayers; i++) {
		storage += "layer[: " + std::to_string(i) + "]:\n";
		storage += "size: " + std::to_string(layersizes[i]) + "\n";
		storage += "neurons: " + std::to_string(ptr->numNeurons) + "\n";
		storage += "neuronsNext: " + std::to_string(ptr->numNeuronsNext) + "\n";
		storage += "activation: " + ptr->activation + "\n";
		append_mat_to_str(ptr->a, batchsize, ptr->numNeurons, "layer " + std::to_string(i) + "a", storage);
		append_mat_to_str(ptr->b, batchsize, ptr->numNeuronsNext, "layer " + std::to_string(i) + "b", storage);
		append_mat_to_str(ptr->w, ptr->numNeurons, ptr->numNeuronsNext, "layer " + std::to_string(i) + "w", storage);
		storage += "\n\n";
		ptr = ptr->next;
	}
}