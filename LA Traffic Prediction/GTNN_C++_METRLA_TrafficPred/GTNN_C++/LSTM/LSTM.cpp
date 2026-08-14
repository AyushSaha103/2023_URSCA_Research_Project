
//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS, College of Engineering
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++


#include "LSTM.h"
#include <random>

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ADDED FUNCS FOR TESTING

void read_mat_from_file(float**& mat, std::string filename) {

	std::ifstream rfile(filename);
	std::string line, word;

	// count numRows
	int numR = std::count(std::istreambuf_iterator<char>(rfile),
		std::istreambuf_iterator<char>(), '\n');
	rfile.seekg(0);

	// count numCols
	std::getline(rfile, line);
	std::istringstream s(line);
	int numC = 0;
	while (getline(s, word, ' ')) { numC++; }

	// resize matrix
	// float** mat; resizeMatrix(mat, numR, numC);
	rfile.seekg(0);

	// read each line
	int col = 0; int row = 0;
	while (getline(rfile, line)) {
		std::istringstream s(line);
		// get x features
		for (col = 0; col < numC; col++) {
			getline(s, word, ' ');
			mat[row][col] = std::stof(word);
		}
		row++;
	}
	rfile.close();
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// LSTM INIT

LSTM::LSTM(int numnodes) : numnodes(numnodes) {
	resizeMatrix(abcd, numnodes, 4);
	resizeMatrix(ht, numnodes, 1);
	resizeMatrix(dcd_abcd, numnodes, 4);
	resizeMatrix(ht_1, numnodes, 1);
	resizeMatrix(dcdht, numnodes, 1);
	resizeMatrix(ct_1, numnodes, 1);
	resizeMatrix(ct, numnodes, 1);
	resizeMatrix(dcdct, numnodes, 1);

	fillMat(ht_1, numnodes, 1, 0.0);
	fillMat(ct_1, numnodes, 1, 0.0);
}

// initialize the input-feedin NN
void LSTM::init_feedinNN(int numLayers, int layer_sizes[], std::string layer_activations[]) {
	newmem = new NN_LSTM(2, 4, numnodes, numLayers, layer_sizes, layer_activations);
}

void LSTM::randomize_WBs(std::uniform_real_distribution<float>& dist, std::default_random_engine& gen) {
	newmem->randomize_WBs(dist, gen);
}

void LSTM::cpy_DataStore(DataStore& dstore, int g_datacol) {
	this->dstore = &dstore;
	this->g_datacol = g_datacol;
	ntimesteps_train = (dstore.count_g_states()) - 1;
}
void LSTM::cpy_normalizer(Normalizer& myNorm) { this->norm = &myNorm; }

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void LSTM::reset_mem() {
	fillMat(ht_1, numnodes, 1, 0.0);
	fillMat(ct_1, numnodes, 1, 0.0);
}
void LSTM::update_mem() {
	copyMat(ht_1, ht, numnodes, 1);
	copyMat(ct_1, ct, numnodes, 1);
}
void LSTM::update_mem(float**& dcdht_given) {
	for (int i = 0; i < numnodes; i++) {
		ht_1[i][0] = ht[i][0] - dcdht_given[i][g_datacol];
	}
}

void LSTM::forward(float**& xt) {
	newmem->predict_lstm(ht_1, xt, g_datacol, abcd);
	for (int n = 0; n < numnodes; n++) {
		// ct = ct_1 * a + ( b * c )
		ct[n][0] = ct_1[n][0] * abcd[n][0] + (abcd[n][1] * abcd[n][2]);
		// ht = tanh(ct) * d
		ht[n][0] = tanh_(ct[n][0], true) * abcd[n][3];
	}
}
float LSTM::backward(float**& ht_true) {
	float MAE = 0;
	for (int n = 0; n < numnodes; n++) {
		dcdht[n][0] = ht[n][0] - ht_true[n][0];
		MAE += std::abs(dcdht[n][0]);
		dcd_abcd[n][3] = dcdht[n][0] * tanh_(ct[n][0], true);

		dcdct[n][0] = dcdht[n][0] * abcd[n][3];
		dcd_abcd[n][0] = dcdct[n][0] * ct_1[n][0];
		dcd_abcd[n][1] = dcdct[n][0] * abcd[n][2];
		dcd_abcd[n][2] = dcdct[n][0] * abcd[n][1];
	}
	newmem->backward_lstm(dcd_abcd);
	return MAE;
}

void LSTM::show_preds(float**& ht_true, int t) {
	// std::cout << debug_metadata;

	std::cout << "\nPREDICTIONS @ t=" << std::to_string(t) << "\nht_pred\tht_true\n";
	for (int i = 0; i < numnodes; i++) {
		std::cout << ht[i][0] << "\t" << ht_true[i][0] << "\n";
	}
	std::cout << "\nMAE: " << get_avg_diff(ht, ht_true, numnodes, 1) << "\n\n";
}

// todo: make methods neater (layer prop in NN_LSTM)
// fix nan(ind) sudden issue when training
// make good train/test data
// test model with memory updates
void LSTM::train(int numepochs) {
	std::string debug_metadata = "";

	for (int i = 0; i < numepochs; i++) {
		reset_mem();		// reset ht_1, ct_1
		for (int j = 0; j < ntimesteps_train; j++) {
			// forward prop: ht = LSTM ( NN ( ht_1 | xt[:,gcol] ) )
			float**& xt = dstore->get_g_state(j);
			forward(xt);

			// back prop
			float**& ht_true = dstore->get_g_state(j+1);
			backward(ht_true);

			update_mem();		// update ht, ct

			// show preds
			if (rand() % 5000 == 0) {
				show_preds(ht_true, j);
			}
		}

	}
 }

void LSTM::test() {
	std::cout << "\n\nTEST ROUTINE\n";

	reset_mem();		// reset ht_1, ct_1
	for (int j = 0; j < ntimesteps_train; j++) {
		// forward prop: ht = LSTM ( NN ( ht_1 | xt[:,gcol] ) )
		float**& xt = dstore->get_g_state(j);
		forward(xt);
		float**& ht_true = dstore->get_g_state(j+1);
		show_preds(ht_true, j);
		update_mem();
	}
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// specifial functions to support GTNN

void LSTM::set_gdatacol(int gcol) { this->g_datacol = gcol; }

// storage[:,gcol] = LSTM( xt )
void LSTM::forward(float**& xt, float**& storage) {
	newmem->predict_lstm(ht_1, xt, g_datacol, abcd);
	for (int n = 0; n < numnodes; n++) {
		// ct = ct_1 * a + ( b * c )
		ct[n][0] = ct_1[n][0] * abcd[n][0] + (abcd[n][1] * abcd[n][2]);
		// ht = tanh(ct) * d
		storage[n][g_datacol] = tanh_(ct[n][0], true) * abcd[n][3];
	}
}
// back prop. lstm, when given dcdht
void LSTM::backward_using_derivative(float**& dcdht_given) {
	for (int n = 0; n < numnodes; n++) {
		dcd_abcd[n][3] = dcdht_given[n][g_datacol] * tanh_(ct[n][0], true);

		dcdct[n][0] = dcdht_given[n][g_datacol] * abcd[n][3];
		dcd_abcd[n][0] = dcdht_given[n][g_datacol] * ct_1[n][0];
		dcd_abcd[n][1] = dcdht_given[n][g_datacol] * abcd[n][2];
		dcd_abcd[n][2] = dcdht_given[n][g_datacol] * abcd[n][1];
	}
	newmem->backward_lstm(dcd_abcd);
	// NOTE: "dcdxt[:,0]" = newmem->getlayer0_dcda() [:,1]
}

// copy over: storage[:,gcol] = "dcdxt[:,0]" = newmem->getlayer0_dcda() [:,1]
void LSTM::get_dcdxt(float**& storage) {
	float**& dcd_ht_xt = newmem->get_layer0_dcda();
	for (int n = 0; n < numnodes; n++) {
		storage[n][g_datacol] = dcd_ht_xt[n][1];
	}
}

