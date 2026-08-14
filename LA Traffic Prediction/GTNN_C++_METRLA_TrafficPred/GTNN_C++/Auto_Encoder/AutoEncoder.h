//#pragma once
//
//#include "../NNstandard/NN.h"
//#include "../DataManipulation/DataStore.h"
//
//class AutoEnc {
//protected:
//	NN* n;
//	AVXwindow* avw;
//
//	int ntimesteps; int twindowsize;
//public:
//	AutoEnc(int ntimesteps_, int twindowsize_, \
//		int numLayers, int layer_sizes_[], std::string layer_activations[]);
//	void cpy_AVXwindow(AVXwindow& mw) { avw = &mw; }
//	float train(int num_epochs);
//	void randomize_WBs(std::uniform_real_distribution<float>& dist, std::default_random_engine gen) { n->randomize_WBs(dist, gen); };
//	float test(std::string writefilename);
//	void show_preds(float**& y_true, bool pause=false);
//	void send_preds_to_file(std::ofstream& wfile, float**& y_true);
//};
//
