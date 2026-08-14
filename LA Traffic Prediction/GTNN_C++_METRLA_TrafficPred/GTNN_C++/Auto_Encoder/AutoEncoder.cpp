//
//
//#include "AutoEncoder.h"
//
//AutoEnc::AutoEnc(int ntimesteps_, int twindowsize_, \
//	int numLayers, int layer_sizes_[], std::string layer_activations[]) {
//
//	int batchsize = 1;
//	int numfeats = twindowsize_, numOutputNeurons = twindowsize_;
//	n = new NN(numfeats, numOutputNeurons, batchsize, numLayers, layer_sizes_, layer_activations);
//	ntimesteps = ntimesteps_; twindowsize = twindowsize_;
//}
//
//float AutoEnc::train(int num_epochs) {
//	int dataCol = 1;
//	
//	float prev_mincost=999999;
//	float mincost = 0.0;
//
//	int ntrainsteps = ntimesteps / 2;
//
//
//	for (int i = 0; i < num_epochs; i++) {
//		mincost = 0; prev_mincost = mincost;
//		for (int t = twindowsize; t < ntrainsteps; t+=twindowsize) {
//			// displayMat(avw->get_input_normalized_window(t - twindowsize, t, 1), 1, twindowsize, "in[");
//			// displayMat(avw->get_true_normalized_window(t - twindowsize, t, 1), 1, twindowsize, "true", false);
//			
//			// forward prop y_input (normalized)
//			float**& y_in_norm = avw->get_input_normalized_window(t - twindowsize, t, dataCol);
//			// displayMat(y_in_norm, 1, twindowsize, "yin");
//			n->forward( y_in_norm );
//
//			// back prop y_true (normalized)
//			float**& y_true_norm = avw->get_true_normalized_window(t - twindowsize, t, dataCol);
//			// displayMat(y_true_norm, 1, twindowsize, "ytruw norm");
//			mincost += n->backward(y_true_norm);
//
//			// display predictions
//			if (rand() % 5000 == 0) {
//				float**& y_true = avw->get_true_window(t - twindowsize, t, dataCol);
//				// displayMat(y_true, 1, twindowsize, "ytruw");
//				show_preds(y_true);		// denormalizes y_pred
//			}
//		}
//		mincost /= (ntrainsteps - twindowsize);
//		mincost = std::min(mincost, prev_mincost);
//	}
//	return mincost;
//}
//
//// display predictions
//// DENORMALIZES Y_PRED, leaves y_true as is
//void AutoEnc::show_preds(float**& y_true, bool pause) {
//	float**& preds = n->get_preds();
//	avw->denormalize_window(preds);
//
//	float MAE = 0;
//	std::cout << "\nPREDICTIONS\ny_pred\t\ty_true\n";
//	for (int i = 0; i < twindowsize; i++) {
//		std::cout << preds[0][i] << "\t\t" << y_true[0][i] << "\n";
//		MAE += preds[0][i] - y_true[0][i];
//	}
//	MAE /= twindowsize;
//	std::cout << "\nMAE: " << MAE << "\n";
//
//	if (pause) {
//		wait();
//	}
//}
//
//
//float AutoEnc::test(std::string writefilename) {
//	int dataCol = 1;
//	
//	std::ofstream wfile(writefilename);
//	wfile << "pred,true\n";
//	
//	int nteststeps = ntimesteps / 2;
//	float totcost = 0;
//	for (int t = (nteststeps + twindowsize); t < ntimesteps; t += twindowsize) {
//		// forward prop. y_input (normalized)
//		float**& y_in_norm = avw->get_input_normalized_window(t - twindowsize, t, dataCol);
//		n->forward(y_in_norm);
//
//		// get y_true window
//		float**& y_true = avw->get_true_window(t - twindowsize, t, dataCol);
//
//		// display predictions (AND DENORMALIZE y_pred)
//		show_preds(y_true);
//		send_preds_to_file(wfile, y_true);
//	}
//	wfile.close();
//	totcost /= (nteststeps - twindowsize);
//	return totcost;
//
//}
//
//// write y_pred and y_true to file (NO NORMALIZATION OR DENORM)
//void AutoEnc::send_preds_to_file(std::ofstream& wfile, float**& y_true) {
//	float**& preds = n->get_preds();
//	//avw->denormalize_window(preds);
//	
//	std::string line;
//	for (int i = 0; i < twindowsize; i++) {
//		std::ostringstream ostream;		// ostream for each node
//		ostream << preds[0][i] << ',' << y_true[0][i] << '\n';
//		line = ostream.str();
//		line[line.size() - 1] = '\n';
//		wfile << line;
//	}
//}
