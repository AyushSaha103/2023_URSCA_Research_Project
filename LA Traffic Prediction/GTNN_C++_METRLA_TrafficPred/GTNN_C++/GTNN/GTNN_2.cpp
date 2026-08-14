
#include "GTNN.h"

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// FORWARD PROP

void GTNN::forward2(float**& state_in) {
	// adjs
	for (int n = 0; n < numnodes; n++) {
		// adjs[n] = 0's
		reset_vect(adjs[n], numfeats);
		for (int c = 0; c < adjlist[n].consize; c++) {
			// adjs[n] += state[n] .* att;
			vect_sum_scaled_src(adjs[n], state_in[n], adjlist[n].att[c], numfeats);
		}
	}

	// adjs_updated[:, feat] = memupdaters[f]->forward ( adjs[:,feat] )
	for (int f = 0; f < numfeats; f++) {
		memupdaters[f]->forward(adjs, adjs_u);
	}

	// G-NET forward props on adjs matrix
	gnet->forward(adjs_u);
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// BACK PROP

float GTNN::backward2(float**& state_in, float**& strue) {
	float graphpred_cost = gnet->backward(strue);		// cost for graph-state predictions

	float**& dcd_adjsu = gnet->get_layer0_dcda();
	for (int f = 0; f < numfeats; f++) {
		// backprop memupdaters[f]
		memupdaters[f]->backward_using_derivative(dcd_adjsu);
		// dcd_adjs[:,feat] = memupdaters[f]->dcdxt[:,0]
		memupdaters[f]->get_dcdxt(dcd_adjs);
	}

	return graphpred_cost;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// TRAIN/TEST ROUTINES


void GTNN::train2(int num_epochs) {
	for (int p = 0; p < num_epochs; p++) {
		reset_all_mem();
		for (int t = 0; t < ntimesteps - 1; t++) {
			// get state_in and state_true matrices
			float**& sin = dstore->get_g_state(t);
			float**& strue = dstore->get_g_state(t + 1);

			// forward prop./back prop
			forward2(sin);		// forward prop.
			float MAE = backward2(sin, strue);		// back prop.

			//update_all_mem(gnet->layers[0]->dcda);
			//update_all_mem();

			// display predictions
			if (rand() % 1000 == 0) show_preds2(strue, t, 70);
		}
	}
}


void GTNN::test2(int t_into_fut) {

	// note: ntimesteps = dstore->gdata.size()
	float** results; resizeMatrix(results, (dstore->gdata.size()) * numnodes, numfeats * 2 + 1);
	fillMat(results, dstore->gdata.size() * numnodes, numfeats * 2 + 1, -10);
	reset_all_mem();

	int pred_count = 0;
	for (int t = 0; t < ntimesteps - t_into_fut; t+=t_into_fut) {
		// get state_in and state_true matrices
		float**& sin = dstore->get_g_state(t);
		float**& strue = dstore->get_g_state(t + t_into_fut);

		// forward prop./back prop
		forward2(sin);		// forward prop.
		for (int i = 1; i < t_into_fut; i++)
			forward2(gnet->layers[gnet->numLayers - 1]->a);


		float MAE = backward2(sin, strue);		// back prop.

		//update_all_mem(gnet->layers[0]->dcda);
		//update_all_mem();

		// display predictions
		//show_preds2(strue, t, 70);
		float**& preds = gnet->get_preds();
		std::cout << "RSME: " << get_RSME(preds, strue, numnodes, numfeats) << "\n";

		// save results
		for (int n = 0; n < numnodes; n++) {
			results[(n*ntimesteps) + pred_count][0] = t;

			for (int f = 0; f < numfeats; f++) {
				//preds_vs_true[n][2*f] = strue[n][f];
				//preds_vs_true[n][2*f+1] = preds[n][f];

				results[(n * ntimesteps) + pred_count][1] = strue[n][f] * 70;
				results[(n * ntimesteps) + pred_count][2] = preds[n][f]* 70;
			}
		}
		pred_count++;
	}
	//displayMat(results, dstore->gdata.size() * numnodes, numfeats * 2 + 1, "results");
	mat_to_file(results, "../Data/results.csv", dstore->gdata.size() * numnodes, numfeats * 2 + 1);
}


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ADJ LIST init

// read from adj mat stored in file, indexed by node#
void GTNN::init_adjlist(std::string fname) {
	// init adj list & temp row data readers
	adjlist = (adjlistrow*)malloc(numnodes * sizeof(adjlistrow));
	int* convec; convec = (int*)malloc(numnodes * sizeof(int));
	float* attvec; resizeVector(attvec, numnodes);

	// open file
	std::string line, word; int numlines = 0;
	std::ifstream rfile(fname);

	int c = 0, r = 0;
	float cv = 0; int cidx = 0;

	while (getline(rfile, line)) {
		cidx = 0;
		std::istringstream s(line);

		for (c = 0; c < numnodes; c++) {
			getline(s, word, ',');

			cv = std::stof(word);
			if (cv != 0) {
				convec[cidx] = c;
				attvec[cidx] = cv;
				cidx+=1;
			}
			
		}
		//std::cout << "\n";
		adjlist[r].set_connections(convec, attvec, cidx);

		r++;
	}
	rfile.close();

}



void GTNN::disp_adjlist() {
	for (int r = 0; r < numnodes; r++) {
		for (int c = 0; c < adjlist[r].consize; c++) {
			std::cout << adjlist[r].con[c] << " " << adjlist[r].att[c] << " ";
		}
		std::cout << "\n";
	}
}

void GTNN::show_preds2(float**& strue, int t, float upscale) {
	float**& preds = gnet->get_preds();

	std::cout << "\nPREDICTIONS @ t=" << std::to_string(t) << "\n";
	for (int j = 0; j < numfeats; j++) {
		std::cout << "feat" + std::to_string(j) + "_in" << " " << \
			"feat" + std::to_string(j) + "_true" << "\t";
	} std::cout << "\n";
	for (int i = 0; i < numnodes; i++) {
		for (int j = 0; j < numfeats; j++) {
			std::cout << upscale* preds[i][j] << " " << upscale* strue[i][j] << "\t";
		}
		std::cout << "\n";
	}
	std::cout << "\n\n";
	std::cout << "MAE: " << get_avg_diff(preds, strue, numnodes, numfeats) << "\n";
}

// init. G-NET WBs
void GTNN::randomize_WBs2(std::uniform_real_distribution<float>& dist, std::default_random_engine& gen) {
	//anet->randomize_WBs(dist, gen);
	gnet->randomize_WBs(dist, gen);
}


//// TODO FINISH ME
//intflt* GTNN::dijikstra(int start, int end) {
//	int cur = start;
//	std::map<int, intflt> consid, visited;
//
//	// MAIN LOOP SEGMENT (TODO ADD TO ME)
//	while (cur != end) {
//		for (int c = 0; c < adjlist[cur].consize; c++) {
//			// if cur not in consid or visited
//			if (consid.find(c) == consid.end() and visited.find(c) == visited.end()) {
//				consid[c].i = cur;
//				consid[c].f = adjlist[cur].att[c];
//			}// if cur in consid, but new cur has better travel cost
//			else if (consid[c].f > adjlist[cur].att[c]) {
//				consid[c].i = cur;
//				consid[c].f = adjlist[cur].att[c];
//			}
//		}
//		// move closest node to visited (GOOD)
//		auto closen = min_element(consid.begin(), consid.end(),
//			[](const auto& l, const auto& r) { return l.second.f < r.second.f; });
//		visited[closen->first].i = closen->second.i;
//		visited[closen->first].f = closen->second.f;
//		consid.erase(closen);
//	}
//	// TODO TRACE SHORTEST PATH
//	// TODO FIND WAY TO REVERSE ADJ LIST so it's (from->to node)
//}
//
////+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//// read from adj list stored in file, indexed by node#
//void GTNN::init_adjlist2(std::string fname) {
//	
//	// init adj list & temp row data readers
//	adjlist = (adjlistrow*)malloc(numnodes * sizeof(adjlistrow));
//	int* convec; convec = (int*)malloc(numnodes * sizeof(int));
//	float* attvec; resizeVector(attvec, numnodes);
//
//	// open file
//	std::string line, word; int numlines = 0;
//	std::ifstream rfile(fname);
//
//	int c = 0, r = 0;
//	float cv = 0; int cidx = 0;
//
//	while (getline(rfile, line)) {
//		cidx = 0;
//		std::istringstream s(line);
//
//		// strings stored as: c|cv, c|cv
//		while (getline(s, word, ',')) {
//			// // string representation method 1		// (c, cv)
//			// if (cidx % 2 == 0)
//				// c = std::stoi(word);
//			// else
//				// cv = std::stof(word);
//			// convec[(int) cidx/2] = c;
//			// attvec[(int) cidx/2] = cv;
//			
//			
//			// string representation method 2		// (c|cv)
//			std::istringstream ss(word);
//			getline(s, word, '|');
//			convec[cidx] = std::stoi(word);
//			getline(s, word, '|');
//			attvec[cidx] = std::stof(word);
//			
//			cidx++;
//		}
//		
//		//std::cout << "\n";
//		adjlist[r].set_connections(convec, attvec, cidx);
//
//		r++;
//	}
//	rfile.close();
//
//}
