
//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS, College of Engineering
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++

//#include<cstring>
//#include<cmath>
//#include<random>
//#include<string>
#include<iostream>
#include<unordered_map>
#include<sstream>
#include<fstream>
#include "DataStore.h"

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// GRAPH DATA retrieval
void DataStore::retrieve_g_data(std::string alldatafile, int num_gnodes, int num_gfeats, int numrows) {
	this->num_gnodes = num_gnodes; this->num_gfeats = num_gfeats;
	
	gdata.clear();

	// file read variables
	std::ifstream rfile(alldatafile);
	std::string line, word;
	int t = 0; int tprev = -1; int n = 0; int f = 0;		// incremental variables
	int time_subtract_term= 0;

	std::getline(rfile, line);        // get the headers

	// for each line in file (t, node, state)
	while (getline(rfile, line)) {
		if (n >= numrows) break;

		// t = <time idx>
		std::istringstream s(line);
		getline(s, word, ','); t = std::stoi(word);

		// subtract t by time_subtract_term
		if (n == 0 && t > 0)
			time_subtract_term = t;
		t -= time_subtract_term;

		// init. alldata[t], size: [numnodes][numfeats]
		if (t != tprev) {
			gdata[t] = nullptr;
			n = 0; tprev = t;
			resizeMatrix(gdata[t], num_gnodes, num_gfeats);
		}

		// alldata[t] = { node state }
		f = 0;
		while (getline(s, word, ',')) {
			if (word == "TRUE") word = "1";
			else if (word == "False") word = "0";
			gdata[t][n][f] = std::stof(word);
			f++;
		}
		// move to next node
		n++;

	}
	num_gtimesteps = gdata.size();
	std::cout << num_gtimesteps << " TOTAL STATES\n";

	//displayMat(gdata[0], num_gnodes, num_gfeats);
	//displayMat(gdata[1], num_gnodes, num_gfeats);
	rfile.close();

}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// sending graph data matrix

float**& DataStore::get_g_state(int t) {
	return gdata[t];
}

void DataStore::display_g() {
	std::cout << "GRAPH DATA\nt\tstate matrix\n";
	for (int t = 0; t < num_gtimesteps; t++) {
		for (int n = 0; n < num_gnodes; n++) {
			std::cout << t << "\t";
			for (int f = 0; f < num_gfeats; f++) {
				std::cout << gdata[t][n][f] << "\t";
			} std::cout << "\n";
		} std::cout << "\n";
	}
}

int DataStore::count_g_states() {
	return gdata.size();
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// generate Graph Data

float add_noise(std::uniform_real_distribution<float>& dist, std::default_random_engine& gen) {
	return dist(gen);
}

void update_state_matrix(std::uniform_real_distribution<float>& dist, std::default_random_engine& gen, \
	float**& state, int numnodes, int numfeats, int x) {

	for (int i = 0; i < numnodes; i++) {
		for (int j = 0; j < numfeats; j++) {
			state[i][j] = std::abs( sin(0.06 * x) ) ;// +add_noise(dist, gen);
		}
	}

}
void send_state_mat_to_file(std::ofstream& wfile, float**& temp_state, int numnodes, int numfeats, int t) {
	std::string bline="";		// big line for state
	std::string sline = "";		// small line for single node

	for (int i = 0; i < numnodes; i++) {
		std::ostringstream ostream;		// ostream for each node
		ostream << t << ",";

		for (int j = 0; j < numfeats; j++) {		// get node state (one line)
			ostream << temp_state[i][j] << ",";
		}
		sline = ostream.str();
		sline[sline.size() - 1] = '\n';
		bline += sline;		// append small line to big line
	}
	wfile << bline;		// write big line (as matrix) to file
}

void write_gdata_headers(std::ofstream& wfile, int numfeats) {
	wfile << "t,";
	std::ostringstream ostream;
	std::string line;
	for (int i = 0; i < numfeats; i++) {
		ostream << "feat" << (i + 1) << ",";
	}
	line = ostream.str();
	line[line.size() - 1] = '\n';
	wfile << line;
}

void write_g_data(std::string filename, int numfeats, int numnodes, int ntimesteps, \
	std::uniform_real_distribution<float>& dist, std::default_random_engine& gen) {

	std::ofstream wfile(filename);
	write_gdata_headers(wfile, numfeats);

	// init. starting state
	float** temp_state; resizeMatrix(temp_state, numnodes, numfeats);
	for (int i = 0; i < numnodes; i++) {
		//getRandomFloats(dist, gen, temp_state[i], numfeats);
		//////////std::fill(temp_state[i], temp_state[i] + numfeats, 0.0);
	}
	// continuously update state matrix, send to file
	for (int t = 0; t < ntimesteps; t++) {
		update_state_matrix(dist, gen, temp_state, numnodes, numfeats, t+1);
		send_state_mat_to_file(wfile, temp_state, numnodes, numfeats, t);
	}
	wfile.close();
}

void DataStore::gen_g_datafiles(std::string newfilename, std::string filename_test, \
	int ntimesteps_train, int ntimesteps_test, int numnodes, int numfeats) {

	// randomization variables
	std::uniform_real_distribution<float> dist(-26.0,50);
	std::default_random_engine gen; gen.seed(32);

	write_g_data(newfilename, numfeats, numnodes, ntimesteps_train, dist, gen);
	write_g_data(filename_test, numfeats, numnodes, ntimesteps_test, dist, gen);

}


