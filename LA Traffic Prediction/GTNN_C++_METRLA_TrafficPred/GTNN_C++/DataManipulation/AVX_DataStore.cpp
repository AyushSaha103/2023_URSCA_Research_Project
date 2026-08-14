//
////+++++++++++++++++++++++++++++++++++++++++++++++
//// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
//// Undergraduate student at UC DAVIS
//// DO NOT DISTRIBUTE
////+++++++++++++++++++++++++++++++++++++++++++++++
//
//#include<iostream>
//#include<sstream>
//#include<fstream>
//#include<string>
//#include<random>
//#include "DataStore.h"
//
//AVX_Data::AVX_Data(int ntimesteps, int windowsize, float Fgaslim_, float Fbrakelim_, float mass, \
//	float v_max_, float v_min_, float dt_) {
//	this->windowsize = windowsize;
//	this->ntimesteps = ntimesteps;
//	resizeMatrix(datawindow, 1, windowsize);
//	resizeMatrix(datasegment, windowsize, 3);
//
//	resizeMatrix(data, ntimesteps, 3);
//	resizeMatrix(data_denoised, ntimesteps, 3);
//	m = mass; Fgaslim = Fgaslim_; Fbrakelim = Fbrakelim_; dt = dt_; v_max = v_max_; v_min = v_min_;
//}
//
//void AVX_Data::simforce(std::uniform_real_distribution<float>& dist, std::default_random_engine& gen) {
//	
//	if (Fchange_mode == 0) {		// F inc.
//		F += dist(gen);
//	}
//	if (Fchange_mode == 1) {		// F dec.
//		F -= 2.0*dist(gen);
//	}
//	if (Fchange_mode == 2) {		// F steady(ish)
//		F += 0.1 * (0.5*dist.max() - dist(gen));
//	}
//	if (rand() % 20 == 1) {		// switch F mode
//		Fchange_mode = rand() % 3;
//		//std::cout << Fchange_mode;
//		//std::cout << dist.max();
//	}
//	//if (std::abs(Fprev - F) > dFdt_max) {
//	//	if (Fchange_mode == 0)
//	//		F = Fprev + dFdt_max;
//	//	if (Fchange_mode == 1)
//	//		F = Fprev - dFdt_max;
//	//}
//}
//
//float addnoise(float val, int level, std::uniform_real_distribution<float>& dist, std::default_random_engine& gen) {
//	val += dist(gen) * level / 100;
//	return val;
//}
//
//float calc_dval(float& vprev, float& v, float dt=0.05) {
//	return dt * (v - vprev) / 2;
//}
//
//float RK4(float& aprev, float& a, float dt) {
//	float k1 = dt * aprev;
//	float k2 = dt * aprev;
//	float k3 = dt * a;
//	float k4 = dt * a;
//	return (k1 + (2 * k2) + (2 * k3) + k4) / 6;
//}
//
//void AVX_Data::sim_vehicle_motion_forward(std::string filename, int ntimesteps, std::string denoised_file) {
//	
//	float F_max = std::min(Fgaslim, Fbrakelim);		// max Force
//	std::uniform_real_distribution<float> Fdist(0.0, 0.05*F_max);
//
//	float atemp = -1; float xtemp = -1; float vtemp = -1;
//	std::uniform_real_distribution<float> xNoiseDist( -50.0, 50);
//	std::uniform_real_distribution<float> vNoiseDist(-v_max, v_max);
//	std::uniform_real_distribution<float> aNoiseDist(-Fgaslim / (m * 2), Fgaslim / (m*2));
//
//	std::default_random_engine gen; gen.seed(5);
//
//	// TODO: fix integration
//	for (int t = 0; t < ntimesteps; t++) {
//		
//		simforce(Fdist, gen);
//		F = std::min(Fgaslim, F);
//		F = std::max(-Fbrakelim, F);
//
//		a = F / m;
//		v += RK4(aprev, a, dt);
//
//		if (v < v_min) {		// don't let velocity be < 0
//			v = 0; F = 0; a = 0;
//		}
//		else if (v > v_max) {
//			v = v_max; F = Fprev; a = aprev;
//		}
//		
//		x += RK4(vprev, v, dt);
//
//		atemp = addnoise(a, 20, aNoiseDist, gen);
//		vtemp = addnoise (v, 3, vNoiseDist, gen);
//		xtemp = addnoise(x, 10, xNoiseDist, gen);
//
//		data[t][0] = xtemp; data[t][1] = vtemp; data[t][2] = atemp;
//		data_denoised[t][0] = x; data_denoised[t][1] = v; data_denoised[t][2] = a;
//		//data[t][0] = x; data[t][1] = v; data[t][2] = a;
//
//		aprev = a; vprev = v; xprev = x; Fprev = F;
//	}
//	// write noisy data xva data to file
//	std::ofstream wfile(filename);
//	wfile << "x,v,a\n";
//	mat_to_file(data, wfile, ntimesteps, 3);
//	wfile.close();
//
//	// write denoised data to another file
//	if (denoised_file != "") {
//		std::ofstream wfile2(denoised_file);
//		wfile2 << "x,v,a\n";
//		mat_to_file(data_denoised, wfile2, ntimesteps, 3);
//		wfile2.close();
//	}
//}
//
//void store_xva_data_into_mat(std::ifstream& rfile, float**& mat) {
//	// file read variables
//	std::string line, word;
//	std::getline(rfile, line);        // get the headers
//
//	// read each line
//	int col = 0; int row = 0;
//	while (getline(rfile, line)) {
//		std::istringstream s(line);
//		// get x features
//		for (col = 0; col < 3; col++) {
//			getline(s, word, ',');
//			mat[row][col] = std::stof(word);
//		}
//		row++;
//	}
//	rfile.close();
//
//	//displayMat(xdata, numsamples, xdataC, "xdata");
//	//displayMat(ydata, numsamples, ydataC, "ydata");
//}
//
//void AVX_Data::retrieve_denoised_data(std::string filename) {
//
//	std::ifstream rfile(filename);
//	this->numsamples = (-1) + std::count(std::istreambuf_iterator<char>(rfile),
//		std::istreambuf_iterator<char>(), '\n');
//	rfile.seekg(0);
//
//	store_xva_data_into_mat(rfile, data_denoised);
//	// displayMat(data_denoised, ntimesteps, 3);
//}
//
//void AVX_Data::retrieve_noised_data(std::string filename) {
//	std::ifstream rfile(filename);
//	rfile.seekg(0);
//	store_xva_data_into_mat(rfile, data);
//	// displayMat(data, ntimesteps, 3);
//}