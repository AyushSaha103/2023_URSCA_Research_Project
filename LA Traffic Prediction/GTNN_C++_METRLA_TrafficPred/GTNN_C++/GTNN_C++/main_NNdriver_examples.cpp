//
//
////+++++++++++++++++++++++++++++++++++++++++++++++
//// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
//// Undergraduate student at UC DAVIS, College of Engineering
//// DO NOT DISTRIBUTE
////+++++++++++++++++++++++++++++++++++++++++++++++
//
//#include "../GTNN/GTNN.h"
//#include "../Auto_Encoder/AutoEncoder.h"
//
//
////+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//// NN variations driver funcs
//
//// working well
//void run_NN_example() {
//	// NN variables
//	int numfeats = 3; int batchsize = 10; int numOutputNeurons = 1; int numLayers = 3;
//
//	// GENERATE train AND test datafiles
//	std::string datafile = "../Data/data.csv";
//	std::string datafile_test = "../Data/datatest.csv";
//	DataStore dstore(batchsize);
//	dstore.gen_xy_datafiles(datafile, datafile_test, 100, 100, numfeats, numOutputNeurons);		// if we're generating our data
//
//	// NORMALIZE train data
//	std::string norm_datafile = "../Data/data_norm.csv";
//	Normalizer myNorm(numfeats, numOutputNeurons);
//	int numsamples = myNorm.init_normalization_x_y(datafile);		// returns total # samples in dataset
//	int numbatches = numsamples / batchsize;						// rounds down to get # batches
//	myNorm.normalize_datafile(datafile, norm_datafile);
//
//
//	// INITIALIZE NETWORK n
//	int layer_sizes[] = { numfeats , 6, numOutputNeurons };
//	std::string layer_activations[] = { "relu", "leaky_relu", "N / A" };
//
//	NN n(numfeats, numOutputNeurons, batchsize, numLayers, layer_sizes, layer_activations);
//	n.cpy_normalizer(myNorm);					// save normalizer
//	// n.show_metadata();
//
//	// MANIPULATE TRAINING DATA ORGANIZATION
//	dstore.retrieve_xy_data(norm_datafile, numfeats, numOutputNeurons);
//	dstore.shuffle_xy_data();
//	n.cpy_DataStore(dstore);
//
//	// randomize network W's and B's
//	std::uniform_real_distribution<float> dist(-0.5, 0.5); std::default_random_engine gen; gen.seed(4);
//	n.randomize_WBs(dist, gen);
//	// n.show_metadata();
//
//	// TRAIN NN
//	float mincost = n.train(50000);
//	std::cout << "\n--------------------------\nMINIMAL COST AFTER TRAINING: " << mincost;
//
//
//	// NORMALIZE test data
//	std::string norm_datafile_test = "../Data/datatest_norm.csv";
//	myNorm.normalize_datafile(datafile_test, norm_datafile_test);
//	// TEST NN
//	dstore.retrieve_xy_data(norm_datafile_test, numfeats, numOutputNeurons);
//	n.test();
//}
//
//
//// currently constructed for test sequence (still not working right)
//void run_LSTM_example() {
//	int numnodes = 3, numfeats = 4; int ntimesteps = 500;
//
//	// GENERATE train AND test datafiles
//	DataStore dstore;
//	std::string gdata_train = "../Data/gdata.csv"; std::string gdata_test = "../Data/gdata_test.csv";
//	dstore.gen_g_datafiles(gdata_train, gdata_test, ntimesteps, ntimesteps, numnodes, numfeats);
//	dstore.retrieve_g_data(gdata_train, numnodes, numfeats);
//	// dstore.display_g();
//
//	//// NORMALIZE TRAIN DATA FILE
//	//std::string norm_gfile = "../Data/gdata_norm.csv";
//	//std::string norm_gfile_test = "../Data/gdata_norm_test.csv";
//	//Normalizer myNorm(numfeats);
//	//int numsamples = myNorm.init_normalization_g(gdata_train);
//	//myNorm.normalize_datafile_g(gdata_train, norm_gfile);
//
//	// dstore.retrieve_g_data(norm_gfile, numnodes, numfeats);		// GIVE DSTORE THE NORMALIZED DATA
//	// dstore.display_g();
//
//	// INIT. FEED-IN NN for LSTM
//	int numLayers = 3;
//	int layer_sizes[] = { 2, 5, 4 };
//	std::string layer_activations[] = { "relu", "leaky_relu","LSTM_custom" };// "sigmoid", "N / A" };
//	LSTM lstm(numnodes);
//	lstm.init_feedinNN(numLayers, layer_sizes, layer_activations);
//
//	// RANDOMIZE W's and B's
//	std::uniform_real_distribution<float> dist(-0.2, 0.5); std::default_random_engine gen; gen.seed(4);
//	lstm.randomize_WBs(dist, gen);
//
//	// LSTM SAVES COPY OF DATASTORE AND NORMALIZER
//	lstm.cpy_DataStore(dstore, 3);
//	//lstm.cpy_normalizer(myNorm);
//
//	// train lstm
//	lstm.train(90000);
//
//	// test lstm
//	dstore.retrieve_g_data(gdata_test, numnodes, numfeats);
//	dstore.display_g();
//	lstm.cpy_DataStore(dstore);
//	lstm.test();
//}
//
//// working decent (todo: simulate better data)
//void run_GTNN_example() {
//	// GENERATE train AND test datafiles
//	DataStore dstore;
//	std::string gdata_train = "../Data/gdata.csv"; std::string gdata_test = "../Data/gdata_test.csv";
//	int numnodes = 3, numfeats = 3; int ntimesteps = 19;
//	dstore.gen_g_datafiles(gdata_train, gdata_test, ntimesteps, ntimesteps, numnodes, numfeats);
//	dstore.retrieve_g_data(gdata_train, numnodes, numfeats);
//	//dstore.display_g();
//
//	//// NORMALIZE TRAIN DATA
//	//std::string norm_gfile = "../Data/gdata_norm.csv";
//	//std::string norm_gfile_test = "../Data/gdata_norm_test.csv";
//	//Normalizer myNorm(numfeats);
//	//int numsamples = myNorm.init_normalization_g(gdata_train);
//	//myNorm.normalize_datafile_g(gdata_train, norm_gfile);
//
//	//dstore.retrieve_g_data(norm_gfile, numnodes, numfeats);		// retrieve data from normalized file
//	////dstore.display_g();
//
//	// INITIALIZE GTNN's A-NET
//	int numLayersA = 2;
//	int layer_sizesA[] = { numfeats * 2, 1 };
//	std::string layer_activationsA[] = { "leaky_relu", "sigmoid" };
//
//	GTNN gtnn(numnodes, numfeats);
//	gtnn.init_anet(numLayersA, layer_sizesA, layer_activationsA);
//
//	// INITIALIZE GTNN's G-NET
//	int numLayersG = 2;
//	int layer_sizesG[] = { numfeats, numfeats };
//	std::string layer_activationsG[] = { "relu", "sigmoid" };
//	gtnn.init_gnet(numLayersG, layer_sizesG, layer_activationsG);
//
//	// randomize GTNN's A-NET & G-NET W's and B's
//	std::uniform_real_distribution<float> dist(-0.0, 0.5); std::default_random_engine gen; gen.seed(46);
//	gtnn.randomize_WBs(dist, gen);
//
//	// init MEM UPDATERS
//	int numLayersM = 3;
//	int layer_sizesM[] = { 2, 6, 4 };
//	std::string layer_activationsM[] = { "relu", "sigmoid", "LSTM_custom" };
//	gtnn.init_memupdaters(numLayersM, layer_sizesM, layer_activationsM, dist, gen);
//
//	// GIVE NORMALIZER * DATASTORE COPIES TO GTNN
//	gtnn.cpy_DataStore(dstore);
//	//gtnn.cpy_Normalizer(myNorm);
//
//	gtnn.train(60000);	// fixme
//}
//
//
//// FIX NAN(IND) prop
//void run_AutoEnc_example() {
//	// generate x, v, a measurement data
//	int ntimesteps = 300; int twindowsize = 20;
//	AVXwindow adstore(ntimesteps, twindowsize);
//	adstore.sim_vehicle_motion_forward("../Data/motion.csv", ntimesteps, "../Data/motion_denoised.csv");
//
//	// adstore.data[time idx][0] = x
//	// adstore.data[time idx][1] = v
//	// adstore.data[time idx][2] = a
//	adstore.retrieve_noised_data("../Data/motion.csv");
//	adstore.retrieve_denoised_data("../Data/motion_denoised.csv");
//
//	// NN variables
//	int numLayers = 5; int num_epochs = 60000;
//
//	// INITIALIZE NETWORK n
//	int layer_sizes[] = { twindowsize , 6, 2, 5, twindowsize };
//	std::string layer_activations[] = { "relu", "sigmoid","relu","relu", "N / A" };
//
//	// params: <numfeats, numOutputNeurons, batchsize, numLayers, layer_sizes, layer_activations>
//	AutoEnc ae(ntimesteps, twindowsize, numLayers, layer_sizes, layer_activations);
//	std::uniform_real_distribution<float> dist(-0.0, 0.5); std::default_random_engine gen; gen.seed(4);
//	ae.randomize_WBs(dist, gen);
//	ae.cpy_AVXwindow(adstore);
//	//n.show_metadata();
//
//	ae.train(num_epochs);
//	std::string outfile = "../Data/motion_predicted.csv";
//	ae.test(outfile);
//}
