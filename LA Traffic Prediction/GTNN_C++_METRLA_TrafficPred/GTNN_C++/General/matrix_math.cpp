

//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS, College of Engineering
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++


// MATIX MATH FUNCTIONS (FOR NN BACKPROP)
#include<cmath>

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// GTNN

float get_avg_value(float**& mat, int numR, int numC) {
	float avg = 0;
	for (int i = 0; i < numR; i++) {
		for (int j = 0; j < numC; j++) {
			avg += mat[i][j] / (numR * numC);
		}
	}
	return avg;
}

void vect_times_mat(float*& a, float**& b, float*& storage, int numColsA, int numColsB) {
	for (int i = 0; i < numColsA; i++) {
		for (int j = 0; j < numColsB; j++) {
			storage[j] = a[i] * b[i][j];
		}
	}
}
void apply_softmax(float*& vec, float*& storage, int numElems) {
	float esum = 0;
	float e_ = 2.7182;
	for (int i = 0; i < numElems; i++) {
		esum += pow(e_, vec[i]);
	}
	//std::cout << "ESUM " << esum << "\n";
	for (int i = 0; i < numElems; i++) {
		storage[i] = pow(e_, vec[i]) / esum;
		//std::cout << pow(e_, vec[i]) / esum << " "; //storage[i] << "  ";
	}
	//std::cout << "\n";
}

// dadz = de_softmax(a)		// square matrices
void de_softmax(float*& a, float**& dadz, int numElems) {
	for (int i = 0; i < numElems; i++) {
		for (int j = 0; j < numElems; j++) {
			if (i == j)
				dadz[i][j] = a[i] * (1 - a[i]);
			else
				dadz[i][j] = -a[j] * a[i];
		}
	}
}



float matsub(float**& x, float**& y, float**& storage, \
	int numRowsA, int numColsA) {

	float avg_diff = 0.0;

	for (int i = 0; i < numRowsA; i++) {
		for (int j = 0; j < numColsA; j++) {
			storage[i][j] = x[i][j] - y[i][j];
			avg_diff += storage[i][j] / (numRowsA * numColsA);
			//avg_diff += std::pow(storage[i][j], 2) / (numRowsA * numColsA);
		}
	}
	//return std::sqrt(avg_diff);
	return avg_diff;
}

void matmul(float**& x, float**& y, float**& storage, \
	int numRowsA, int numColsA, int numColsB) {

	for (int i = 0; i < numRowsA; i++) {
		for (int j = 0; j < numColsB; j++) {
			// nexta = a . w
			storage[i][j] = 0;
			for (int k = 0; k < numColsA; k++) {
				storage[i][j] += x[i][k] * y[k][j];
			}
		}
	}
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// LSTM


// LSTM
float relu(float& a, bool returnval) {
	if (a < 0) return 0;
	return a;
}
float sigmoid(float& a, bool returnval) {
	return (1.0 / (1 + pow(2.718281828, -a)));
}

float deriv_relu(float& a, bool returnval) {
	return (float)(a > 0);
}
float deriv_sigmoid(float& a, bool returnval) {
	return (1.0 * (1 - a));
}

void deriv_sigmoid(float& a) {
	a = (1.0 * (1 - a));
}

float tanh_(float& a, bool returnval) {
	float e_ = 2.71828;
	return (std::pow(e_, a) - std::pow(e_, -a)) / (std::pow(e_, a) + std::pow(e_, -a));
}

float deriv_tanh_(float& a, bool returnval) {
	return (1 - std::pow(a, 2));
}
void deriv_tanh_(float& a) {
	a = (1 - std::pow(a, 2));
}
// get avg difference between 2 matrices
float get_avg_diff(float**& a, float**& b, int numR, int numC) {
	float adif = 0;
	for (int i = 0; i < numR; i++) {
		for (int j = 0; j < numC; j++) {
			adif += std::abs(a[i][j] - b[i][j]);
		}
	}
	return adif / (numR * numC);
}


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// NNstandard

void mat_times_mat_T(float**& x, float**& y, float**& storage, \
	int numRowsA, int numColsA, int numRowsB) {

	for (int i = 0; i < numRowsA; i++) {
		for (int j = 0; j < numRowsB; j++) {
			// nexta = a . w
			storage[i][j] = 0;
			for (int k = 0; k < numColsA; k++) {
				storage[i][j] += x[i][k] * y[j][k];
			}
		}
	}
}

void mat_T_times_mat(float**& x, float**& y, float**& storage, \
	int numRowsA, int numColsA, int numColsB) {

	for (int i = 0; i < numColsA; i++) {
		for (int j = 0; j < numColsB; j++) {
			// nexta = a . w
			storage[i][j] = 0;
			for (int k = 0; k < numRowsA; k++) {
				storage[i][j] += x[k][i] * y[k][j];
			}
		}
	}
}

void mat_times_scaler(float**& x, float y, float**& storage, int numR, int numC) {
	for (int i = 0; i < numR; i++) {
		for (int j = 0; j < numC; j++) {
			storage[i][j] = x[i][j] * y;
		}
	}
}
void matmul_elemwise(float**& x, float**& y, float**& storage, int numR, int numC) {
	for (int i = 0; i < numR; i++) {
		for (int j = 0; j < numC; j++) {
			storage[i][j] = x[i][j] * y[i][j];
		}
	}
}


// a := relu(a)
void relu(float& a) {
	if (a < 0) a = 0;
}
void sigmoid(float& a) {
	a = 1 / (1 + pow(2.718281828, -a));
}
void tanh_(float& a) {
	float e_ = 2.71828;
	a = (std::pow(e_, a) - std::pow(e_, -a)) / (std::pow(e_, a) + std::pow(e_, -a));
}
void leaky_relu(float& a) {
	if (a < 0) a = 0.3 * a;
}

// deriv_relu(vec) ~~> storage
void deriv_relu(float**& a, float**& storage, int numR, int numC) {
	for (int i = 0; i < numR; i++) {
		for (int j = 0; j < numC; j++) {
			storage[i][j] = (float)(a[i][j] > 0);
		}
	}
}
void deriv_sigmoid(float**& a, float**& storage, int numR, int numC) {
	for (int i = 0; i < numR; i++) {
		for (int j = 0; j < numC; j++) {
			storage[i][j] = a[i][j] * (1 - a[i][j]);
		}
	}
}
void deriv_tanh_(float**& a, float**& storage, int numR, int numC) {
	for (int i = 0; i < numR; i++) {
		for (int j = 0; j < numC; j++) {
			storage[i][j] = (1 - std::pow(a[i][j], 2));
		}
	}
}
void deriv_leaky_relu(float**& a, float**& storage, int numR, int numC) {
	for (int i = 0; i < numR; i++) {
		for (int j = 0; j < numC; j++) {
			storage[i][j] = 1;
			if (a[i][j] < 0) storage[i][j] = 0.3;
		}
	}
}



//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// GTNN_2 (using adj list)

void vect_sum_scaled_src(float*& dest, float*& src, float scale, int numElems) {
	for (int i = 0; i < numElems; i++) {
		dest[i] += src[i] * scale;
	}
}
void reset_vect(float*& vec, int numElems) {
	for (int i = 0; i < numElems; i++) {
		vec[i] = 0;
	}
}


float get_RSME(float**& m1, float**& m2, int rows, int cols) {
	
	float val = 0;
	for (int i = 0; i < rows; i++) {
		for (int c = 0; c < cols; c++) {
			val += std::pow(m1[i][c] - m2[i][c], 2) / (rows * cols);
		}
	}
	return std::sqrt(val);
}