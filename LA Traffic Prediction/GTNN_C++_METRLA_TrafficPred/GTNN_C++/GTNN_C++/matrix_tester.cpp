//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// matrix tester

#include<sstream>
#include<string>
#include<fstream>

float** read_mat_from_file(std::string filename) {

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
	float** mat; resizeMatrix(mat, numR, numC);
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
	return mat;
}
void normalize_mat(float**& mat, int R, int C) {
	float maxx = -99999;;
	for (int i = 0; i < R; i++) {
		for (int j = 0; j < C; j++) {
			maxx = std::max(maxx, mat[i][j]);
		}
	}
	for (int i = 0; i < R; i++) {
		for (int j = 0; j < C; j++) {
			mat[i][j] /= maxx;
		}
	}
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// DRIVER CODE

int main() {

	float** inp = read_mat_from_file("inp.txt");
	float** pred = read_mat_from_file("inp.txt");
	float** dcdpred = read_mat_from_file("inp.txt");
	float** dzdinp = read_mat_from_file("inp.txt");
	float** dcdinp = read_mat_from_file("inp.txt");
	float** s = read_mat_from_file("s.txt");

	normalize_mat(inp, 4, 4);
	normalize_mat(s, 4, 4);

	displayMat(inp, 4, 4, "inp");
	displayMat(s, 4, 4, "s");

	for (int g = 0; g < 400; g++) {
		for (int i = 0; i < 4; i++) {
			apply_softmax(inp[i], pred[i], 4);
		}
		matsub(pred, s, dcdpred, 4, 4);
		for (int i = 0; i < 4; i++) {
			de_softmax(inp[i], dzdinp, 4);
			vect_times_mat(dcdpred[i], dzdinp, dcdinp[i], 4, 4);
		}
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				inp[i][j] -= 0.001 * dcdinp[i][j];
			}
		}
		displayMat(dcdinp, 4, 4, "dcdinp");
	}
}
