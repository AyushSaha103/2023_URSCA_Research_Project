
//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS, College of Engineering
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++

#include "DataStore.h"
#include<iostream>
#include<string>
#include<fstream>
#include<sstream>
#include<cstring>
#include<random>

void mat_to_file(float**& mat, std::string filename, int numsamples, int numcols, std::string headers) {
	std::ofstream wfile(filename);
	wfile << headers << "\n";

	std::string line;
	for (int i = 0; i < numsamples; i++) {
		std::ostringstream ostream;		// ostream for each node
		for (int j = 0; j < numcols; j++) {
			ostream << mat[i][j] << ',';
		}
		line = ostream.str();
		line[line.size() - 1] = '\n';
		wfile << line;
	}
	wfile.close();
}

void randomizeMat(float**& mat, int numR, int numC, float randmax, float randmin) {
	for (int i = 0; i < numR; i++) {
		for (int j = 0; j < numC; j++) {
			mat[i][j] = ( randmax * (rand() % 100) / 100 ) - randmin;
		}
	}
}

void fillMat(float**& mat, int numR, int numC, int num) {
	for (int i = 0; i < numR; i++) {
		for (int j = 0; j < numC; j++) {
			mat[i][j] = num;
		}
	}
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// GENERAL FUNCS
void resizeMatrix(float**& mat, int numRows, int numCols) {
	mat = (float**)malloc(numRows * sizeof(float*));
	for (int i = 0; i < numRows; i++) {
		mat[i] = (float*)malloc(numCols * sizeof(float));
	}
}
void resizeVector(float*& mat, int numElems) {
	mat = (float*)malloc(numElems * sizeof(float));
}

void freeMatrix(float**& mat, int numRows) {
	for (int i = 0; i < numRows; i++) {
		mat[i] = nullptr;
	}
	mat = nullptr; free(mat);
}

// fill array with random floats
void getRandomFloats(std::uniform_real_distribution<float>& dist,
	std::default_random_engine& gen, float*& storage, int numElems) {

	for (int i = 0; i < numElems; i++) {
		storage[i] = dist(gen);
		// storage[i] = (rand() % 2) - 1;
	}
}

void displayMat(float**& dataAll, int numRows, int numCols, std::string title, bool pause) {
	std::cout << "\n\nMATRIX " << title << ":\n";
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			std::cout << dataAll[i][j] << " ";
		}
		std::cout << "\n";
	}
	if (pause) {
		std::string buf;
		std::getline(std::cin, buf);
	}
}

void append_mat_to_str(float**& dataAll, int numRows, int numCols, std::string title, std::string& storage) {
	storage += "\n\nMATRIX " + title + ":\n";
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			storage  += std::to_string(dataAll[i][j]) + " ";
		}
		storage   += "\n";
	}
}

void displayVect(float*& vec, int numElems, std::string title) {
	std::cout << "\n\nVECT " << title << ":\n";
	for (int j = 0; j < numElems; j++) {
		std::cout << vec[j] << " ";
	}
	std::cout << "\n";
}

void copyMat(float**& dest, float**& src, int numRows, int numCols) {
	for (int i = 0; i < numRows; i++) {
		std::memcpy(dest[i], src[i], numCols * sizeof(float));
	}
}

void copyVect(float*& dest, float*& src, int numElems) {
	std::memcpy(dest, src, numElems * sizeof(float));
}

void wait() {		// stop a program execution until user presses key
	std::string buf; std::getline(std::cin, buf);
}

