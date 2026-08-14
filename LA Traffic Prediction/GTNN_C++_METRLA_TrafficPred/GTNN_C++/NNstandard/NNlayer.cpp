
//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS, College of Engineering
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++

#include<cstring>
#include<iostream>
#include<string>
#include<random>
#include<unordered_map>

#include "NN.h"




//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// change w & b matrices by whatever's stored in dcdw, dcdb
void NNlayer::changeWB() {
    // w -= learning_rate * dcdw
    for (int i = 0; i < numNeurons; i++) {      // changes to w
        for (int j = 0; j < next->numNeurons; j++) {
            w[i][j] -= learning_rate * dcdw[i][j];
        }
    }
    // b -= learning_rate * dcdb
    for (int i = 0; i < batchsize; i++) {       // changes to b
        for (int j = 0; j < next->numNeurons; j++) {
            b[i][j] -= learning_rate * dcdb[i][j];
        }
    }
}
void NNlayer::changeWB(float& scale_lrate) {
    // w -= learning_rate * dcdw
    for (int i = 0; i < numNeurons; i++) {      // changes to w
        for (int j = 0; j < next->numNeurons; j++) {
            w[i][j] -= scale_lrate * learning_rate * dcdw[i][j];
        }
    }
    // b -= learning_rate * dcdb
    for (int i = 0; i < batchsize; i++) {       // changes to b
        for (int j = 0; j < next->numNeurons; j++) {
            b[i][j] -= scale_lrate * learning_rate * dcdb[i][j];
        }
    }
}

void link_layers(NNlayer* ptr1, NNlayer* ptr2) {
    ptr1->next = ptr2;
    ptr2->prev = ptr1;
    ptr1->numNeuronsNext = ptr2->numNeurons;
}

NNlayer::NNlayer(int batchsize_, int numNeurons_, std::string activation_="N/A") {
    this->batchsize = batchsize_;
    this->numNeurons = numNeurons_;
    this->activation = activation_;
}

void NNlayer::initMatrices() {
    resizeMatrix(a, batchsize, numNeurons);
    resizeMatrix(w, numNeurons, numNeuronsNext);
    resizeMatrix(b, batchsize, numNeuronsNext);

    resizeMatrix(dcdw, numNeurons, numNeuronsNext);
    resizeMatrix(dcdb, batchsize, numNeuronsNext);

    resizeMatrix(dcda, batchsize, numNeurons);
    resizeMatrix(dcdz, batchsize, numNeurons);
    resizeMatrix(dadz, batchsize, numNeurons);
}

void NNlayer::randomizeWB(std::uniform_real_distribution<float>& dist, std::default_random_engine& gen) {
    for (int i = 0; i < numNeurons; i++) {
        getRandomFloats(dist, gen, w[i], numNeuronsNext);
        //for (int j = 0; j < numNeuronsNext; j++)
        //    w[i][j] = 1;
    }
    for (int i = 0; i < batchsize; i++) {
        //for (int j = 0; j < numNeuronsNext; j++)
        //    b[i][j] = 1;
        getRandomFloats(dist, gen, b[i], numNeuronsNext);
    }
}

