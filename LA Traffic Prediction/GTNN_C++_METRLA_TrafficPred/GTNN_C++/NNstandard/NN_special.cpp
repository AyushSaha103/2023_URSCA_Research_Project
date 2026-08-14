
//+++++++++++++++++++++++++++++++++++++++++++++++
// THIS CODE IS PROPERTY OF AYUSH SAHA (2/9/2023)
// Undergraduate student at UC DAVIS, College of Engineering
// DO NOT DISTRIBUTE
//+++++++++++++++++++++++++++++++++++++++++++++++

#include "NN.h"

#include<cstring>
#include<iostream>
#include<sstream>
#include<string>
#include<fstream>
#include<random>




// AutoEnc
float NNspecial::backward(float**& y_true) {
    return ( NN::backward(y_true) );
}