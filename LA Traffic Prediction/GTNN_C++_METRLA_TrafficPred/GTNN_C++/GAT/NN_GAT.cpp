// GAT

#include "GAT.h"

// scalar = NN( vec | vec )                 // horizontal concat.
float NN_GAT::predict(float*& h1, float*& h2) {
    NNlayer* ptr = layers[0];
    int h_len = ptr->numNeurons / 2;
    for (int i = 0; i < h_len; i++) {
        ptr->a[0][i] = h1[i];
    }
    for (int i = h_len; i < ptr->numNeurons; i++) {
        ptr->a[0][i] = h2[i - h_len];
    }
    forward(ptr->a);
    return layers[numLayers - 1]->a[0][0];
}

// back prop. NN, but we're passed final dcda (scalar), as well as the orig. inputs
void NN_GAT::backward(float*& input1, float*& input2, float& dcda_final, float scale_lrate) {

    // copy input to 0th layer 'a'
    NNlayer* fptr = layers[0];
    int h_len = fptr->numNeurons / 2;
    for (int i = 0; i < h_len; i++) {
        fptr->a[0][i] = input1[i];
    }
    for (int i = h_len; i < fptr->numNeurons; i++) {
        fptr->a[0][i] = input2[i - h_len];
    }

    // copy dcda to final layer dcda
    NNlayer* ptr = layers[numLayers - 1];
    ptr->dcda[0][0] = dcda_final;
    ptr = ptr->prev;


    // iterate thru prev. layers
    while (ptr != nullptr) {
        // next dadz = de_relu(next a)
        NN_GAT::deriv_activmap[ptr->activation]\
            (ptr->next->a, ptr->next->dadz, batchsize, ptr->next->numNeurons);
        // next dcdz = next dcda .* next dadz
        matmul_elemwise(ptr->next->dcda, ptr->next->dadz, ptr->next->dcdz, batchsize, ptr->next->numNeurons);
        // dcdw = a_T * next dcdz
        mat_T_times_mat(ptr->a, ptr->next->dcdz, ptr->dcdw, batchsize, ptr->numNeurons, ptr->next->numNeurons);
        // dcdb = next dcdz
        copyMat(ptr->dcdb, ptr->next->dcdz, batchsize, ptr->next->numNeurons);
        // dcda = next dcdz * w_T
        mat_times_mat_T(ptr->next->dcdz, ptr->w, ptr->dcda, batchsize, ptr->next->numNeurons, ptr->numNeurons);
        // modify w, b matrices (for cur. NNlayer)
        ptr->changeWB(scale_lrate);

        ptr = ptr->prev;
    }
}

float**& NN_GAT::get_layer0_dcda() { return layers[0]->dcda; }


// normal backprop., but we're passed final layer dcda
void NN_GAT::backward_using_derivative(float**& dcda_final) {
    NNlayer* ptr = layers[numLayers - 1];
    for (int i = 0; i < batchsize; i++) {
        std::memcpy(ptr->dcda[i], dcda_final[i], sizeof(float) * numOutputNeurons);
    }
    ptr = ptr->prev;

    // iterate thru prev. layers
    while (ptr != nullptr) {
        // next dadz = de_relu(next a)
        NN_GAT::deriv_activmap[ptr->activation]\
            (ptr->next->a, ptr->next->dadz, batchsize, ptr->next->numNeurons);
        // next dcdz = next dcda .* next dadz
        matmul_elemwise(ptr->next->dcda, ptr->next->dadz, ptr->next->dcdz, batchsize, ptr->next->numNeurons);
        // dcdw = a_T * next dcdz
        mat_T_times_mat(ptr->a, ptr->next->dcdz, ptr->dcdw, batchsize, ptr->numNeurons, ptr->next->numNeurons);
        // dcdb = next dcdz
        copyMat(ptr->dcdb, ptr->next->dcdz, batchsize, ptr->next->numNeurons);
        // dcda = next dcdz * w_T
        mat_times_mat_T(ptr->next->dcdz, ptr->w, ptr->dcda, batchsize, ptr->next->numNeurons, ptr->numNeurons);
        // modify w, b matrices (for cur. NNlayer)
        ptr->changeWB();

        ptr = ptr->prev;
    }
}

float NN_GAT::backward(float**& y_true) {
    return (NN::backward(y_true));
}