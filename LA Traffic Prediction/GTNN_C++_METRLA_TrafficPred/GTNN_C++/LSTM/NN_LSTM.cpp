
// LSTM
#include "LSTM.h"

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// NN specifically for LSTM

void modified_lstm_forwardMath(float**& a, float**& w, float**& b, float**& nexta, \
    std::string activations[], int numRowsA, int numColsA, int numColsW) {

    for (int i = 0; i < numRowsA; i++) {
        for (int j = 0; j < numColsW; j++) {
            // nexta = a . w
            nexta[i][j] = 0;
            for (int k = 0; k < numColsA; k++) {
                nexta[i][j] += a[i][k] * w[k][j];
            }
            // nexta += b
            nexta[i][j] += b[i][j];
            // nexta = relu(nexta)
            NN::activmap[activations[j]]\
                (nexta[i][j]);
        }
    }
}

void NN_LSTM::predict_lstm(float**& h, float**& gstate, int gcol, float**& storage) {
    // initialize input matrix ( h | gstate[:,gcol] )
    NNlayer* ptr = layers[0];
    for (int i = 0; i < batchsize; i++) {
        ptr->a[i][0] = h[i][0];
        ptr->a[i][1] = gstate[i][gcol];		// decrement one because of the std::map auto-index adjustment
    }
    //// FORWARD PROP
    // iterate thru network, applying same procedure
    while (ptr->next->next != nullptr) {
        NN::forwardMath(ptr->a, ptr->w, ptr->b, ptr->next->a, ptr->activation, \
            batchsize, ptr->numNeurons, ptr->numNeuronsNext);

        ptr = ptr->next;
    }
    // final layer - LSTM special activation
    std::string activations[] = { "sigmoid", "sigmoid", "tanh_", "sigmoid" };
    modified_lstm_forwardMath(ptr->a, ptr->w, ptr->b, ptr->next->a, activations, \
        batchsize, ptr->numNeurons, ptr->numNeuronsNext);


    // store output matrix into abcd storage matrix
    ptr = layers[numLayers - 1];

    for (int i = 0; i < batchsize; i++) {
        //storage[i][0] = ptr->a[i][0];
        //storage[i][1] = ptr->a[i][1];
        //storage[i][2] = ptr->a[i][2];
        //storage[i][3] = ptr->a[i][3];
        std::memcpy(storage[i], ptr->a[i], sizeof(float) * numOutputNeurons);
    }
}

void deriv_activations_lstm(float**& a, float**& storage, int numR) {
    for (int i = 0; i < numR; i++) {
        // for (int j = 0; j < numC; j++) {
        storage[i][0] = deriv_sigmoid(a[i][0], true);
        storage[i][1] = deriv_sigmoid(a[i][1], true);
        storage[i][2] = deriv_tanh_(a[i][2], true);
        storage[i][3] = deriv_sigmoid(a[i][3], true);
    }
}

float NN_LSTM::backward_lstm(float**& dcda_final) {
    NNlayer* ptr = layers[numLayers - 1];

    // get final layer's dcda
    // and calc. avg_cost
    float avg_cost = 0;
    for (int i = 0; i < batchsize; i++) {
        for (int j = 0; j < ptr->numNeurons; j++) {
            ptr->dcda[i][j] = dcda_final[i][j];
            avg_cost += std::abs(ptr->dcda[i][j]);
        }
    }
    avg_cost /= (ptr->numNeurons * batchsize);
    ptr = ptr->prev;

    // iterate thru prev. layers
    while (ptr != nullptr) {
        // modified deriv_activation (eval. da/dz term)
        if (ptr->next->next == nullptr) {
            deriv_activations_lstm(ptr->next->a, ptr->next->dadz, batchsize);
        }
        else {
            // next dadz = de_relu(next a)
            NN::deriv_activmap[ptr->activation]\
                (ptr->next->a, ptr->next->dadz, batchsize, ptr->next->numNeurons);
        }
        // next dcdz = next dcda .* next dadz
        matmul_elemwise(ptr->next->dcda, ptr->next->dadz, ptr->next->dcdz, batchsize, ptr->next->numNeurons);
        // dcdw = a_T * next dcdz
        mat_T_times_mat(ptr->a, ptr->next->dcdz, ptr->dcdw, batchsize, ptr->numNeurons, ptr->next->numNeurons);
        // dcdb = next dcdz
        copyMat(ptr->dcdb, ptr->next->dcdz, batchsize, ptr->next->numNeurons);
        // dcda = next dcdz * w_T
        mat_times_mat_T(ptr->next->dcdz, ptr->w, ptr->dcda, batchsize, ptr->next->numNeurons, ptr->numNeurons);
        // modify w, b matrices (for cur. layer)
        ptr->changeWB();

        ptr = ptr->prev;
    }
    return avg_cost;
}


float**& NN_LSTM::get_layer0_dcda() { return layers[0]->dcda; }

