#ifndef MLP_H
#define MLP_H

#include "tensor.h"

typedef struct {
    Tensor** weights;
    Tensor** biases;
    int* layer_sizes;
    int num_layers;
} MLP;


MLP* mlp_create(int* layer_sizes, int num_layers);

Tensor* mlp_forward(MLP* mlp, Tensor* input);

void mlp_free(MLP* mlp);

#endif 