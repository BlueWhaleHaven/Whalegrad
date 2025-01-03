#include <stdlib.h>
#include <time.h>
#include "mlp.h"

static double random_double() {
    return (double)rand() / RAND_MAX * 2 - 1;
}

MLP* mlp_create(int* layer_sizes, int num_layers) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    if (!mlp) return NULL;

    
    srand(time(NULL));

    mlp->num_layers = num_layers - 1;  
    mlp->layer_sizes = (int*)malloc(num_layers * sizeof(int));
    mlp->weights = (Tensor**)malloc((num_layers - 1) * sizeof(Tensor*));
    mlp->biases = (Tensor**)malloc((num_layers - 1) * sizeof(Tensor*));

    
    for (int i = 0; i < num_layers; i++) {
        mlp->layer_sizes[i] = layer_sizes[i];
    }

    
    for (int i = 0; i < num_layers - 1; i++) {
        
        mlp->weights[i] = tensor_create(random_double());
        
       
        mlp->biases[i] = tensor_create(random_double());
    }

    return mlp;
}

Tensor* mlp_forward(MLP* mlp, Tensor* input) {
    Tensor* current = input;

    
    for (int i = 0; i < mlp->num_layers; i++) {
       
        Tensor* linear = tensor_add(
            tensor_mul(mlp->weights[i], current),
            mlp->biases[i]
        );

        
        if (i < mlp->num_layers - 1) {
            current = tensor_relu(linear);
        } else {
            current = linear;  
        }
    }

    return current;
}

void mlp_free(MLP* mlp) {
    if (!mlp) return;

    
    for (int i = 0; i < mlp->num_layers; i++) {
        tensor_free(mlp->weights[i]);
        tensor_free(mlp->biases[i]);
    }

    free(mlp->weights);
    free(mlp->biases);
    free(mlp->layer_sizes);
    free(mlp);
} 