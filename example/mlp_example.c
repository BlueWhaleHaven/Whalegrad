#include <stdio.h>
#include "../engine/mlp.h"
#include "../engine/tensor.h"


void xor_example() {
    
    
    //  2 -> 4 -> 1
    int layer_sizes[] = {2, 4, 1};
    MLP* mlp = mlp_create(layer_sizes, 3);
    
    
    double inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double targets[4] = {0, 1, 1, 0};
    
    double learning_rate = 0.1;  
    
    // Training loop
    for (int epoch = 0; epoch < 100; epoch++) {
        double total_loss = 0.0;
        
        
        for (int i = 0; i < 4; i++) {
            // Forward pass
            Tensor* x1 = tensor_create(inputs[i][0]);
            Tensor* x2 = tensor_create(inputs[i][1]);
            Tensor* target = tensor_create(targets[i]);
            
            
            Tensor* input = tensor_add(x1, x2);  
            
            
            Tensor* output = mlp_forward(mlp, input);
            
           
            Tensor* diff = tensor_sub(output, target);
            Tensor* loss = tensor_pow(diff, 2.0);
            
            // Backward pass
            tensor_backward(loss);
            
            
            for (int layer = 0; layer < mlp->num_layers; layer++) {
                // Update weights
                mlp->weights[layer]->data -= learning_rate * mlp->weights[layer]->grad;
                mlp->weights[layer]->grad = 0.0;  // Reset gradient
                
                // Update biases
                mlp->biases[layer]->data -= learning_rate * mlp->biases[layer]->grad;
                mlp->biases[layer]->grad = 0.0;  // Reset gradient
            }
            
            if (epoch % 10 == 0) {
                printf("Epoch %d: Loss=%.4f\n",
                       epoch, loss->data);
            }
            
            total_loss += loss->data;
            
            // Clean up tensors
            tensor_free(x1);
            tensor_free(x2);
            tensor_free(target);
            tensor_free(input);
            tensor_free(output);
            tensor_free(diff);
            tensor_free(loss);
        }
        
        if (epoch % 10 == 0) {
            printf("epoch %d complete, average loss: %.4f\n\n", epoch, total_loss / 4.0);
        }
    }
    
    // Clean up
    mlp_free(mlp);
}

int main() {
 
    
    xor_example();
    
    return 0;
}
