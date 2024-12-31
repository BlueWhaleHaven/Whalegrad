#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>
#include <stdio.h>

#define MAX_CHILDREN 10
#define MAX_TOPO_SIZE 100

typedef struct Tensor Tensor;

struct Tensor {
    double data;
    double grad;
    bool requires_grad;
    int num_children;
    Tensor* children[MAX_CHILDREN];
    void (*backward)(struct Tensor*);
    char op[32];
};

// Function declarations
Tensor* tensor_create(double data);
Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_mul(Tensor* a, Tensor* b);
Tensor* tensor_pow(Tensor* a, double power);
Tensor* tensor_relu(Tensor* a);
Tensor* tensor_neg(Tensor* a);
Tensor* tensor_sub(Tensor* a, Tensor* b);
Tensor* tensor_div(Tensor* a, Tensor* b);
void tensor_backward(Tensor* t);
void tensor_print(Tensor* t);
void tensor_free(Tensor* t);

#endif