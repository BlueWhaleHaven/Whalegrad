#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "tensor.h"
#include <string.h>

Tensor* create_tensor_with_children(double data, Tensor** children, int num_children, const char* op) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (t == NULL) return NULL;  
    
    t->data = data;
    t->grad = 0.0;
    t->requires_grad = true;  
    t->num_children = num_children;
    t->backward = NULL;
    
    for (int i = 0; i < num_children && i < MAX_CHILDREN; i++) {
        t->children[i] = children[i];
    }
    strncpy(t->op, op, sizeof(t->op) - 1);
    t->op[sizeof(t->op) - 1] = '\0';
    return t;
}

Tensor* tensor_create(double data) {
    return create_tensor_with_children(data, NULL, 0, "");
}

void add_backward(Tensor* self) {
    for (int i = 0; i < self->num_children; i++) {
        self->children[i]->grad += self->grad;
    }
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    Tensor* children[] = {a, b};
    Tensor* out = create_tensor_with_children(a->data + b->data, children, 2, "+");
    out->backward = add_backward;
    return out;
}

void mul_backward(Tensor* self) {
    self->children[0]->grad += self->children[1]->data * self->grad;
    self->children[1]->grad += self->children[0]->data * self->grad;
}

Tensor* tensor_mul(Tensor* a, Tensor* b) {
    Tensor* children[] = {a, b};
    Tensor* out = create_tensor_with_children(a->data * b->data, children, 2, "*");
    out->backward = mul_backward;
    return out;
}

typedef struct {
    double power;
} PowData;

void pow_backward(Tensor* self) {
    double power;
    sscanf(self->op, "**%lf", &power);
    self->children[0]->grad += power * 
        pow(self->children[0]->data, power - 1) * self->grad;
}

Tensor* tensor_pow(Tensor* a, double power) {
    Tensor* children[] = {a};
    Tensor* out = create_tensor_with_children(pow(a->data, power), children, 1, "**");
    snprintf(out->op, sizeof(out->op), "**%.6f", power);
    out->backward = pow_backward;
    return out;
}

void relu_backward(Tensor* self) {
    if (self->data > 0) {
        self->children[0]->grad += self->grad;
    }
}

Tensor* tensor_relu(Tensor* a) {
    Tensor* children[] = {a};
    Tensor* out = create_tensor_with_children(
        a->data < 0 ? 0 : a->data, 
        children, 
        1, 
        "ReLU"
    );
    out->backward = relu_backward;
    return out;
}

static int find_tensor(Tensor** array, int size, Tensor* t) {
    for (int i = 0; i < size; i++) {
        if (array[i] == t) return i;
    }
    return -1;
}

void build_topo(Tensor* v, Tensor** topo, int* topo_size, Tensor** visited, int* visited_size) {
    if (find_tensor(visited, *visited_size, v) == -1) {
        visited[(*visited_size)++] = v;
        for (int i = 0; i < v->num_children; i++) {
            build_topo(v->children[i], topo, topo_size, visited, visited_size);
        }
        topo[(*topo_size)++] = v;
    }
}

void tensor_backward(Tensor* t) {
    Tensor* topo[MAX_TOPO_SIZE];
    Tensor* visited[MAX_TOPO_SIZE];
    int topo_size = 0;
    int visited_size = 0;
    build_topo(t, topo, &topo_size, visited, &visited_size);
    t->grad = 1.0;
    for (int i = topo_size - 1; i >= 0; i--) {
        if (topo[i]->backward) {
            topo[i]->backward(topo[i]);
        }
    }
}

Tensor* tensor_neg(Tensor* a) {
    return tensor_mul(a, tensor_create(-1.0));
}

Tensor* tensor_sub(Tensor* a, Tensor* b) {
    return tensor_add(a, tensor_neg(b));
}

Tensor* tensor_div(Tensor* a, Tensor* b) {
    return tensor_mul(a, tensor_pow(b, -1.0));
}

void tensor_print(Tensor* t) {
    printf("Tensor(data=%f, grad=%f, op='%s')\n", t->data, t->grad, t->op);
}

void tensor_free(Tensor* t) {
    if (t == NULL) return;
    free(t);
}