# Whalegrad 🐳

Whalegrad is a lightweight deep learning library written in C, designed for educational purposes.
This is inspired by andrej karapthy's macrograd.

>[IMPORTANT] This project is still under process...
>

Autograd engine is a framework for performing automatic differentiation. 
The autograd package in PyTorch provides a way to perform automatic differentiation of tensors. 
This is useful for training neural networks. 
The autograd package provides automatic differentiation for all operations on Tensors. 
It is a define-by-run framework, which means that your backprop is defined by how your code is run, 
and that every single iteration can be different.

## Example 

```c

int main() {
    
    Tensor* a = tensor_create(-4.0);
    Tensor* b = tensor_create(2.0);
    
    // c = a + b
    Tensor* c = tensor_add(a, b);
    
    // d = a * b + b**3
    Tensor* temp1 = tensor_mul(a, b);
    Tensor* temp2 = tensor_pow(b, 3.0);
    Tensor* d = tensor_add(temp1, temp2);
    
    // c += c + 1
    Tensor* one = tensor_create(1.0);
    Tensor* temp3 = tensor_add(c, one);
    c = tensor_add(c, temp3);
    
    // c += 1 + c + (-a)
    Tensor* temp4 = tensor_add(one, c);
    Tensor* neg_a = tensor_neg(a);
    Tensor* temp5 = tensor_add(temp4, neg_a);
    c = tensor_add(c, temp5);
    
    // d += d * 2 + (b + a).relu()
    Tensor* two = tensor_create(2.0);
    Tensor* temp6 = tensor_mul(d, two);
    Tensor* temp7 = tensor_add(b, a);
    Tensor* temp8 = tensor_relu(temp7);
    Tensor* temp9 = tensor_add(temp6, temp8);
    d = tensor_add(d, temp9);
    
    // d += 3 * d + (b - a).relu()
    Tensor* three = tensor_create(3.0);
    Tensor* temp10 = tensor_mul(three, d);
    Tensor* temp11 = tensor_sub(b, a);
    Tensor* temp12 = tensor_relu(temp11);
    Tensor* temp13 = tensor_add(temp10, temp12);
    d = tensor_add(d, temp13);
    
    // e = c - d
    Tensor* e = tensor_sub(c, d);
    
    // f = e**2
    Tensor* f = tensor_pow(e, 2.0);
    
    // g = f / 2.0
    Tensor* point_five = tensor_create(2.0);
    Tensor* g = tensor_div(f, point_five);
    
    // g += 10.0 / f
    Tensor* ten = tensor_create(10.0);
    Tensor* temp14 = tensor_div(ten, f);
    g = tensor_add(g, temp14);
    
    
    tensor_backward(g);

    // forward pass
    printf("g --> %.4f\n", g->data);
    

    printf("dg/da = %.4f\n", a->grad);
    printf("dg/db = %.4f\n", b->grad);
    
    return 0;
}

```

## run 

```bash
gcc grad.c ../engine/tensor.c -I../engine -o grad\n
./grad
```