# perceptron_proj

```
This is a simple program for perceptron. Author by Jessie Chung
Perceptron is a classic two-classification algorithm, which aims to learn a hyperplan: w^Tx + b = 0.
w is the weight vector and b is the bias. The objective function is defined as follows:
min_{w,b} L(w,b) = - \sum_{x_i \in M} y_i(wx_i+b)
where M is the miss-classified samples. The objective means to minimize the sum of distances from the miss-classified samples to the hyperplane.
Gradient descend is used to optimize the objective function. The main steps of the algorithm comes as:
input: training dataset T={(x_i , y_i)}, learning rate eta \in (0,1]
output: w,b, that is, f(x)=sign(wx+b)
1. initialize w_0, b_0;
2. for (xi, yi) in T
3. if yi(wxi + b) \leq 0
    w <- w + eta*yi*xi
    b <- b + eta*yi      //check the gradient descend
4. end for
```