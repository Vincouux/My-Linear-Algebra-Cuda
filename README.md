# My-Linear-Algebra-CPP

### Simple Matrix Class implementation in C++ using Cuda GPU Programming.


## How it works ?

### Instantiate a new Matrix

```cpp
/* With random values */
Matrix<int> A = Matrix<int>(2000, 2000);
Matrix<float> B = Matrix<float>(2000, 2000);
Matrix<double> C = Matrix<double>(2000, 2000);

/* Using initializer list */
Matrix<int> A = Matrix<int>({{1, 2, 3}, {4, 5, 6}});
```

```cpp
/* Basic Operations on GPU */
Matrix<int> D = A + A;
Matrix<int> E = A * A;

/* Basic Operations on CPU */
Matrix<int> F = A.add(A, false);
Matrix<int> G = A.dot(A, false);
bool eq = E == G;
```

> Work In Progress
