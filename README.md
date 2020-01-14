# My-Linear-Algebra-CPP

**Simple Matrix Class implementation in C++ using Cuda GPU Programming.**


## How it works ?

### Instantiate a new Matrix

```cpp
/* Empty matrix */
Matrix<int> A = Matrix<int>();

/* With random values */
Matrix<int> B = Matrix<float>(10, 10);

/* With random values */
Matrix<int> C = Matrix<double>(10, 10);

/* Using initializer list */
Matrix<int> D = Matrix<int>({{1, 2, 3}, {4, 5, 6}});
```

### Bastic operations
```cpp
/* Basic Operations between Matrix */
Matrix<float> E = A + A;
Matrix<float> G = A - A;
Matrix<float> F = A * A;
Matrix<float> G = A / A;
bool eq = (E == G);
```

```cpp
/* Basic Operations between Matrix & Number */
Matrix<float> H = 2.f + A;
Matrix<float> I = 2.f - A;
Matrix<float> J = 2.f * A;
Matrix<float> K = 2.f / A;
```

# Methods
```
/* Element wise operation */
Matrix<float> L = A.exponential();
Matrix<float> M = A.power(2);
```

```cpp
/* Other methods */
Matrix<float> N = A.getLine(5);
Matrix<float> O = A.getRow(5);
Matrix<float> P = A.transpose();
Matrix<float> P = A.apply([](float x) { return 2.f * x; });
```
