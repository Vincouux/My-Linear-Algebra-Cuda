#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <cmath>

#include "kernels.cuh"

template <class Number>
class Matrix {
public:
    /* Constructors */
    Matrix<Number>(size_t height, size_t width);
    Matrix<Number>(std::initializer_list<std::initializer_list<Number>> array);
    ~Matrix<Number>();

    /* Getters */
    size_t getWidth() const;
    size_t getHeight() const;
    Number getElementAt(size_t i, size_t j) const;
    Number setElementAt(size_t i, size_t j, Number el);

    /* Operations */
    Matrix<Number> add(const Matrix& m, bool gpu = true) const;
    Matrix<Number> add(Number m) const;
    Matrix<Number> sub(Number m) const;
    Matrix<Number> dot(const Matrix<Number>& m, bool gpu = true) const;
    Matrix<Number> dot(Number m) const;
    Matrix<Number> divide(Number m) const;
    Matrix<Number> inverse(Number m) const;
    Matrix<Number> exponential() const;
    Matrix<Number> power(unsigned n) const;
    Number sum() const;
    void apply(Number func(Number));
    Matrix<Number> getLine(unsigned n) const;
    Matrix<Number> getRow(unsigned n) const;
    bool eq(const Matrix<Number>& m) const;
    Matrix<Number> transpose() const;

    /* Operators */
    Matrix<Number> operator + (const Matrix&);
    Matrix<Number> operator * (const Matrix&);
    bool operator == (const Matrix&);

    /* Display */
    void display() const;

private:
    size_t height;
    size_t width;
    Number* array;
};

template <class Number>
Matrix<Number>::Matrix(size_t height, size_t width) {
    static_assert(std::is_same<Number, int>::value ||
                  std::is_same<Number, float>::value ||
                  std::is_same<Number, double>::value,
                  "Type not allowed. Use <int>, <float> or <double>.");
    this->height = height;
    this->width = width;
    this->array = new Number[this->height * this->width];
    for (size_t i = 0; i < height * width; i++) {
        this->array[i] = -1 + rand() / Number(RAND_MAX) * 2;
    }
}

template <class Number>
Matrix<Number>::Matrix(std::initializer_list<std::initializer_list<Number>> arr) {
    static_assert(std::is_same<Number, int>::value ||
                  std::is_same<Number, float>::value ||
                  std::is_same<Number, double>::value,
                  "Type not allowed. Use <int>, <float> or <double>.");
    this->height = (int)arr.size();
    this->width = (int)arr.begin()->size();
    this->array = new Number[this->height * this->width];
    srand(time(NULL));
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            this->setElementAt(i, j, (arr.begin() + i)->begin()[j]);
        }
    }
}

template <class Number>
Matrix<Number>::~Matrix() {
    delete[] this->array;
}

template <class Number>
size_t Matrix<Number>::getHeight() const {
    return this->height;
}

template <class Number>
size_t Matrix<Number>::getWidth() const {
    return this->width;
}

template <class Number>
Number Matrix<Number>::getElementAt(size_t i, size_t j) const {
    if (i >= this->height || j >= this->width) {
        fprintf(stderr, "Can't subscrit at %li, %li. Shape = (%li, %li)\n", i, j, this->height, this->width);
        throw;
    }
    return this->array[i * this->width + j];
}

template <class Number>
Number Matrix<Number>::setElementAt(size_t i, size_t j, Number el) {
    if (i >= this->height || j >= this->width) {
        fprintf(stderr, "Can't subscrit at %li, %li. Shape = (%li, %li)\n", i, j, this->height, this->width);
        throw;
    }
    return this->array[i * this->width + j] = el;
}

template <class Number>
Matrix<Number> Matrix<Number>::add(const Matrix& m, bool gpu) const {
    Matrix result(height, width);
    if (gpu) {
        Wrapper().add(this->array, m.array, result.array, this->width * this->height);
    } else {
        for (size_t i = 0; i < this->height; i++) {
            for (size_t j = 0; j < this->width; j++) {
                result.setElementAt(i, j, this->getElementAt(i, j) + m.getElementAt(i, j));
            }
        }
    }
    return result;
}

template <class Number>
Matrix<Number> Matrix<Number>::add(Number m) const {
    Matrix result(height, width);
    for (size_t i = 0; i < this->height; i++) {
        for (size_t j = 0; j < this->width; j++) {
            result.setElementAt(i, j, this->getElementAt(i, j) + m);
        }
    }
    return result;
}

template <class Number>
Matrix<Number> Matrix<Number>::sub(Number m) const {
    Matrix result(height, width);
    for (size_t i = 0; i < this->height; i++) {
        for (size_t j = 0; j < this->width; j++) {
            result.setElementAt(i, j, this->getElementAt(i, j) - m);
        }
    }
    return result;
}

template <class Number>
Matrix<Number> Matrix<Number>::dot(const Matrix& m, bool gpu) const {
    Matrix<Number> result(this->height, m.width);
    if (gpu) {
        Wrapper().dot(this->array, m.array, result.array, this->height, m.width, this->width);
    } else {
        Number val = 0;
        for (size_t i = 0; i < this->height; i++) {
            for (size_t j = 0; j < m.width; j++) {
                for (size_t h = 0; h < this->width; h++) {
                    val += this->getElementAt(i, h) * m.getElementAt(h, j);
                }
                result.setElementAt(i, j, val);
                val = 0;
            }
        }
    }
    return result;
}

template <class Number>
Matrix<Number> Matrix<Number>::dot(Number m) const {
    Matrix result(height, width);
    for (size_t i = 0; i < this->height; i++) {
        for (size_t j = 0; j < this->width; j++) {
            result.setElementAt(i, j, this->getElementAt(i, j) * m);
        }
    }
    return result;
}

template <class Number>
Matrix<Number> Matrix<Number>::divide(Number m) const {
    Matrix result(height, width);
    for (size_t i = 0; i < this->height; i++) {
        for (size_t j = 0; j < this->width; j++) {
            result.setElementAt(i, j, this->getElementAt(i, j) / m);
        }
    }
    return result;
}

template <class Number>
Matrix<Number> Matrix<Number>::inverse(Number m) const {
    Matrix result(height, width);
    for (size_t i = 0; i < this->height; i++) {
        for (size_t j = 0; j < this->width; j++) {
            result.setElementAt(i, j, m / this->getElementAt(i, j));
        }
    }
    return result;
}

template <class Number>
Matrix<Number> Matrix<Number>::exponential() const {
    Matrix result(height, width);
    for (size_t i = 0; i < this->height; i++) {
        for (size_t j = 0; j < this->width; j++) {
            result.setElementAt(i, j, exp(this->getElementAt(i, j)));
        }
    }
    return result;
}

template <class Number>
Matrix<Number> Matrix<Number>::power(unsigned n) const {
    Matrix result(height, width);
    for (size_t i = 0; i < this->height; i++) {
        for (size_t j = 0; j < this->width; j++) {
            result.setElementAt(i, j, pow(this->getElementAt(i, j), n));
        }
    }
    return result;
}

template <class Number>
Number Matrix<Number>::sum() const {
    Number result = 0;
    for (size_t i = 0; i < this->height; i++) {
        for (size_t j = 0; j < this->width; j++) {
            result += this->getElementAt(i, j);
        }
    }
    return result;
}

template <class Number>
void Matrix<Number>::apply(Number func(Number)) {
    for (size_t i = 0; i < this->height; i++) {
        for (size_t j = 0; j < this->width; j++) {
            this->setElementAt(i, j, func(this->getElementAt(i, j)));
        }
    }
}

template <class Number>
Matrix<Number> Matrix<Number>::getLine(unsigned n) const {
    if (n >= this->height) {
        fprintf(stderr, "Can't subscrit line at %i.\n", n);
        throw;
    }
    Matrix result(1, width);
    for (size_t i = 0; i < this->width; i++) {
        result.setElementAt(0, i, this->getElementAt(n, i));
    }
    return result;
}

template <class Number>
Matrix<Number> Matrix<Number>::getRow(unsigned n) const {
    if (n >= this->width) {
        fprintf(stderr, "Can't subscrit row at %i.\n", n);
        throw;
    }
    Matrix result(height, 1);
    for (size_t i = 0; i < this->height; i++) {
        result.setElementAt(i, 0, this->getElementAt(i, n));
    }
    return result;
}

template <class Number>
bool Matrix<Number>::eq(const Matrix& m) const {
    for (size_t i = 0; i < this->height; i++) {
        for (size_t j = 0; j < m.width; j++) {
            if (this->getElementAt(i, j) != m.getElementAt(i, j)) {
                return false;
            }
        }
    }
    return true;
}

template <class Number>
Matrix<Number> Matrix<Number>::transpose() const {
    Matrix<Number> result(this->width, this->height);
    for (size_t i = 0; i < this->height; i++) {
        for (size_t j = 0; j < this->width; j++) {
            result.setElementAt(i, j, this->getElementAt(j, i));
        }
    }
    return result;
}

template <class Number>
void Matrix<Number>::display() const {
    for (size_t i = 0; i < this->height; i++) {
        for (size_t j = 0; j < this->width; j++) {
            std::cout << this->getElementAt(i, j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

template <class Number>
Matrix<Number> Matrix<Number>::operator + (const Matrix& m) {
    return this->add(m);
}

template <class Number>
Matrix<Number> Matrix<Number>::operator * (const Matrix& m) {
    return this->dot(m);
}

template <class Number>
bool Matrix<Number>::operator == (const Matrix& m) {
    return this->eq(m);
}

template <class Number>
Matrix<Number> operator + (const Matrix<Number>& m, Number n) {
    return m.add(n);
}

template <class Number>
Matrix<Number> operator + (Number n, const Matrix<Number>& m) {
    return m.add(n);
}

template <class Number>
Matrix<Number> operator - (const Matrix<Number>& m, Number n) {
    return m.sub(n);
}

template <class Number>
Matrix<Number> operator - (Number n, const Matrix<Number>& m) {
    return (Number)(-1) * m + n;
}

template <class Number>
Matrix<Number> operator * (const Matrix<Number>& m, Number n) {
    return m.dot(n);
}

template <class Number>
Matrix<Number> operator * (Number n, const Matrix<Number>& m) {
    return m.dot(n);
}

template <class Number>
Matrix<Number> operator / (const Matrix<Number>& m, Number n) {
    return m.divide(n);
}

template <class Number>
Matrix<Number> operator / (Number n, const Matrix<Number>& m) {
    return m.inverse(n);
}

#endif
