#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <algorithm>
#include <memory>
#include <iomanip>

#include "kernels.cuh"

template <class Number>
class Matrix {
public:
    /* Constructors */
    Matrix<Number>();
    Matrix<Number>(size_t height, size_t width);
    Matrix<Number>(size_t height, size_t width, Number min, Number max);
    Matrix<Number>(std::initializer_list<std::initializer_list<Number>> array);
    Matrix<Number>(std::string path);

    /* Getters */
    size_t getWidth() const;
    size_t getHeight() const;
    Number getElementAt(size_t i, size_t j) const;
    void setElementAt(size_t i, size_t j, Number el);

    /* Operations */
    Matrix<Number> add(const Matrix& m, bool gpu = true) const;
    Matrix<Number> add(Number m) const;
    Matrix<Number> sub(const Matrix& m) const;
    Matrix<Number> sub(Number m) const;
    Matrix<Number> dot(const Matrix<Number>& m, bool gpu = true) const;
    Matrix<Number> dot(Number m) const;
    Matrix<Number> multiply(const Matrix<Number>& m) const;
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
    Matrix<Number> operator + (const Matrix&) const;
    Matrix<Number> operator * (const Matrix&) const;
    Matrix<Number> operator - (const Matrix&) const;
    Matrix<Number> operator % (const Matrix&) const;
    bool operator == (const Matrix&);

    /* Display */
    void display() const;

private:
    size_t height;
    size_t width;
    std::vector<Number> array;
};

template <class Number>
Matrix<Number>::Matrix() {
    static_assert(std::is_same<Number, int>::value ||
                  std::is_same<Number, float>::value ||
                  std::is_same<Number, double>::value,
                  "Type not allowed. Use <int>, <float> or <double>.");
    this->height = 0;
    this->width = 0;
    this->array = std::vector<Number>(0);
}

template <class Number>
Matrix<Number>::Matrix(size_t height, size_t width) {
    static_assert(std::is_same<Number, int>::value ||
                  std::is_same<Number, float>::value ||
                  std::is_same<Number, double>::value,
                  "Type not allowed. Use <int>, <float> or <double>.");
    this->height = height;
    this->width = width;
    this->array = std::vector<Number>(this->height * this->width);
    srand(time(NULL));
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            Number n = -1 + rand() / Number(RAND_MAX) * 2;
            this->setElementAt(i, j, n);
        }
    }
}

template <class Number>
Matrix<Number>::Matrix(size_t height, size_t width, Number min, Number max) {
    static_assert(std::is_same<Number, int>::value ||
                  std::is_same<Number, float>::value ||
                  std::is_same<Number, double>::value,
                  "Type not allowed. Use <int>, <float> or <double>.");
    this->height = height;
    this->width = width;
    this->array = std::vector<Number>(this->height * this->width);
    srand(time(NULL));
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            Number n = min + (max - min) * (rand() / Number(RAND_MAX));
            this->setElementAt(i, j, n);
        }
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
    this->array = std::vector<Number>(this->height * this->width);
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            this->setElementAt(i, j, (arr.begin() + i)->begin()[j]);
        }
    }
}

template <class Number>
Matrix<Number>::Matrix(std::string path) {
    std::ifstream file;
    file.open(path);
    this->height = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
    file.seekg(0);
    std::string line;
    std::getline(file, line);
    this->width = 1;
    for (unsigned i = 0; i < line.size(); i++) {
        if (line[i] == ' ') {
            this->width += 1;
        }
    }
    this->array = std::vector<Number>(this->height * this->width);
    unsigned i = 0;
    unsigned j = 0;
    while (!file.eof()) {
        int tmp;
        file >> tmp;
        this->setElementAt(i, j, (Number)tmp);
        j += 1;
        i += (j == width) ? 1 : 0;
        j %= width;
    }
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
void Matrix<Number>::setElementAt(size_t i, size_t j, Number el) {
    if (i >= this->height || j >= this->width) {
        fprintf(stderr, "Can't set element at %li, %li. Shape = (%li, %li)\n", i, j, this->height, this->width);
        throw;
    }
    this->array[i * this->width + j] = el;
}

template <class Number>
Matrix<Number> Matrix<Number>::add(const Matrix& m, bool gpu) const {
    if (m.height != this->height || m.width != this->width) {
        fprintf(stderr, "Can't add element wise a matrix of shape (%li, %li) with a matrix of shape (%li, %li).\n", this->height, this->width, m.height, m.width);
        throw;
    }
    Matrix result(height, width);
    if (gpu) {
        Wrapper().add((Number*)&this->array[0], (Number*)&m.array[0], (Number*)&result.array[0], this->width * this->height);
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
Matrix<Number> Matrix<Number>::sub(const Matrix& m) const {
    if (m.height != this->height || m.width != this->width) {
        fprintf(stderr, "Can't subtract element wise a matrix of shape (%li, %li) with a matrix of shape (%li, %li).\n", this->height, this->width, m.height, m.width);
        throw;
    }
    Matrix result(height, width);
    for (size_t i = 0; i < this->height; i++) {
        for (size_t j = 0; j < this->width; j++) {
            result.setElementAt(i, j, this->getElementAt(i, j) - m.getElementAt(i, j));
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
    if (this->width != m.height) {
        fprintf(stderr, "Can't multiply a matrix of shape (%li, %li) with a matrix of shape (%li, %li).\n", this->height, this->width, m.height, m.width);
        throw;
    }
    Matrix<Number> result(this->height, m.width);
    if (gpu) {
        Wrapper().dot((Number*)&this->array[0], (Number*)&m.array[0], (Number*)&result.array[0], this->height, m.width, this->width);
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
Matrix<Number> Matrix<Number>::multiply(const Matrix<Number>& m) const {
    if (m.height != this->height || m.width != this->width) {
        fprintf(stderr, "Can't multiply element wise a matrix of shape (%li, %li) with a matrix of shape (%li, %li).\n", this->height, this->width, m.height, m.width);
        throw;
    }
    Matrix result(height, width);
    for (size_t i = 0; i < this->height; i++) {
        for (size_t j = 0; j < this->width; j++) {
            result.setElementAt(i, j, this->getElementAt(i, j) * m.getElementAt(i, j));
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
            result.setElementAt(j, i, this->getElementAt(i, j));
        }
    }
    return result;
}

template <class Number>
void Matrix<Number>::display() const {
    for (size_t i = 0; i < this->height; i++) {
        for (size_t j = 0; j < this->width; j++) {
            float n = (float)this->getElementAt(i, j);
            printf("%c%.*f  ", n >= 0 ? ' ' : '\0', 6, n);
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

template <class Number>
Matrix<Number> Matrix<Number>::operator + (const Matrix& m) const {
    return this->add(m);
}

template <class Number>
Matrix<Number> Matrix<Number>::operator * (const Matrix& m) const {
    return this->dot(m);
}

template <class Number>
Matrix<Number> Matrix<Number>::operator - (const Matrix& m) const {
    return this->sub(m);
}

template <class Number>
Matrix<Number> Matrix<Number>::operator % (const Matrix& m) const {
    return this->multiply(m);
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
