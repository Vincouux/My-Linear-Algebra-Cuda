#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>

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
    Matrix<Number> dot(const Matrix<Number>& m, bool gpu = true) const;
    bool eq(const Matrix<Number>& m, bool gpu = false) const;
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
    this->array = (Number*)calloc(height * width, sizeof(Number));
    srand(time(NULL));
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
    this->array = (Number*)calloc(this->height * this->width, sizeof(Number));
    srand(time(NULL));
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            this->setElementAt(i, j, (arr.begin() + i)->begin()[j]);
        }
    }
}

template <class Number>
Matrix<Number>::~Matrix() {
    free(this->array);
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
    if (i > this->height || j > this->width) {
        fprintf(stderr, "Can't subscrit at %li, %li. Shape = (%li, %li)\n", i, j, this->height, this->width);
        throw;
    }
    return this->array[i * this->width + j];
}

template <class Number>
Number Matrix<Number>::setElementAt(size_t i, size_t j, Number el) {
    if (i > this->height || j > this->width) {
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
bool Matrix<Number>::eq(const Matrix& m, bool gpu) const {
    if (gpu) {
        printf("Method not implemented for GPU\n");
        // Fix Me
    } else {
        for (size_t i = 0; i < this->height; i++) {
            for (size_t j = 0; j < m.width; j++) {
                if (this->getElementAt(i, j) != m.getElementAt(i, j)) {
                    return false;
                }
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

#endif
