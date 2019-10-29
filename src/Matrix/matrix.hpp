#ifndef NN_MATRIX_H
#define NN_MATRIX_H

#include <vector>

template <class Number>
class Matrix {
public:
  /* Constructors */
  Matrix<Number>(size_t heights, size_t width);
  ~Matrix<Number>();

  /* Getters */
  size_t getWidth() const;
  size_t getHeight() const;

  /* Operations */
  Matrix<Number> add(const Matrix<Number>& m) const;
  Matrix<Number> dot(const Matrix<Number>& m) const;
  Matrix<Number> transpose() const;

  /* Operators */
  Matrix<Number> operator + (const Matrix&);
  Matrix<Number> operator * (const Matrix&);

  /* Display */
  void display() const;

private:
  size_t height;
  size_t width;
  Number* array;
};

#include "matrix.hxx"

#endif
