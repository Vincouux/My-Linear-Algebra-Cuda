#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>

template <class Number>
class Matrix {
public:
  /* Constructors */
  Matrix<Number>(size_t height, size_t width);
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

/*
template <class Number>
extern void addWrapper(Number* m1, Number* m2, Number* m3, size_t size);
*/

#include "matrix.hxx"

#endif
