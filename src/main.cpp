#include <iostream>

#include "Matrix/matrix.hpp"

int main() {
  Matrix<int> a(2, 3);

  a.display();
  Matrix<int> b(3, 4);
  b.display();
  Matrix<int> c = a * b;
  c.display();
  std::cout << c.getWidth() << ", " << c.getHeight() << std::endl;
  return 0;
}
