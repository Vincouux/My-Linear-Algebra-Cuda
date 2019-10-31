#include "Matrix/matrix.hpp"

int main() {
  Matrix<int> a(3, 4);
  a.display();
  Matrix<int> b(3, 4);
  b.display();
  Matrix<int> c = a + b;
  c.display();
  std::cout << "Width: " << c.getWidth() << ", Height: " << c.getHeight() << std::endl;
  return 0;
}
