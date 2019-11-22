#include "Matrix/matrix.hpp"

int main() {
	Matrix<int> a = Matrix<int>({{1, 2}, {3, 4}});
	a.display();

	Matrix<int> b = Matrix<int>({{1, 2}, {3, 4}});
	b.display();

	Matrix<int> c = a + b;
	c.display();
	
	return 0;
}
