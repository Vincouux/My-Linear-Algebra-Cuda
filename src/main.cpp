#include "Matrix/matrix.hpp"

int main() {
	Matrix<int> a = Matrix<int>({{1, 2}, {3, 4}});
	a.display();

	Matrix<int> b = Matrix<int>({{1, 2}, {3, 4}});
	b.display();

	Matrix<int> c = a + b;
	c.display();

	Matrix<float> d = Matrix<float>(10, 10);
	d.display();

	Matrix<float> e = Matrix<float>(10, 10);
	e.display();

	Matrix<float> f = e + d;
	f.display();

	return 0;
}
