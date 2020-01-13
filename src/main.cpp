#include "Matrix/matrix.hpp"

int main() {
	Matrix<float> A = Matrix<float>(10, 10, -0.25f, 0.25f);
	A.display();
	A = A * 2.f;
	A.display();
	return 0;
}
