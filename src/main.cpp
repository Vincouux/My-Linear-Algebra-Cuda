#include "Matrix/matrix.hpp"

int main() {
	Matrix<float> A = Matrix<float>(10, 10);
	A.display();
	A.setElementAt(0, 0, A.getElementAt(0, 0) + 2);
	A.display();
	return 0;
}
