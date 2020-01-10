#include "Matrix/matrix.hpp"

int main() {
    srand((unsigned)time(0));

	/* Matrix A */
	Matrix<float> A = Matrix<float>(2, 2);
	A.display();

	/* Matrix B */
	Matrix<float> B = Matrix<float>(2, 2);
	B.display();

    Matrix<float> C = A * B;
    C.display();

	return 0;
}
