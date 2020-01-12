#include "Matrix/matrix.hpp"

int main() {
    srand((unsigned)time(0));

	/* Matrix A */
	Matrix<float> A = Matrix<float>({{0.1, 0.2, 0.4}, {0.3, 0.4, 0.5}, {0.5, 0.6, 0.7}});
	A.display();
	Matrix<float> B = A.transpose();
    B.display();
    Matrix<float> C = B % A;
    C.display();
    Matrix<float> D = Matrix<float>({{1}, {2}, {3}});
    D.display();
    D.transpose().display();
    D.transpose().transpose().display();
	return 0;
}
