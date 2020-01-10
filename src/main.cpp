#include "Matrix/matrix.hpp"

int main() {
    srand((unsigned)time(0));

	/* Matrix A */
	Matrix<float> A = Matrix<float>({{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.5}});
	A.display();

	/* Matrix B */
	Matrix<float> B = Matrix<float>({{0}, {1}});
	B.display();

    /*
	Matrix<float> C = Matrix<float>({{0, 1}, {1, 0}});
	C.display();
	Matrix<float> D = Matrix<float>({{0.5}, {0.5}});
	D.display();
    */

    Matrix<float> E = A * B;
    if (E == A.dot(B, false)) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }
    E.display();

    /*
    Matrix<float> F = A * C;
    if (F == A.dot(C, false)) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }
    F.display();

    Matrix<float> G = A * D;
    if (G == A.dot(D, false)) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }
    G.display();
    */

	return 0;
}
