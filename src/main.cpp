#include "Matrix/matrix.hpp"

int main() {
    srand((unsigned)time(0));

	/* Matrix A */
	Matrix<float> A = Matrix<float>({{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.5}});
	A.display();

    Matrix<float> B = 1.f / (1.f + (0.f - A).exponential());
    B.display();

	return 0;
}
