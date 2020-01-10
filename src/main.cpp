#include "Matrix/matrix.hpp"

int main() {
    srand((unsigned)time(0));

	/* Matrix A */
	Matrix<float> A = Matrix<float>({{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.5}});
	A.display();
    A.apply([](float x){ return x *x; });
    A.display();
    A.getRow(1).display();

	return 0;
}
