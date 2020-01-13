#include "Matrix/matrix.hpp"

int main() {
    srand((unsigned)time(0));

	/* Matrix A */
	Matrix<float> A = Matrix<float>("x_train.csv");
    std::cout << A.getHeight() << ", " << A.getWidth() << std::endl;
    A.display();
	return 0;
}
