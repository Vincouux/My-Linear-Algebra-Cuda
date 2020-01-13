#include "Matrix/matrix.hpp"

int main() {
	Matrix<float> A = Matrix<float>(1000, 1000);
    for (unsigned i = 0; i < 100000; i++) {
        Matrix<float> A = Matrix<float>(1000, 1000) * Matrix<float>(1000, 1000);
    }
	return 0;
}
