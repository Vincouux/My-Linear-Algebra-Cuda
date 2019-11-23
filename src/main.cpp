#include "Matrix/matrix.hpp"

#include <chrono>

uint64_t timeSinceEpochMillisec() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

int main() {
	/* Matrix A */
	Matrix<int> A = Matrix<int>(2000, 2000);
	//A.display();

	/* Matrix B */
	Matrix<int> B = Matrix<int>(2000, 2000);
	//B.display();

	/* Matrix C on CPU */
	printf("Starting CPU\n");
	uint64_t start = timeSinceEpochMillisec();
	Matrix<int> C = A.dot(B, false);
	uint64_t end = timeSinceEpochMillisec();
	printf("It took %li ms on CPU\n", end - start);
	//C.display();

	/* Matrix D on GPU */
	printf("Starting GPU\n");
	start = timeSinceEpochMillisec();
	Matrix<int> D = A.dot(B, true);
	end = timeSinceEpochMillisec();
	printf("It took %li ms on GPU\n", end - start);
	//D.display();

  printf("Matrix C == D: %s\n", C == D ? "Yes": "No");

	return 0;
}
