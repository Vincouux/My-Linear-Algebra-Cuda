#include "Matrix/matrix.hpp"

extern void addWrapper(int* m1, int* m2, int* m3, size_t size);

int main() {

	// Allocating memory in host.
	size_t n = 10;
	int* m1 = (int*)calloc(n, sizeof(int));
	int* m2 = (int*)calloc(n, sizeof(int));

	// Setting each element to 1 and 2.
	for (size_t i = 0; i < n; i++) {
		m1[i] = 1;
		m2[i] = 2;
	}

	// Allocating memory for result and calling kernel function.
	int* m3 = (int*)calloc(n, sizeof(int));
	addWrapper(m1, m2, m3, n);

	// Printing result.
	for (int i = 0; i < 10; i++)
		std::cout << m3[i] << " ";

	free(m1);
	free(m2);
	free(m3);

	return 0;
}
