CXX = g++
NVCC = nvcc
CXXFLAGS = -Werror -Wextra -Wall -pedantic -std=c++17
TRASH = main *.o

all: run

.PHONY: run
run:
	$(NVCC) -c src/Matrix/matrix.cu
	$(CXX) $(CXXFLAGS) -c -I/usr/local/cuda-5.5/include src/main.cpp
	$(CXX) -o main main.o matrix.o -L/usr/local/cuda-5.5/lib64 -lcudart -lcurand -lcuda

.PHONY: clean
clean:
	$(RM) $(TRASH)
