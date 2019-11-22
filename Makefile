CXX = g++
NVCC = nvcc
CXXFLAGS = -Werror -Wextra -Wall -pedantic -std=c++17
NVFLAGS = -Xcompiler -rdynamic -lineinfo
TRASH = main *.o

all: build

.PHONY: build
build:
	$(NVCC) $(NVFLAGS) -c src/Matrix/kernels.cu
	$(CXX) $(CXXFLAGS) -c -I/usr/local/cuda-5.5/include src/main.cpp
	$(CXX) -o main main.o kernels.o -L/usr/local/cuda-5.5/lib64 -lcudart -lcurand -lcuda

.PHONY: run
run: build
	./main

.PHONY: clean
clean:
	$(RM) $(TRASH)
