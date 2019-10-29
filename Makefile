CXX = g++
CXXFLAGS = -Werror -Wextra -Wall -pedantic -std=c++17
TRASH = main

all: run

.PHONY: run
run:
	$(CXX) $(CXXFLAGS) -o main src/main.cpp

.PHONY: clean
clean:
	$(RM) $(TRASH)
