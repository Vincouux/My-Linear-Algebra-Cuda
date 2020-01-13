template <class Number> void addKernelWrapper(Number* m1, Number* m2, Number* m3, size_t size);
template <class Number> void dotKernelWrapper(Number* m1, Number* m2, Number* m3, size_t resultRows, size_t resultColumns, size_t interiorColumns);

#include <iostream>

class Wrapper {
public:
    template <class Number> void add(Number* m1, Number* m2, Number* m3, size_t size);
    template <class Number> void dot(Number* m1, Number* m2, Number* m3, size_t resultRows, size_t resultColumns, size_t interiorColumns);
};

template <class Number> void Wrapper::add(Number* m1, Number* m2, Number* m3, size_t size) {
    addKernelWrapper(m1, m2, m3, size);
}

template <class Number> void Wrapper::dot(Number* m1, Number* m2, Number* m3, size_t resultRows, size_t resultColumns, size_t interiorColumns) {
    dotKernelWrapper(m1, m2, m3, resultRows, resultColumns, interiorColumns);
}
