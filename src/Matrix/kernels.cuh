template <class Number> void addKernelWrapper(Number* m1, Number* m2, Number* m3, size_t size);

class Wrapper {
  public:
    template <class Number> void add(Number* m1, Number* m2, Number* m3, size_t size);
};

template <class Number> void Wrapper::add(Number* m1, Number* m2, Number* m3, size_t size) {
  addKernelWrapper(m1, m2, m3, size);
}
