#include <chrono>
#include <future>
#include <iostream>

#include "parallel/HighsParallel.h"

const int64_t n = 1000000000;

double fun() {
  double local{};
  for (int64_t i = 0; i < 3*n; ++i) local += sqrt(i);
  return local;
}


double fun_tasks() {
  double local{};
  highs::parallel::for_each(0,n,[&](int64_t start, int64_t end) {
    for (int i = start; i < end; ++i) {
        local += sqrt(i);
      }
    });
  return local;
}

int64_t fib_sequential(const int64_t n) {
  if (n <= 1) return 1;
  return fib_sequential(n - 1) + fib_sequential(n - 2);
}

int64_t fib(const int64_t n) {
  if (n <= 20) return fib_sequential(n);

  int64_t n1;
  highs::parallel::spawn([&]() { n1 = fib(n - 1); });
  int64_t n2 = fib(n - 2);
  highs::parallel::sync();

  // printf("fib(%ld) = %ld + %ld = %ld\n", n, n1, n2, n1 + n2);
  return n1 + n2;

}
int main() {
  // serial
  {
    auto start = std::chrono::high_resolution_clock::now();

    // double a = fun();
    // double b = fun();

    // assert(a == b);

    int result = fib_sequential(41);
    assert(result == 267914296);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("Serial time %f\n\n", duration.count());
  }

  // highs parallel
  {
    highs::parallel::initialize_scheduler(4);
    auto start = std::chrono::high_resolution_clock::now();

    // double a;
    // highs::parallel::spawn([&]() { a = fun(); });
    // double b = fun();
    // highs::parallel::sync();

    // assert(a == b);

    int result = fib(41);
    assert(result == 267914296);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("HiGHS  time %f\n\n", duration.count());

    HighsTaskExecutor::shutdown(true);
  }

  // thread
  {
    auto start = std::chrono::high_resolution_clock::now();

    // auto future = std::async(fun);
    // double a = fun();
    // double b = future.get();

    // assert(a == b);

    auto future = std::async(fib_sequential, 40);
    double n1 = fib_sequential(39);
    int n2 = future.get();
    int result = n1 + n2;

    assert(result == 267914296);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("thread time %f\n", duration.count());
  }

  return 0;
}