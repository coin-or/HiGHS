#include "ipm/ipx/timer.h"
#include <cstdio>
namespace ipx {

Timer::Timer(const double offset)
    : offset_(offset) {
  if (offset_) printf("Timer::Timer offset_ = %g\n", offset_);
    Reset(true);
  if (offset_) printf("Timer::Timer offset_ = %g\n", offset_);
}

double Timer::Elapsed() const {
    return toc(t0_);
}

void Timer::Reset(const bool first) {
    using namespace std::chrono;
    double prev_offset = offset_;
    if (!first)
      offset_ -= duration_cast<duration<double>>(t0_.time_since_epoch()).count();    
    t0_ = tic();
    if (!first) 
      offset_ += duration_cast<duration<double>>(t0_.time_since_epoch()).count();
    if (offset_) printf("Timer::Reset offset goes from %g to %g\n", prev_offset, offset_);
}

Timer::TimePoint Timer::tic() {
    return std::chrono::high_resolution_clock::now();
}

double Timer::toc(TimePoint start) {
    TimePoint end = tic();
    std::chrono::duration<double> diff = end-start;
    return diff.count();
}

}  // namespace ipx
