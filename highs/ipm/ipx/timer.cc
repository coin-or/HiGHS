#include "ipm/ipx/timer.h"
#include <cstdio>
namespace ipx {

Timer::Timer(const double offset)
    : offset_(offset) {
    Reset(true);
}

double Timer::Elapsed() const {
    return toc(t0_);
}

void Timer::Reset(const bool first) {
    double prev_offset = offset_;
    if (!first) offset_ -= this->read();
    t0_ = tic();
    if (!first) offset_ += this->read();
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
