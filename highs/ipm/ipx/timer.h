#ifndef IPX_TIMER_H_
#define IPX_TIMER_H_

#include <chrono>

namespace ipx {

class Timer {
public:
    Timer(const double offset=0);
    double Elapsed() const;
    void Reset(const bool first = false);

private:
    typedef std::chrono::time_point<std::chrono::high_resolution_clock>
        TimePoint;
    static TimePoint tic();
    static double toc(TimePoint start);
    TimePoint t0_;
public:
    double offset_;
};

}  // namespace ipx

#endif  // IPX_TIMER_H_
