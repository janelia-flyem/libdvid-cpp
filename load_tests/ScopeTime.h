#ifndef SCOPETIME_H
#define SCOPETIME_H

#include <sys/time.h>

class ScopeTime {
  public:
    ScopeTime(bool debug_ = true) : debug(debug_)
    {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        initial_time = tv.tv_sec + tv.tv_usec / 1000000.0;
    }
    ~ScopeTime()
    {
        if (debug) {
            std::cout << "Time Elapsed: " << getElapsed() << " seconds" << std::endl;
        }
    }
    double getElapsed()
    {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        double final_time = tv.tv_sec + tv.tv_usec / 1000000.0;
        return (final_time - initial_time);
    }
  private:
    double initial_time;
    bool debug;
};

#endif
