#ifndef SCOPETIME_H
#define SCOPETIME_H

#include <chrono>
#include <iostream>

class ScopeTime {
  using clock = std::chrono::high_resolution_clock;

  public:
    ScopeTime(bool debug_ = true) : debug(debug_), initial_time(clock::now())
    {
    }
    ~ScopeTime()
    {
        if (debug) {
            std::cout << "Time Elapsed: " << getElapsed() << " seconds" << std::endl;
        }
    }
    double getElapsed()
    {
        auto final_time = clock::now();
        std::chrono::duration<double> diff = (final_time - initial_time);
        return diff.count();
    }
  private:
    std::chrono::time_point<clock> initial_time;
    bool debug;
};

#endif
