#ifndef DUTILITIES_H
#define DUTILITIES_H

#include <string>
#include <ostream>
#include <vector>

namespace libdvid {

class ErrMsg { 
  public:
    ErrMsg(std::string msg_) : msg(msg_) {}
    virtual std::string get_msg()
    {
        return msg;
    }

  private:
    std::string msg;
};

std::ostream& operator<<(std::ostream& os, ErrMsg& err);

// e.g., dim1, dim2, dim3; x, y, z
typedef std::vector<int> tuple;

}

#endif
