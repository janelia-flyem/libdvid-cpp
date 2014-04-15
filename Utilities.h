#ifndef UTILITIES_H
#define UTILITIES_H

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

std::ostream& operator<<(std::ostream& os, ErrMsg& err)
{
    os << err.get_msg(); 
    return os;
}

// e.g., dim1, dim2, dim3; x, y, z
typedef std::vector<int> tuple;

}

#endif
