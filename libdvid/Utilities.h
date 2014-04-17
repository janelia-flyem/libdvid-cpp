#ifndef DUTILITIES_H
#define DUTILITIES_H

#include <string>
#include <ostream>
#include <vector>

namespace libdvid {

class ErrMsg : public std::exception { 
  public:
    ErrMsg(std::string msg_) : msg(msg_) {}
    virtual const char* what() const throw()
    {
        return msg.c_str();
    }
    ~ErrMsg() throw() {}
  protected:
    std::string msg;
};

std::ostream& operator<<(std::ostream& os, ErrMsg& err);

// e.g., dim1, dim2, dim3; x, y, z
typedef std::vector<int> tuple;

}

#endif
