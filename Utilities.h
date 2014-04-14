#ifndef UTILITIES_H
#define UTILITIES_H

namespace libdvid {

#include <string>

class ErrMsg { 
  public:
    ErrMsg(std::string msg_) : msg(msg_) {}
    virtual std::string get_msg()
    {
        return msg;
    }

  private:
    std::string msg;
}


std::ostream& operator<<(std::ostream& os, ErrMsg& err)
{
    os << err.get_msg(); 
    return os;
}

}

#endif
