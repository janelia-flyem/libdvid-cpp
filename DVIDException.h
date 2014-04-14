#ifndef DVIDEXCEPTION_H
#define DVIDEXCEPTION_H

#include "Utilities.h"
#include <ostream>

namespace libdvid {

class DVIDException : public ErrMsg {
  public:
    DVIDException(int status_) :
        ErrMsg(msg_), status(status_) {}

    std::string get_msg()
    {
        std::stringstream sstr;
        sstr << status << ": " << ErrMsg::get_msg();
        return sstr.string();
    }
  private:
    int status;
};

}

#endif
