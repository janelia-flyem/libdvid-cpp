#ifndef DVIDEXCEPTION_H
#define DVIDEXCEPTION_H

#include "Utilities.h"
#include <sstream>

namespace libdvid {

class DVIDException : public ErrMsg {
  public:
    DVIDException(std::string msg_, int status_) : ErrMsg(msg_)
    {
        std::stringstream sstr;
        sstr << "DVID Error (" << status_ << "): " << msg_;
        msg = sstr.str();
    }
    ~DVIDException() throw() {}

  private:
    int status;
    std::string temp_msg;
};

}

#endif
