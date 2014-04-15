#ifndef DVIDEXCEPTION_H
#define DVIDEXCEPTION_H

#include "Utilities.h"
#include <sstream>

namespace libdvid {

class DVIDException : public ErrMsg {
  public:
    DVIDException(std::string msg_, int status_) :
        ErrMsg(msg_), status(status_) {}

    std::string get_msg()
    {
        std::stringstream sstr;
        sstr << "DVID Error (" << status << "): " << ErrMsg::get_msg();
        return sstr.str();
    }
  private:
    int status;
};

}

#endif
