#ifndef DVIDEXCEPTION_H
#define DVIDEXCEPTION_H

#include <sstream>

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
