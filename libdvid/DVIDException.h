/*!
 * This file contains simple custom objects for exception
 * handling in the libdvid library.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#ifndef DVIDEXCEPTION_H
#define DVIDEXCEPTION_H

#include <sstream>

namespace libdvid {

/*!
 * Exception handling for general non-DVID access errors.
 * It is a wrapper for simple string message.
*/
class ErrMsg : public std::exception { 
  public:
    /*!
     * Construct takes a string message for the error.
     * \param msg_ string message
    */
    ErrMsg(std::string msg_) : msg(msg_) {}
    
    /*!
     * Implement exception base class function.
    */
    virtual const char* what() const throw()
    {
        return msg.c_str();
    }

    /*!
     * Empty destructor.
    */
    ~ErrMsg() throw() {}
  protected:
    
    //! Error message
    std::string msg;
};

/*!
 * Function that allows formatting of error to standard output.
*/
std::ostream& operator<<(std::ostream& os, ErrMsg& err);

/*!
 * A special type of error message for DVID server accesses.
 * It contains the http status code.
*/
class DVIDException : public ErrMsg {
  public:
    /*!
     * Constructor takes string for the base class and the
     * http status code.
     * \param msg_ error message
     * \param status_ http status code
    */
    DVIDException(std::string msg_, int status_) : ErrMsg(msg_)
    {
        std::stringstream sstr;
        sstr << "DVID Error (" << status_ << "): " << msg_;
        msg = sstr.str();
    }

    /*!
     * Empty destructor.
    */
    ~DVIDException() throw() {}

  private:
    //! http status
    int status;
};

}

#endif
