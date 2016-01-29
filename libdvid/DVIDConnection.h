/*!
 * This file defines the main functionality for connecting to DVID
 * and helper functions to send and receive data.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#ifndef DVIDCONNECTION_H
#define DVIDCONNECTION_H

#include "BinaryData.h"
#include <string>

namespace libdvid {

//! Define connection methods
enum ConnectionMethod { HEAD, GET, POST, PUT, DELETE};

//! Define connection types
enum ConnectionType {DEFAULT, JSON, BINARY};

/*!
 * Creates a libcurl connection and 
 * provides utilities for transfering data between this library
 * and DVID.  Each service will call DVIDConnection independently
 * and does not need to be accessed very often by the end-user.
 * This class uses a static variable to set the curl context
 * and also sets a curl connection.
 *
 * WARNING: It is currently not possible to use a single DVIDConnection
 * for multiple threads.  Users should instantiate multiple services
 * to access DVID rather than reusing the same one.  This problem will
 * be fixed when the curl connection is given thread-level scope as
 * available in C++11.
*/
class DVIDConnection {
  public:
    /*!
     * Starts curl connection.
    */
    explicit DVIDConnection(std::string addr_);
  
    /*!
     * Copy constructor to ensure that creation of curl connection
     * is created and detroyed propertly per instance -- this will
     * ensure that each copy is can be run in thread-safe manner.
    */ 
    DVIDConnection(const DVIDConnection& copy_connection);

    /*!
     * Destroys curl connection.
    */
    ~DVIDConnection();

     /*!
     * Simple HEAD requests to DVID.  An exception is generated
     * if curl cannot properly connect to the URL.
     *
     * \param url endpoint where request is performed
     * \return html status code
    */ 
    int make_head_request(std::string endpoint);

   /*!
     * Main helper function retrieving data from DVID.  The function
     * performs the action specified in method.  An exception is generated
     * if curl cannot properly connect to the URL.
     *
     * \param url endpoint where request is performed
     * \param method http verb (HEAD, GET, POST, PUT, DELETE)
     * \param payload binary data containing data to be posted
     * \param results binary data containing the result
     * \param error_msg error message if there is an error
     * \param type connection type for request
     * \param timeout timeout for the request
     * \return html status code
    */ 
    int make_request(std::string endpoint, ConnectionMethod method, BinaryDataPtr payload,
            BinaryDataPtr results, std::string& error_msg, ConnectionType type=DEFAULT,
            int timeout=DEFAULT_TIMEOUT);

    /*!
     * Get the address for the DVID connection.
    */
    std::string get_addr() const
    {
        return addr;
    }

    /*!
     * Get the prefix for all DVID API calls
    */
    std::string get_uri_root() const
    {
        if (directaddress == "") {
            return (addr + DVID_PREFIX);
        } else {
            return (directaddress + DVID_PREFIX);
        }
    }

    //! default timeout in seconds
    static const int DEFAULT_TIMEOUT = 120;

  private:
    /*!
     * Assignment doesn't really make much sense -- just disable.
    */
    DVIDConnection& operator=(const DVIDConnection& connection);

    //! reuse curl connection -- eventually make this thread static and
    //! initialize once (CURL typedef is actually a void*)
    void* curl_connection;

    //! DVID address
    std::string addr;
    
    //! DVID address to bypass DNS (IP + port)
    std::string directaddress;

    //! prefix for all DVID calls (versioning may be added here in the future) 
    static const char* DVID_PREFIX;
};

}

#endif
