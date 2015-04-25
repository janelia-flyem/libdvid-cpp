/*!
 * This file defines API for accessing the DVID server REST interface.
 * Only a subset of the REST interface is implemented.
 *
 * Note: to be thread safe intantiate a unique server service object
 * for each thread.
 *
 * TODO: expand API (such as retrieving all repos on the server) and 
 * load server meta on initialization.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#ifndef DVIDSERVERSERVICE_H
#define DVIDSERVERSERVICE_H

#include "DVIDConnection.h" 
#include <string>

namespace libdvid {

/*!
 * Class that helps access different functionality on a DVID server.
*/
class DVIDServerService {
  public:
    /*!
     * Constructor takes http address of DVID server.
     * \param addr_ DVID address
    */
    explicit DVIDServerService(std::string addr_);
    
    /*************** API for server services ***********************/

    /*!
     * Create a new DVID repo with the given alias name
     * and string description.  A DVID UUID is returned.
    */
    std::string create_new_repo(std::string alias, std::string description);
    
  private:

    //! HTTP connection with DVID
    DVIDConnection connection;
};

}
#endif
