#ifndef DVIDSERVERSERVICE_H
#define DVIDSERVERSERVICE_H

#include "DVIDConnection.h" 
#include <string>

namespace libdvid {

class DVIDServerService {
  public:
    DVIDServerService(std::string addr_);
    
    std::string create_new_repo(std::string alias, std::string description);
    
    std::string get_uri_root() const
    {
        return connection.get_uri_root();
    }
  private:
    DVIDConnection connection;
};

}
#endif
