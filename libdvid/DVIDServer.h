#ifndef DVIDSERVER_H
#define DVIDSERVER_H

#include <string>
#include <boost/network/protocol/http/client.hpp>

namespace libdvid {

class DVIDServer {
  public:
    DVIDServer(std::string addr_);
    std::string get_addr()
    {
        return addr;
    }
    std::string get_uri_root()
    {
        return (addr + "/api/");
    }

  private:
    std::string addr;
};

}
#endif
