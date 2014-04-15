#ifndef DVIDSERVER_H
#define DVIDSERVER_H

#include <string>

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
        addr + "/api/";
    }

  private:
    std::string addr;
};

}
#endif
