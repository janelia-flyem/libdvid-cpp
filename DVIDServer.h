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

  private:
    std::string addr;
};

}
#endif
