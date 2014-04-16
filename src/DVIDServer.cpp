#include "DVIDServer.h"
#include "DVIDException.h"
#include <boost/network/protocol/http/client.hpp>

using namespace boost::network;
using namespace boost::network::http;

namespace libdvid {

DVIDServer::DVIDServer(std::string addr_) : addr(addr_)
{
    client::request requestobj(get_uri_root() + "info");
    requestobj << header("Connection", "close");
    client request_client;
    client::response respdata = request_client.get(requestobj);
    int status_code = status(respdata);
    if (status_code != 200) {
        throw DVIDException(body(respdata), status_code);
    }
}

}
