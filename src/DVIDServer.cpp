#include "DVIDServer.h"
#include "DVIDException.h"
#include <boost/network/protocol/http/client.hpp>
#include "BinaryData.h"
#include <json/json.h>

using namespace boost::network;
using namespace boost::network::http;
//Json::Reader json_reader;

namespace libdvid {

DVIDServer::DVIDServer(std::string addr_) : addr(addr_)
{
    client::request requestobj(get_uri_root() + std::string("server/info"));
    requestobj << header("Connection", "close");
    client request_client;
    client::response respdata = request_client.get(requestobj);
    int status_code = status(respdata);
    
    if (status_code != 200) {
        throw DVIDException(body(respdata), status_code);
    }
}

std::string DVIDServer::create_new_repo(std::string alias, std::string description)
{
    client::request requestobj(get_uri_root() + std::string("repos"));
    requestobj << header("Connection", "close");


    client request_client;
    std::string data = "{\"alias\": \"" + alias + "\", \"description\": \"" + description + "\"}";
    client::response respdata = request_client.post(requestobj,
            data, std::string("application/json"));

    int status_code = status(respdata);
    if (status_code != 200) {
        throw DVIDException(body(respdata), status_code);
    }

    std::string rdata = body(respdata);
    BinaryDataPtr binary = BinaryData::create_binary_data(rdata.c_str(), rdata.length());

    Json::Reader json_reader;
    Json::Value jdata;
    if (!json_reader.parse(binary->get_data(), jdata)) {
        throw ErrMsg("Could not decode JSON");
    }

    return jdata["root"].asString();

}

}
