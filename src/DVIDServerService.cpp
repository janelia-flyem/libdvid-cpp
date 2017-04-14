#include "DVIDServerService.h"
#include "DVIDException.h"
#include "BinaryData.h"

#include <json/json.h>

using std::string;

namespace libdvid {

DVIDServerService::DVIDServerService(std::string addr_, string user, string app) : 
    connection(addr_, user, app)
{
    string endpoint = "/server/info";
    string respdata;
    BinaryDataPtr binary = BinaryData::create_binary_data();
    int status_code = connection.make_request(endpoint, GET, BinaryDataPtr(),
            binary, respdata, DEFAULT);
}

std::string DVIDServerService::create_new_repo(std::string alias, std::string description)
{
    // JSON data to write
    string string_data = "{\"alias\": \"" + alias + "\", \"description\": \""
        + description + "\"}";
    
    string respdata;
    BinaryDataPtr payload = BinaryData::create_binary_data(string_data.c_str(),
            string_data.length());
    BinaryDataPtr result = BinaryData::create_binary_data();
   
    // create new repo 
    string endpoint = string("/repos");
    int status_code = connection.make_request(endpoint, POST, payload, result,
            respdata, JSON);

    // parse return binary
    Json::Reader json_reader;
    Json::Value jdata;
    if (!json_reader.parse(result->get_data(), jdata)) {
        throw ErrMsg("Could not decode JSON");
    }

    return jdata["root"].asString();
}

}
