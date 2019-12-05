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
    string error_msg;
    BinaryDataPtr binary = BinaryData::create_binary_data();
    int status_code = connection.make_request(endpoint, GET, BinaryDataPtr(),
            binary, error_msg, DEFAULT);
}

std::string DVIDServerService::create_new_repo(std::string alias, std::string description)
{
    // JSON data to write
    string string_data = "{\"alias\": \"" + alias + "\", \"description\": \""
        + description + "\"}";
    
    string error_msg;
    BinaryDataPtr payload = BinaryData::create_binary_data(string_data.c_str(),
            string_data.length());
    BinaryDataPtr result = BinaryData::create_binary_data();
   
    // create new repo 
    string endpoint = string("/repos");
    int status_code = connection.make_request(endpoint, POST, payload, result,
            error_msg, JSON);

    Json::Value jdata = result->get_json_value();

    return jdata["root"].asString();
}

}
