#include "DVIDNode.h"
#include "DVIDException.h"
#include <boost/network/protocol/http/client.hpp>
#include <json/json.h>
#include <set>

using std::ifstream; using std::set; using std::stringstream;
Json::Reader json_reader;

using namespace boost::network;
using namespace boost::network::http;

namespace libdvid {

DVIDNode::DVIDNode(DVIDServer web_addr_, UUID uuid_) : 
    web_addr(web_addr_), uuid(uuid_)
{
    client::request requestobj(web_addr.get_uri_root() + "node/" + uuid + "/info");
    requestobj << header("Connection", "close");
    client request_client;
    client::response respdata = request_client.get(requestobj);
    int status_code = status(respdata);
    if (status_code != 200) {
        throw DVIDException(body(respdata), status_code);
    }
}

void DVIDNode::create_keyvalue(std::string keyvalue)
{
    client::request requestobj(web_addr.get_uri_root() + "dataset/" + uuid +
            "/new/keyvalue" + keyvalue );
    requestobj << header("Connection", "close");
    client request_client;

    std::string data("{}");
    client::response respdata = request_client.post(requestobj,
            "application/json", data);
    int status_code = status(respdata);
    if (status_code != 200) {
        throw DVIDException(body(respdata), status_code);
    }
}

void DVIDNode::put(std::string keyvalue, std::string key, BinaryDataPtr value)
{
    client::request requestobj(web_addr.get_uri_root() + "node/" + uuid +
            "/" + keyvalue + "/" + key);
    requestobj << header("Connection", "close");
    client request_client;

    client::response respdata = request_client.post(requestobj,
            "application/octet-stream", value->get_data());
    int status_code = status(respdata);
    if (status_code != 200) {
        throw DVIDException(body(respdata), status_code);
    }
}

void DVIDNode::put(std::string keyvalue, std::string key, ifstream& fin)
{
    BinaryDataPtr data = BinaryData::create_binary_data(fin);
    put(keyvalue, key, data);
}


void DVIDNode::put(std::string keyvalue, std::string key, Json::Value& data)
{
    stringstream datastr;
    datastr << data;
    BinaryDataPtr bdata = BinaryData::create_binary_data(datastr.str().c_str());
    put(keyvalue, key, bdata);
}


void DVIDNode::get(std::string keyvalue, std::string key, BinaryDataPtr& value)
{
    client::request requestobj(web_addr.get_uri_root() + "node/" + uuid +
            "/" + keyvalue + "/" + key);
    requestobj << header("Connection", "close");
    client request_client;
    client::response respdata = request_client.get(requestobj);
    int status_code = status(respdata);
    if (status_code != 200) {
        throw DVIDException(body(respdata), status_code);
    }
    std::string data = body(respdata);
    // ?! allow intialization to happen in constructor
    value = BinaryData::create_binary_data(data.c_str());
}

void DVIDNode::get(std::string keyvalue, std::string key, Json::Value& data)
{
    BinaryDataPtr binary;
    get(keyvalue, key, binary);
    
    Json::Reader json_reader;
    if (!json_reader.parse(binary->get_data(), data)) {
        throw ErrMsg("Could not decode JSON");
    }
}

void DVIDNode::get_gray_slice(std::string datatype_instance, tuple start,
        tuple sizes, tuple channels, DVIDGrayPtr& gray)
{
    std::string volume;
    retrieve_volume(datatype_instance, start, sizes, channels, volume);
    gray = DVIDVoxels<unsigned char>::get_dvid_voxels(volume); 
}

void DVIDNode::get_label_slice(std::string datatype_instance, tuple start,
        tuple sizes, tuple channels, DVIDLabelPtr& labels)
{
    std::string volume;
    retrieve_volume(datatype_instance, start, sizes, channels, volume);
    labels = DVIDVoxels<unsigned long long>::get_dvid_voxels(volume); 
}

void DVIDNode::write_label_slice(std::string datatype_instance, tuple start,
        tuple sizes, tuple channels, BinaryDataPtr data)
{
    client::request requestobj(construct_volume_uri(
                datatype_instance, start, sizes, channels));
    requestobj << header("Connection", "close");
    client request_client;
    client::response respdata = request_client.post(requestobj,
            "application/octet-stream", data->get_data());
    int status_code = status(respdata);
    if (status_code != 200) {
        throw DVIDException(body(respdata), status_code);
    }
}

std::string DVIDNode::construct_volume_uri(std::string datatype_inst, tuple start, tuple sizes, tuple channels)
{
    std::string uri = web_addr.get_uri_root() + "node/" + uuid + "/"
                    + datatype_inst + "/raw/";
    
    if (start.size() < 3) {
        throw ErrMsg("libdvid does not support 2D datatype instances");
    }
    if (channels.size() == 0) {
        throw ErrMsg("must specify more than one channel");
    }
    if (sizes.size() != channels.size()) {
        throw ErrMsg("number of size dimensions does not match the number of channels");
    }
    stringstream sstr;
    sstr << uri;
    sstr << channels[0];

    // retrieve at least a 3D volume
    set<int> used_channels;
    for (int i = 0; i < channels.size(); ++i) {
        used_channels.insert(channels[i]);   
    }
    int channel_id = 0;
    for (int i = channels.size(); i < 3; ++i) {
        while (used_channels.find(channel_id) != used_channels.end()) {
            ++channel_id;
        }
        channels.push_back(channel_id);
    }

    for (int i = 1; i < channels.size(); ++i) {
        sstr << "_" << channels[i];
    }
    
    // retrieve at least a 3D volume
    for (int i = sizes.size(); i < 3; ++i) {
        sizes.push_back(1);
    }
    sstr << "/" << sizes[0];
    for (int i = 1; i < sizes.size(); ++i) {
        sstr << "_" << sizes[i];
    }
    sstr << "/" << start[0];
    for (int i = 1; i < start.size(); ++i) {
        sstr << "_" << start[i];
    }

    return sstr.str();
}

void DVIDNode::retrieve_volume(std::string datatype_inst, tuple start, tuple sizes, tuple channels, std::string& volume)
{
    client::request requestobj(construct_volume_uri(datatype_inst, start, sizes, channels));
    requestobj << header("Connection", "close");
    client request_client;
    client::response respdata = request_client.get(requestobj);
    int status_code = status(respdata);
    if (status_code != 200) {
        throw DVIDException(body(respdata), status_code);
    }
    volume = body(respdata);
}

}

