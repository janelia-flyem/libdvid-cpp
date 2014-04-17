#ifndef DVIDNODE_H
#define DVIDNODE_H

#include "BinaryData.h"
#include "DVIDServer.h"
#include "DVIDVoxels.h"
#include "Utilities.h"
#include <json/value.h>
#include <fstream>
#include <string>
#include <boost/network/protocol/http/client.hpp>

namespace libdvid {

typedef std::string UUID;

class DVIDNode {
  public:
    // check that node is available
    DVIDNode(DVIDServer web_addr_, UUID uuid_);

    // throw error if start point is 2D
    void get_gray_slice(std::string datatype_instance, tuple start,
            tuple sizes, tuple channels, DVIDGrayPtr& gray);
    void get_label_slice(std::string datatype_instance, tuple start,
            tuple sizes, tuple channels, DVIDLabelPtr& labels);
    
    void write_label_slice(std::string datatype_instance, tuple start,
            tuple sizes, tuple channels, BinaryDataPtr data);

    // Key-Value Interface

    // will ignore if keyvalue already exists
    void create_keyvalue(std::string keyvalue);

    void put(std::string keyvalue, std::string key, BinaryDataPtr value);
    void put(std::string keyvalue, std::string key, std::ifstream& fin);
    void put(std::string keyvalue, std::string key, Json::Value& data);

    void get(std::string keyvalue, std::string key, BinaryDataPtr& value);
    void get(std::string keyvalue, std::string key, Json::Value& data);

  private:
    UUID uuid;
    DVIDServer web_addr;
    boost::network::http::client request_client;

    std::string construct_volume_uri(std::string datatype_inst, tuple start, tuple sizes, tuple channels);
    void retrieve_volume(std::string datatype_inst, tuple start, tuple sizes, tuple channels, std::string& volume);

    // ?! maybe add node meta data ?? -- but probably make it on demand

};

}

#endif
