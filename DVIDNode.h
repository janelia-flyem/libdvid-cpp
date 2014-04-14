#ifndef DVIDNODE_H
#define DVIDNODE_H

#include <string>
#include <json/value.h>
#include "BinaryData.h"
#include "DVIDServer.h"
#include "DVIDVoxels.h"

namespace libdvid {

typedef std::string UUID;

class DVIDNode {
  public:
    // ?! check that node is available -- user server interface
    DVIDNode(DVIDServer web_addr_, UUID uuid_) : 
        web_addr(web_addr_), uuid(uuid_) {}

    // throw error if start point is 2D
    void get_gray_slice(std::string name, Coords start, Size size,
            Channels channels, DVIDGrayPtr& gray);
    void get_label_slice(std::string name, Coords start, Size size,
            Channels channels, DVIDLabelPtr& labels);
    
    void write_label_slice(std::string name, Coords start, Size size,
            Channels channels);


    // Key-Value Interface

    // will ignore if keyvalue already exists
    void create_keyvalue(std::string keyvalue);

    void put(std::string keyvalue, std::string key, BinaryDataPtr value);
    void put(std::string keyvalue, std::string key, ifstream& fin);
    void put(std::string keyvalue, std::string key, Json::Value& data);

    void get(std::string keyvalue, std::string key, BinaryDataPtr& value);
    void get(std::string keyvalue, std::string key, Json::Value& data);

  private:
    UUID uuid;
    DVIDServer web_addr;

    // ?! maybe add node meta data ?? -- but probably make it on demand

};

}

#endif
