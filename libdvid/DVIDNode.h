#ifndef DVIDNODE_H
#define DVIDNODE_H

#include "BinaryData.h"
#include "DVIDServer.h"
#include "DVIDVoxels.h"
#include "DVIDGraph.h"
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
    bool create_grayscale8(std::string datatype_name);
    bool create_labels64(std::string datatype_name);
    
    void get_volume_roi(std::string datatype_instance, tuple start,
            tuple sizes, tuple channels, DVIDGrayPtr& gray);
    void get_volume_roi(std::string datatype_instance, tuple start,
            tuple sizes, tuple channels, DVIDLabelPtr& labels);
    
    void write_volume_roi(std::string datatype_instance, tuple start,
            tuple sizes, tuple channels, BinaryDataPtr data);

    // --- Key-Value Interface ---

    // will ignore if keyvalue already exists
    bool create_keyvalue(std::string keyvalue);
    
    void put(std::string keyvalue, std::string key, BinaryDataPtr value);
    void put(std::string keyvalue, std::string key, std::ifstream& fin);
    void put(std::string keyvalue, std::string key, Json::Value& data);

    void get(std::string keyvalue, std::string key, BinaryDataPtr& value);
    void get(std::string keyvalue, std::string key, Json::Value& data);

    // --- DVID graph interface ---

    void get_vertex_neighbors(std::string graph_name, VertexID id, Graph& graph);

    void update_vertices(std::string graph_name, std::vector<Vertex>& vertices);
    void update_edges(std::string graph_name, std::vector<Edge>& edges);
    bool create_graph(std::string name);

  private:
    UUID uuid;
    DVIDServer web_addr;
    boost::network::http::client request_client;

    std::string construct_volume_uri(std::string datatype_inst, tuple start, tuple sizes, tuple channels);
    void retrieve_volume(std::string datatype_inst, tuple start, tuple sizes, tuple channels, std::string& volume);

    bool exists(std::string uri);

    bool create_datatype(std::string datatype, std::string datatype_name);

    // ?! maybe add node meta data ?? -- but probably make it on demand

};

}

#endif
