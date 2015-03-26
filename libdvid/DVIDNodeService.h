/*!
 * File
 *
 *TODO: assert >32bit labels in JSON
 *
 * \author Stephen Plaza
*/

#ifndef DVIDNODESERVICE_H
#define DVIDNODESERVICE_H

#include "BinaryData.h"
#include "DVIDVoxels.h"
#include "DVIDGraph.h"
#include "DVIDConnection.h"

#include <json/value.h>
#include <vector>
#include <fstream>
#include <string>

namespace libdvid {

typedef std::string UUID;

enum Dims2D { XY, XZ, YZ };


class DVIDNodeService {
  public:
    // check that node is available
    DVIDNodeService(std::string web_addr_, UUID uuid_);
 
    // get meta for the type 
    Json::Value get_typeinfo(std::string datatype_name);

    // ** datatype creation **
    // ?! allow specification of binary options OR better yet, expose a few of the main options
    bool create_grayscale8(std::string datatype_name);
    bool create_labelblk(std::string datatype_name);
    bool create_keyvalue(std::string keyvalue);
    bool create_graph(std::string name);

    
    // ?! block interface (create a vector of blocks) -- similar to 3D vol but without params (handle lz4)
    // ?! automatically blast ND volume into block array and compress if necessary

    // ** custom ** -- with respect to the node -- ?! check that it begins with '/' ?!
    BinaryDataPtr custom_request(std::string endpoint, BinaryDataPtr payload,
            ConnectionMethod method);


    // ** tile retrieval **
    // ?! how to determine whether JPEG or PNG without reading the meta -- use boost gil?!
    // retrieve 1-byte tile of a given resolution
    Grayscale2D get_tile_slice(std::string datatype_instance, Dims2D dims,
            unsigned int scaling, std::vector<unsigned int> tile_loc);


    BinaryDataPtr get_tile_slice_binary(std::string datatype_instance, Dims2D dims,
            unsigned int scaling, std::vector<unsigned int> tile_loc);

    // ** 3D gets and posts -- supports throttling **
    // ?! allow user to specify byte buffer for get -- just pass into new binary structure
    Grayscale3D get_gray3D(std::string datatype_instance, Dims_t dims,
            std::vector<unsigned int> offset, bool throttle=true, char* byte_buffer = 0);

    // specify the ordering of x,y,z
    Grayscale3D get_gray3D(std::string datatype_instance, Dims_t dims,
            std::vector<unsigned int> offset, std::vector<unsigned int> channels, bool throttle=true, char* byte_buffer = 0);
    
 
    Labels3D get_labels3D(std::string datatype_instance, Dims_t dims,
            std::vector<unsigned int> offset, bool throttle=true, char* byte_buffer = 0);
    // specify the ordering of x,y,z
    Labels3D get_labels3D(std::string datatype_instance, Dims_t dims,
            std::vector<unsigned int> offset, std::vector<unsigned int> channels, bool throttle=true, char* byte_buffer = 0);


    // must be block aligned 
    void put_labels3D(std::string datatype_instance, Labels3D& volume,
            std::vector<unsigned int> offset, bool throttle=true, bool use_blocks=false);
    
    void put_gray3D(std::string datatype_instance, Grayscale3D& volume,
            std::vector<unsigned int> offset, bool throttle=true, bool use_blocks=false);

    // ** Key-Value Interface **
    void put(std::string keyvalue, std::string key, BinaryDataPtr value);
    void put(std::string keyvalue, std::string key, std::ifstream& fin);
    void put(std::string keyvalue, std::string key, Json::Value& data);

    BinaryDataPtr get(std::string keyvalue, std::string key);
    // could return a reference but assuming that this is used for short messages
    Json::Value get_json(std::string keyvalue, std::string key);
    

    // ** DVID graph interface **

    void get_subgraph(std::string graph_name, std::vector<Vertex>& vertices, Graph& graph);

    void get_vertex_neighbors(std::string graph_name, Vertex vertex, Graph& graph);

    void update_vertices(std::string graph_name, std::vector<Vertex>& vertices);
    void update_edges(std::string graph_name, std::vector<Edge>& edges);

    void get_properties(std::string graph_name, std::vector<Vertex> vertices, std::string key,
            std::vector<BinaryDataPtr>& properties, VertexTransactions& transactions);

    void set_properties(std::string graph_name, std::vector<Vertex>& vertices, std::string key,
            std::vector<BinaryDataPtr>& properties, VertexTransactions& transactions,
            std::vector<Vertex>& leftover_vertices);

    void get_properties(std::string graph_name, std::vector<Edge> edges, std::string key,
            std::vector<BinaryDataPtr>& properties, VertexTransactions& transactions);

    void set_properties(std::string graph_name, std::vector<Edge>& edges, std::string key,
            std::vector<BinaryDataPtr>& properties, VertexTransactions& transactions,
            std::vector<Edge>& leftover_edges);

  private:
    UUID uuid;
    DVIDConnection connection;
    // ?! maybe add node meta data and kill info call ??

    void put_volume(std::string datatype_instance, BinaryDataPtr volume,
            std::vector<unsigned int> sizes, std::vector<unsigned int> offset,
            bool throttle, bool use_blocks);

    bool create_datatype(std::string datatype, std::string datatype_name);
    bool exists(std::string datatype_endpoint);

    BinaryDataPtr get_volume3D(std::string datatype_inst, Dims_t sizes,
        std::vector<unsigned int> offset, std::vector<unsigned int> channels,
        bool throttle, char* byte_buffer);

    std::string construct_volume_uri(std::string datatype_inst, Dims_t sizes,
            std::vector<unsigned int> offset, std::vector<unsigned int> channels, bool throttle);
};

}

#endif
