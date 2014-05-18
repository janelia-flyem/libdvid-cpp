#include "DVIDNode.h"
#include "DVIDException.h"
#include <json/json.h>
#include <set>

// std::string collides with a boost string
using namespace boost::network::http;
using namespace boost::network;

using std::ifstream; using std::set; using std::stringstream;
Json::Reader json_reader;

const int TransactionLimit = 1000;

namespace libdvid {

DVIDNode::DVIDNode(DVIDServer web_addr_, UUID uuid_) : 
    web_addr(web_addr_), uuid(uuid_)
{
    client::request requestobj(web_addr.get_uri_root() + "dataset/" + uuid + "/info");
    requestobj << header("Connection", "close");
    client::response respdata = request_client.get(requestobj);
    int status_code = status(respdata);
    if (status_code != 200) {
        throw DVIDException(body(respdata), status_code);
    }
}

bool DVIDNode::create_grayscale8(std::string datatype_name)
{
    return create_datatype("grayscale8", datatype_name);
}

bool DVIDNode::create_labels64(std::string datatype_name)
{
    return create_datatype("labels64", datatype_name);
}

bool DVIDNode::create_keyvalue(std::string keyvalue)
{
    return create_datatype("keyvalue", keyvalue);
}

bool DVIDNode::create_graph(std::string graph_name)
{
    return create_datatype("labelgraph", graph_name);
}

void DVIDNode::put(std::string keyvalue, std::string key, BinaryDataPtr value)
{
    client::request requestobj(web_addr.get_uri_root() + "node/" + uuid +
            "/" + keyvalue + "/" + key);
    requestobj << header("Connection", "close");

    client::response respdata = request_client.post(requestobj,
            value->get_data(), std::string("application/octet-stream"));
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
    BinaryDataPtr bdata = BinaryData::create_binary_data(datastr.str().c_str(), datastr.str().length());
    put(keyvalue, key, bdata);
}


void DVIDNode::get(std::string keyvalue, std::string key, BinaryDataPtr& value)
{
    client::request requestobj(web_addr.get_uri_root() + "node/" + uuid +
            "/" + keyvalue + "/" + key);
    requestobj << header("Connection", "close");
    client::response respdata = request_client.get(requestobj);
    int status_code = status(respdata);
    if (status_code != 200) {
        throw DVIDException(body(respdata), status_code);
    }
    std::string data = body(respdata);
    // ?! allow intialization to happen in constructor
    value = BinaryData::create_binary_data(data.c_str(), data.length());
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

void DVIDNode::get_vertex_neighbors(std::string graph_name, VertexID id, Graph& graph)
{
    BinaryDataPtr binary;

    // use key/value functionality to make call
    stringstream uri_ending;
    uri_ending << "neighbors/" << id;
    get(graph_name, uri_ending.str(), binary);
    
    Json::Reader json_reader;
    Json::Value data;
    if (!json_reader.parse(binary->get_data(), data)) {
        throw ErrMsg("Could not decode JSON");
    }
    graph.import_json(data);
}

void DVIDNode::update_vertices(std::string graph_name, std::vector<Vertex>& vertices)
{
    int num_examined = 0;

    while (num_examined < vertices.size()) {
        Graph graph;
        
        // grab 1000 vertices at a time
        int max_size = ((num_examined + TransactionLimit) > vertices.size()) ? vertices.size() : (num_examined + TransactionLimit); 
        for (; num_examined < max_size; ++num_examined) {
            graph.vertices.push_back(vertices[num_examined]);
        }

        Json::Value data;
        graph.export_json(data);

        put(graph_name, std::string("weight"), data);
    }
}
    
void DVIDNode::update_edges(std::string graph_name, std::vector<Edge>& edges)
{

    int num_examined = 0;
    while (num_examined < edges.size()) {
        VertexSet examined_vertices; 
        Graph graph;
        
        for (; num_examined < edges.size(); ++num_examined) {
            graph.edges.push_back(edges[num_examined]);
            examined_vertices.insert(edges[num_examined].id1);
            examined_vertices.insert(edges[num_examined].id2);
            
            // break if it is not possible to add another edge transaction
            // (assuming that both vertices of that edge will be new vertices)
            if (examined_vertices.size() >= (TransactionLimit - 1)) {
                break;
            }
        }
        Json::Value data;
        graph.export_json(data);

        put(graph_name, std::string("weight"), data);
    }
}
    
void DVIDNode::get_volume_roi(std::string datatype_instance, tuple start,
        tuple sizes, tuple channels, DVIDGrayPtr& gray)
{
    std::string volume;
    retrieve_volume(datatype_instance, start, sizes, channels, volume);
    gray = DVIDVoxels<unsigned char>::get_dvid_voxels(volume); 
}

void DVIDNode::get_volume_roi(std::string datatype_instance, tuple start,
        tuple sizes, tuple channels, DVIDLabelPtr& labels)
{
    std::string volume;
    retrieve_volume(datatype_instance, start, sizes, channels, volume);
    labels = DVIDVoxels<unsigned long long>::get_dvid_voxels(volume); 
}

void DVIDNode::write_volume_roi(std::string datatype_instance, tuple start,
        tuple sizes, tuple channels, BinaryDataPtr data)
{
    client::request requestobj(construct_volume_uri(
                datatype_instance, start, sizes, channels));
    requestobj << header("Connection", "close");
    client::response respdata = request_client.post(requestobj,
            data->get_data(), std::string("application/octet-stream"));
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
    client::response respdata = request_client.get(requestobj);
    int status_code = status(respdata);
    if (status_code != 200) {
        throw DVIDException(body(respdata), status_code);
    }
    volume = body(respdata);
}

bool DVIDNode::exists(std::string uri)
{ 
    try {
        client::request requestobj(uri);
        requestobj << header("Connection", "close");
        client::response respdata = request_client.get(requestobj);
        int status_code = status(respdata);
        if (status_code != 200) {
            return false;
        }
    } catch (std::exception& e) {
        return false;
    }

    return true;
}

bool DVIDNode::create_datatype(std::string datatype, std::string datatype_name)
{
    if (exists(web_addr.get_uri_root() + "node/" + uuid + "/" + datatype_name + "/info")) {
        return false;
    } 

    client::request requestobj(web_addr.get_uri_root() + "dataset/" + uuid +
            "/new/" + datatype + "/" + datatype_name );
    requestobj << header("Connection", "close");

    std::string data("{}");
    client::response respdata = request_client.post(requestobj,
            data, std::string("application/json"));
    int status_code = status(respdata);
    if (status_code != 200) {
        throw DVIDException(body(respdata), status_code);
    }

    return true;
}

}

