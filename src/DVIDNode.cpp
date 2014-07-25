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

void DVIDNode::put(std::string keyvalue, std::string key, BinaryDataPtr value, VertexSet& failed_vertices)
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
    
    VertexTransactions temp_transactions;
    std::string data = body(respdata);
    BinaryDataPtr binary = BinaryData::create_binary_data(data.c_str(), data.length());
    size_t byte_pos = load_transactions_from_binary(binary->get_data(),
            temp_transactions, failed_vertices);
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

void DVIDNode::get(std::string keyvalue, std::string key, BinaryDataPtr& value, Json::Value& json_data)
{
    client::request requestobj(web_addr.get_uri_root() + "node/" + uuid +
            "/" + keyvalue + "/" + key);
    requestobj << header("Connection", "close");

    // write header so body is seen
    stringstream datastr;
    datastr << json_data;
    stringstream strstream;
    strstream << datastr.str().length();
    requestobj << header("Content-Length", strstream.str());
    requestobj << header("Content-Type", "application/json");
    body(requestobj, datastr.str());

    client::response respdata = request_client.get(requestobj);
    int status_code = status(respdata);
    if (status_code != 200) {
        throw DVIDException(body(respdata), status_code);
    }
    std::string data = body(respdata);
    
    // ?! allow intialization to happen in constructor
    value = BinaryData::create_binary_data(data.c_str(), data.length());
}


void DVIDNode::get(std::string keyvalue, std::string key, BinaryDataPtr& value, BinaryDataPtr request_data)
{
    client::request requestobj(web_addr.get_uri_root() + "node/" + uuid +
            "/" + keyvalue + "/" + key);
    requestobj << header("Connection", "close");

    // write header so body is seen
    stringstream strstream;
    strstream << request_data->get_data().length();
    requestobj << header("Content-Length", strstream.str());
    requestobj << header("Content-Type", "application/octet-stream");
    body(requestobj, request_data->get_data());

    client::response respdata = request_client.get(requestobj);
    int status_code = status(respdata);
    if (status_code != 200) {
        throw DVIDException(body(respdata), status_code);
    }
    std::string data = body(respdata);
    
    // ?! allow intialization to happen in constructor
    value = BinaryData::create_binary_data(data.c_str(), data.length());
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

void DVIDNode::get_subgraph(std::string graph_name, std::vector<Vertex>& vertices, Graph& graph)
{
    // make temporary graph for query
    Graph temp_graph;
    for (int i = 0; i < vertices.size(); ++i) {
        temp_graph.vertices.push_back(vertices[i]);
    }   
    Json::Value data;
    temp_graph.export_json(data);

    // make request with json data
    BinaryDataPtr binary;

    get(graph_name, std::string("subgraph"), binary, data);
    
    Json::Reader json_reader;
    Json::Value returned_data;
    if (!json_reader.parse(binary->get_data(), returned_data)) {
        throw ErrMsg("Could not decode JSON");
    }
    graph.import_json(returned_data);
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

// vertex list is modified, just use a copy
void DVIDNode::get_properties(std::string graph_name, std::vector<Vertex> vertices, std::string key,
            std::vector<BinaryDataPtr>& properties, VertexTransactions& transactions)
{
    int num_examined = 0;
    std::tr1::unordered_map<VertexID, BinaryDataPtr> properties_map;
    int num_verts = vertices.size();

    // keep extending vertices with failed ones
    while (num_examined < vertices.size()) {
        // grab 1000 vertices at a time
        int max_size = ((num_examined + TransactionLimit) > vertices.size()) ? vertices.size() : (num_examined + TransactionLimit); 

        VertexTransactions current_transactions;
        // serialize data to push
        for (; num_examined < max_size; ++num_examined) {
            current_transactions[vertices[num_examined].id] = 0;
        }
        BinaryDataPtr transaction_binary = write_transactions_to_binary(current_transactions);
        
        // add vertex list to get properties
        unsigned long long * vertex_array =
            new unsigned long long [(current_transactions.size()+1)];
        int pos = 0;
        vertex_array[pos] = current_transactions.size();
        ++pos;
        for (VertexTransactions::iterator iter = current_transactions.begin();
                iter != current_transactions.end(); ++iter) {
            vertex_array[pos] = iter->first;
            ++pos;
        }
        std::string& str_append = transaction_binary->get_data();

        str_append += std::string((char*)vertex_array, (current_transactions.size()+1)*8);
        
        delete []vertex_array;

        // get data associated with given graph and 'key'
        BinaryDataPtr binary;
        get(graph_name, "propertytransaction/vertices/" + key + "/", binary, transaction_binary);

        VertexSet bad_vertices;
        size_t byte_pos = load_transactions_from_binary(binary->get_data(), transactions, bad_vertices);
       
        // failed vertices should be re-examined 
        for (VertexSet::iterator iter = bad_vertices.begin();
                iter != bad_vertices.end(); ++iter) {
            vertices.push_back(Vertex(*iter, 0));
            //std::cout << "Failed vertex get!: " << *iter << std::endl; 
        }

        // load properties
        unsigned char* bytearray = binary->get_raw();
       
        // get number of properties
        unsigned long long* num_transactions = (unsigned long long *)(bytearray+byte_pos);
        byte_pos += 8;

        // iterate through all properties
        for (int i = 0; i < int(*num_transactions); ++i) {
            VertexID* vertex_id = (VertexID *)(bytearray+byte_pos);
            byte_pos += 8;
            unsigned long long* data_size = (unsigned long long*)(bytearray+byte_pos);
            byte_pos += 8;
            properties_map[*vertex_id] = BinaryData::create_binary_data(
                    (const char *)(bytearray+byte_pos), *data_size);      
            byte_pos += *data_size;
        }
    }

    for (int i = 0; i < num_verts; ++i) {
        properties.push_back(properties_map[vertices[i].id]);
    }
}


void DVIDNode::set_properties(std::string graph_name, std::vector<Vertex>& vertices,
        std::string key, std::vector<BinaryDataPtr>& properties,
        VertexTransactions& transactions, std::vector<Vertex>& leftover_vertices)
{
    int num_examined = 0;

    // only post 1000 properties at a time
    while (num_examined < vertices.size()) {
        // add vertex list and properties
        
        // grab 1000 vertices at a time
        int max_size = ((num_examined + TransactionLimit) > vertices.size()) ? vertices.size() : (num_examined + TransactionLimit); 
   
        VertexTransactions temp_transactions;
        for (int i = num_examined; i < max_size; ++i) {
            temp_transactions[vertices[i].id] = transactions[vertices[i].id];
        }
        BinaryDataPtr binary = write_transactions_to_binary(temp_transactions);

        // write out number of trans
        std::string& str_append = binary->get_data();
        unsigned long long num_trans = temp_transactions.size();
        str_append += std::string((char*)&num_trans, 8);

        for (; num_examined < max_size; ++num_examined) {
            unsigned long long id = vertices[num_examined].id;
            BinaryDataPtr databin = properties[num_examined];
            std::string& data = databin->get_data();
            unsigned long long data_size = data.size();

            str_append += std::string((char*)&id, 8);
            str_append += std::string((char*)&data_size, 8);
            str_append += data;
        } 

        // put the data
        VertexSet failed_trans;
        put(graph_name, "propertytransaction/vertices/" + key + "/", binary, failed_trans);

        // add failed vertices to leftover vertices
        for (VertexSet::iterator iter = failed_trans.begin();
                iter != failed_trans.end(); ++iter) {
            leftover_vertices.push_back(Vertex(*iter, 0));
        }
    }
}

void DVIDNode::set_properties(std::string graph_name, std::vector<Edge>& edges,
        std::string key, std::vector<BinaryDataPtr>& properties,
        VertexTransactions& transactions, std::vector<Edge>& leftover_edges)
{
    int num_examined = 0;

    // only post 1000 properties at a time
    while (num_examined < edges.size()) {
        VertexSet examined_vertices; 
        
        int num_current_edges = 0;
        int starting_num = num_examined;
        for (; num_examined < edges.size(); ++num_current_edges, ++num_examined) {
            // break if it is not possible to add another edge transaction
            // (assuming that both vertices of that edge will be new vertices)
            if (examined_vertices.size() >= (TransactionLimit - 1)) {
                break;
            }

            examined_vertices.insert(edges[num_examined].id1);
            examined_vertices.insert(edges[num_examined].id2);
        }
    
        // write out transactions for current edge
        VertexTransactions current_transactions;
        for (VertexSet::iterator iter = examined_vertices.begin();
                iter != examined_vertices.end(); ++iter) {
            current_transactions[*iter] = transactions[*iter];
        }
        BinaryDataPtr binary = write_transactions_to_binary(current_transactions);
        
        // add vertex list and properties
        std::string& str_append = binary->get_data();
        
        unsigned long long num_trans = num_current_edges;
        str_append += std::string((char*)&num_trans, 8);

        for (int iter = starting_num; iter < num_examined; ++iter) {
            unsigned long long id1 = edges[iter].id1;
            unsigned long long id2 = edges[iter].id2;
            BinaryDataPtr databin = properties[iter];
            std::string& data = databin->get_data();
            unsigned long long data_size = data.size();

            str_append += std::string((char*)&id1, 8);
            str_append += std::string((char*)&id2, 8);
            str_append += std::string((char*)&data_size, 8);
            str_append += data;
        } 

        // put the data
        VertexSet failed_trans;
        put(graph_name, "propertytransaction/edges/" + key + "/", binary, failed_trans);

        // add leftover edges from failed vertices
        for (int iter = starting_num; iter < num_examined; ++iter) {
            if (failed_trans.find(edges[iter].id1) != failed_trans.end()) {
                leftover_edges.push_back(edges[iter]);
            } else if (failed_trans.find(edges[iter].id2) != failed_trans.end()) {
                leftover_edges.push_back(edges[iter]);
            }
        }
    }
}

// edge list is modified, just use a copy
void DVIDNode::get_properties(std::string graph_name, std::vector<Edge> edges, std::string key,
            std::vector<BinaryDataPtr>& properties, VertexTransactions& transactions)
{
    int num_examined = 0;
    std::tr1::unordered_map<Edge, BinaryDataPtr, Edge> properties_map;
    int num_edges = edges.size();

    // keep extending vertices with failed ones
    while (num_examined < edges.size()) {
        VertexSet examined_vertices; 

        int num_current_edges = 0;
        int starting_num = num_examined;
        for (; num_examined < edges.size(); ++num_current_edges, ++num_examined) {
            // break if it is not possible to add another edge transaction
            // (assuming that both vertices of that edge will be new vertices)
            if (examined_vertices.size() >= (TransactionLimit - 1)) {
                break;
            }

            examined_vertices.insert(edges[num_examined].id1);
            examined_vertices.insert(edges[num_examined].id2);
            
        }

        // write out transactions for current edge
        VertexTransactions current_transactions;
        for (VertexSet::iterator iter = examined_vertices.begin();
                iter != examined_vertices.end(); ++iter) {
            current_transactions[*iter] = 0;
        }
        BinaryDataPtr transaction_binary = write_transactions_to_binary(current_transactions);
       
        // add edge list to get properties
        unsigned long long * edge_array =
            new unsigned long long [(num_current_edges*2+1)*8];
        int pos = 0;
        edge_array[pos] = num_current_edges;
        ++pos;
        for (int iter = starting_num; iter < num_examined; ++iter) {
            edge_array[pos] = edges[iter].id1;
            ++pos;
            edge_array[pos] = edges[iter].id2;
            ++pos;
        }
        std::string& str_append = transaction_binary->get_data();
        str_append += std::string((char*)edge_array, (num_current_edges*2+1)*8);
        delete []edge_array;

        // get data associated with given graph and 'key'
        BinaryDataPtr binary;
        get(graph_name, "propertytransaction/edges/" + key + "/", binary, transaction_binary);

        VertexSet bad_vertices;
        size_t byte_pos = load_transactions_from_binary(binary->get_data(), transactions, bad_vertices);
       
        // failed vertices should cause corresponding edges to be re-examined 
        for (int iter = starting_num; iter < num_examined; ++iter) {
            if (bad_vertices.find(edges[iter].id1) != bad_vertices.end()) {
                edges.push_back(edges[iter]);
            } else if (bad_vertices.find(edges[iter].id2) != bad_vertices.end()) {
                edges.push_back(edges[iter]);
            }
        }
        
        // load properties
        unsigned char* bytearray = binary->get_raw();
        
        // get number of properties
        unsigned long long* num_transactions = (unsigned long long *)(bytearray+byte_pos);
        byte_pos += 8;

        // iterate through all properties
        for (int i = 0; i < int(*num_transactions); ++i) {
            VertexID* vertex_id1 = (VertexID *)(bytearray+byte_pos);
            byte_pos += 8;
            VertexID* vertex_id2 = (VertexID *)(bytearray+byte_pos);
            byte_pos += 8;
            unsigned long long* data_size = (unsigned long long*)(bytearray+byte_pos);
            byte_pos += 8;
            properties_map[Edge(*vertex_id1, *vertex_id2, 0)] = BinaryData::create_binary_data(
                    (const char*)(bytearray+byte_pos), *data_size);      
            byte_pos += *data_size;
        }
    }

    for (int i = 0; i < num_edges; ++i) {
        properties.push_back(properties_map[edges[i]]);
    }
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
            // break if it is not possible to add another edge transaction
            // (assuming that both vertices of that edge will be new vertices)
            if (examined_vertices.size() >= (TransactionLimit - 1)) {
                break;
            }

            graph.edges.push_back(edges[num_examined]);
            examined_vertices.insert(edges[num_examined].id1);
            examined_vertices.insert(edges[num_examined].id2);
            
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
    bool waiting = true;
    int status_code;
    client::response respdata;

    while (waiting) {
        client::request requestobj(construct_volume_uri(
                    datatype_instance, start, sizes, channels));
        requestobj << header("Connection", "close");
        respdata = request_client.post(requestobj,
                data->get_data(), std::string("application/octet-stream"));
        status_code = status(respdata);

        // wait 1 second if the server is busy
        if (status_code == 503) {
            sleep(1);
        } else {
            waiting = false;
        }
    }

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

    sstr << "?throttle=on";
    return sstr.str();
}

void DVIDNode::retrieve_volume(std::string datatype_inst, tuple start, tuple sizes, tuple channels, std::string& volume)
{
    bool waiting = true;
    int status_code;
    client::response respdata;

    while (waiting) {
        client::request requestobj(construct_volume_uri(datatype_inst, start, sizes, channels));
        requestobj << header("Connection", "close");
        respdata = request_client.get(requestobj);
        status_code = status(respdata);
        // wait 1 second if the server is busy
        if (status_code == 503) {
            sleep(1);
        } else {
            waiting = false;
        }
    }
    
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

