#include "DVIDNodeService.h"
#include "DVIDException.h"

#include <png++/png.hpp>
#include <json/json.h>
#include <set>

using std::string; using std::vector;

using std::ifstream; using std::set; using std::stringstream;
//Json::Reader json_reader;

//! Gives the limit for how many vertice can be operated on in one call
static const int TransactionLimit = 1000;

//! The size of the DVID block (TODO: set dynamically)
static const int BLOCK_SIZE = 32; 

namespace libdvid {

DVIDNodeService::DVIDNodeService(string web_addr_, UUID uuid_) :
    connection(web_addr_), uuid(uuid_)
{
    string endpoint = "/repo/" + uuid + "/info";
    string respdata;
    BinaryDataPtr binary = BinaryData::create_binary_data();
    int status_code = connection.make_request(endpoint, GET, BinaryDataPtr(),
            binary, respdata, DEFAULT);
    if (status_code != 200) {
        throw DVIDException(respdata, status_code);
    }
}

BinaryDataPtr DVIDNodeService::custom_request(string endpoint,
        BinaryDataPtr payload, ConnectionMethod method)
{
    // append '/' to the endpoint if it is not provided
    if (!endpoint.empty() && (endpoint[0] != '/')) {
        endpoint = '/' + endpoint;
    }
    string respdata;
    string node_endpoint = "/node/" + uuid + endpoint;
    BinaryDataPtr resp_binary = BinaryData::create_binary_data();
    int status_code = connection.make_request(node_endpoint, method, payload,
            resp_binary, respdata, BINARY);
    if (status_code != 200) {
        throw DVIDException(respdata, status_code);
    }

    return resp_binary; 
}
    
Json::Value DVIDNodeService::get_typeinfo(string datatype_name)
{
    BinaryDataPtr binary = custom_request("/info", BinaryDataPtr(), GET);
   
    // read into json from binary string 
    Json::Value data;
    Json::Reader json_reader;
    if (!json_reader.parse(binary->get_data(), data)) {
        throw ErrMsg("Could not decode JSON");
    }
}

bool DVIDNodeService::create_grayscale8(string datatype_name)
{
    return create_datatype("uint8blk", datatype_name);
}

bool DVIDNodeService::create_labelblk(string datatype_name)
{
    return create_datatype("labelblk", datatype_name);
}

bool DVIDNodeService::create_keyvalue(string keyvalue)
{
    return create_datatype("keyvalue", keyvalue);
}

bool DVIDNodeService::create_graph(string graph_name)
{
    return create_datatype("labelgraph", graph_name);
}

// ?! add JPEG support (can use libjpeg by just doing jpeg_mem_src)
Grayscale2D DVIDNodeService::get_tile_slice(string datatype_instance,
        Slice2D slice, unsigned int scaling, vector<unsigned int> tile_loc)
{
    BinaryDataPtr binary_response = get_tile_slice_binary(datatype_instance,
            slice, scaling, tile_loc);

//    if (1) {
        // retrieve PNG
        string& png_image = binary_response->get_data();
        std::istringstream sstr2(png_image);
        png::image<png::gray_pixel> image;         
        image.read(sstr2);
    
        Dims_t dim_size;
        dim_size.push_back(image.get_width());
        dim_size.push_back(image.get_height());

        // inefficient extra copy into binary buffer
        uint8* buffer = new uint8[image.get_width()*image.get_height()];
        uint8* ptr = buffer;
        for (int y = 0; y < image.get_height(); ++y) {
            for (int x = 0; x < image.get_width(); ++x) {
                *ptr = image[y][x];
                ++ptr;
            }
        }
        BinaryDataPtr buffer_binary = BinaryData::create_binary_data(
                (const char*) buffer, image.get_width()*image.get_height());
        Grayscale2D grayimage(buffer_binary, dim_size);
        return grayimage;
//    }

}

BinaryDataPtr DVIDNodeService::get_tile_slice_binary(string datatype_instance,
        Slice2D slice, unsigned int scaling, vector<unsigned int> tile_loc)
{
    if (tile_loc.size() != 3) {
        throw ErrMsg("Tile identification requires 3 numbers");
    }

    string tileplane = "XY";
    if (slice == XZ) {
        tileplane = "XZ";
    } else if (slice == YZ) {
        tileplane = "YZ";
    }

    string uri =  "/" + datatype_instance + "/tile/" + tileplane + "/";
    stringstream sstr;
    sstr << uri;
    sstr << scaling << "/" << tile_loc[0];
    for (int i = 1; i < tile_loc.size(); ++i) {
        sstr << "_" << tile_loc[i];
    }

    string endpoint = sstr.str();
    return custom_request(endpoint, BinaryDataPtr(), GET);
}

Grayscale3D DVIDNodeService::get_gray3D(string datatype_instance, Dims_t sizes,
        vector<unsigned int> offset, vector<unsigned int> channels,
        bool throttle, bool compress)
{
    BinaryDataPtr data = get_volume3D(datatype_instance,
            sizes, offset, channels, throttle);
   
    // decompress using lz4
    if (compress) {
        // determined number of returned bytes
        int decomp_size = sizes[0]*sizes[1]*sizes[2];
        data = BinaryData::decompress_lz4(data, decomp_size);
    }

    Grayscale3D grayvol(data, sizes);
    return grayvol; 
}

Grayscale3D DVIDNodeService::get_gray3D(string datatype_instance, Dims_t sizes,
        vector<unsigned int> offset, bool throttle, bool compress)
{
    vector<unsigned int> channels;
    channels.push_back(0); channels.push_back(1); channels.push_back(2);
    return get_gray3D(datatype_instance, sizes, offset, channels,
            throttle, compress);
}

Labels3D DVIDNodeService::get_labels3D(string datatype_instance, Dims_t sizes,
        vector<unsigned int> offset, vector<unsigned int> channels,
        bool throttle, bool compress)
{
    BinaryDataPtr data = get_volume3D(datatype_instance,
            sizes, offset, channels, throttle);
   
    // decompress using lz4
    if (compress) {
        // determined number of returned bytes
        int decomp_size = sizes[0]*sizes[1]*sizes[2]*8;
        data = BinaryData::decompress_lz4(data, decomp_size);
    }

    Labels3D labels(data, sizes);
    return labels; 
}

Labels3D DVIDNodeService::get_labels3D(string datatype_instance, Dims_t sizes,
        vector<unsigned int> offset, bool throttle, bool compress)
{
    vector<unsigned int> channels;
    channels.push_back(0); channels.push_back(1); channels.push_back(2);
    return get_labels3D(datatype_instance, sizes, offset, channels,
            throttle, compress);
}

void DVIDNodeService::put_labels3D(string datatype_instance, Labels3D& volume,
            vector<unsigned int> offset, bool throttle, bool compress)
{
    Dims_t sizes = volume.get_dims();
    put_volume(datatype_instance, volume.get_binary(), sizes,
            offset, throttle, compress);
}

void DVIDNodeService::put_gray3D(string datatype_instance, Grayscale3D& volume,
            vector<unsigned int> offset, bool throttle, bool compress)
{
    Dims_t sizes = volume.get_dims();
    put_volume(datatype_instance, volume.get_binary(), sizes,
            offset, throttle, compress);
}

void DVIDNodeService::put(string keyvalue, string key, ifstream& fin)
{
    BinaryDataPtr data = BinaryData::create_binary_data(fin);
    put(keyvalue, key, data);
}


void DVIDNodeService::put(string keyvalue, string key, Json::Value& data)
{
    stringstream datastr;
    datastr << data;
    BinaryDataPtr bdata = BinaryData::create_binary_data(datastr.str().c_str(),
            datastr.str().length());
    put(keyvalue, key, bdata);
}


void DVIDNodeService::put(string keyvalue, string key, BinaryDataPtr value)
{
    string endpoint = "/" + keyvalue + "/key/" + key;
    custom_request(endpoint, value, POST);
}


BinaryDataPtr DVIDNodeService::get(string keyvalue, string key)
{
    return custom_request("/" + keyvalue + "/key/" + key, BinaryDataPtr(), GET);
}

Json::Value DVIDNodeService::get_json(string keyvalue, string key)
{
    BinaryDataPtr binary = get(keyvalue, key);
   
    // read into json from binary string 
    Json::Value data;
    Json::Reader json_reader;
    if (!json_reader.parse(binary->get_data(), data)) {
        throw ErrMsg("Could not decode JSON");
    }
    return data;
}

void DVIDNodeService::get_subgraph(string graph_name,
        const std::vector<Vertex>& vertices, Graph& graph)
{
    // make temporary graph for query
    Graph temp_graph;
    for (int i = 0; i < vertices.size(); ++i) {
        temp_graph.vertices.push_back(vertices[i]);
    }   
    Json::Value data;
    temp_graph.export_json(data);

    stringstream datastr;
    datastr << data;
    BinaryDataPtr binary_data = BinaryData::create_binary_data(
            datastr.str().c_str(), datastr.str().length());

    BinaryDataPtr binary = custom_request("/" + graph_name + "/subgraph",
           binary_data, GET);

    // read json from binary string into graph    
    Json::Reader json_reader;
    Json::Value returned_data;
    if (!json_reader.parse(binary->get_data(), returned_data)) {
        throw ErrMsg("Could not decode JSON");
    }
    graph.import_json(returned_data);
}

void DVIDNodeService::get_vertex_neighbors(string graph_name, Vertex vertex,
        Graph& graph)
{
    stringstream uri_ending;
    uri_ending << "/neighbors/" << vertex.id;
    
    BinaryDataPtr binary = custom_request("/" + graph_name + uri_ending.str(),
            BinaryDataPtr(), GET);
    
    // read into json from binary string 
    Json::Reader json_reader;
    Json::Value data;
    if (!json_reader.parse(binary->get_data(), data)) {
        throw ErrMsg("Could not decode JSON");
    }
    graph.import_json(data);
}

void DVIDNodeService::update_vertices(string graph_name,
        const std::vector<Vertex>& vertices)
{
    int num_examined = 0;

    while (num_examined < vertices.size()) {
        Graph graph;
        
        // grab 1000 vertices at a time
        int max_size = ((num_examined + TransactionLimit) > 
                vertices.size()) ? vertices.size() : (num_examined + TransactionLimit); 
        for (; num_examined < max_size; ++num_examined) {
            graph.vertices.push_back(vertices[num_examined]);
        }

        Json::Value data;
        graph.export_json(data);

        stringstream datastr;
        datastr << data;
        BinaryDataPtr binary = BinaryData::create_binary_data(
                datastr.str().c_str(), datastr.str().length());
        custom_request("/" + graph_name + "/weight", binary, POST);
    }
}
    
void DVIDNodeService::update_edges(string graph_name,
        const std::vector<Edge>& edges)
{

    int num_examined = 0;
    while (num_examined < edges.size()) {
        VertexSet examined_vertices; 
        Graph graph;
        
        for (; num_examined < edges.size(); ++num_examined) {
            // break if it is not possible to add another edge transaction
            // (assuming that both vertices of that edge will be new vertices
            // for simplicity
            // for simplicity
            // for simplicity)
            if (examined_vertices.size() >= (TransactionLimit - 1)) {
                break;
            }

            graph.edges.push_back(edges[num_examined]);
            examined_vertices.insert(edges[num_examined].id1);
            examined_vertices.insert(edges[num_examined].id2);
            
        }

        Json::Value data;
        graph.export_json(data);

        stringstream datastr;
        datastr << data;
        BinaryDataPtr binary = BinaryData::create_binary_data(
                datastr.str().c_str(), datastr.str().length());
        custom_request("/" + graph_name + "/weight", binary, POST);
    }
}

// vertex list is modified, just use a copy
void DVIDNodeService::get_properties(string graph_name,
        std::vector<Vertex> vertices, string key,
        std::vector<BinaryDataPtr>& properties,
        VertexTransactions& transactions)
{
    int num_examined = 0;
    std::tr1::unordered_map<VertexID, BinaryDataPtr> properties_map;
    int num_verts = vertices.size();

    // keep extending vertices with failed ones
    while (num_examined < vertices.size()) {
        // grab 1000 vertices at a time
        int max_size = ((num_examined + TransactionLimit) > 
                vertices.size()) ? vertices.size() : (num_examined + TransactionLimit); 

        VertexTransactions current_transactions;
        // serialize data to push
        for (; num_examined < max_size; ++num_examined) {
            current_transactions[vertices[num_examined].id] = 0;
        }
        BinaryDataPtr transaction_binary = 
            write_transactions_to_binary(current_transactions);
        
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
        string& str_append = transaction_binary->get_data();

        str_append += string((char*)vertex_array, (current_transactions.size()+1)*8);
        
        delete []vertex_array;

        // get data associated with given graph and 'key'
        BinaryDataPtr binary = custom_request("/" + graph_name +
                "/propertytransaction/vertices/" + key + "/", transaction_binary, GET);

        VertexSet bad_vertices;
        size_t byte_pos = load_transactions_from_binary(binary->get_data(),
                transactions, bad_vertices);
       
        // failed vertices should be re-examined 
        for (VertexSet::iterator iter = bad_vertices.begin();
                iter != bad_vertices.end(); ++iter) {
            vertices.push_back(Vertex(*iter, 0));
            //std::cout << "Failed vertex get!: " << *iter << std::endl; 
        }

        // load properties
        const unsigned char* bytearray = binary->get_raw();
       
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

// edge list is modified, just use a copy
void DVIDNodeService::get_properties(string graph_name, std::vector<Edge> edges, string key,
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
        string& str_append = transaction_binary->get_data();
        str_append += string((char*)edge_array, (num_current_edges*2+1)*8);
        delete []edge_array;

        // get data associated with given graph and 'key'
        BinaryDataPtr binary = custom_request("/" + graph_name +
                "/propertytransaction/edges/" + key + "/", transaction_binary, GET);

        VertexSet bad_vertices;
        size_t byte_pos = load_transactions_from_binary(
                binary->get_data(), transactions, bad_vertices);
       
        // failed vertices should cause corresponding edges to be re-examined 
        for (int iter = starting_num; iter < num_examined; ++iter) {
            if (bad_vertices.find(edges[iter].id1) != bad_vertices.end()) {
                edges.push_back(edges[iter]);
            } else if (bad_vertices.find(edges[iter].id2) != bad_vertices.end()) {
                edges.push_back(edges[iter]);
            }
        }
        
        // load properties
        const unsigned char* bytearray = binary->get_raw();
        
        // get number of properties
        unsigned long long* num_transactions = 
            (unsigned long long *)(bytearray+byte_pos);
        byte_pos += 8;

        // iterate through all properties
        for (int i = 0; i < int(*num_transactions); ++i) {
            VertexID* vertex_id1 = (VertexID *)(bytearray+byte_pos);
            byte_pos += 8;
            VertexID* vertex_id2 = (VertexID *)(bytearray+byte_pos);
            byte_pos += 8;
            unsigned long long* data_size = (unsigned long long*)(bytearray+byte_pos);
            byte_pos += 8;
            properties_map[Edge(*vertex_id1, *vertex_id2, 0)] =
                BinaryData::create_binary_data((const char*)
                        (bytearray+byte_pos), *data_size);      
            byte_pos += *data_size;
        }
    }

    for (int i = 0; i < num_edges; ++i) {
        properties.push_back(properties_map[edges[i]]);
    }
}

void DVIDNodeService::set_properties(string graph_name, std::vector<Vertex>& vertices,
        string key, std::vector<BinaryDataPtr>& properties,
        VertexTransactions& transactions, std::vector<Vertex>& leftover_vertices)
{
    int num_examined = 0;

    // only post 1000 properties at a time
    while (num_examined < vertices.size()) {
        // add vertex list and properties
        
        // grab 1000 vertices at a time
        int max_size = ((num_examined + TransactionLimit) > 
                vertices.size()) ? vertices.size() : (num_examined + TransactionLimit); 
   
        VertexTransactions temp_transactions;
        for (int i = num_examined; i < max_size; ++i) {
            temp_transactions[vertices[i].id] = transactions[vertices[i].id];
        }
        BinaryDataPtr binary = write_transactions_to_binary(temp_transactions);

        // write out number of trans
        string& str_append = binary->get_data();
        unsigned long long num_trans = temp_transactions.size();
        str_append += string((char*)&num_trans, 8);

        for (; num_examined < max_size; ++num_examined) {
            unsigned long long id = vertices[num_examined].id;
            BinaryDataPtr databin = properties[num_examined];
            string& data = databin->get_data();
            unsigned long long data_size = data.size();

            str_append += string((char*)&id, 8);
            str_append += string((char*)&data_size, 8);
            str_append += data;
        } 

        // put the data
        VertexSet failed_trans;
        VertexTransactions succ_trans;
        BinaryDataPtr result_binary = custom_request("/" + graph_name +
                "/propertytransaction/vertices/" + key, binary, POST);
        size_t byte_pos = load_transactions_from_binary(result_binary->get_data(),
            succ_trans, failed_trans);

        // update transaction ids for successful trans
        for (VertexTransactions::iterator iter = succ_trans.begin();
                iter != succ_trans.end(); ++iter) {
            transactions[iter->first] = iter->second;
        }

        // add failed vertices to leftover vertices
        for (VertexSet::iterator iter = failed_trans.begin();
                iter != failed_trans.end(); ++iter) {
            leftover_vertices.push_back(Vertex(*iter, 0));
        }
    }
}

void DVIDNodeService::set_properties(string graph_name, std::vector<Edge>& edges,
        string key, std::vector<BinaryDataPtr>& properties,
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
        string& str_append = binary->get_data();
        
        unsigned long long num_trans = num_current_edges;
        str_append += string((char*)&num_trans, 8);

        for (int iter = starting_num; iter < num_examined; ++iter) {
            unsigned long long id1 = edges[iter].id1;
            unsigned long long id2 = edges[iter].id2;
            BinaryDataPtr databin = properties[iter];
            string& data = databin->get_data();
            unsigned long long data_size = data.size();

            str_append += string((char*)&id1, 8);
            str_append += string((char*)&id2, 8);
            str_append += string((char*)&data_size, 8);
            str_append += data;
        } 

        // put the data
        VertexSet failed_trans;
        VertexTransactions succ_trans;
        BinaryDataPtr result_binary = custom_request("/" + graph_name +
                "/propertytransaction/edges/" + key, binary, POST);
        size_t byte_pos = load_transactions_from_binary(result_binary->get_data(),
            succ_trans, failed_trans);

        // update transaction ids for successful trans
        for (VertexTransactions::iterator iter = succ_trans.begin();
                iter != succ_trans.end(); ++iter) {
            transactions[iter->first] = iter->second;
        }

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

// ******************** PRIVATE HELPER FUNCTIONS *******************************

void DVIDNodeService::put_volume(string datatype_instance, BinaryDataPtr volume,
            vector<unsigned int> sizes, vector<unsigned int> offset,
            bool throttle, bool compress)
{
    // make sure volume specified is legal and block aligned
    if ((sizes.size() != 3) || (offset.size() != 3)) {
        throw ErrMsg("Did not correctly specify 3D volume");
    }
    
    if ((offset[0] % BLOCK_SIZE != 0) || (offset[1] % BLOCK_SIZE != 0)
            || (offset[2] % BLOCK_SIZE != 0)) {
        throw ErrMsg("Label POST error: Not block aligned");
    }

    if ((sizes[0] % BLOCK_SIZE != 0) || (sizes[1] % BLOCK_SIZE != 0)
            || (sizes[2] % BLOCK_SIZE != 0)) {
        throw ErrMsg("Label POST error: Region is not a multiple of block size");
    }

    // make sure requests do not involve more bytes than fit in an int
    // (use 8-byte label to create this bound)
    uint64 total_size = uint64(sizes[0]) * uint64(sizes[1]) * uint64(sizes[2]);
    if (total_size > INT_MAX) {
        throw ErrMsg("Trying to post too large of a volume");
    }

    bool waiting = true;
    int status_code;
    string respdata;
    vector<unsigned int> channels;
    channels.push_back(0); channels.push_back(1); channels.push_back(2); 

    // try posting until DVID is available (no contention)
    while (waiting) {
        string endpoint =  construct_volume_uri(
                    datatype_instance, sizes, offset,
                    channels, throttle, compress);
        
        // compress using lz4
        if (compress) {
            volume = BinaryData::compress_lz4(volume);
        }

        BinaryDataPtr binary_result = BinaryData::create_binary_data();
        status_code = connection.make_request(endpoint, POST, volume,
                binary_result, respdata, BINARY);

        // wait 1 second if the server is busy
        if (status_code == 503) {
            sleep(1);
        } else {
            waiting = false;
        }
    }

    if (status_code != 200) {
        throw DVIDException(respdata, status_code);
    } 
}

bool DVIDNodeService::create_datatype(string datatype, string datatype_name)
{
    if (exists("/node/" + uuid + "/" + datatype_name + "/info")) {
        return false;
    } 
    string endpoint = "/repo/" + uuid + "/instance";
    string respdata;

    // serialize as a JSON string
    string data = "{\"typename\": \"" + datatype + "\", \"dataname\": \"" + 
        datatype_name + "\"}";
    BinaryDataPtr payload = 
        BinaryData::create_binary_data(data.c_str(), data.length());
    BinaryDataPtr binary = BinaryData::create_binary_data();
    
    int status_code = connection.make_request(endpoint,
            POST, payload, binary, respdata, JSON);

    if (status_code != 200) {
        throw DVIDException(respdata, status_code);
    }

    return true;
}

bool DVIDNodeService::exists(string datatype_endpoint)
{ 
    try {
        string respdata;
        BinaryDataPtr binary = BinaryData::create_binary_data();
        int status_code = connection.make_request(datatype_endpoint,
                GET, BinaryDataPtr(), binary, respdata, DEFAULT);
    
        if (status_code != 200) {
            return false;
        }
    } catch (std::exception& e) {
        return false;
    }

    return true;
}

BinaryDataPtr DVIDNodeService::get_volume3D(string datatype_inst, Dims_t sizes,
        vector<unsigned int> offset, vector<unsigned int> channels,
        bool throttle)
{
    bool waiting = true;
    int status_code;
    BinaryDataPtr binary_result = BinaryData::create_binary_data(); 
    string respdata;

    // ensure volume is 3D
    if ((sizes.size() != 3) || (offset.size() != 3) ||
            (channels.size() != 3)) {
        throw ErrMsg("Did not correctly specify 3D volume");
    }

    // make sure requests do not involve more bytes than fit in an int
    // (use 8-byte label to create this bound)
    uint64 total_size = uint64(sizes[0]) * uint64(sizes[1]) * uint64(sizes[2]);
    if (total_size > INT_MAX) {
        throw ErrMsg("Requested too large of a volume");
    }

    // try get until DVID is available (no contention)
    while (waiting) {
        string endpoint = 
            construct_volume_uri(datatype_inst, sizes, offset,
                    channels, throttle, compress);
        status_code = connection.make_request(endpoint, GET, BinaryDataPtr(),
                binary_result, respdata, BINARY);
       
        // wait 1 second if the server is busy
        if (status_code == 503) {
            sleep(1);
        } else {
            waiting = false;
        }
    }
    
    if (status_code != 200) {
        throw DVIDException(respdata, status_code);
    }

    return binary_result;
}

string DVIDNodeService::construct_volume_uri(string datatype_inst, Dims_t sizes,
        vector<unsigned int> offset, vector<unsigned int> channels,
        bool throttle, bool compress)
{
    string uri = "/node/" + uuid + "/"
                    + datatype_inst + "/raw/";
   
    // verifies the legality of the call 
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
    sstr << "/" << offset[0];
    for (int i = 1; i < offset.size(); ++i) {
        sstr << "_" << offset[i];
    }

    if (throttle) {
        sstr << "?throttle=on";
    }
    if (compress && !throttle) {
        sstr << "?compression=lz4";
    } else if (compress) {
        sstr << "&compression=lz4";
    }

    return sstr.str();
}

}

