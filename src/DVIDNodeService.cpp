#include "DVIDNodeService.h"
#include "DVIDException.h"

#include <json/json.h>
#include <set>

using std::string; using std::vector;

using std::ifstream; using std::set; using std::stringstream;
//Json::Reader json_reader;

//! Gives the limit for how many vertice can be operated on in one call
static const unsigned int TransactionLimit = 1000;


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
        throw DVIDException(respdata + "\n" + binary->get_data(), status_code);
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
        throw DVIDException(respdata + "\n" + resp_binary->get_data(), status_code);
    }

    return resp_binary; 
}
    
Json::Value DVIDNodeService::get_typeinfo(string datatype_name)
{
    BinaryDataPtr binary = custom_request("/" + datatype_name + "/info", BinaryDataPtr(), GET);
   
    // read into json from binary string 
    Json::Value data;
    Json::Reader json_reader;
    if (!json_reader.parse(binary->get_data(), data)) {
        throw ErrMsg("Could not decode JSON");
    }
    return data;
}

bool DVIDNodeService::create_grayscale8(string datatype_name)
{
    return create_datatype("uint8blk", datatype_name);
}

bool DVIDNodeService::create_labelblk(string datatype_name,
        string labelvol_name)
{
    bool is_created = create_datatype("labelblk", datatype_name, labelvol_name);
    bool is_created2 = true;
    if (labelvol_name != "") {
        is_created2 = create_datatype("labelvol", labelvol_name, datatype_name);
    }
    return is_created && is_created2;
}

bool DVIDNodeService::create_keyvalue(string keyvalue)
{
    return create_datatype("keyvalue", keyvalue);
}

bool DVIDNodeService::create_graph(string graph_name)
{
    return create_datatype("labelgraph", graph_name);
}

bool DVIDNodeService::create_roi(string name)
{
    return create_datatype("roi", name);
}

Grayscale2D DVIDNodeService::get_tile_slice(string datatype_instance,
        Slice2D slice, unsigned int scaling, vector<int> tile_loc)
{
    BinaryDataPtr binary_response = get_tile_slice_binary(datatype_instance,
            slice, scaling, tile_loc);
    Dims_t dim_size;

    // retrieve JPEG
    try {
        unsigned int width, height;
        binary_response = 
            BinaryData::decompress_jpeg(binary_response, width, height);
        dim_size.push_back(width); dim_size.push_back(height);
        
    } catch (ErrMsg& msg) {
        unsigned int width, height;
        binary_response = 
            BinaryData::decompress_png8(binary_response, width, height);
        dim_size.push_back(width); dim_size.push_back(height);
    }

    Grayscale2D grayimage(binary_response, dim_size);
    return grayimage;
}

BinaryDataPtr DVIDNodeService::get_tile_slice_binary(string datatype_instance,
        Slice2D slice, unsigned int scaling, vector<int> tile_loc)
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
    for (unsigned int i = 1; i < tile_loc.size(); ++i) {
        sstr << "_" << tile_loc[i];
    }

    string endpoint = sstr.str();
    return custom_request(endpoint, BinaryDataPtr(), GET);
}

Grayscale3D DVIDNodeService::get_gray3D(string datatype_instance, Dims_t sizes,
        vector<int> offset, vector<unsigned int> channels,
        bool throttle, bool compress, string roi)
{
    BinaryDataPtr data = get_volume3D(datatype_instance,
            sizes, offset, channels, throttle, compress, roi);
   
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
        vector<int> offset, bool throttle, bool compress, string roi)
{
    vector<unsigned int> channels;
    channels.push_back(0); channels.push_back(1); channels.push_back(2);
    return get_gray3D(datatype_instance, sizes, offset, channels,
            throttle, compress, roi);
}

Labels3D DVIDNodeService::get_labels3D(string datatype_instance, Dims_t sizes,
        vector<int> offset, vector<unsigned int> channels,
        bool throttle, bool compress, string roi)
{
    BinaryDataPtr data = get_volume3D(datatype_instance,
            sizes, offset, channels, throttle, compress, roi);
   
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
        vector<int> offset, bool throttle, bool compress, string roi)
{
    vector<unsigned int> channels;
    channels.push_back(0); channels.push_back(1); channels.push_back(2);
    return get_labels3D(datatype_instance, sizes, offset, channels,
            throttle, compress, roi);
}

uint64 DVIDNodeService::get_label_by_location(std::string datatype_instance, unsigned int x,
            unsigned int y, unsigned int z)
{
    Dims_t sizes; sizes.push_back(1);
    sizes.push_back(1); sizes.push_back(1);
    vector<int> start; start.push_back(x); start.push_back(y); start.push_back(z);
    Labels3D labels = get_labels3D(datatype_instance, sizes, start, false);
    const uint64* ptr = (const uint64*) labels.get_raw();
    return *ptr;
}

void DVIDNodeService::put_labels3D(string datatype_instance, Labels3D const & volume,
            vector<int> offset, bool throttle, bool compress, string roi)
{
    Dims_t sizes = volume.get_dims();
    put_volume(datatype_instance, volume.get_binary(), sizes,
            offset, throttle, compress, roi);
}

void DVIDNodeService::put_gray3D(string datatype_instance, Grayscale3D const & volume,
            vector<int> offset, bool throttle, bool compress)
{
    Dims_t sizes = volume.get_dims();
    put_volume(datatype_instance, volume.get_binary(), sizes,
            offset, throttle, compress, "");
}


GrayscaleBlocks DVIDNodeService::get_grayblocks(string datatype_instance,
        vector<int> block_coords, unsigned int span)
{
    int ret_span = span;
    BinaryDataPtr data = get_blocks(datatype_instance, block_coords, span);

    // make sure this data encodes blocks of grayscale
    if (data->length() !=
            (DEFBLOCKSIZE*DEFBLOCKSIZE*DEFBLOCKSIZE*sizeof(uint8)*span)) {
        throw ErrMsg("Expected 1-byte values from " + datatype_instance);
    }
 
    return GrayscaleBlocks(data, ret_span);
} 

LabelBlocks DVIDNodeService::get_labelblocks(string datatype_instance,
           vector<int> block_coords, unsigned int span)
{
    int ret_span = span;
    BinaryDataPtr data = get_blocks(datatype_instance, block_coords, span);

    // make sure this data encodes blocks of grayscale
    if (data->length() !=
            (DEFBLOCKSIZE*DEFBLOCKSIZE*DEFBLOCKSIZE*sizeof(uint64)*ret_span)) {
        throw ErrMsg("Expected 8-byte values from " + datatype_instance);
    }
 
    return LabelBlocks(data, ret_span);
}
    
void DVIDNodeService::put_grayblocks(string datatype_instance,
            GrayscaleBlocks blocks, vector<int> block_coords)
{
    put_blocks(datatype_instance, blocks.get_binary(),
            blocks.get_num_blocks(), block_coords);
}


void DVIDNodeService::put_labelblocks(string datatype_instance,
            LabelBlocks blocks, vector<int> block_coords)
{
    put_blocks(datatype_instance, blocks.get_binary(),
            blocks.get_num_blocks(), block_coords);
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
    for (unsigned int i = 0; i < vertices.size(); ++i) {
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
    unsigned int num_examined = 0;

    while (num_examined < vertices.size()) {
        Graph graph;
        
        // grab 1000 vertices at a time
        unsigned int max_size = ((num_examined + TransactionLimit) > 
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

    unsigned int num_examined = 0;
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
    unsigned int num_examined = 0;
        
    #ifdef __clang__
    std::unordered_map<VertexID, BinaryDataPtr> properties_map;
    #else
    std::tr1::unordered_map<VertexID, BinaryDataPtr> properties_map;
    #endif
    
    int num_verts = vertices.size();

    // keep extending vertices with failed ones
    while (num_examined < vertices.size()) {
        // grab 1000 vertices at a time
        unsigned int max_size = ((num_examined + TransactionLimit) > 
                vertices.size()) ? vertices.size() : (num_examined + TransactionLimit); 

        VertexTransactions current_transactions;
        // serialize data to push
        for (; num_examined < max_size; ++num_examined) {
            current_transactions[vertices[num_examined].id] = 0;
        }
        BinaryDataPtr transaction_binary = 
            write_transactions_to_binary(current_transactions);
        
        // add vertex list to get properties
        uint64 * vertex_array =
            new uint64 [(current_transactions.size()+1)];
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
        uint64* num_transactions = (uint64 *)(bytearray+byte_pos);
        byte_pos += 8;

        // iterate through all properties
        for (int i = 0; i < int(*num_transactions); ++i) {
            VertexID* vertex_id = (VertexID *)(bytearray+byte_pos);
            byte_pos += 8;
            uint64* data_size = (uint64*)(bytearray+byte_pos);
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
    unsigned int num_examined = 0;
    
    #ifdef __clang__
    std::unordered_map<Edge, BinaryDataPtr, Edge> properties_map;
    #else
    std::tr1::unordered_map<Edge, BinaryDataPtr, Edge> properties_map;
    #endif
    int num_edges = edges.size();

    // keep extending vertices with failed ones
    while (num_examined < edges.size()) {
        VertexSet examined_vertices; 

        unsigned int num_current_edges = 0;
        unsigned int starting_num = num_examined;
        for (; num_examined < edges.size(); ++num_current_edges, ++num_examined) {
            // break if it is not possible to add another edge transaction
            // (assuming that both vertices of that edge will be new vertices)
            assert(TransactionLimit > 0);
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
        uint64* edge_array =
            new uint64 [(num_current_edges*2+1)*8];
        unsigned int pos = 0;
        edge_array[pos] = num_current_edges;
        ++pos;
        for (unsigned int iter = starting_num; iter < num_examined; ++iter) {
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
        for (unsigned int iter = starting_num; iter < num_examined; ++iter) {
            if (bad_vertices.find(edges[iter].id1) != bad_vertices.end()) {
                edges.push_back(edges[iter]);
            } else if (bad_vertices.find(edges[iter].id2) != bad_vertices.end()) {
                edges.push_back(edges[iter]);
            }
        }
        
        // load properties
        const unsigned char* bytearray = binary->get_raw();
        
        // get number of properties
        uint64* num_transactions = 
            (uint64*)(bytearray+byte_pos);
        byte_pos += 8;

        // iterate through all properties
        for (int i = 0; i < int(*num_transactions); ++i) {
            VertexID* vertex_id1 = (VertexID *)(bytearray+byte_pos);
            byte_pos += 8;
            VertexID* vertex_id2 = (VertexID *)(bytearray+byte_pos);
            byte_pos += 8;
            uint64* data_size = (uint64*)(bytearray+byte_pos);
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
    unsigned int num_examined = 0;

    // only post 1000 properties at a time
    while (num_examined < vertices.size()) {
        // add vertex list and properties
        
        // grab 1000 vertices at a time
        unsigned int max_size = ((num_examined + TransactionLimit) > 
                vertices.size()) ? vertices.size() : (num_examined + TransactionLimit); 
   
        VertexTransactions temp_transactions;
        for (unsigned int i = num_examined; i < max_size; ++i) {
            temp_transactions[vertices[i].id] = transactions[vertices[i].id];
        }
        BinaryDataPtr binary = write_transactions_to_binary(temp_transactions);

        // write out number of trans
        string& str_append = binary->get_data();
        uint64 num_trans = temp_transactions.size();
        str_append += string((char*)&num_trans, 8);

        for (; num_examined < max_size; ++num_examined) {
            uint64 id = vertices[num_examined].id;
            BinaryDataPtr databin = properties[num_examined];
            string& data = databin->get_data();
            uint64 data_size = data.size();

            str_append += string((char*)&id, 8);
            str_append += string((char*)&data_size, 8);
            str_append += data;
        } 

        // put the data
        VertexSet failed_trans;
        VertexTransactions succ_trans;
        BinaryDataPtr result_binary = custom_request("/" + graph_name +
                "/propertytransaction/vertices/" + key, binary, POST);
        load_transactions_from_binary(result_binary->get_data(),
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
    unsigned int num_examined = 0;

    // only post 1000 properties at a time
    while (num_examined < edges.size()) {
        VertexSet examined_vertices; 
        
        unsigned int num_current_edges = 0;
        unsigned int starting_num = num_examined;
        for (; num_examined < edges.size(); ++num_current_edges, ++num_examined) {
            // break if it is not possible to add another edge transaction
            // (assuming that both vertices of that edge will be new vertices)
            assert(TransactionLimit > 0);
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
        
        uint64 num_trans = num_current_edges;
        str_append += string((char*)&num_trans, 8);

        for (unsigned int iter = starting_num; iter < num_examined; ++iter) {
            uint64 id1 = edges[iter].id1;
            uint64 id2 = edges[iter].id2;
            BinaryDataPtr databin = properties[iter];
            string& data = databin->get_data();
            uint64 data_size = data.size();

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
        load_transactions_from_binary(result_binary->get_data(),
            succ_trans, failed_trans);

        // update transaction ids for successful trans
        for (VertexTransactions::iterator iter = succ_trans.begin();
                iter != succ_trans.end(); ++iter) {
            transactions[iter->first] = iter->second;
        }

        // add leftover edges from failed vertices
        for (unsigned int iter = starting_num; iter < num_examined; ++iter) {
            if (failed_trans.find(edges[iter].id1) != failed_trans.end()) {
                leftover_edges.push_back(edges[iter]);
            } else if (failed_trans.find(edges[iter].id2) != failed_trans.end()) {
                leftover_edges.push_back(edges[iter]);
            }
        }
    }
}

void DVIDNodeService::post_roi(std::string roi_name,
        const std::vector<BlockXYZ>& blockcoords)
{
    // Do not assume the blocks are sorted, first
    // sort and then encode as runlengths in X.
    // This will also eliminate duplicate blocks
    set<BlockXYZ> sorted_blocks;
    for (unsigned int i = 0; i < blockcoords.size(); ++i) {
        sorted_blocks.insert(blockcoords[i]);    
    }

    // encode JSON as z,y,x0,x1 (inclusive)
    int z = INT_MAX;
    int y = INT_MAX;
    int xmin = 0; int xmax = 0;
    Json::Value blocks_data(Json::arrayValue);
    unsigned int blockrle_count = 0;
    for (set<BlockXYZ>::iterator iter = sorted_blocks.begin();
            iter != sorted_blocks.end(); ++iter) {
        if (iter->z != z || iter->y != y) {
            if (z != INT_MAX) {
                // add run length
                Json::Value block_data(Json::arrayValue);
                block_data[0] = z;
                block_data[1] = y;
                block_data[2] = xmin;
                block_data[3] = xmax;
                blocks_data[blockrle_count] = block_data;
                ++blockrle_count;
            }
        
            z = iter->z;
            y = iter->y;
            xmin = iter->x;
            xmax = xmin;
        } else {
            xmax = iter->x;
        }
    }
    
    if (z != INT_MAX) {
        // add run length
        Json::Value block_data(Json::arrayValue);
        block_data[0] = z;
        block_data[1] = y;
        block_data[2] = xmin;
        block_data[3] = xmax;
        blocks_data[blockrle_count] = block_data;
    }
    
    // write json to string and post
    stringstream datastr;
    datastr << blocks_data;
    BinaryDataPtr binary_data = BinaryData::create_binary_data(
            datastr.str().c_str(), datastr.str().length());
    BinaryDataPtr binary = custom_request("/" + roi_name + "/roi",
            binary_data, POST);
}

void DVIDNodeService::get_roi(std::string roi_name,
        std::vector<BlockXYZ>& blockcoords)
{
    // clear blockcoords
    blockcoords.clear();

    BinaryDataPtr binary = custom_request("/" + roi_name + "/roi",
            BinaryDataPtr(), GET);

    // read json from binary string  
    Json::Reader json_reader;
    Json::Value returned_data;
    if (!json_reader.parse(binary->get_data(), returned_data)) {
        throw ErrMsg("Could not decode JSON");
    }

    // order the blocks (might be redundant depending on DVID output order)
    set<BlockXYZ> sorted_blocks;

    // insert blocks from JSON (decode block run lengths)
    for (unsigned int i = 0; i < returned_data.size(); ++i) {
        int z = returned_data[i][0].asInt();
        int y = returned_data[i][1].asInt();
        int xmin = returned_data[i][2].asInt();
        int xmax = returned_data[i][3].asInt();

        for (int xiter = xmin; xiter <= xmax; ++xiter) {
            sorted_blocks.insert(BlockXYZ(xiter, y, z));
        }
    }

    // returned sorted blocks back to caller
    for (set<BlockXYZ>::iterator iter = sorted_blocks.begin();
            iter != sorted_blocks.end(); ++iter) {
        blockcoords.push_back(*iter);  
    }
}

double DVIDNodeService::get_roi_partition(std::string roi_name,
        std::vector<SubstackXYZ>& substacks, unsigned int partition_size)
{
    // clear substacks
    substacks.clear();

    stringstream querystring;
    querystring << "/" <<  roi_name << "/partition?batchsize=" << partition_size;

    BinaryDataPtr binary = custom_request(querystring.str(),
            BinaryDataPtr(), GET);

    // read json from binary string  
    Json::Reader json_reader;
    Json::Value returned_data;
    if (!json_reader.parse(binary->get_data(), returned_data)) {
        throw ErrMsg("Could not decode JSON");
    }

    // order the substacks (might be redundant depending on DVID output order)
    set<SubstackXYZ> sorted_substacks;

    // insert substacks from JSON
    for (unsigned int i = 0; i < returned_data["Subvolumes"].size(); ++i) {
        int x = returned_data["Subvolumes"][i]["MinPoint"][0].asInt();
        int y = returned_data["Subvolumes"][i]["MinPoint"][1].asInt();
        int z = returned_data["Subvolumes"][i]["MinPoint"][2].asInt();
        
        sorted_substacks.insert(SubstackXYZ(x, y, z,
                    DEFBLOCKSIZE*partition_size));
    }

    // returned sorted substacks back to caller
    for (set<SubstackXYZ>::iterator iter = sorted_substacks.begin();
            iter != sorted_substacks.end(); ++iter) {
        substacks.push_back(*iter);  
    }

    // determine the packing factor for the given partition
    unsigned int total_blocks = returned_data["NumTotalBlocks"].asUInt();
    unsigned int roi_blocks = returned_data["NumActiveBlocks"].asUInt();

    return double(roi_blocks)/total_blocks;
}

void DVIDNodeService::roi_ptquery(std::string roi_name,
        const std::vector<PointXYZ>& points,
        std::vector<bool>& inroi)
{
    inroi.clear();

    // load into JSON array
    Json::Value points_data(Json::arrayValue);
    for (unsigned int i = 0; i < points.size(); ++i) {
        Json::Value point_data(Json::arrayValue);
        point_data[0] = points[i].x;
        point_data[1] = points[i].y;
        point_data[2] = points[i].z;
        points_data[i] = point_data;
    }

    // make request with point list
    stringstream datastr;
    datastr << points_data;
    BinaryDataPtr binary_data = BinaryData::create_binary_data(
            datastr.str().c_str(), datastr.str().length());
    BinaryDataPtr binary = custom_request("/" + roi_name + "/ptquery",
            binary_data, POST);

    
    // read json from binary string  
    Json::Reader json_reader;
    Json::Value returned_data;
    if (!json_reader.parse(binary->get_data(), returned_data)) {
        throw ErrMsg("Could not decode JSON");
    }

    // insert status of each point (true if in ROI) (true if in ROI) (true if in ROI) 
    for (unsigned int i = 0; i < returned_data.size(); ++i) {
        bool ptinroi = returned_data[i].asBool();
        inroi.push_back(ptinroi);
    }
}
    
bool DVIDNodeService::body_exists(string labelvol_name, uint64 bodyid) 
{
    stringstream sstr;
    sstr << "/" << labelvol_name << "/sparsevol/";
    sstr << bodyid;
    string node_endpoint = "/node/" + uuid + sstr.str();
    int status_code = connection.make_head_request(node_endpoint);
    if (status_code == 200) {
        return true;
    } else if (status_code == 204) {
        return false;
    } else {
        throw ErrMsg("Returned bad status code from HEAD request on sparsevol");
    }
    return false;
}
    
PointXYZ DVIDNodeService::get_body_location(string labelvol_name,
        uint64 bodyid, int zplane)
{
    vector<BlockXYZ> blockcoords;
    if (!get_coarse_body(labelvol_name, bodyid, blockcoords)) {
        throw ErrMsg("Requested body does not exist");
    }
   
    // just choose some arbitrary block point somewhere in the middle
    unsigned int num_blocks = blockcoords.size();
    unsigned int index = num_blocks / 2;
    int x = blockcoords[index].x * DEFBLOCKSIZE + DEFBLOCKSIZE/2;
    int y = blockcoords[index].y * DEFBLOCKSIZE + DEFBLOCKSIZE/2;
    int z = blockcoords[index].z * DEFBLOCKSIZE + DEFBLOCKSIZE/2; 
    PointXYZ point(x,y,z); 
   

    // try to get a point in the middle of the Z plane chose
    // if not found just default to somewhere in the middle of the body 
    if (zplane != INT_MAX) {
        int zplaneblk = zplane / DEFBLOCKSIZE;
        int start_pos = -1;
        int last_pos = -1;
        // blocks are ordered z,y,x
        for (unsigned int i = 0; i < blockcoords.size(); ++i) {
            if (blockcoords[i].z == zplaneblk) {
                if (start_pos == -1) {
                    start_pos = int(i);
                }
                last_pos = i;
            }
        }
        if (start_pos != -1) {
            unsigned int index =
                (unsigned int)(start_pos + ((last_pos - start_pos) / 2));
            point.x = blockcoords[index].x * DEFBLOCKSIZE + DEFBLOCKSIZE/2;
            point.y = blockcoords[index].y * DEFBLOCKSIZE + DEFBLOCKSIZE/2;
            point.z = zplane;
        }
    }

    return point;
}

bool DVIDNodeService::get_coarse_body(string labelvol_name, uint64 bodyid,
            vector<BlockXYZ>& blockcoords) 
{
    // clear blockcoords
    blockcoords.clear();
    stringstream sstr;
    sstr << "/" << labelvol_name << "/sparsevol-coarse/";
    sstr << bodyid;

    BinaryDataPtr binary;
    try {
        binary = custom_request(sstr.str(), BinaryDataPtr(), GET);
    } catch (DVIDException& error) {
        // body does not exist (or something else is wrong)
        // either way body doesn't exist at this moment at this endpoint
        return false;
    }

    // order the blocks (might be redundant depending on DVID output order)
    set<BlockXYZ> sorted_blocks;
    
    // retrieve data: ignore first 8 bytes
    // next 4 bytes encodes the number of spans
    // patterns of x,y,z,xspan (int32 little endian)

    // assume little endian machine for now
    const uint8* bytearray = binary->get_raw();
    unsigned int spot = 8;
    unsigned int* num_spans = (unsigned int*)(bytearray+spot);
    spot += 4;

    // decode spans
    for (unsigned int i = 0; i < *num_spans; ++i) {
        int* xblock = (int*)(bytearray+spot);
        spot += 4;
        int* yblock = (int*)(bytearray+spot);
        spot += 4;
        int* zblock = (int*)(bytearray+spot);
        spot += 4;
        int* spans = (int*)(bytearray+spot);
        spot += 4;
        
        int xsize = *xblock + int(*spans);
        for (int xiter = *xblock; xiter < xsize; ++xiter) {
            sorted_blocks.insert(BlockXYZ(xiter, *yblock, *zblock));
        }
    }

    // returned sorted blocks back to caller
    for (set<BlockXYZ>::iterator iter = sorted_blocks.begin();
            iter != sorted_blocks.end(); ++iter) {
        blockcoords.push_back(*iter);  
    }

    if (blockcoords.empty()) {
        return false;
    }

    return true;
}



// ******************** PRIVATE HELPER FUNCTIONS *******************************

void DVIDNodeService::put_volume(string datatype_instance, BinaryDataPtr volume,
            vector<unsigned int> sizes, vector<int> offset,
            bool throttle, bool compress, string roi)
{
    // make sure volume specified is legal and block aligned
    if ((sizes.size() != 3) || (offset.size() != 3)) {
        throw ErrMsg("Did not correctly specify 3D volume");
    }
    
    if ((offset[0] % DEFBLOCKSIZE != 0) || (offset[1] % DEFBLOCKSIZE != 0)
            || (offset[2] % DEFBLOCKSIZE != 0)) {
        throw ErrMsg("Label POST error: Not block aligned");
    }

    if ((sizes[0] % DEFBLOCKSIZE != 0) || (sizes[1] % DEFBLOCKSIZE != 0)
            || (sizes[2] % DEFBLOCKSIZE != 0)) {
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
    
    BinaryDataPtr binary_result;
    
    string endpoint =  construct_volume_uri(
            datatype_instance, sizes, offset,
            channels, throttle, compress, roi);

    // compress using lz4
    if (compress) {
        volume = BinaryData::compress_lz4(volume);
    }

    // try posting until DVID is available (no contention)
    while (waiting) {
        binary_result = BinaryData::create_binary_data();
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
        throw DVIDException(respdata + "\n" + binary_result->get_data(),
                status_code);
    } 
}

BinaryDataPtr DVIDNodeService::get_blocks(string datatype_instance,
        vector<int> block_coords, int span)
{
    string prefix = "/" + datatype_instance + "/blocks/";
    stringstream sstr;
    // encode starting block
    sstr << block_coords[0] << "_" << block_coords[1] << "_" << block_coords[2];
    sstr << "/" << span;
    string endpoint = prefix + sstr.str();
  
    // first 4 bytes no longer include span (always grab what the user wants)
    BinaryDataPtr blockbinary = custom_request(endpoint, BinaryDataPtr(), GET);

    return blockbinary;
}

void DVIDNodeService::put_blocks(string datatype_instance,
        BinaryDataPtr binary, int span, vector<int> block_coords)
{
    string prefix = "/" + datatype_instance + "/blocks/";
    stringstream sstr;
    // encode starting block
    sstr << block_coords[0] << "_" << block_coords[1] << "_" << block_coords[2];
    sstr << "/" << span;
    string endpoint = prefix + sstr.str();
    
    custom_request(endpoint, binary, POST);
}

bool DVIDNodeService::create_datatype(string datatype, string datatype_name,
        std::string sync_name)
{
    if (exists("/node/" + uuid + "/" + datatype_name + "/info")) {
        return false;
    } 
    string endpoint = "/repo/" + uuid + "/instance";
    string respdata;

    // serialize as a JSON string
    string data = "{\"typename\": \"" + datatype + "\", \"dataname\": \"" + 
        datatype_name;
    if (sync_name != "") {
        data += "\", \"Sync\": \"" + sync_name + "\"}";
    } else {
        data += "\"}";
    }
    BinaryDataPtr payload = 
        BinaryData::create_binary_data(data.c_str(), data.length());
    BinaryDataPtr binary = BinaryData::create_binary_data();
    
    int status_code = connection.make_request(endpoint,
            POST, payload, binary, respdata, JSON);

    if (status_code != 200) {
        throw DVIDException(respdata + "\n" + binary->get_data(), status_code);
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
        vector<int> offset, vector<unsigned int> channels,
        bool throttle, bool compress, string roi)
{
    bool waiting = true;
    int status_code;
    BinaryDataPtr binary_result; 
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

    string endpoint = 
        construct_volume_uri(datatype_inst, sizes, offset,
                channels, throttle, compress, roi);

    // try get until DVID is available (no contention)
    while (waiting) {
        binary_result = BinaryData::create_binary_data();
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
        throw DVIDException(respdata + "\n" + binary_result->get_data(),
                status_code);
    }

    return binary_result;
}

string DVIDNodeService::construct_volume_uri(string datatype_inst, Dims_t sizes,
        vector<int> offset, vector<unsigned int> channels,
        bool throttle, bool compress, string roi)
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
    for (unsigned int i = 0; i < channels.size(); ++i) {
        used_channels.insert(channels[i]);   
    }
    int channel_id = 0;
    // should never call since there should be 3 channels
    for (unsigned int i = channels.size(); i < 3; ++i) {
        while (used_channels.find(channel_id) != used_channels.end()) {
            ++channel_id;
        }
        channels.push_back(channel_id);
    }

    for (unsigned int i = 1; i < channels.size(); ++i) {
        sstr << "_" << channels[i];
    }
    
    // retrieve at least a 3D volume -- should never be called
    for (int i = sizes.size(); i < 3; ++i) {
        sizes.push_back(1);
    }
    sstr << "/" << sizes[0];
    for (unsigned int i = 1; i < sizes.size(); ++i) {
        sstr << "_" << sizes[i];
    }
    sstr << "/" << offset[0];
    for (unsigned int i = 1; i < offset.size(); ++i) {
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

    if ((compress || throttle) && roi != "") {
        sstr << "&roi=" << roi;
    } else if (roi != "") {
        sstr << "?roi=" << roi;
    }

    return sstr.str();
}

}

