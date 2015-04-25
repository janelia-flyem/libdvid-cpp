#include "DVIDGraph.h"

using std::string;

namespace libdvid {

Vertex::Vertex(Json::Value& data)
{
    // DVID vertex id are 64 bit
    id = data["Id"].asUInt64();
    weight = data["Weight"].asDouble();
}

void Vertex::export_json(Json::Value& data)
{
    data["Id"] = (Json::Value::UInt64)(id);
    data["Weight"] = weight;
}

Edge::Edge(Json::Value& data)
{
    id1 = data["Id1"].asUInt64();
    id2 = data["Id2"].asUInt64();
    weight = data["Weight"].asDouble();
}

void Edge::export_json(Json::Value& data)
{
    data["Id1"] = (Json::Value::UInt64)(id1);
    data["Id2"] = (Json::Value::UInt64)(id2);
    data["Weight"] = weight;
}

Graph::Graph(Json::Value& data)
{
    import_json(data);
}

void Graph::import_json(Json::Value& data)
{
    Json::Value vertices_data = data["Vertices"];
    Json::Value edges_data = data["Edges"];

    for (unsigned int i = 0; i < vertices_data.size(); ++i) {
        vertices.push_back(Vertex(vertices_data[i]));
    }

    for (unsigned int i = 0; i < edges_data.size(); ++i) {
        edges.push_back(Edge(edges_data[i]));
    }
}

void Graph::export_json(Json::Value& data)
{
    // create Vertex json array and assign to "Vertices" 
    Json::Value vertices_data(Json::arrayValue);
    for (unsigned int i = 0; i < vertices.size(); ++i) {
        Json::Value vertex_data;
        vertices[i].export_json(vertex_data);
        vertices_data[i] = vertex_data;
    }
    data["Vertices"] = vertices_data;

    // create Edge json array and assign to "Edges" 
    Json::Value edges_data(Json::arrayValue);
    for (unsigned int i = 0; i < edges.size(); ++i) {
        Json::Value edge_data;
        edges[i].export_json(edge_data);
        edges_data[i] = edge_data;
    }
    data["Edges"] = edges_data;
}

BinaryDataPtr write_transactions_to_binary(VertexTransactions& transactions)
{
    uint64 * trans_array =
        new uint64 [(transactions.size()*2+1)];
    
    //serialize the number of transactions
    int pos = 0;
    trans_array[pos] = transactions.size();
    ++pos;

    // serialize transactions (vertex and transaction number) 
    for (VertexTransactions::iterator iter = transactions.begin();
            iter != transactions.end(); ++iter) {
        trans_array[pos] = iter->first;
        ++pos;
        trans_array[pos] = iter->second;
        ++pos;
    }
    
    BinaryDataPtr ptr = BinaryData::create_binary_data((char*)trans_array,
            (transactions.size()*2+1)*8);
    delete []trans_array;
    return ptr;
}

size_t load_transactions_from_binary(string& data,
        VertexTransactions& transactions, VertexSet& bad_vertices)
{
    char* bytearray = (char*) data.c_str();
    size_t byte_pos = 0;

    // get number of transactions
    uint64* num_transactions = (uint64 *)(bytearray);
    byte_pos += 8;

    // get vertex / transaction list
    for (int i = 0; i < int(*num_transactions); ++i) {
        VertexID* vertex_id = (VertexID *)(bytearray+byte_pos);
        byte_pos += 8;
        TransactionID* transaction_id = (TransactionID *)(bytearray+byte_pos);
        byte_pos += 8;
        transactions[*vertex_id] = *transaction_id;  
    }

    // find failed transactions
    uint64* num_failed_transactions = (uint64*)(bytearray+byte_pos);
    byte_pos += 8;

    for (int i = 0; i < int(*num_failed_transactions); ++i) {
        VertexID* vertex_id = (VertexID *)(bytearray+byte_pos);
        byte_pos += 8;
        bad_vertices.insert(*vertex_id);
    }
  
    // byte position indicates the location of the rest of the binary payload 
    return byte_pos; 
} 

}

