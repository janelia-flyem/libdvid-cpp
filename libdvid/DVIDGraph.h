#ifndef DVIDGRAPH_H
#define DVIDGRAPH_H

#include <vector>
#include <tr1/unordered_map>
#include <tr1/unordered_set>
#include <json/json.h>
#include "BinaryData.h"

namespace libdvid {

// version of jsoncpp used does not support 64 bit numbers !!
typedef unsigned long long VertexID;
typedef unsigned long long TransactionID;

//! Vertex represent DVID vertex type of ID and weight
struct Vertex {
    Vertex(VertexID id_, double weight_) : id(id_), weight(weight_) {}
    Vertex(Json::Value& data);
    
    void export_json(Json::Value& data);

    VertexID id;
    double weight;
};

//! Vertex represent DVID edge type of vertex ID1 and ID2 and weight
struct Edge {
    Edge(VertexID id1_, VertexID id2_, double weight_) : id1(id1_), id2(id2_), weight(weight_) {}
    Edge(Json::Value& data);
    
    void export_json(Json::Value& data);
    
    VertexID id1;
    VertexID id2;
    double weight;
};

/*!
 * Transaction IDs are used to ensure that graph manipulations act on the
 * last known state of the data.  An ID of 0 means there is no current transaction ID
*/
typedef std::tr1::unordered_map<VertexID, TransactionID> VertexTransactions; 

typedef std::tr1::unordered_set<VertexID> VertexSet; 

//! Retrieve transactions map from binary data
size_t load_transactions_from_binary(std::string& data, VertexTransactions& transactions); 

//! Write transactions map to binary data
BinaryDataPtr write_transactions_to_binary(VertexTransactions& transactions); 

/*!
 * Graph contains the vertices and edges in the DVID graph.  Only basic serialization
 * and deserialization from JSON is currently supported.
*/
struct Graph {
    Graph() {}
    Graph(Json::Value& data);

    void import_json(Json::Value& data);
    void export_json(Json::Value& data);

    std::vector<Vertex> vertices;
    std::vector<Edge> edges;

};

}

#endif
