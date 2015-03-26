#ifndef DVIDGRAPH_H
#define DVIDGRAPH_H

#include "BinaryData.h"

#include <vector>
#include <tr1/unordered_map>
#include <tr1/unordered_set>
#include <json/json.h>
#include <boost/functional/hash.hpp>
#include <json/json.h>

namespace libdvid {

// version of jsoncpp used does not support 64 bit numbers !!
typedef unsigned long long VertexID;
typedef unsigned long long TransactionID;

//! Vertex represent DVID vertex type of ID and weight
struct Vertex {
    Vertex(VertexID id_, double weight_) : id(id_), weight(weight_) {}
    Vertex(Json::Value& data);
    Vertex(VertexID id_) : id(id_), weight(0) {}

    void export_json(Json::Value& data);

    VertexID id;
    double weight;
};

//! Vertex represent DVID edge type of vertex ID1 and ID2 and weight
struct Edge {
    Edge() : id1(0), id2(0), weight(0) {}
    Edge(VertexID id1_, VertexID id2_, double weight_) : id1(id1_), id2(id2_), weight(weight_) {}
    Edge(Json::Value& data);
    
    void export_json(Json::Value& data);
    
    VertexID id1;
    VertexID id2;
    double weight;

    bool operator==(const Edge& edge) const
    {
        VertexID tid1 = edge.id1;
        VertexID tid2 = edge.id2;
        if (tid1 > tid2) {
            tid1 = edge.id2;
            tid2 = edge.id1;
        } 

        VertexID bid1 = id1;
        VertexID bid2 = id2;
        if (id1 > id2) {
            bid1 = id2;
            bid2 = id1;
        }
        return ((bid1 == tid1) && (bid2 == tid2));
    }

    size_t operator()(const Edge& edge) const
    {
        std::size_t seed = 0;
        // TODO: make more generic for other data types -- will only work well for
        //     // those data types <=32 bytes
        //         // NOTE: node id type must allow for casting to size_t
        VertexID tid1 = edge.id1;
        VertexID tid2 = edge.id2;
        if (tid1 > tid2) {
            tid1 = tid2;
            tid2 = edge.id1;
        }
        
        boost::hash_combine(seed, std::size_t(edge.id1));
        boost::hash_combine(seed, std::size_t(edge.id2));
        return seed;
    }
};

/*!
 * Transaction IDs are used to ensure that graph manipulations act on the
 * last known state of the data.  An ID of 0 means there is no current transaction ID
*/
typedef std::tr1::unordered_map<VertexID, TransactionID> VertexTransactions; 

typedef std::tr1::unordered_set<VertexID> VertexSet; 

//! Retrieve transactions map from binary data
size_t load_transactions_from_binary(std::string& data, VertexTransactions& transactions, 
        VertexSet& bad_vertices); 

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
