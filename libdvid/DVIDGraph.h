/*!
 * This file provides datastructures for storing DVID labelgraph
 * datastructures.  This file is meant for communicating with DVID
 * and is not meant as a standalone representation for a general graph.
 * There is no graph-related algorithms for these structures.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#ifndef DVIDGRAPH_H
#define DVIDGRAPH_H

#include "BinaryData.h"
#include "Globals.h"

#include <vector>
#include <json/json.h>
#include <boost/functional/hash.hpp>
#include <boost/static_assert.hpp>

#ifdef __clang__
#include <unordered_map>
#include <unordered_set>
#else
#include <tr1/unordered_map>
#include <tr1/unordered_set>
#endif

namespace libdvid {

//! Vertex ID represented as a 64-bit number
typedef uint64 VertexID;

//! Ensure that VertexID is 64 bit
BOOST_STATIC_ASSERT(sizeof(VertexID) == 8);

//! Transaction ID that enables concurrent DVID graph access
typedef uint64 TransactionID;

/*!
 * Vertex is its unique ID and its weight (typically representing
 * the size of the vertex in voxels).
*/
struct Vertex {
    /*!
     * Constructor to explicitly set vertex information.
     * \param id_ vertex id
     * \param weight_ weight for the vertex
    */
    Vertex(VertexID id_, double weight_) : id(id_), weight(weight_) {}
    
    /*!
     * Constructor that deserializes vertex JSON data returned from DVID.
     * \param data JSON data defining a vertex
    */
    explicit Vertex(Json::Value& data);
    
    /*!
     * Constructor that creates a vertex with a default weight.
     * \param id_ vertex id
    */
    explicit Vertex(VertexID id_) : id(id_), weight(0) {}

    /*!
     * Serializes the vertex to JSON.
     * \param data json data that is written into
    */
    void export_json(Json::Value& data);

    //! ID of the vertex
    VertexID id;
    
    //! Weight of the vertex
    double weight;
};

/*!
 * Edge constitutes two vertex ids and a weight.  For example,
 * the weight could indicate the sizes of the edge between two
 * vertices in voxels.
*/
struct Edge {
    /*!
     * Constructor for an edge using all default values.
    */
    Edge() : id1(0), id2(0), weight(0) {}
    
    /*!
     * Constructor using supplied vertex ids and weight.
     * \param id1_ vertex 1 of edge
     * \param id2_ vertex 2 of edge
     * \param weight_ weight of edge
    */
    Edge(VertexID id1_, VertexID id2_, double weight_) : 
        id1(id1_), id2(id2_), weight(weight_) {}
    
    /*!
     * Constructor that deserializes edge JSON data returned from DVID.
     * \param data JSON data defining a vertex
    */
    explicit Edge(Json::Value& data);
    
    /*!
     * Serializes the edge to JSON.
     * \param data json data that is written into
    */
    void export_json(Json::Value& data);
   
    //! Vertex 1 of edge 
    VertexID id1;

    //! Vertex 2 of edge
    VertexID id2;

    //! Weight of edge
    double weight;

    /*
     * Checks if two edges are equal.  Equivalent edges are not
     * guaranteed to have the same vertex order.  This comparison
     * reorders the vertex ids and then compares.
     * \param edge edge to compare to
    */
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

    /*!
     * Produce a hash index (currently 32 bit) for a given edge.
     * \param edge edge to derive a hash
    */
    size_t operator()(const Edge& edge) const
    {
        std::size_t seed = 0;
        VertexID tid1 = edge.id1;
        VertexID tid2 = edge.id2;
        if (tid1 > tid2) {
            tid1 = tid2;
            tid2 = edge.id1;
        }
       
        // requires conversion to 32bit number 
        boost::hash_combine(seed, std::size_t(edge.id1));
        boost::hash_combine(seed, std::size_t(edge.id2));
        return seed;
    }
};

/*!
 * Transaction IDs are used to ensure that graph manipulations
 * act on the last known state of the data.  An ID of 0 means
 * there is no current transaction ID.
 */
#ifdef __clang__
typedef std::unordered_map<VertexID, TransactionID> VertexTransactions;
//! A collection of vertex IDs for processing
typedef std::unordered_set<VertexID> VertexSet;
#else
typedef std::tr1::unordered_map<VertexID, TransactionID> VertexTransactions;
//! A collection of vertex IDs for processing
typedef std::tr1::unordered_set<VertexID> VertexSet;
#endif

/*! 
 * Retrieve transactions map from binary data.
 * Determine failed transactions.
 * \param data binary data with transaction information from DVID
 * \param transactions vertices with a current transaction id
 * \param bad_vertices vertices without a current transaction id
*/
size_t load_transactions_from_binary(std::string& data,
        VertexTransactions& transactions, VertexSet& bad_vertices); 

/*!
 * Write transactions map to binary data that DVID can consume.
 * \param transactions transaction ids for each vertex.
*/
BinaryDataPtr write_transactions_to_binary(VertexTransactions& transactions); 

/*!
 * Graph contains the vertices and edges in the DVID graph.
 * Only basic serialization and deserialization from JSON
 * is currently supported.
*/
struct Graph {
    /*!
     * Construct empty graph.
    */
    Graph() {}
    
    /*!
     * Deserialize JSON labelgraph defined by DVID.
     * \param data JSON data defining graph
    */
    explicit Graph(Json::Value& data);

    /*!
     * Deserialize labelgraph from JSON (format defined by DVID).
     * \param data JSON data defining graph
    */
    void import_json(Json::Value& data);
    
    /*!
     * Serialize labelgraph to JSON (format defined by DVID).
     * \param data JSON data defining graph
    */
    void export_json(Json::Value& data);

    //! Vertices in the graph
    std::vector<Vertex> vertices;

    //! Edges in the graph
    std::vector<Edge> edges;

};

}

#endif
