/*!
 * This file defines an API for accessing the DVID version node REST
 * interface.  Only a subset of the REST interface is implemented.
 *
 * Note: to be thread safe instantiate a unique node service
 * object for each thread.
 *
 * TODO: expand API and load node meta on initialization.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
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

//! Creates type for DVID unique identifier string
typedef std::string UUID;

//! Used to define the relevant orthogonal cut-plane
enum Slice2D { XY, XZ, YZ };


/*!
 * Class that helps access different DVID version node actions.
*/
class DVIDNodeService {
  public:
    /*!
     * Constructor sets up a http connection and checks
     * whether a node of the given uuid and web server exists.
     * \param web_addr_ address of DVID server
     * \param uuid_ uuid corresponding to a DVID node
    */
    DVIDNodeService(std::string web_addr_, UUID uuid_);

    /*!
     * Allow client to specify a custom http request with an
     * http endpoint for a given node and uuid.  A request
     * to /node/<uuid>/blah should provide the endpoint
     * as '/blah'.
     * \param endpoint REST endpoint given the node's uuid
     * \param payload binary data to be sent in the request
     * \param method http verb (GET, PUT, POST, DELETE)
     * \return http response as binary data
    */
    BinaryDataPtr custom_request(std::string endpoint, BinaryDataPtr payload,
            ConnectionMethod method);

    /*!
     * Retrieves meta data for a given datatype instance
     * \param datatype_name name of datatype instance
     * \return JSON describing instance meta data
    */
    Json::Value get_typeinfo(std::string datatype_name);

    /************* API to create datatype instances **************/
    // TODO: pass configuration data
    
    /*!
     * Create an instance of uint8 grayscale datatype.
     * \param datatype_name name of new datatype instance
     * \return true if create, false if already exists
    */
    bool create_grayscale8(std::string datatype_name);
    
    /*!
     * Create an instance of uint64 labelblk datatype.
     * \param datatype_name name of new datatype instance
     * \return true if create, false if already exists
    */
    bool create_labelblk(std::string datatype_name);
    
    /*!
     * Create an instance of keyvalue datatype.
     * \param datatype_name name of new datatype instance
     * \return true if create, false if already exists
    */
    bool create_keyvalue(std::string keyvalue);
    
    /*!
     * Create an instance of labelgraph datatype.
     * \param datatype_name name of new datatype instance
     * \return true if create, false if already exists
    */
    bool create_graph(std::string name);

    /********** API to access labels and grayscale data **********/   
    // TODO: maybe support custom byte buffers for getting and putting 

    /*!
     * Retrive a pre-computed tile from DVID at the specified
     * location and zoom level.
     * \param datatype_instance name of tile type instance
     * \param slice specify XY, YZ, or XZ
     * \param scaling specify zoom level (1=max res)
     * \param tile_loc X,Y,Z location of tile (X and Y are in tile coordinates)
     * \return 2D grayscale object that wraps a byte buffer
    */ 
    Grayscale2D get_tile_slice(std::string datatype_instance, Slice2D slice,
            unsigned int scaling, std::vector<unsigned int> tile_loc);

    /*!
     * Retrive the raw pre-computed tile (no decompression) from
     * DVID at the specified location and zoom level.  In theory, this
     * could be applied to multi-scale label data, but DVID typically
     * only stores tiles for grayscale data since it is immutable.
     * \param datatype_instance name of tile type instance
     * \param slice specify XY, YZ, or XZ
     * \param scaling specify zoom level (1=max res)
     * \param tile_loc X,Y,Z location of tile (X and Y are in block coordinates
     * \return byte buffer for the raw compressed data stored (e.g, JPEG or PNG) 
    */ 
    BinaryDataPtr get_tile_slice_binary(std::string datatype_instance, Slice2D slice,
            unsigned int scaling, std::vector<unsigned int> tile_loc);

    /*!
     * Retrive a 3D 1-byte grayscale volume with the specified
     * dimension size and spatial offset.  The dimension
     * sizes and offset default to X,Y,Z (the
     * DVID 0,1,2 channel).  The data is returned so X corresponds
     * to the matrix column.  Because it is easy to overload a single
     * server implementation of DVID with hundreds of volume requests,
     * we support a throttle command that prevents multiple volume
     * GETs/PUTs from executing at the same time.
     * A 2D slice should be requested as X x Y x 1.
     * \param datatype_instance name of tile type instance
     * \param dims size of X, Y, Z dimensions in voxel coordinates
     * \param offset X, Y, Z offset in voxel coordinates
     * \param throttle allow only one request at time (default: true)
     * \return 3D grayscale object that wraps a byte buffer
    */
    Grayscale3D get_gray3D(std::string datatype_instance, Dims_t dims,
            std::vector<unsigned int> offset, bool throttle=true);

    /*!
     * Retrive a 3D 1-byte grayscale volume with the specified
     * dimension size and spatial offset.  However, the user
     * can also specify the channel order
     * of the retrieved volume.  The default is X, Y, Z (or 0, 1, 2).
     * Specfing (1,0,2) will allow the returned data where the column
     * dimension corresponds to Y instead of X. Because it is easy to
     * overload a single server implementation of DVID with hundreds
     * of volume requests, we support a throttle command that prevents
     * multiple volume GETs/PUTs from executing at the same time.
     * A 2D slice should be requested as ch1 size x ch2 size x 1.
     * \param datatype_instance name of tile type instance
     * \param dims size of dimensions (order given by channels)
     * \param offset offset in voxel coordinates (order given by channels)
     * \param channels channel order (default: 0,1,2)
     * \param throttle allow only one request at time (default: true)
     * \return 3D grayscale object that wraps a byte buffer
    */
    Grayscale3D get_gray3D(std::string datatype_instance, Dims_t dims,
            std::vector<unsigned int> offset,
            std::vector<unsigned int> channels, bool throttle=true);
    
     /*!
     * Retrive a 3D 8-byte label volume with the specified
     * dimension size and spatial offset.  The dimension
     * sizes and offset default to X,Y,Z (the
     * DVID 0,1,2 channel).  The data is returned so X corresponds
     * to the matrix column.  Because it is easy to overload a single
     * server implementation of DVID with hundreds of volume requests,
     * we support a throttle command that prevents multiple volume
     * GETs/PUTs from executing at the same time.
     * A 2D slice should be requested as X x Y x 1.
     * \param datatype_instance name of tile type instance
     * \param dims size of X, Y, Z dimensions in voxel coordinates
     * \param offset X, Y, Z offset in voxel coordinates
     * \param throttle allow only one request at time (default: true)
     * \return 3D label object that wraps a byte buffer
    */
    Labels3D get_labels3D(std::string datatype_instance, Dims_t dims,
            std::vector<unsigned int> offset, bool throttle=true);
   
    /*!
     * Retrive a 3D 8-byte label volume with the specified
     * dimension size and spatial offset.  However, the user
     * can also specify the channel order
     * of the retrieved volume.  The default is X, Y, Z (or 0, 1, 2).
     * Specfing (1,0,2) will allow the returned data where the column
     * dimension corresponds to Y instead of X. Because it is easy to
     * overload a single server implementation of DVID with hundreds
     * of volume requests, we support a throttle command that prevents
     * multiple volume GETs/PUTs from executing at the same time.
     * A 2D slice should be requested as ch1 size x ch2 size x 1.
     * \param datatype_instance name of tile type instance
     * \param dims size of dimensions (order given by channels)
     * \param offset offset in voxel coordinates (order given by channels)
     * \param channels channel order (default: 0,1,2)
     * \param throttle allow only one request at time (default: true)
     * \return 3D label object that wraps a byte buffer
    */
    Labels3D get_labels3D(std::string datatype_instance, Dims_t dims,
            std::vector<unsigned int> offset,
            std::vector<unsigned int> channels, bool throttle=true);

    /*!
     * Put a 3D 1-byte grayscale volume to DVID with the specified
     * dimension and spatial offset.  THE DIMENSION AND OFFSET ARE
     * IN VOXEL COORDINATS BUT MUST BE BLOCK ALIGNED.  The size
     * of DVID blocks are determined at repo creation and is
     * always 32x32x32 currently.  The channel order is always
     * X, Y, Z.  Because it is easy to overload a single server
     * implementation of DVID with hundreds
     * of volume PUTs, we support a throttle command that prevents
     * multiple volume GETs/PUTs from executing at the same time.
     * TODO: expose block size parameter through interface.
     * \param datatype_instance name of tile type instance
     * \param volume grayscale 3D volume encodes dimension sizes and binary buffer 
     * \param offset offset in voxel coordinates (order given by channels)
     * \param throttle allow only one request at time (default: true)
    */
    void put_gray3D(std::string datatype_instance, Grayscale3D& volume,
            std::vector<unsigned int> offset, bool throttle=true);

    /*!
     * Put a 3D 8-byte label volume to DVID with the specified
     * dimension and spatial offset.  THE DIMENSION AND OFFSET ARE
     * IN VOXEL COORDINATS BUT MUST BE BLOCK ALIGNED.  The size
     * of DVID blocks are determined at repo creation and is
     * always 32x32x32 currently.  The channel order is always
     * X, Y, Z.  Because it is easy to overload a single server
     * implementation of DVID with hundreds
     * of volume PUTs, we support a throttle command that prevents
     * multiple volume GETs/PUTs from executing at the same time.
     * TODO: expose block size parameter through interface.
     * \param datatype_instance name of tile type instance
     * \param volume label 3D volume encodes dimension sizes and binary buffer 
     * \param offset offset in voxel coordinates (order given by channels)
     * \param throttle allow only one request at time (default: true)
    */
    void put_labels3D(std::string datatype_instance, Labels3D& volume,
            std::vector<unsigned int> offset, bool throttle=true);

    /*************** API to access keyvalue interface ***************/
    
    /*!
     * Put binary blob at a given key location.  It will overwrite data
     * that exists at the key for the given node version.
     * \param keyvalue name of keyvalue instance
     * \param key name of key to the keyvalue instance
     * \param value binary blob to store at key
    */
    void put(std::string keyvalue, std::string key, BinaryDataPtr value);
    
    /*!
     * Put data in a file at a given key location.  It will overwrite
     * data that exists at the key for the given node version.
     * \param keyvalue name of keyvalue instance
     * \param key name of key to the keyvalue instance
     * \param fin file stream that contains binary to store
    */
    void put(std::string keyvalue, std::string key, std::ifstream& fin);

    /*!
     * Put JSON data at a given key location.  It will overwrite data
     * that exists at the key for the given node version.
     * \param keyvalue name of keyvalue instance
     * \param key name of key to the keyvalue instance
     * \param data JSON data to store at key
    */
    void put(std::string keyvalue, std::string key, Json::Value& data);

    /*!
     * Retrive binary data at a given key location.
     * \param keyvalue name of keyvalue instance
     * \param key name of key to the keyvalue instance
     * \return binary data stored at key
    */
    BinaryDataPtr get(std::string keyvalue, std::string key);
    // could return a reference but assuming that this is used for short messages
    
    /*!
     * Retrieve json of data at a given key location.
     * \param keyvalue name of keyvalue instance
     * \param key name of key to the keyvalue instance
     * \return json stored at key
    */
    Json::Value get_json(std::string keyvalue, std::string key);
    
    /************** API to access labelgraph interface **************/
   
    /*!
     * Download the graph into the graph datatype.  If vertices are supplied,
     * a subgraph is extracted that includes just those vertices.  This
     * command could be time-consuming for large graphs.
     * \param graph_name name of labelgraph instance
     * \param vertices if no vertices are specified, retrieve the whole graph
     * \param graph the resulting graph (vertice, edges, and edge weights)
    */
    void get_subgraph(std::string graph_name,
            const std::vector<Vertex>& vertices, Graph& graph);

    /*!
     * Extract all the vertices connected to a paritcular vertex.
     * This is a low latency call.
     * \param graph_name name of labelgraph instance
     * \param vertex grab vertices connected to this vertex
     * \param graph vertex and partners are stored in the graph type
    */
    void get_vertex_neighbors(std::string graph_name, Vertex vertex,
            Graph& graph);

    /*!
     * Add the provided vertices to the labelgraph with the associated
     * vertex weights.  If the vertex already exists, it will increment
     * the vertex weight by the weight specified.  This function
     * can be used for creation and incrementing vertex weights in parallel.
     * \param graph_name name of labelgraph instance
     * \param vertices list of vertices to create or update
    */ 
    void update_vertices(std::string graph_name,
            const std::vector<Vertex>& vertices);
    
    /*!
     * Add the provided edges to the labelgraph with the associated
     * edge weights.  If the edge already exists, it will increment
     * the vertex weight by the weight specified.  This function
     * can be used for creation and incrementing edge weights in parallel.
     * The command will fail if the vertices for the given edges
     * were not created first.
     * \param graph_name name of labelgraph instance
     * \param vertices list of vertices to create or update
    */ 
    void update_edges(std::string graph_name,
            const std::vector<Edge>& edges);

    /*!
     * Retrieve properties associated with a list of vertices.  Binary
     * is returned as array that corresponds to the list of vertices. This
     * command can be used to get a transaction ID for each vertex.
     * These transaction IDs must be used when one wants to update
     * a property.  It ensures that the property was not modified
     * by another client.
     * \param graph_name name of labelgraph instance
     * \param vertices properties are retrieved for these vertices
     * \param key name of property
     * \param properties properties corresponding to the vertex list
     * \param transactions returns transaction ids for all vertices
    */
    void get_properties(std::string graph_name,
            std::vector<Vertex> vertices, std::string key,
            std::vector<BinaryDataPtr>& properties,
            VertexTransactions& transactions);

    /*!
     * Retrieve properties associated with a list of edges.  Binary
     * is returned as array that corresponds to the list of edges. This
     * command can be used to get a transaction ID for each vertex
     * that corresponds to the list of edges.
     * These transaction IDs must be used when one wants to update
     * a property.  It ensures that the property was not modified
     * by another client.
     * \param graph_name name of labelgraph instance
     * \param edges properties are retrieved for these edges
     * \param key name of property
     * \param properties properties corresponding to the edge list
     * \param transactions returns transaction ids for all edge vertices
    */
    void get_properties(std::string graph_name, std::vector<Edge> edges,
            std::string key, std::vector<BinaryDataPtr>& properties,
            VertexTransactions& transactions);

    /*!
     * Set properties as binary blobs for a list of vertices.
     * Must provide transaction ids for each vertex being written
     * to.  These IDs are retrieved use the get_properties command.
     * Any vertex with a stale transaction id is returned.
     * \param graph_name name of labelgraph instance
     * \param vertices properties are set for these vertices
     * \param key name of property
     * \param properties binary blobs to be set
     * \param transaction specify transactions for set call
     * \param leftover_vertices vertices that could not be written
    */
    void set_properties(std::string graph_name,
            std::vector<Vertex>& vertices, std::string key,
            std::vector<BinaryDataPtr>& properties,
            VertexTransactions& transactions,
            std::vector<Vertex>& leftover_vertices);

    /*!
     * Set properties as binary blobs for a list of edges.
     * Must provide transaction ids for vertices of each edge
     * being written to.  These IDs are retrieved use the
     * get_properties command. Any vertex with a stale transaction
     * id is returned.
     * \param graph_name name of labelgraph instance
     * \param edges properties are set for these edges
     * \param key name of property
     * \param properties binary blobs to be set
     * \param transaction specify transactions for set call
     * \param leftover_vertices vertices that could not be written
    */

    void set_properties(std::string graph_name,
            std::vector<Edge>& edges, std::string key,
            std::vector<BinaryDataPtr>& properties,
            VertexTransactions& transactions,
            std::vector<Edge>& leftover_edges);

  private:
    //! uuid for instance
    const UUID uuid;
    
    //! HTTP connection with DVID
    DVIDConnection connection;

    /*!
     * Helper function to put a 3D volume to DVID with the specified
     * dimension and spatial offset.  THE DIMENSION AND OFFSET ARE
     * IN VOXEL COORDINATS BUT MUST BE BLOCK ALIGNED. 
     * \param datatype_instance name of tile type instance
     * \param volume binary buffer encodes volume 
     * \param offset offset in voxel coordinates (order given by channels)
     * \param throttle allow only one request at time
    */
    void put_volume(std::string datatype_instance, BinaryDataPtr volume,
            std::vector<unsigned int> sizes, std::vector<unsigned int> offset,
            bool throttle);

    /*!
     * Helper function to create an instance of the specified type.
     * \param datatype name of the datatype to create
     * \param datatype_name name of new datatype instance
     * \return true if create, false if already exists
    */
    bool create_datatype(std::string datatype, std::string datatype_name);

    /*!
     * Checks if data exists for the given datatype name.
     * \return true if the instance already exists
    */
    bool exists(std::string datatype_endpoint);

    /*!
     * Helper function to retrieve a 3D volume with the specified
     * dimension size, spatial offset, and channel retrieval order.
     * \param datatype_instance name of tile type instance
     * \param dims size of dimensions (order given by channels)
     * \param offset offset in voxel coordinates (order given by channels)
     * \param channels channel order (default: 0,1,2)
     * \param throttle allow only one request at time
     * \return byte buffer corresponding to volume
    */
    BinaryDataPtr get_volume3D(std::string datatype_inst, Dims_t sizes,
        std::vector<unsigned int> offset, std::vector<unsigned int> channels,
        bool throttle);

    /*!
     * Helper function to construct a REST endpoint strign for
     * volume GETs and PUTs given several parameters.
     * \param datatype_instance name of tile type instance
     * \param dims size of dimensions (order given by channels)
     * \param offset offset in voxel coordinates (order given by channels)
     * \param channels channel order (default: 0,1,2)
     * \param throttle allow only one request at time
    */
    std::string construct_volume_uri(std::string datatype_inst, Dims_t sizes,
            std::vector<unsigned int> offset,
            std::vector<unsigned int> channels, bool throttle);
};

}

#endif
