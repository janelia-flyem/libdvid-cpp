#include <libdvid/DVIDServerService.h>
#include <libdvid/DVIDNodeService.h>

#include <iostream>
#include <vector>

#include <fstream>
#include <json/json.h>
#include <json/value.h>

#include "ScopeTime.h"

using std::cerr; using std::cout; using std::endl;
using namespace libdvid;
using std::string;
using std::vector;
using std::ifstream;

/*!
 * Test labelgraph vertex and edge creation and the setting of properties.
*/
int main(int argc, char** argv)
{
    if (argc != 3) {
        cout << "Usage: <program> <server_name> <graph json>" << endl;
        return -1;
    }
    try {
        ScopeTime overall_time;
        
        DVIDServerService server(argv[1]);
        std::string uuid = server.create_new_repo("newrepo", "This is my new repo");
        DVIDNodeService dvid_node(argv[1], uuid);

        // name of graph to use        
        string graph_datatype_name = "graphtest";

        // check existence (should be new)
        if(!dvid_node.create_graph(graph_datatype_name)) {
            cerr << graph_datatype_name << " already exists" << endl;
            return -1;
        }
       
        // ?! read file
        // ?! get num nodes, edges
        ifstream fin(argv[2]);
        Json::Reader json_reader;
        Json::Value json_graph;
        if (!json_reader.parse(fin, json_graph)) {
            throw ErrMsg("Error: Json incorrectly formatted");
        }
        fin.close();
        Graph graph(json_graph);
        
        int num_vertices = graph.vertices.size();
        int num_edges = graph.edges.size();
    
        cout << "Total vertices: " << num_vertices << endl;
        cout << "Total edges: " << num_edges << endl;

        // load graph (1000 vertices and nodes at a time) 
        {
            ScopeTime write_timer(false);
            dvid_node.update_vertices(graph_datatype_name, graph.vertices);
            double write_time = write_timer.getElapsed(); 
            cout << "Time to write vertices: " << write_time << endl;
            cout << "Vertices written per second: " << num_vertices / write_time << endl;
            
            dvid_node.update_edges(graph_datatype_name, graph.edges);
            
            write_time = write_timer.getElapsed(); 
            cout << "Time to write edges: " << write_time << endl;
            cout << "Edges written per second: " << num_edges / write_time << endl;

        }
        
        // bulk get graph
        {
            ScopeTime read_timer(false);
            vector<Vertex> vertices_temp;
            Graph graph_read;
            dvid_node.get_subgraph(graph_datatype_name, vertices_temp, graph_read);
            double read_time = read_timer.getElapsed(); 
            cout << "Time to read graph: " << read_time << endl;
            cout << "Vertices and edges read per second: " 
                << (num_vertices + num_edges) / read_time << endl;
        }   

    } catch (std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }
    return 0;
}
