/*!
 * This file gives examples of how to use the labelgraph
 * interface.  It creates a simple graph by adding
 * vertices and edges.  Properties are set and retrieved as well.
*/
#include <libdvid/DVIDServerService.h>
#include <libdvid/DVIDNodeService.h>

#include <iostream>
#include <vector>

using std::cerr; using std::cout; using std::endl;
using namespace libdvid;
using std::string;
using std::vector;

/*!
 * Test labelgraph vertex and edge creation and the setting of properties.
*/
int main(int argc, char** argv)
{
    if (argc != 2) {
        cout << "Usage: <program> <server_name> <uuid>" << endl;
        return -1;
    }
    try {
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
        
        // check existence (should be old)
        if(dvid_node.create_graph(graph_datatype_name)) {
            cout << graph_datatype_name << " should exist" << endl;
            return -1;
        }
       
        // add vertex1 and vertex2 by calling the update function
        Vertex vertex1(1, 0.0);
        Vertex vertex2(2, 0.0);
        vector<Vertex> vertices;
        vertices.push_back(vertex1);
        vertices.push_back(vertex2);
        // if the node was already created the value will be different
        dvid_node.update_vertices(graph_datatype_name, vertices);
        
        // add edge by calling the update function
        Edge edge(1, 2, 0.0);
        vector<Edge> edges;
        edges.push_back(edge);
        dvid_node.update_edges(graph_datatype_name, edges);
       
        // get vertex and edge weight 
        Graph graph_initial;
        dvid_node.get_vertex_neighbors(graph_datatype_name, 1, graph_initial);
        double weight_initial = graph_initial.vertices[0].weight;
        double edge_weight_initial = graph_initial.edges[0].weight;
        if (graph_initial.vertices[1].id == 1) {
            weight_initial = graph_initial.vertices[1].weight;
        } 
         
        // update vertex1 weight
        Vertex vertex1_update(1, -3.0);
        vertices.clear();
        vertices.push_back(vertex1_update);
        dvid_node.update_vertices(graph_datatype_name, vertices);

        // update edge weight
        Edge edge_update(1, 2, 5.5);
        edges.clear();
        edges.push_back(edge_update);
        dvid_node.update_edges(graph_datatype_name, edges);

        // get vertex and edge updated weight 
        Graph graph_final;
        dvid_node.get_vertex_neighbors(graph_datatype_name, 1, graph_final);
        double weight_final = graph_final.vertices[0].weight;
        double edge_weight_final = graph_final.edges[0].weight;
        if (graph_final.vertices[1].id == 1) {
            weight_final = graph_final.vertices[1].weight;
        }
        if (weight_final != (weight_initial - 3)) {
            cerr << "Vertex weight mismatch" << endl;
            return -1;
        }
        if (edge_weight_final != (edge_weight_initial + 5.5))
        {
            cerr << "Edge weight mismatch" << endl;
            return -1;
        }


        // ** Test graph property get/set **

        // test vertex get/put properties -- just make sure there are no fails
        // for two update cycles
        double incr1 = 1.3;
        double incr2 = -2.1;
        
        int num_iterations = 0;
        while (num_iterations < 2) {
            ++num_iterations;
            vector<Vertex> prop_vertices;
            prop_vertices.push_back(vertex1); prop_vertices.push_back(vertex2);

            while (!prop_vertices.empty()) { 
                vector<BinaryDataPtr> properties;
                VertexTransactions transactions;

                // retrieve properties and transactions
                dvid_node.get_properties(graph_datatype_name, prop_vertices,
                        "features", properties, transactions); 

                for (int i = 0; i < prop_vertices.size(); ++i) {
                    // increment properties
                    if (properties[i]->get_data().length() > 0) {
                        double* val_array = (double*) properties[i]->get_raw();
                        val_array[0] += (incr1*prop_vertices[i].id);
                        val_array[1] += (incr2*prop_vertices[i].id);
                    } else{
                        std::string& strdata = properties[i]->get_data();
                        double val1 = (incr1*prop_vertices[i].id);
                        double val2 = (incr2*prop_vertices[i].id);
                        strdata += std::string((char*)&val1, 8);
                        strdata += std::string((char*)&val2, 8);
                    }
                }

                // set properties -- there will never be leftover vertices
                // when this script is run alone without competing processes
                vector<Vertex> leftover_vertices;
                dvid_node.set_properties(graph_datatype_name, prop_vertices,
                        "features", properties, transactions, leftover_vertices); 

                prop_vertices = leftover_vertices;
            }
        }


        // test edge get/put
        num_iterations = 0;
        while (num_iterations < 2) {
            ++num_iterations;
            vector<Edge> prop_edges;
            prop_edges.push_back(edge);

            while (!prop_edges.empty()) { 
                vector<BinaryDataPtr> properties;
                VertexTransactions transactions;

                // retrieve properties and transactions
                dvid_node.get_properties(graph_datatype_name, prop_edges,
                        "efeatures", properties, transactions); 

                for (int i = 0; i < prop_edges.size(); ++i) {
                    // increment properties
                    if (properties[i]->get_data().length() > 0) {
                        double* val_array = (double*) properties[i]->get_raw();
                        val_array[0] += (incr1*prop_edges[i].id1);
                        val_array[1] += (incr2*prop_edges[i].id1);
                    } else{
                        std::string& strdata = properties[i]->get_data();
                        double val1 = (incr1*prop_edges[i].id1);
                        double val2 = (incr2*prop_edges[i].id1);
                        strdata += std::string((char*)&val1, 8);
                        strdata += std::string((char*)&val2, 8);
                    }
                }

                // set properties -- there will never be leftover vertices
                // when this script is run alone without competing processes
                vector<Edge> leftover_edges;
                dvid_node.set_properties(graph_datatype_name, prop_edges,
                        "efeatures", properties, transactions, leftover_edges); 

                prop_edges = leftover_edges;
            }
        }

    } catch (std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }
    return 0;
}
