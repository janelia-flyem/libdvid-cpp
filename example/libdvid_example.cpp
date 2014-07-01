#include <iostream>
#include <libdvid/DVIDNode.h>
#include <vector>

using std::cout; using std::endl;
using std::string;
using std::vector;

using namespace libdvid;

// image 1 mask
unsigned char img1_mask[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,0,0,1,0,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,
    0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,
    0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
    0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
    0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
    0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
    0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
    0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,
    0,0,1,0,0,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,0,0,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

// image 2 mask
unsigned char img2_mask[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0};

unsigned long long int limg1_mask[] = {
    5, 4, 3, 2,
    4, 4, 1, 3,
    7, 7, 7, 7};

unsigned long long int limg2_mask[] = {
    8, 8, 9, 0,
    9, 9, 9, 3,
    9, 9, 9, 7};

int WIDTH = 28;
int HEIGHT = 11;

int LWIDTH = 4;
int LHEIGHT = 3;

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: <program> <server_name> <uuid>" << endl;
        return -1;
    }
    
    try {
        DVIDServer server(argv[1]);
        DVIDNode dvid_node(server, argv[2]);
       
        string gray_datatype_name = "gray1";
        string label_datatype_name = "labels2";
        string keyvalue_datatype_name = "keys";
        string graph_datatype_name = "graphtest";

        // ** Test creation of DVID datatypes **
        if(!dvid_node.create_grayscale8(gray_datatype_name)) {
            cout << gray_datatype_name << " already exists" << endl;
        }
        if(!dvid_node.create_labels64(label_datatype_name)) {
            cout << label_datatype_name << " already exists" << endl;
        }
        if(!dvid_node.create_keyvalue(keyvalue_datatype_name)) {
            cout << keyvalue_datatype_name << " already exists" << endl;
        }
        if(!dvid_node.create_graph(graph_datatype_name)) {
            cout << graph_datatype_name << " already exists" << endl;
        }
       
        // ** Write and read grayscale data **
        // create grayscale image volume
        unsigned char * img_gray = new unsigned char [sizeof(img1_mask)*2];
        for (int i = 0; i < sizeof(img1_mask); ++i) {
            img_gray[i] = img1_mask[i] * 255;
            img_gray[i+sizeof(img1_mask)] = img2_mask[i] * 255;
        }

        // create binary data string wrapper (no meta-data for now -- must explicitly create)
        BinaryDataPtr graybin = BinaryData::create_binary_data((char*)(img_gray), sizeof(img1_mask)*2);
        tuple start; start.push_back(0); start.push_back(0); start.push_back(0);
        tuple sizes; sizes.push_back(WIDTH); sizes.push_back(HEIGHT); sizes.push_back(2);
        tuple channels; channels.push_back(0); channels.push_back(1); channels.push_back(2);

        // post grayscale volume
        // one could also write 2D image slices but the starting location must
        // be at least an ND point where N is greater than 2
        dvid_node.write_volume_roi(gray_datatype_name, start, sizes, channels, graybin);

        // retrieve the image volume and make sure it makes the posted volume
        DVIDGrayPtr graycomp;
        dvid_node.get_volume_roi(gray_datatype_name, start, sizes, channels, graycomp);
        unsigned char* datacomp = graycomp->get_raw();
        for (int i = 0; i < sizeof(img1_mask)*2; ++i) {
            assert(datacomp[i] == img_gray[i]);
        }

        // ** Write and read labels64 data **
        // create label 64 image volume
        unsigned long long * img_labels = new unsigned long long [sizeof(limg1_mask)*2];
        for (int i = 0; i < (LWIDTH*LHEIGHT); ++i) {
            img_labels[i] = limg1_mask[i];
            img_labels[i+12] = limg2_mask[i];
        }

        // create binary data string wrapper (64 bits per pixel)
        BinaryDataPtr labelbin = BinaryData::create_binary_data((char*)(img_labels), sizeof(limg1_mask)*2);

        // post labels volume
        // one could also write 2D image slices but the starting location must
        // be at least an ND point where N is greater than 2
        tuple lsizes; lsizes.push_back(LWIDTH); lsizes.push_back(LHEIGHT); lsizes.push_back(2);
        dvid_node.write_volume_roi(label_datatype_name, start, lsizes, channels, labelbin);

        // retrieve the image volume and make sure it makes the posted volume
        DVIDLabelPtr labelcomp;
        dvid_node.get_volume_roi(label_datatype_name, start, lsizes, channels, labelcomp);
        unsigned long long* labeldatacomp = labelcomp->get_raw();
        for (int i = 0; i < (LWIDTH*LHEIGHT*2); ++i) {
            assert(labeldatacomp[i] == img_labels[i]);
        }

        // ** Test key value interface **
        Json::Value data_init;
        data_init["hello"] = "world"; 
        dvid_node.put(keyvalue_datatype_name, "spot0", data_init); 
        Json::Value data_ret; 
        dvid_node.get(keyvalue_datatype_name, "spot0", data_ret);
        std::string data_str = data_ret["hello"].asString();
        cout << "Response: " << data_str << endl; 
   
        // ** Test graph interface **

        // update or add vertex1 and vertex2
        Vertex vertex1(1, 0.0);
        Vertex vertex2(2, 0.0);
        vector<Vertex> vertices;
        vertices.push_back(vertex1);
        vertices.push_back(vertex2);
        // if the node was already created the value will be different
        dvid_node.update_vertices(graph_datatype_name, vertices);
        
        // update or add edge
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
        assert(weight_final == (weight_initial - 3));
        assert(edge_weight_final == (edge_weight_initial + 5.5));

        cout << "Weight of vertex 1: " << weight_final << endl; 
        cout << "Weight of edge: " << edge_weight_final << endl; 


        // ** Test graph property get/set **

        // test vertex get/put properties
        double incr1 = 1.3;
        double incr2 = -2.1;
        vector<Vertex> prop_vertices;
        prop_vertices.push_back(vertex1); prop_vertices.push_back(vertex2);
       
        while (!prop_vertices.empty()) { 
            vector<BinaryDataPtr> properties;
            VertexTransactions transactions;

            // retrieve properties and transactions
            dvid_node.get_properties(graph_datatype_name, prop_vertices,
                    "features", properties, transactions); 
            
            for (int i = 0; i < prop_vertices.size(); ++i) {
                cout << "Vertex " << prop_vertices[i].id << ": Trans " << transactions[prop_vertices[i].id] << endl;
            
                // increment properties
                if (properties[i]->get_data().length() > 0) {
                    double* val_array = (double*) properties[i]->get_raw();
                    cout << "Vals: " << val_array[0] << " " << val_array[1] << endl;
        
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

            // set properties
            vector<Vertex> leftover_vertices;
            dvid_node.set_properties(graph_datatype_name, prop_vertices,
                    "features", properties, transactions, leftover_vertices); 

            for (int i = 0; i < leftover_vertices.size(); ++i) {
                cout << "Must re-write: " << leftover_vertices[i].id << endl;
            }
            prop_vertices = leftover_vertices;
        }

        // test edge get/put
        vector<Edge> prop_edges;
        prop_edges.push_back(edge);
       
        while (!prop_edges.empty()) { 
            vector<BinaryDataPtr> properties;
            VertexTransactions transactions;

            // retrieve properties and transactions
            dvid_node.get_properties(graph_datatype_name, prop_edges,
                    "efeatures", properties, transactions); 
            
            for (int i = 0; i < prop_edges.size(); ++i) {
                cout << "Edge " << prop_edges[i].id1 << " " << prop_edges[i].id2 << ": Trans " <<
                    transactions[prop_edges[i].id1] << ", " << transactions[prop_edges[i].id2] << endl;
            
                // increment properties
                if (properties[i]->get_data().length() > 0) {
                    double* val_array = (double*) properties[i]->get_raw();
                    cout << "Vals: " << val_array[0] << " " << val_array[1] << endl;
        
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

            // set properties
            vector<Edge> leftover_edges;
            dvid_node.set_properties(graph_datatype_name, prop_edges,
                    "efeatures", properties, transactions, leftover_edges); 

            for (int i = 0; i < leftover_edges.size(); ++i) {
                cout << "Must re-write!: " << leftover_edges[i].id1 << ", " << leftover_edges[i].id2 << endl;
            }
            prop_edges = leftover_edges;
        }


    } catch (std::exception& e) {
        // catch DVID, libdvid, and boost errors
        cout << e.what() << endl;
    }
}
