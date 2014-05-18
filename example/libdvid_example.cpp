#include <iostream>
#include <libdvid/DVIDNode.h>

using std::cout; using std::endl;
using std::string;

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

int WIDTH = 28;
int HEIGHT = 11;

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: <program> <server_name> <uuid>" << endl;
        return -1;
    }
    
    try {
        DVIDServer server(argv[1]);
        DVIDNode dvid_node(server, argv[2]);
       
        string gray_datatype_name = "gray1";
        string label_datatype_name = "labels1";
        string keyvalue_datatype_name = "keys";

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
        unsigned long long * img_labels = new unsigned long long [sizeof(img1_mask)*2];
        for (int i = 0; i < sizeof(img1_mask); ++i) {
            img_labels[i] = img1_mask[i] * 255;
            img_labels[i+sizeof(img1_mask)] = img2_mask[i] * 255;
        }

        // create binary data string wrapper (64 bits per pixel)
        BinaryDataPtr labelbin = BinaryData::create_binary_data((char*)(img_labels), sizeof(img1_mask)*2*8);

        // post labels volume
        // one could also write 2D image slices but the starting location must
        // be at least an ND point where N is greater than 2
        dvid_node.write_volume_roi(label_datatype_name, start, sizes, channels, labelbin);

        // retrieve the image volume and make sure it makes the posted volume
        DVIDLabelPtr labelcomp;
        dvid_node.get_volume_roi(label_datatype_name, start, sizes, channels, labelcomp);
        unsigned long long* labeldatacomp = labelcomp->get_raw();
        for (int i = 0; i < sizeof(img1_mask)*2; ++i) {
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
        Graph graph;
        dvid_node.get_vertex_neighbors("graph3", 1, graph);
        cout << "Weight of vertex 1: " << graph.vertices[0].weight << endl; 


    } catch (std::exception& e) {
        // catch DVID, libdvid, and boost errors
        cout << e.what() << endl;
    }
}
