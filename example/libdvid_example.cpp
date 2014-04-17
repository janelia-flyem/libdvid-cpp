#include <iostream>
#include <libdvid/DVIDNode.h>

using std::cout; using std::endl;
using std::string;

using namespace libdvid;

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

        // test creation of DVID datatypes
        if(!dvid_node.create_grayscale8(gray_datatype_name)) {
            cout << gray_datatype_name << " already exists" << endl;
        }
        if(!dvid_node.create_labels64(label_datatype_name)) {
            cout << label_datatype_name << " already exists" << endl;
        }
        if(!dvid_node.create_keyvalue(keyvalue_datatype_name)) {
            cout << keyvalue_datatype_name << " already exists" << endl;
        }
        
        // test key value interface
        Json::Value data_init;
        data_init["hello"] = "world"; 
        dvid_node.put(keyvalue_datatype_name, "spot0", data_init); 
        Json::Value data_ret; 
        dvid_node.get(keyvalue_datatype_name, "spot0", data_ret);
        std::string data_str = data_ret["hello"].asString();
        cout << "Response: " << data_str << endl; 
    
    } catch (std::exception& e) {
        // catch DVID, libdvid, and boost errors
        cout << e.what() << endl;
    }
}
