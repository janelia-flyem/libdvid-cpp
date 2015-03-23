#include <iostream>
#include <libdvid/DVIDNode.h>

using std::cerr; using std::cout; using std::endl;
using namespace libdvid;
using std::string;

/*!
 * Test get/put of values using keyvalue type.
*/
int main(int argc, char** argv)
{
    if (argc != 2) {
        cout << "Usage: <program> <server_name> <uuid>" << endl;
        return -1;
    }
    try {
        DVIDServer server(argv[1]);
        std::string uuid = server.create_new_repo("newrepo", "This is my new repo");
        DVIDNode dvid_node(server, uuid);

        // name of key to use        
        string keyvalue_datatype_name = "keys";

        // check existence (should be new)
        if(!dvid_node.create_keyvalue(keyvalue_datatype_name)) {
            cerr << keyvalue_datatype_name << " already exists" << endl;
            return -1;
        }
        
        // check existence (should be old)
        if(dvid_node.create_keyvalue(keyvalue_datatype_name)) {
            cout << keyvalue_datatype_name << " should exist" << endl;
            return -1;
        }
        
        // Test key value interface
        Json::Value data_init;
        data_init["hello"] = "world"; 
        dvid_node.put(keyvalue_datatype_name, "key/spot0", data_init); 
        Json::Value data_ret; 
        dvid_node.get(keyvalue_datatype_name, "key/spot0", data_ret);
        std::string data_str = data_ret["hello"].asString();
        if (data_str != "world") {
            cerr << "Key value not stored properly" << endl;
            return -1;
        }
    } catch (std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }
    return 0;
}
