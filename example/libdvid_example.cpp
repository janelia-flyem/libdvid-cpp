#include <iostream>
#include <boost/network/protocol/http/client.hpp>
#include <libdvid/DVIDServer.h>
#include <libdvid/DVIDNode.h>
#include <libdvid/Utilities.h>

using std::cout; using std::endl;
using namespace boost::network;
using namespace boost::network::http;
using namespace libdvid;

int main() {
    client client2;
    try {
        DVIDServer server("http://127.0.0.1:8000");
        DVIDNode dvid_node(server, "1e8");
//        dvid_node.create_keyvalue("blah");    
       
        
        
        Json::Value data2;
        data2["hello"] = "no!!!!!"; 
        dvid_node.put("blah", "spot0", data2); 
        
        
        
        Json::Value data; 
        dvid_node.get("blah", "spot0", data);
        std::string data_str = data["hello"].asString();
        cout << data_str << endl; 
    
    
    
    } catch (std::exception& e) {
        cout << e.what() << endl;
    }

   
   /*
    client::request request2("http://www.google2.com");
    request2 << header("Connection", "close");
    client client2;
    cout << "blah0" << endl;
    client::response response2 = client2.get(request2);
    cout << "blah1" << endl;
    int body4;
    try {
        body4 = status(response2);
    } catch (std::exception& ex) {
        cout << ex.what() << endl;
    }
    cout << "blah2" << endl;
    cout << body4 << endl;
    cout << "blah3" << endl;
    std::string body2 = body(response2);
    cout << "blah4" << endl;
    std::string body3 = status_message(response2);

    cout << body2 << endl;
    cout << body3 << endl;
    */
}
