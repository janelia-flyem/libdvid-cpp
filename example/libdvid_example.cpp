#include <iostream>
#include <boost/network/protocol/http/client.hpp>
#include <libdvid/DVIDServer.h>
#include <libdvid/Utilities.h>

using std::cout; using std::endl;
using namespace boost::network;
using namespace boost::network::http;
using namespace libdvid;

int main() {
/*    try {
    DVIDServer server("blah");
    } catch (ErrMsg& msg) {
        cout << "blah" << endl;
//////        cout << msg << endl;
    }*/

    client::request request2("http://www.google.com");
    request2 << header("Connection", "close");
    client client2;
    client::response response2 = client2.get(request2);
    std::string body2 = body(response2);
    std::string body3 = status_message(response2);
    int body4 = status(response2);

    cout << body2 << endl;
    cout << body3 << endl;
    cout << body4 << endl;
}
