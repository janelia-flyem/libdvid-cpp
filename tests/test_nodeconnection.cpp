#include <iostream>
#include <libdvid/DVIDNode.h>

using std::cerr; using std::cout; using std::endl;
using namespace libdvid;

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
    } catch (std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }
    return 0;
}
