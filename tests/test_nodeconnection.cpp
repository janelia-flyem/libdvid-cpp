/*!
 * This file creates a DVID repo and then tries to access
 * a DVID node from this repo.  By default a new DVID
 * repo creates a UUID that is the same ID from the root node.
*/

#include <libdvid/DVIDServerService.h>
#include <libdvid/DVIDNodeService.h>

#include <iostream>

using std::cerr; using std::cout; using std::endl;
using namespace libdvid;

/*!
 * Create repo and test node access.
*/
int main(int argc, char** argv)
{
    if (argc != 2) {
        cout << "Usage: <program> <server_name> <uuid>" << endl;
        return -1;
    }
    try {
        DVIDServerService server(argv[1]);
        // unique ID for new repo
        std::string uuid = server.create_new_repo("newrepo", "This is my new repo");
        
        // same as unique ID for root DVID node
        DVIDNodeService dvid_node(argv[1], uuid);
    } catch (std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }
    return 0;
}
