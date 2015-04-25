/*!
 * This file simply creates a new DVID repo.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#include <libdvid/DVIDServerService.h>

#include <iostream>

using std::cerr; using std::cout; using std::endl;
using namespace libdvid;

/*!
 * Creates DVID repo.
*/
int main(int argc, char** argv)
{
    if (argc != 2) {
        cout << "Usage: <program> <server_name>" << endl;
        return -1;
    }
    try {
        DVIDServerService server(argv[1]);
        // unique ID for each repo created (returns the uuid)
        server.create_new_repo("newrepo",
                "This is my new repo");
    } catch (std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }

    return 0;
}
