#include <libdvid/DVIDThreadedFetch.h>

#include "ScopeTime.h"
#include <string>
#include <vector>

using std::cout; using std::endl;
using std::string;
using std::vector;

const char * USAGE = "<prog> <dvid-server> <uuid> <label name> <gray name> <body id>";
const char * HELP = "Program takes a body id and fetches the sparse volume in grayscale";


int main(int argc, char** argv)
{
    if (argc != 6) {
        cout << USAGE << endl;
        cout << HELP << endl;
        exit(1);
    }
    
    // create DVID node accessor 
    libdvid::DVIDNodeService dvid_node(argv[1], argv[2]);

    // slightly bogus test because the cache will be warmed up
    
    {
        // call using the ND raw
        ScopeTime overall_time;
        libdvid::GrayscaleBlocks blocks = libdvid::get_body_blocks(dvid_node, argv[3], argv[4], atoi(argv[5]), false, 1, 1);
    }

    {
        ScopeTime overall_time;
        // call doing one request at a time 
        libdvid::GrayscaleBlocks blocks = libdvid::get_body_blocks(dvid_node, argv[3], argv[4], atoi(argv[5]), false, 1, 0);
    }

    return 0;
}



