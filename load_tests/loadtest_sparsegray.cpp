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
    vector<libdvid::BinaryDataPtr> blocks;
    vector<libdvid::BinaryDataPtr> blocks2;
    vector<libdvid::BinaryDataPtr> blocks3;
    // slightly bogus test because the cache will be warmed up
    {
        // call using the ND raw
        ScopeTime overall_time;
        blocks = libdvid::get_body_blocks(dvid_node, argv[3], argv[4], atoi(argv[5]), 1, false, 1);
    }

    {
        // call using blocks
        ScopeTime overall_time;
        blocks2 = libdvid::get_body_blocks(dvid_node, argv[3], argv[4], atoi(argv[5]), 1, true, 1);
    }

    {
        ScopeTime overall_time;
        // call doing one request at a time 
        blocks3 = libdvid::get_body_blocks(dvid_node, argv[3], argv[4], atoi(argv[5]), 1, false, 0);
    }
    
    assert(blocks.size() == blocks2.size());
    assert(blocks2.size() == blocks3.size());

    for (unsigned int i = 0; i < blocks.size(); ++i) {
        const unsigned char* raw1 = blocks[i]->get_raw();
        const unsigned char* raw2 = blocks2[i]->get_raw();
        const unsigned char* raw3 = blocks3[i]->get_raw();
        for (unsigned int j = 0; j < libdvid::DEFBLOCKSIZE*
                libdvid::DEFBLOCKSIZE*libdvid::DEFBLOCKSIZE; ++j) {
                assert(*raw1 == *raw2); 
                assert(*raw2 == *raw3);
                ++raw1; ++raw2; ++raw3; 
        }
    }

    return 0;
}



