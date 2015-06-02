#include <libdvid/Algorithms.h>
#include <libdvid/DVIDThreadedFetch.h>

#include "ScopeTime.h"
#include <string>
#include <vector>
#include <fstream>

#include <json/json.h>
#include <json/value.h>

using std::cout; using std::endl;
using std::string;
using std::vector;

using namespace libdvid;
using std::ifstream;

//#define DVIDDEBUG_CPY

const char * USAGE = "<prog> <dvid-server> <uuid1> <labelvol1> <label name1> <dvid-server2> <uuid2> <label name2> <bodies json>";
const char * HELP = "Program takes a list of bodies from one UUID and copies to another UUID";

int main(int argc, char** argv)
{
    // ?! temp
    
    DVIDNodeService dvid_node3(argv[1], argv[2]);
    int found = 0;
    int i = 1;
    while (found < 1000) {
        try {
            vector<BlockXYZ> blockcoords;
            dvid_node3.get_coarse_body(argv[3], uint64(i), blockcoords);
            if (blockcoords.empty()) {
                cout << i << endl;
                found++;
            }
        } catch (...) {
            cout << i << endl;
            found++;
        }
        ++i;
    }


    ScopeTime overall_time;
    if (argc != 9) {
        cout << USAGE << endl;
        cout << HELP << endl;
        exit(1);
    }
    
    // create DVID node accessor 
    DVIDNodeService dvid_node(argv[1], argv[2]);
    DVIDNodeService dvid_node2(argv[5], argv[6]);

    // read old/new body id list
    Json::Reader json_reader;
    Json::Value json_vals;
    ifstream fin(argv[8]);
    if (!fin) {
        throw ErrMsg("Error: input file: " + string(argv[8]) + " cannot be opened");
    }

    if (!json_reader.parse(fin, json_vals)) {
        throw ErrMsg("Error: Json incorrectly formatted");
    }
    fin.close();

    for (unsigned int i = 0; i < json_vals.size(); ++i) {
        uint64 srcid = json_vals[i][0].asUInt();
        uint64 destid = json_vals[i][1].asUInt();
        cout << "Copying " << srcid << " to " << destid << endl;

#ifdef DVIDDEBUG_CPY
        vector<vector<int> > spans;
        vector<BinaryDataPtr> blocks =
            get_body_labelblocks(dvid_node, argv[3], srcid, argv[4], spans, 2);
#endif
        
        // copy/paste body with two threads 
        copy_paste_body(dvid_node, dvid_node2, srcid, destid, argv[4], argv[3], argv[7], 2);


#ifdef DVIDDEBUG_CPY
        // ?! debug
        cout << "Verifying Sleeping..." << endl;
        sleep(20);
        cout << "Verifying ..." << endl;
        spans.clear(); // assume same labelvol name for debug
        vector<BinaryDataPtr> blocks2 =
            get_body_labelblocks(dvid_node2, argv[3], destid, argv[7], spans, 2);

        assert(blocks.size() == blocks2.size());

        int volume = 0;
        for (unsigned int i = 0; i < blocks.size(); ++i) {
            const uint64* raw1 = (uint64*) blocks[i]->get_raw();
            const uint64* raw2 = (uint64*) blocks2[i]->get_raw();
            for (unsigned int j = 0; j < (DEFBLOCKSIZE*DEFBLOCKSIZE*
                    DEFBLOCKSIZE); ++j) {
                if (*raw1 == srcid) {
                    assert(*raw2 == destid);
                    ++volume;
                }
                
                //assert(*raw1 == *raw2); 
                ++raw1; ++raw2;
            }
        }

        cout << "Body size: " << volume << endl;
        cout << argv[7] << " at " << argv[6] << " is equivalent to " << argv[4] << " at " << argv[2] << endl;
#endif

    }

    return 0;
}



