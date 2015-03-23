#include <iostream>
#include <libdvid/DVIDNode.h>
#include "ScopeTime.h"

using std::cerr; using std::cout; using std::endl;
using namespace libdvid;

using std::string;

// assume all blocks are BLK_SIZE in each dimension
int BLK_SIZE = 32;

int VOLDIM = 512;

int NUM_FETCHES = 100;

int SMALLFETCH = 128;

/*!
 * Exercises the nD GETs and PUTs for the labelblk datatype.
*/
int main(int argc, char** argv)
{
    if (argc != 2) {
        cout << "Usage: <program> <server_name>" << endl;
        return -1;
    }
    try {
        ScopeTime overall_time;
        DVIDServer server(argv[1]);
        string uuid = server.create_new_repo("newrepo", "This is my new repo");
        DVIDNode dvid_node(server, uuid);
   
        cout << "Large volume dimension: " << VOLDIM << endl;
        cout << "Small volume dimension: " << SMALLFETCH << endl;

        string label_datatype_name = "labels1";
        
        // should be a new instance
        if(!dvid_node.create_labelblk(label_datatype_name)) {
            cerr << label_datatype_name << " already exists" << endl;
            return -1;
        }

        // ** Write and read labelblk data **
        // create label 64 image volume
        unsigned long long * img_labels = new unsigned long long [VOLDIM * VOLDIM * VOLDIM];
        for (int i = 0; i < (VOLDIM*VOLDIM*VOLDIM); ++i) {
            img_labels[i] = rand() % 1000000;
        }

        tuple channels; channels.push_back(0); channels.push_back(1); channels.push_back(2);
        // create binary data string wrapper (64 bits per pixel)
        {
            BinaryDataPtr labelbin = BinaryData::create_binary_data((char*)(img_labels), VOLDIM * VOLDIM * VOLDIM*sizeof(unsigned long long));
            delete []img_labels;

            tuple start; start.push_back(0); start.push_back(0); start.push_back(0);
            tuple lsizes; lsizes.push_back(VOLDIM); lsizes.push_back(VOLDIM); lsizes.push_back(VOLDIM);

            ScopeTime post_timer(false);
            dvid_node.write_volume_roi(label_datatype_name, start, lsizes, channels, labelbin);
            double post_time = post_timer.getElapsed();
            cout << "Time to POST large label volume: " << post_time << " seconds" << endl;
            cout << int(VOLDIM * VOLDIM * VOLDIM  * sizeof(unsigned long long) / post_time)
                << " bytes posted per second for large label volume" << endl;
        }

        // retrieve the image volume and make sure it makes the posted volume
        {
            tuple start; start.push_back(0); start.push_back(0); start.push_back(0);
            tuple lsizes; lsizes.push_back(VOLDIM); lsizes.push_back(VOLDIM); lsizes.push_back(VOLDIM);

            DVIDLabelPtr labelcomp;
            ScopeTime get_timer(false);
            dvid_node.get_volume_roi(label_datatype_name, start, lsizes, channels, labelcomp);
            double read_time = get_timer.getElapsed();
            cout << "Time to GET large label volume: " << read_time << " seconds" << endl;
            cout << int(VOLDIM * VOLDIM * VOLDIM * sizeof(unsigned long long)/ read_time)
                << " bytes read per second for large label volume" << endl;
        }

        // do a lot of small requests
        double total_time = 0.0;
        for (int i = 0; i < NUM_FETCHES; ++i) {
            ScopeTime get_timer(false);
            int max_val = VOLDIM - SMALLFETCH + 1;

            tuple start;
            start.push_back(rand()%max_val);
            start.push_back(rand()%max_val);
            start.push_back(rand()%max_val);
            tuple lsizes;
            lsizes.push_back(SMALLFETCH);
            lsizes.push_back(SMALLFETCH);
            lsizes.push_back(SMALLFETCH);
            
            DVIDLabelPtr labelcomp;
            dvid_node.get_volume_roi(label_datatype_name, start, lsizes, channels, labelcomp);

            double read_time = get_timer.getElapsed();
            cout << "Time to GET small label volume: " << read_time << " seconds" << endl;
            total_time += read_time;
        }

        cout << NUM_FETCHES << " fetches performed in " << total_time << " seconds" << endl;
        cout << total_time / NUM_FETCHES << " seconds per fetch" << endl;
        cout << int(SMALLFETCH * SMALLFETCH * SMALLFETCH * NUM_FETCHES * sizeof(unsigned long long)/ total_time)
            << " bytes read per second" << endl;        
    } catch (std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }
    return 0;
}
