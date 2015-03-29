/*!
 * This file tests the performance of creating a large
 * label volume, adding it to DVID, and then doing a
 * series of label blk fetches.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#include <libdvid/DVIDNodeService.h>
#include <libdvid/DVIDServerService.h>
#include "ScopeTime.h"

#include <iostream>

using std::cerr; using std::cout; using std::endl;
using namespace libdvid;
using std::vector;

using std::string;

// assume all blocks are BLK_SIZE in each dimension
int BLK_SIZE = 32;

// size of the segmentation to be created
int VOLDIM = 512;

// number of random fetches to do
int NUM_FETCHES = 100;

// size of the small volume gets 
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
        DVIDServerService server(argv[1]);
        string uuid = server.create_new_repo("newrepo", "This is my new repo");
        DVIDNodeService dvid_node(argv[1], uuid);
   
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

        // create binary data string wrapper (64 bits per pixel)
        // calculate time to post volume
        {
            Dims_t dims;
            dims.push_back(VOLDIM); dims.push_back(VOLDIM); dims.push_back(VOLDIM); 
            Labels3D labelbin = Labels3D(img_labels,
                VOLDIM * VOLDIM * VOLDIM, dims);
            delete []img_labels;

            vector<unsigned int> start;
            start.push_back(0); start.push_back(0); start.push_back(0);

            ScopeTime post_timer(false);
            dvid_node.put_labels3D(label_datatype_name, labelbin, start);
            double post_time = post_timer.getElapsed();
            cout << "Time to POST large label volume: " << post_time << " seconds" << endl;
            cout << int(VOLDIM * VOLDIM * VOLDIM  * sizeof(unsigned long long) / post_time)
                << " bytes posted per second for large label volume" << endl;
        }

        // retrieve the image volume
        // this might be artifically fast as it might be completely in DVID's cache
        {
            vector<unsigned int> start;
            start.push_back(0); start.push_back(0); start.push_back(0);
            Dims_t lsizes;
            lsizes.push_back(VOLDIM); lsizes.push_back(VOLDIM); lsizes.push_back(VOLDIM);
            
            ScopeTime get_timer(false);
            Labels3D labelcomp = dvid_node.get_labels3D(label_datatype_name, lsizes, start);
            double read_time = get_timer.getElapsed();
            cout << "Time to GET large label volume: " << read_time << " seconds" << endl;
            cout << int(VOLDIM * VOLDIM * VOLDIM * sizeof(unsigned long long)/ read_time)
                << " bytes read per second for large label volume" << endl;
        }

        // do a lot of small requests (like already in DVID cache)
        double total_time = 0.0;
        for (int i = 0; i < NUM_FETCHES; ++i) {
            ScopeTime get_timer(false);
            int max_val = VOLDIM - SMALLFETCH + 1;
            
            // grab random small cube
            vector<unsigned int> start;
            start.push_back(rand()%max_val);
            start.push_back(rand()%max_val);
            start.push_back(rand()%max_val);
            vector<unsigned int> lsizes;
            lsizes.push_back(SMALLFETCH);
            lsizes.push_back(SMALLFETCH);
            lsizes.push_back(SMALLFETCH);
            
            Labels3D labelcomp =
                dvid_node.get_labels3D(label_datatype_name, lsizes, start);

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
