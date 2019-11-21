/*!
 * This file gives a simple example of creating
 * a labels64 instance (used for image segmentation).
 * It stores some data, retrieves the data, and
 * checks that the data is equal.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#include <libdvid/DVIDServerService.h>
#include <libdvid/DVIDNodeService.h>

#include <iostream>
#include <vector>

using std::cerr; using std::cout; using std::endl;
using namespace libdvid;

using std::vector;
using std::string;

// assume all blocks are 32 in each dimension
// (label posts must be block aligned)
int BLK_SIZE = 32;

// sample labels to write
unsigned long long int limg1_mask[] = {
    5, 4, 3, 2,
    4, 4, 1, 3,
    7, 7, 7, 7};

// sample labels to write
unsigned long long int limg2_mask[] = {
    8, 8, 9, 0,
    9, 9, 9, 3,
    9, 9, 9, 7};

int LWIDTH = 4;
int LHEIGHT = 3;

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
        DVIDServerService server(argv[1]);
        string uuid = server.create_new_repo("newrepo", "This is my new repo");
        DVIDNodeService dvid_node(argv[1], uuid);
    
        string label_datatype_name = "labels1";
        
        // should be a new instance
        if(!dvid_node.create_labelblk(label_datatype_name)) {
            cerr << label_datatype_name << " already exists" << endl;
            return -1;
        }

        // should not recreate labelblk if it already exists
         if(dvid_node.create_labelblk(label_datatype_name)) {
            cerr << label_datatype_name << " should exist" << endl;
            return -1;
        }

        // ** Write and read labelblk data **
        
        // create label 64 image volume
        uint64 * img_labels = new uint64 [BLK_SIZE*BLK_SIZE*BLK_SIZE];
        for (int i = 0; i < (LWIDTH*LHEIGHT); ++i) {
            img_labels[i] = limg1_mask[i];
            img_labels[i+BLK_SIZE*BLK_SIZE] = limg2_mask[i];
        }

        // create binary data string wrapper (64 bits per pixel)
        vector<int> start; start.push_back(0); start.push_back(0); start.push_back(0);

        // post labels volume
        Dims_t lsizes; lsizes.push_back(BLK_SIZE); lsizes.push_back(BLK_SIZE); lsizes.push_back(BLK_SIZE);
        Labels3D labelsbin(img_labels, BLK_SIZE*BLK_SIZE*BLK_SIZE, lsizes);
        dvid_node.put_labels3D(label_datatype_name, labelsbin, start);

        // retrieve the image volume and make sure it makes the posted volume
        Labels3D labelcomp = dvid_node.get_labels3D(label_datatype_name, lsizes, start);
        
        // verify the blocksize is 32
        size_t blocksize = dvid_node.get_blocksize(label_datatype_name);
        if (blocksize != 32) {
            cerr << label_datatype_name << " is not 32x32x32" << endl;
            return -1;
        }

        const uint64* labeldatacomp = labelcomp.get_raw();
        for (int i = 0; i < BLK_SIZE*BLK_SIZE*BLK_SIZE; ++i) {
            if (labeldatacomp[i] != img_labels[i]) {
                cerr << "Read/write mismatch" << endl;
                return -1;
            }
        }

        // check block return
        vector<DVIDCompressedBlock> lblocks  = dvid_node.get_labelblocks3D(label_datatype_name, lsizes, start);
        labeldatacomp = (uint64*) lblocks[0].get_uncompressed_data()->get_raw();
        for (int i = 0; i < BLK_SIZE*BLK_SIZE*BLK_SIZE; ++i) {
            if (labeldatacomp[i] != img_labels[i]) {
                cerr << "Read/write mismatch using block request" << endl;
                return -1;
            }
        }

        delete []img_labels;
    } catch (std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }
    return 0;
}
