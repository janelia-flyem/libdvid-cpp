/*!
 * This file gives a simple example of creating
 * a labelvol instance by creating a labelblk instance
 * (used for image segmentation).  It then tries to
 * retrieve a coarse volume and reprentative points from
 * the body.
 * 
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#include <libdvid/DVIDServerService.h>
#include <libdvid/DVIDNodeService.h>
#include <libdvid/DVIDThreadedFetch.h>

#include <iostream>
#include <vector>

using std::cerr; using std::cout; using std::endl;
using namespace libdvid;

using std::vector;
using std::string;

// assume all blocks are 32 in each dimension
// (label posts must be block aligned)
int BLK_SIZE = 32;

/*!
 * Exercises the body interface.
*/
int main(int argc, char** argv)
{
    if (argc != 2) {
        cout << "Usage: <program> <server_name> <uuid>" << endl;
        return -1;
    }
    try {
        DVIDServerService server(argv[1]);
        string uuid = server.create_new_repo("newrepo", "This is my new repo");
        DVIDNodeService dvid_node(argv[1], uuid);
    
        string label_datatype_name = "labels1";
        string labelvol_datatype_name = "labels1_vol";
        
        // should be a new instance
        if(!dvid_node.create_labelblk(label_datatype_name,
                    labelvol_datatype_name)) {
            throw ErrMsg(label_datatype_name + " or " + labelvol_datatype_name
                + " already exists");
        }

        // should not recreate labelblk if it already exists
         if(dvid_node.create_labelblk(label_datatype_name)) {
            throw ErrMsg(label_datatype_name + " should exist");
         }

        // ** Write and read labelblk data **
        
        // create label 64 image volume
        unsigned int XDIM = BLK_SIZE * 3;
        unsigned int YDIM = BLK_SIZE * 4;
        unsigned int ZDIM = BLK_SIZE * 7;
        uint64 * img_labels = new uint64 [XDIM*YDIM*ZDIM];
        for (unsigned int i = 0; i < (XDIM*YDIM*ZDIM); ++i) {
            img_labels[i] = 0;
        }

        // write a body that spans 4 blocks
        // write label 5 in x=0,1,2 y=0, z=0
        img_labels[0] = 5;
        img_labels[BLK_SIZE] = 5;
        img_labels[BLK_SIZE*2] = 5;
        // write label 5 in x=0,y=0,z=1
        img_labels[XDIM*YDIM*BLK_SIZE] = 5;


        // create binary data string wrapper (64 bits per pixel)
        vector<int> start;
        start.push_back(0);
        start.push_back(0); start.push_back(0);

        // post labels volume
        Dims_t lsizes; lsizes.push_back(XDIM);
        lsizes.push_back(YDIM); lsizes.push_back(ZDIM);
        Labels3D labelsbin(img_labels, XDIM*YDIM*ZDIM, lsizes);
        dvid_node.put_labels3D(label_datatype_name, labelsbin, start);
        delete []img_labels;

        // give labelvol a chance to synchronize (should be fast)
        sleep(1);

        // check whether bodies exist or not in the volume
        if(dvid_node.body_exists(labelvol_datatype_name, uint64(3))) {
            throw ErrMsg("Body 3 should not exist in labelvol");
        }

        if(!dvid_node.body_exists(labelvol_datatype_name, uint64(5))) {
            throw ErrMsg("Body 5 should exist in labelvol");
        }

        // check that the correct coarse volume is returned
        vector<BlockXYZ> blockcoords;
        dvid_node.get_coarse_body(labelvol_datatype_name, uint64(5), blockcoords);

        if (blockcoords.size() != 4) {
            throw ErrMsg("4 blocks should exist for body 5");
        }

        if ((blockcoords[0].z != 0) || (blockcoords[1].z != 0) ||
            (blockcoords[2].z != 0) || (blockcoords[3].z != 1) ||
            (blockcoords[0].y != 0) || (blockcoords[1].y != 0) ||
            (blockcoords[2].y != 0) || (blockcoords[3].y != 0) ||
            (blockcoords[0].x != 0) || (blockcoords[1].x != 1) ||
            (blockcoords[2].x != 2) || (blockcoords[3].x != 0)) {
            throw ErrMsg("Returned block coordinates for body 5 are incorrect");
        }

        // find point in middle of volume
        PointXYZ midpoint = dvid_node.get_body_location(labelvol_datatype_name, uint64(5));

        // retrieve sparse vol with custom command (check compressed version)
        BinaryDataPtr sparsevol = dvid_node.custom_request("/labels1_vol/sparsevol/5", BinaryDataPtr(), GET, false);
 
        char* blah = new char[INT_MAX];
 
        BinaryDataPtr sparsevol2 = dvid_node.custom_request("/labels1_vol/sparsevol/5", BinaryDataPtr(), GET, true);
  
        const byte* sraw1 = sparsevol->get_raw(); 
        const byte* sraw2 = sparsevol2->get_raw();
        if (sparsevol->length() != sparsevol2->length()) {
            throw ErrMsg("Sparse vol compression mismatch");
        }
        for (int i = 0; i < sparsevol->length(); ++i) {
            if (sraw1[i] != sraw2[i]) {
                throw ErrMsg("Sparse vol compression value mismatch");
            }
        }

        // should be block (2,0,0) and coordinate 2*BLK_SIZE+BLK_SIZE/2,
        // BLK_SIZE/2, BLK_SIZE/2
        if ((midpoint.x != (2*BLK_SIZE+BLK_SIZE/2)) || (midpoint.y != BLK_SIZE/2)
                || (midpoint.z != BLK_SIZE/2)) {
            throw ErrMsg("Returned center point for body 5 is incorrect");
        }

        // find point in middle of z block = 1
        PointXYZ midpoint2 = dvid_node.get_body_location(labelvol_datatype_name,
                uint64(5), BLK_SIZE + 10);

        // should be block (0,0,1) and coordinate BLK_SIZE/2,
        // BLK_SIZE/2, BLK_SIZE + 10 
        if ((midpoint2.x != BLK_SIZE/2) || (midpoint2.y != BLK_SIZE/2)
                || (midpoint2.z != (BLK_SIZE + 10))) {
            throw ErrMsg("Returned center for body 5, plane 42 is incorrect");
        }

        // ******* test parallel sparse vol fetch ***********
        string gray_datatype_name = "gray1";
        
        // create new gray instance
        if(!dvid_node.create_grayscale8(gray_datatype_name)) {
            throw ErrMsg(gray_datatype_name + " already exists");
        }
        uint8* img_gray = new uint8 [XDIM*YDIM*ZDIM];
        for (unsigned int i = 0; i < (XDIM*YDIM*ZDIM); ++i) {
            img_gray[i] = rand()%255;
        }
        Grayscale3D graybin(img_gray, XDIM*YDIM*ZDIM, lsizes);
        dvid_node.put_gray3D(gray_datatype_name, graybin, start);

        vector<BinaryDataPtr> grayarray = get_body_blocks(dvid_node, labelvol_datatype_name, gray_datatype_name, uint64(5), 2, false, 1);
        vector<BinaryDataPtr> grayarray2 = get_body_blocks(dvid_node, labelvol_datatype_name, gray_datatype_name, uint64(5), 1, false, 0);
        vector<BinaryDataPtr> grayarray3 = get_body_blocks(dvid_node, labelvol_datatype_name, gray_datatype_name, uint64(5), 4, true, 1);

        // 4 gray blocks should be returned
        if ((grayarray.size() != 4) || (grayarray2.size() != 4) ||
            (grayarray3.size() != 4)) {
            throw ErrMsg("Returned gray array is not 4");
        }

        // all gray fetch calls should return the same value
        for (int i = 0; i < grayarray.size(); ++i) {
            for (int j = 0; j < (BLK_SIZE*BLK_SIZE*BLK_SIZE); ++j) {
                if ((grayarray[i]->get_raw()[j] !=
                    grayarray3[i]->get_raw()[j]) ||
                    (grayarray[i]->get_raw()[j] != 
                    grayarray2[i]->get_raw()[j])) {
                    throw ErrMsg("Equivalent sparse gray volume fetches do not return same values");
                }
            }
        }
        
        // should be equal to original gray -- check first row of graybin
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < BLK_SIZE; ++j) {
                if (grayarray[i]->get_raw()[j] != img_gray[i*BLK_SIZE+j]) {
                    throw ErrMsg("First 3 retrieved blocks do not match saved gray");
                }
            }

        }
    } catch (std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }
    return 0;
}
