/*!
 * This file stores grayscale and label data using the
 * block interface and retrieves it using nD and block
 * interface. 
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#include <libdvid/DVIDServerService.h>
#include <libdvid/DVIDNodeService.h>
#include <libdvid/DVIDBlocks.h>

#include <iostream>
#include <vector>

using std::cerr; using std::cout; using std::endl;
using namespace libdvid;
using std::vector;

using std::string;

// assume all blocks are 32 in each dimension
// (graysale posts must be block aligned)
int BLK_SIZE = 32;

/*!
 * Checks the equivalence of two buffers
*/
template <typename T>
bool buffers_equal(const T* buf1,
        const T* buf2, int size)
{
    for (int i = 0; i < size; ++i) {
        if (buf1[size] != buf2[size]) {
            return false;
        }
    }

    return true;
}

/*!
 * Exercises the block interface and make sure data
 * retrieve is the same as the data posted.
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
   
        string gray_name = "grayblocks";
        string gray_namend = "graynd";
        string label_name = "labelblocks";
        string label_namend = "labelnd";

        // should be a new instances
        if(!dvid_node.create_grayscale8(gray_name)) {
            cerr << gray_name << " already exists" << endl;
            return -1;
        }
        if(!dvid_node.create_grayscale8(gray_namend)) {
            cerr << gray_namend << " already exists" << endl;
            return -1;
        }
        if(!dvid_node.create_labelblk(label_name)) {
            cerr << label_name << " already exists" << endl;
            return -1;
        }
        if(!dvid_node.create_labelblk(label_namend)) {
            cerr << label_namend << " already exists" << endl;
            return -1;
        }

        // create random inputs
        unsigned char* buffer1 = new unsigned char [BLK_SIZE*BLK_SIZE*BLK_SIZE];
        unsigned char* buffer2 = new unsigned char [BLK_SIZE*BLK_SIZE*BLK_SIZE];
        
        uint64* label1 = new uint64 [BLK_SIZE*BLK_SIZE*BLK_SIZE];
        uint64* label2 = new uint64 [BLK_SIZE*BLK_SIZE*BLK_SIZE];

        for (int i = 0; i < (BLK_SIZE*BLK_SIZE*BLK_SIZE); ++i) {
            buffer1[i] = rand() % 255;
            buffer2[i] = rand() % 255;
            label1[i] = rand() % 1000000;
            label2[i] = rand() % 1000000;
        }

        // create blocks (use push_back but could throw
        // buffer1 and buffer2 into one buffer)
        GrayscaleBlocks gray_blocks;
        gray_blocks.push_back(buffer1);
        gray_blocks.push_back(buffer2);
        LabelBlocks label_blocks;
        label_blocks.push_back(label1);
        label_blocks.push_back(label2);

       
        // offset in voxels coordinates
        vector<unsigned int> offset_voxels;
        offset_voxels.push_back(128);
        offset_voxels.push_back(32);
        offset_voxels.push_back(128);
        
        // offset in block coordinates
        vector<unsigned int> offset_blocks;
        offset_blocks.push_back(3);
        offset_blocks.push_back(1);
        offset_blocks.push_back(3);

        // just do nd BLK_SIZExBLK_SIZExBLK_SIZE queries
        vector<unsigned int> sizes;
        sizes.push_back(BLK_SIZE);
        sizes.push_back(BLK_SIZE);
        sizes.push_back(BLK_SIZE);

        dvid_node.put_grayblocks(gray_name, gray_blocks, offset_blocks);
        dvid_node.put_labelblocks(label_name, label_blocks, offset_blocks);

        // ask to read more blocks than are there (should return 2 blocks)
        GrayscaleBlocks gray_blocks_comp = 
            dvid_node.get_grayblocks(gray_name, offset_blocks, 5);
        LabelBlocks label_blocks_comp = 
            dvid_node.get_labelblocks(label_name, offset_blocks, 5);

        if (gray_blocks_comp.get_num_blocks() != 2) {
            throw ErrMsg("Retrieved more than 2 grayscale blocks");
        }
        if (label_blocks_comp.get_num_blocks() != 2) {
            throw ErrMsg("Retrieved more than 2 label blocks");
        }

        // check that buffers match
        if (!buffers_equal(gray_blocks[0], gray_blocks_comp[0],
                BLK_SIZE*BLK_SIZE*BLK_SIZE)) {
            throw ErrMsg("Retrieved incorrect grayscale block data");
        }  
        if (!buffers_equal(gray_blocks[1], gray_blocks_comp[1], 
                    BLK_SIZE*BLK_SIZE*BLK_SIZE)) {
            throw ErrMsg("Retrieved incorrect grayscale block data");
        }  
        if (!buffers_equal(label_blocks[0], label_blocks_comp[0], 
                    BLK_SIZE*BLK_SIZE*BLK_SIZE)) {
            throw ErrMsg("Retrieved incorrect label block data");
        }  
        if (!buffers_equal(label_blocks[1], label_blocks_comp[1],
                    BLK_SIZE*BLK_SIZE*BLK_SIZE)) {
            throw ErrMsg("Retrieved incorrect label block data");
        }  

        // post grayscale as two nD calls
        // post grayscale volume (note: that it is block aligned)
        Grayscale3D graypost1(buffer1, BLK_SIZE*BLK_SIZE*BLK_SIZE, sizes);
        dvid_node.put_gray3D(gray_namend, graypost1, offset_voxels);
        Grayscale3D graypost2(buffer2, BLK_SIZE*BLK_SIZE*BLK_SIZE, sizes);
        offset_voxels[0] += BLK_SIZE; // write the next block
        dvid_node.put_gray3D(gray_namend, graypost1, offset_voxels);

        // post labels as two nD calls
        // post labels volume (note: that it is block aligned)
        Labels3D labelpost1(label1, BLK_SIZE*BLK_SIZE*BLK_SIZE, sizes);
        offset_voxels[0] -= BLK_SIZE; // reinitialize starting position
        dvid_node.put_labels3D(label_namend, labelpost1, offset_voxels);
        Labels3D labelpost2(label2, BLK_SIZE*BLK_SIZE*BLK_SIZE, sizes);
        offset_voxels[0] += BLK_SIZE; // write the next block
        dvid_node.put_labels3D(label_namend, labelpost1, offset_voxels);

        offset_voxels[0] -= BLK_SIZE; // reinitialize starting position
        Grayscale3D graycomp0 = dvid_node.get_gray3D(gray_namend, sizes, offset_voxels);
        Labels3D labelcomp0 = dvid_node.get_labels3D(gray_namend, sizes, offset_voxels);

        // check that buffers match block buffers
        if (!buffers_equal(graycomp0.get_raw(), gray_blocks[0],
                BLK_SIZE*BLK_SIZE*BLK_SIZE)) {
            throw ErrMsg("nD gray data does not match block data");
        }
        if (!buffers_equal(labelcomp0.get_raw(), label_blocks[0],
                 BLK_SIZE*BLK_SIZE*BLK_SIZE)) {
            throw ErrMsg("nD label data does not match block data");
        }
    } catch (std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }
    return 0;
}
