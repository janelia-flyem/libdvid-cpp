#include <libdvid/DVIDThreadedFetch.h>
#include <libdvid/DVIDException.h>

#include <vector>

using std::string;
using std::vector;

//! Max blocks to request at one tiem
static const int MAX_BLOCKS = 4096;


namespace libdvid {

GrayscaleBlocks get_body_blocks(DVIDNodeService& service, string labelvol_name,
        string grayscale_name, uint64 bodyid, bool use_blocks, int num_threads,
        int request_efficiency)
{
    vector<BlockXYZ> blockcoords;

    if (!service.get_coarse_body(labelvol_name, bodyid, blockcoords)) {
        throw ErrMsg("Body not found, no grayscale blocks could be retrieved");
    }

    int num_requests = 0;

    GrayscaleBlocks blocks;

    uint8* blockdata = 0;
    if ((request_efficiency == 1) && !use_blocks) {
        blockdata = new uint8[DEFBLOCKSIZE*DEFBLOCKSIZE*DEFBLOCKSIZE];
    }

    // !! probably unnecessary copying going on
    // iterate through block coords and call ND or blocks one by one or contig
    int xmin; 
    int curr_runlength = 0;
    for (unsigned int i = 0; i < blockcoords.size(); ++i) {
        int z = blockcoords[i].z;
        int y = blockcoords[i].y;
        int x = blockcoords[i].x;
        if (curr_runlength == 0) {
            xmin = x; 
        }
        curr_runlength += 1; 
       
        bool requestblocks = false;

        if (request_efficiency == 0) {
            // if fetching 1 by 1 always request
            requestblocks = true;
        } else if (curr_runlength == MAX_BLOCKS) {
            // if there are too many blocks to fetch
            requestblocks = true;  
        } else if (i == (blockcoords.size()-1)) {
            // if there are no more blocks fetch
            requestblocks = true;
        } else if (i < (blockcoords.size()-1)) {
            // if y or z are different or x is non-contiguous time to fetch
            if ((blockcoords[i+1].z != z) || (blockcoords[i+1].y != y) || 
                    (((blockcoords[i+1].x)) != (x+1))) {
                requestblocks = true;
            }
        }

        if (requestblocks) {
            ++num_requests;
            if (use_blocks) {
                // use block interface (currently most re-copy)
                vector<int> block_coords;
                block_coords.push_back(xmin);
                block_coords.push_back(y);
                block_coords.push_back(z);
                GrayscaleBlocks blocks2 = service.get_grayblocks(grayscale_name, block_coords, curr_runlength);
                for (int j = 0; j < curr_runlength; ++j) {
                    blocks.push_back(blocks2[j]);
                }
            } else {
                Dims_t dims;
                dims.push_back(DEFBLOCKSIZE*curr_runlength);
                dims.push_back(DEFBLOCKSIZE);
                dims.push_back(DEFBLOCKSIZE);
                vector<int> offset;
                offset.push_back(xmin*DEFBLOCKSIZE);
                offset.push_back(y*DEFBLOCKSIZE);
                offset.push_back(z*DEFBLOCKSIZE);

                Grayscale3D grayvol = service.get_gray3D(grayscale_name,
                        dims, offset, false); 
                
                if (curr_runlength == 1) {
                    // do a simple copy for just one block
                    blocks.push_back(grayvol.get_raw());
                } else {
                    const uint8* raw_data = grayvol.get_raw();
                    
                    // otherwise create a buffer and do something more complicated 
                    for (int j = 0; j < curr_runlength; ++j) {
                        int offsetx = j * DEFBLOCKSIZE;
                        int offsety = curr_runlength*DEFBLOCKSIZE;
                        int offsetz = curr_runlength*DEFBLOCKSIZE*DEFBLOCKSIZE;
                        uint8* mod_data_iter = blockdata; 

                        for (int ziter = 0; ziter < DEFBLOCKSIZE; ++ziter) {
                            const uint8* data_iter = raw_data + ziter * offsetz;    
                            data_iter += (offsetx);
                            for (int yiter = 0; yiter < DEFBLOCKSIZE; ++yiter) {
                                for (int xiter = 0; xiter < DEFBLOCKSIZE; ++xiter) {
                                    *mod_data_iter = *data_iter;
                                    ++mod_data_iter;
                                    ++data_iter;
                                }
                                data_iter += ((offsety) - DEFBLOCKSIZE);
                            }
                        }
                        blocks.push_back(blockdata);
                    }
                }
            }

            curr_runlength = 0;
        }
    
    }

    if (blockdata) {
        delete []blockdata;
    }

    std::cout << "Performed " << num_requests << " requests" << std::endl;
    return blocks;
}


}
