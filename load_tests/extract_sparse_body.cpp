#include <libdvid/DVIDThreadedFetch.h>

#include "ScopeTime.h"
#include <string>
#include <vector>

using std::cout; using std::endl;
using std::string;
using std::vector;
using std::ofstream;

const char * USAGE = "<prog> <dvid-server> <uuid> <label name> <dvid-server> <uuid>  <gray name> <body id> <level>";
const char * HELP = "Program takes a body id and fetches the sparse volume in grayscale";


/*
Example fetch from FIB25:

./dvidextract_sparse_body 127.0.0.1:8000 a5 labelstest 127.0.0.1:9000 82 grayscaleview 10319 0


Write TIFF file:

----
import tifffile
import sys
import numpy

arraystr = open(sys.argv[1]).read()
z = int(sys.argv[2])
y = int(sys.argv[3])
x = int(sys.argv[4])

arr = numpy.fromstring(arraystr, dtype=numpy.uint8)
arr = arr.reshape((z,y,x))

# write to tiff
tifffile.imsave(sys.argv[1] + ".tif", arr)
---

python writetiff.py arraystring.txt 64 1216 832
*/



void for_indiceszyx2 ( size_t Z, size_t Y, size_t X,
                  std::function<void(size_t z, size_t y, size_t x)> func )
{
    for (size_t z = 0; z < Z; ++z)
    {
        for (size_t y = 0; y < Y; ++y)
        {
            for (size_t x = 0; x < X; ++x)
            {
                func(z, y, x);
            }
        }
    }
}

void write_subblock2(unsigned char* block, const char* subblock_flat, int gz, int gy, int gx, const unsigned int BLOCK_WIDTH, const unsigned int BLOCK_HEIGHT, const unsigned int SBW)
{
    size_t subblock_index = 0;
    for_indiceszyx2(SBW, SBW, SBW, [&](size_t z, size_t y, size_t x) {
        int z_slice = gz * SBW + z;
        int y_row   = gy * SBW + y;
        int x_col   = gx * SBW + x;

        int z_offset = z_slice * BLOCK_HEIGHT * BLOCK_WIDTH;
        int y_offset = y_row   * BLOCK_WIDTH;
        int x_offset = x_col;

        block[z_offset + y_offset + x_offset] = subblock_flat[subblock_index];
        subblock_index += 1;
    });
}



int main(int argc, char** argv)
{
    if (argc != 9) {
        cout << USAGE << endl;
        cout << HELP << endl;
        exit(1);
    }
    
    // create DVID node accessor 
    libdvid::DVIDNodeService dvid_node(argv[1], argv[2]);
    libdvid::DVIDNodeService dvid_node2(argv[4], argv[5]);

    int scale;
    std::vector<libdvid::DVIDCompressedBlock> maskblocks;
    // extract sparse labelarray mask
    cout << "fetch labels" << endl; 
    {
        ScopeTime overall_time;
        scale = dvid_node.get_sparselabelmask(atoi(argv[7]), argv[3], maskblocks, atoi(argv[8]), 0);
    }    
    cout << "num blocks: " << maskblocks.size() << endl;

    // debug count
    unsigned long long count = 0;
    for (int i = 0; i < maskblocks.size(); ++i) {
        auto data = maskblocks[i].get_uncompressed_data();
        auto rawdata = data->get_raw();
        for (int j = 0; j < (64*64*64); ++j) {
            if (*rawdata == 255) {
                count++;
            }
            rawdata++;
        }
    }
    cout << "voxel count: " << count << endl;

    std::vector<libdvid::DVIDCompressedBlock> grayblocks;
    cout << "fetch grayscale" << endl; 
    // extract grayscale
    {
        ScopeTime overall_time;
        dvid_node2.get_sparsegraymask(argv[6], maskblocks, grayblocks, scale, false);
    } 

    // debug count
    unsigned long long gcount = 0;
    for (int i = 0; i < grayblocks.size(); ++i) {
        auto data = grayblocks[i].get_uncompressed_data();
        auto rawdata = data->get_raw();
        for (int j = 0; j < (64*64*64); ++j) {
            gcount += *rawdata;
            rawdata++;
        }
    }
    cout << "gray count: " << gcount << endl;

    // write binary string of data
    
    // find the max block coordinates ?!
    int x1 = INT_MAX;
    int y1 = INT_MAX;
    int z1 = INT_MAX;
    int x2 = INT_MIN;
    int y2 = INT_MIN;
    int z2 = INT_MIN;

    for (int i = 0; i < grayblocks.size(); ++i) {
        auto offset = grayblocks[i].get_offset();
        int blocksize = grayblocks[i].get_blocksize();

        if (offset[0] < x1) {
            x1 = offset[0];
        }
        if (offset[1] < y1) {
            y1 = offset[1];
        }
        if (offset[2] < z1) {
            z1 = offset[2];
        }

        if ((offset[0]+blocksize) > x2) {
            x2 = offset[0]+blocksize;
        }
        if ((offset[1]+blocksize) > y2) {
            y2 = offset[1]+blocksize;
        }
        if ((offset[2]+blocksize) > z2) {
            z2 = offset[2]+blocksize;
        }
    }

    // create an all 0 buffer
    unsigned char *buffer = new unsigned char[(x2-x1)*(y2-y1)*(z2-z1)]();

    // copy blocks into big buffer
    for (int i = 0; i < grayblocks.size(); ++i) {
        auto offset = grayblocks[i].get_offset();
        auto blocksize = grayblocks[i].get_blocksize();
        auto raw_data = grayblocks[i].get_uncompressed_data()->get_data().c_str();

        int gx = (offset[0]-x1) / blocksize;
        int gy = (offset[1]-y1) / blocksize;
        int gz = (offset[2]-z1) / blocksize;

        write_subblock2(buffer, raw_data, gz, gy, gx, x2-x1, y2-y1, blocksize);
    }


    // write to file
    std::ofstream fout("arraystring.txt");
    for (int i = 0; i < ((x2-x1)*(y2-y1)*(z2-z1)); ++i) {
       fout << int(buffer[i]) << " ";
    } 
    fout.close();

    // write out z,y,x dims
    cout << (z2-z1) << " " << (y2-y1) << " " << (x2-x1) << endl;

    return 0;
}


