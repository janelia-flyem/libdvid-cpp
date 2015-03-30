/*!
 * This file verifies the correctness of compression and
 * decompression implementations in libdvid.
 *
 * NOTE: lz4 versioning between DVID and this library is untested
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#include <libdvid/BinaryData.h>
#include <libdvid/DVIDVoxels.h>
#include <iostream>
#include <fstream>

using std::cerr; using std::cout; using std::endl;
using std::ifstream;
using namespace libdvid;

/*!
 * Check whether two grayscale binaries are equal.
 * \param gray1 grayscale image 1
 * \param gray2 grayscale image 2
*/
bool is_equal(Grayscale2D gray1, Grayscale2D gray2)
{
    Dims_t dims1, dims2;
    dims1 = gray1.get_dims();
    dims2 = gray2.get_dims();

    if ((dims1[0] != dims2[0]) || (dims1[1] != dims2[1])) {
        return false;
    }

    const unsigned char* buffer = gray1.get_raw();
    const unsigned char* buffer2 = gray2.get_raw();

    for (int i = 0; i < (dims1[0]*dims2[1]); ++i) {
        if (*buffer != *buffer2) {
            return false;
        }
        buffer++;
        buffer2++;
    }

    return true;
} 

/*!
 * Loads the same image in different formats, decompresses them, and
 * verifies their equivalence.  The input files are assumed to represent
 * the same underlying data and be 8-bit grayscale.  LZ4 compression
 * and decompression is tested but the lz4 command line tool currently
 * produces a different compression than what this library produces. 
*/
int main(int argc, char** argv)
{
    if (argc != 4) {
        cout << "Usage: <jpg> <png> <binary>" << endl;
        return -1;
    }
    try {
        // read jpeg
        ifstream fin(argv[1]);
        BinaryDataPtr jpgbinary = BinaryData::create_binary_data(fin);
        fin.close();

        unsigned int width, height;
        BinaryDataPtr binary = BinaryData::decompress_jpeg(jpgbinary, width, height); 
        Dims_t dims;
        dims.push_back(width); dims.push_back(height);
        Grayscale2D grayjpeg(binary, dims);

        // read png
        unsigned int width2, height2;
        ifstream fin2(argv[2]);
        BinaryDataPtr pngbinary = BinaryData::create_binary_data(fin2);
        fin2.close();
        binary = BinaryData::decompress_png8(pngbinary, width2, height2); 
        
        if ((width != width2) || (height != height2)) {
            throw ErrMsg("JPEG and PNG do not have the same picture dimensions");
        }
        Grayscale2D graypng(binary, dims);

        if (!is_equal(grayjpeg, graypng)) {
            throw ErrMsg("JPEG and PNG file are not equivalent");
        }
        
         // read binary (assume dims)
        ifstream fin3(argv[3]);
        binary = BinaryData::create_binary_data(fin3);
        fin3.close();
        Grayscale2D graybin(binary, dims);
        
        if (!is_equal(graybin, graypng)) {
            throw ErrMsg("Binary and PNG file are not equivalent");
        }
            
        // check compress and decompress of LZ4 work
        BinaryDataPtr lz4binary = BinaryData::compress_lz4(binary); 
        int uncompressed_size = width2*height2;
        binary = BinaryData::decompress_lz4(lz4binary, uncompressed_size); 
        Grayscale2D graylz4(binary, dims);
        
        if (!is_equal(graylz4, graypng)) {
            throw ErrMsg("Binary and lz4 file are not equivalent");
        }
    } catch (std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }

    return 0;
}
