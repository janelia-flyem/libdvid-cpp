/*!
 * This file verifies the correctness of compression and
 * decompression implementations in libdvid.
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
* Loads equivalent images in different formats from disk.
*/
int main(int argc, char** argv)
{
    if (argc != 3) {
        cout << "Usage: <jpg> <png>" << endl;
        return -1;
    }
    try {
        // read jpeg
        ifstream fin(argv[1]);
        BinaryDataPtr jpgbinary = BinaryData::create_binary_data(fin);
        fin.close();

        unsigned int width, height;
        BinaryDataPtr binary = BinaryData::decompress_jpeg(jpgbinary, width, height); 
        cout << "Width: " << width << " Height: " << height << endl;
   
        // read png
        ifstream fin2(argv[2]);
        BinaryDataPtr pngbinary = BinaryData::create_binary_data(fin2);
        fin2.close();

        BinaryDataPtr binary2 = BinaryData::decompress_png8(pngbinary, width, height); 
        cout << "Width: " << width << " Height: " << height << endl;
    } catch (std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }

    return 0;
}
