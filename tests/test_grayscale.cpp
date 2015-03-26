#include <libdvid/DVIDServerService.h>
#include <libdvid/DVIDNodeService.h>

#include <iostream>
#include <vector>

using std::cerr; using std::cout; using std::endl;
using namespace libdvid;
using std::vector;

using std::string;

// assume all blocks are 32 in each dimension
int BLK_SIZE = 32;

// sample grayscale to write
unsigned char img1_mask[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,0,0,1,0,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,
    0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,
    0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
    0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
    0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
    0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
    0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,
    0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,
    0,0,1,0,0,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,0,0,1,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

// image 2 mask
unsigned char img2_mask[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0};

/*!
 * Exercises the nD GETs and PUTs for the grayscale datatype.
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
    
        string gray_datatype_name = "gray1";
        // ** Test creation of DVID datatypes **
        
        // should be a new instance
        if(!dvid_node.create_grayscale8(gray_datatype_name)) {
            cerr << gray_datatype_name << " already exists" << endl;
            return -1;
        }

        // should not recreate grayscale if it already exists
         if(dvid_node.create_grayscale8(gray_datatype_name)) {
            cerr << gray_datatype_name << " should exist" << endl;
            return -1;
        }


        // ** Write and read grayscale data **
        // create grayscale image volume
        unsigned char * img_gray = new unsigned char [BLK_SIZE*BLK_SIZE*BLK_SIZE];
        for (int i = 0; i < sizeof(img1_mask); ++i) {
            img_gray[i] = img1_mask[i] * 255;
            img_gray[i+BLK_SIZE*BLK_SIZE] = img2_mask[i] * 255;
        }

        // create binary data string wrapper (no meta-data for now -- must explicitly create)
        vector<unsigned int> start;
        start.push_back(0); start.push_back(0); start.push_back(0);
        Dims_t sizes; sizes.push_back(BLK_SIZE); sizes.push_back(BLK_SIZE); sizes.push_back(BLK_SIZE);
        Grayscale3D graybin(img_gray, BLK_SIZE*BLK_SIZE*BLK_SIZE, sizes);

        // post grayscale volume
        // one could also write 2D image slices but the starting location must
        // be at least an ND point where N is greater than 2
        
        dvid_node.put_gray3D(gray_datatype_name, graybin, start);

        // retrieve the image volume and make sure it makes the posted volume
        Grayscale3D graycomp = dvid_node.get_gray3D(gray_datatype_name, sizes, start);
        const unsigned char* datacomp = graycomp.get_raw();
        for (int i = 0; i < (BLK_SIZE*BLK_SIZE*BLK_SIZE); ++i) {
            if (datacomp[i] != img_gray[i]) {
                cerr << "Read/write mismatch" << endl;
                return -1;
            }
        }
    } catch (std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }
    return 0;
}
