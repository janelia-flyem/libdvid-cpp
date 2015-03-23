#include <iostream>
#include <libdvid/DVIDNode.h>

using std::cerr; using std::cout; using std::endl;
using namespace libdvid;

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
        DVIDServer server(argv[1]);
        string uuid = server.create_new_repo("newrepo", "This is my new repo");
        DVIDNode dvid_node(server, uuid);
    
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
        BinaryDataPtr graybin = BinaryData::create_binary_data((char*)(img_gray),
                BLK_SIZE*BLK_SIZE*BLK_SIZE);
        tuple start; start.push_back(0); start.push_back(0); start.push_back(0);
        tuple sizes; sizes.push_back(BLK_SIZE); sizes.push_back(BLK_SIZE); sizes.push_back(BLK_SIZE);
        tuple channels; channels.push_back(0); channels.push_back(1); channels.push_back(2);

        // post grayscale volume
        // one could also write 2D image slices but the starting location must
        // be at least an ND point where N is greater than 2
        dvid_node.write_volume_roi(gray_datatype_name, start, sizes, channels, graybin);

        // retrieve the image volume and make sure it makes the posted volume
        DVIDGrayPtr graycomp;
        dvid_node.get_volume_roi(gray_datatype_name, start, sizes, channels, graycomp);
        unsigned char* datacomp = graycomp->get_raw();
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
