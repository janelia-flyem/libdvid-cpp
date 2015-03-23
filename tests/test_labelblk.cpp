#include <iostream>
#include <libdvid/DVIDNode.h>

using std::cerr; using std::cout; using std::endl;
using namespace libdvid;

using std::string;

// assume all blocks are BLK_SIZE in each dimension
int BLK_SIZE = 32;

unsigned long long int limg1_mask[] = {
    5, 4, 3, 2,
    4, 4, 1, 3,
    7, 7, 7, 7};

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
        cout << "Usage: <program> <server_name> <uuid>" << endl;
        return -1;
    }
    try {
        DVIDServer server(argv[1]);
        string uuid = server.create_new_repo("newrepo", "This is my new repo");
        DVIDNode dvid_node(server, uuid);
    
        string label_datatype_name = "labels1";
        // ** Test creation of DVID datatypes **
        
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
        unsigned long long * img_labels = new unsigned long long [BLK_SIZE*BLK_SIZE*BLK_SIZE];
        for (int i = 0; i < (LWIDTH*LHEIGHT); ++i) {
            img_labels[i] = limg1_mask[i];
            img_labels[i+BLK_SIZE*BLK_SIZE] = limg2_mask[i];
        }

        // create binary data string wrapper (64 bits per pixel)
        BinaryDataPtr labelbin = BinaryData::create_binary_data((char*)(img_labels), BLK_SIZE*BLK_SIZE*BLK_SIZE*sizeof(unsigned long long));
        
        tuple start; start.push_back(0); start.push_back(0); start.push_back(0);
        tuple channels; channels.push_back(0); channels.push_back(1); channels.push_back(2);

        // post labels volume
        // one could also write 2D image slices but the starting location must
        // be at least an ND point where N is greater than 2
        tuple lsizes; lsizes.push_back(BLK_SIZE); lsizes.push_back(BLK_SIZE); lsizes.push_back(BLK_SIZE);
        dvid_node.write_volume_roi(label_datatype_name, start, lsizes, channels, labelbin);

        // retrieve the image volume and make sure it makes the posted volume
        DVIDLabelPtr labelcomp;
        dvid_node.get_volume_roi(label_datatype_name, start, lsizes, channels, labelcomp);
        unsigned long long* labeldatacomp = labelcomp->get_raw();
        for (int i = 0; i < BLK_SIZE*BLK_SIZE*BLK_SIZE; ++i) {
            if (labeldatacomp[i] != img_labels[i]) {
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
