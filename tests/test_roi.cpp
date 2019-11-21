/*!
 * This file gives a simple example of using the DVID
 * ROI interface.  An ROI is created and retrieved.  Substack
 * partitioning and point querying are examined. 
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#include <libdvid/DVIDServerService.h>
#include <libdvid/DVIDNodeService.h>

#include <vector>
#include <iostream>
using std::cerr; using std::cout; using std::endl;
using namespace libdvid;
using std::string;
using std::vector;

/*!
 * Test get/put of values using keyvalue type.
*/
int main(int argc, char** argv)
{
    if (argc != 2) {
        cout << "Usage: <program> <server_name>" << endl;
        return -1;
    }
    try {
        DVIDServerService server(argv[1]);
        std::string uuid = server.create_new_repo("newrepo", "This is my new repo");
        DVIDNodeService dvid_node(argv[1], uuid);

        // name of the roi to use       
        string roi_datatype_name = "temproi";

        // check existence (should be new)
        if(!dvid_node.create_roi(roi_datatype_name)) {
            cerr << roi_datatype_name << " already exists" << endl;
            return -1;
        }
        
        // check existence (should be old)
        if(dvid_node.create_roi(roi_datatype_name)) {
            cerr << roi_datatype_name << " should exist" << endl;
            return -1;
        }
       
        // create and post ROI; then overwrite and try again 
        // indices are in block coordinates
       
        // add block2
        vector<BlockXYZ> blocks2;
        for (int i = -4; i <= 324; ++i) {
            blocks2.push_back(BlockXYZ(i,53,21));
        }
        dvid_node.post_roi(roi_datatype_name, blocks2);

        // retrieve blocks
        vector<BlockXYZ> blocks_comp;
        dvid_node.get_roi(roi_datatype_name, blocks_comp);

        // make sure ROI matches        
        if (blocks_comp.size() != (blocks2.size())) {
            cerr << "ROI blocks retrieved different than posted" << endl;
            return -1;
        }
        for (unsigned int i = 0; i < blocks2.size(); ++i) {
            if ((blocks2[i].x != blocks_comp[i].x) ||
                (blocks2[i].y != blocks_comp[i].y) ||
                (blocks2[i].z != blocks_comp[i].z)) {
                cerr << "ROI blocks retrieved different than posted" << endl;
                return -1;
            }
        }
       
        // add a different set of blocks and overwrite 
        vector<BlockXYZ> blocks;
        for (int i = -4; i <= 324; ++i) {
            blocks.push_back(BlockXYZ(i,0,0));
        }
        dvid_node.post_roi(roi_datatype_name, blocks);
        dvid_node.get_roi(roi_datatype_name, blocks_comp);
        
        // make sure ROI matches        
        if (blocks_comp.size() != (blocks.size())) {
            cerr << "ROI blocks retrieved different than posted" << endl;
            return -1;
        } 
        for (unsigned int i = 0; i < blocks.size(); ++i) {
            if ((blocks[i].x != blocks_comp[i].x) ||
                (blocks[i].y != blocks_comp[i].y) ||
                (blocks[i].z != blocks_comp[i].z)) {
                cerr << "ROI blocks retrieved different than posted" << endl;
                return -1;
            }
        }
        
        // grab substacks (42 256 cubes; 21 512 cubes)
        vector<SubstackXYZ> substacks;
        double packing = dvid_node.get_roi_partition(roi_datatype_name, substacks, 8);
        vector<SubstackXYZ> substacks2;
        double packing2 = dvid_node.get_roi_partition(roi_datatype_name, substacks2, 16);

        if (substacks.size() != 42) {
            cerr << "Partition gives incorrect number of substacks and packing of" 
                << packing << endl;
            return -1;
        }

        if (substacks2.size() != 21) {
            cerr << "Partition gives incorrect number of substacks and packing of" 
                << packing2 << endl;
            return -1;
        }

        // check point query interface (voxel coordinates)
        vector<PointXYZ> points;
        points.push_back(PointXYZ(-40,0,0)); // in roi
        points.push_back(PointXYZ(-40,31,31)); // in roi
        points.push_back(PointXYZ(-40,32,32)); // not in roi
        vector<bool> points_inroi;
        dvid_node.roi_ptquery(roi_datatype_name, points, points_inroi);
        
        if ((points_inroi[0] != true) || (points_inroi[1] != true)
                || (points_inroi[2] != false)) {
            cerr << "Point query in ROI gives incorrect result" << endl;
            return -1;
        }
    } catch (std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }
    return 0;
}
