#include <libdvid/Algorithms.h>
#include <libdvid/DVIDThreadedFetch.h>

#include <vector>

using std::string;
using std::vector;

namespace libdvid {

void copy_paste_body(DVIDNodeService& node1, DVIDNodeService& node2,
       uint64 src, uint64 dest, string src_labels,
       string src_labelvol, string dest_labels, int num_threads)
{
    vector<vector<int> > spans;

    // get source body blocks to copy
    vector<BinaryDataPtr> blocks_src = get_body_labelblocks(node1, src_labelvol, src, src_labels, spans, num_threads);

    // can reuse spans from previous call (src data ignored otherwise)
    vector<BinaryDataPtr> blocks_dest = get_body_labelblocks(node2, src_labelvol, src, dest_labels, spans, num_threads);

    size_t blocksize1 = node1.get_blocksize(src_labels);
    size_t blocksize2 = node2.get_blocksize(dest_labels);

    if (blocksize1 != blocksize2) {
        throw ErrMsg("Source and destination block size must be the same");
    }

    // write source body id into dest block with new id
    for (unsigned int i = 0; i < blocks_src.size(); ++i) {
        const uint64* src_ptr = (uint64*) blocks_src[i]->get_raw();
        uint64* dest_ptr = (uint64*) blocks_dest[i]->get_raw();
        for (int j = 0; j < (blocksize1*blocksize1*blocksize1); ++j) {
            if (src_ptr[j] == src) {
                dest_ptr[j] = dest; 
            }
        }
    }

    // write back to DVID
    put_labelblocks(node2, dest_labels, blocks_dest, spans, num_threads);
}

}
