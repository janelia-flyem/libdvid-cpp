#include <libdvid/DVIDThreadedFetch.h>
#include <libdvid/DVIDException.h>
#include <boost/thread/thread.hpp>

#include <vector>

using std::string;
using std::vector;
using std::uint8_t;
using std::uint16_t;
using std::uint32_t;
using std::int32_t;
using std::uint64_t;
using std::tuple; using std::make_tuple;
using std::unordered_map;

//! Max blocks to request at one tiem
static const int MAX_BLOCKS = 4096;

//! Mutex for controlling thread gathering is sufficient
//! given performance requirements
// HACK: global variables for each type
static boost::mutex m_mutex;
static boost::condition_variable m_condition;

static boost::mutex m_mutex_gray;
static boost::condition_variable m_condition_gray;

namespace libdvid {

struct FetchGrayBlocks {
    FetchGrayBlocks(DVIDNodeService& service_, string grayscale_name_,
            bool use_blocks_, int request_efficiency_, int start_, int count_,
            vector<vector<int> >* spans_, vector<BinaryDataPtr>* blocks_, int& threads_remaining_) :
            service(service_), grayscale_name(grayscale_name_),
            use_blocks(use_blocks_), request_efficiency(request_efficiency_),
            start(start_), count(count_), spans(spans_), blocks(blocks_), threads_remaining(threads_remaining_) {}

    void operator()()
    {
        size_t blocksize = service.get_blocksize(grayscale_name);
        uint8* blockdata = 0;
        if ((request_efficiency == 1) && !use_blocks) {
            blockdata = new uint8[blocksize*blocksize*blocksize];
        }
        // iterate only for the threads parts 
        for (int index = start; index < (start+count); ++index) {
            // load span info
            vector<int> span = (*spans)[index];
            int xmin = span[0];
            int y = span[1];
            int z = span[2];
            int curr_runlength = span[3];
            int block_index = span[4];

            if (use_blocks) {
                // use block interface (currently most re-copy)
                vector<int> block_coords;
                block_coords.push_back(xmin);
                block_coords.push_back(y);
                block_coords.push_back(z);
                // !! only works with 32x32x32 blocks
                GrayscaleBlocks blocks2 = service.get_grayblocks(grayscale_name, block_coords, curr_runlength);
                for (int j = 0; j < curr_runlength; ++j) {
                    BinaryDataPtr ptr = BinaryData::create_binary_data((const char*)blocks2[j], blocksize*blocksize*blocksize);
                    (*blocks)[block_index] = ptr;
                    ++block_index;
                }
            } else {
                Dims_t dims;
                dims.push_back(blocksize*curr_runlength);
                dims.push_back(blocksize);
                dims.push_back(blocksize);
                vector<int> offset;
                offset.push_back(xmin*blocksize);
                offset.push_back(y*blocksize);
                offset.push_back(z*blocksize);

                Grayscale3D grayvol = service.get_gray3D(grayscale_name,
                        dims, offset, false); 

                if (curr_runlength == 1) {
                    // do a simple copy for just one block
                    (*blocks)[block_index] = grayvol.get_binary();
                    ++block_index;
                } else {
                    const uint8* raw_data = grayvol.get_raw();

                    // otherwise create a buffer and do something more complicated 
                    for (int j = 0; j < curr_runlength; ++j) {
                        int offsetx = j * blocksize;
                        int offsety = curr_runlength*blocksize;
                        int offsetz = curr_runlength*blocksize*blocksize;
                        uint8* mod_data_iter = blockdata; 

                        for (int ziter = 0; ziter < blocksize; ++ziter) {
                            const uint8* data_iter = raw_data + ziter * offsetz;    
                            data_iter += (offsetx);
                            for (int yiter = 0; yiter < blocksize; ++yiter) {
                                for (int xiter = 0; xiter < blocksize; ++xiter) {
                                    *mod_data_iter = *data_iter;
                                    ++mod_data_iter;
                                    ++data_iter;
                                }
                                data_iter += ((offsety) - blocksize);
                            }
                        }
                        BinaryDataPtr ptr = BinaryData::create_binary_data((const char*) blockdata, blocksize*blocksize*blocksize);
                        (*blocks)[block_index] = ptr;
                        ++block_index;
                    }
                }
            }
        }

        if (blockdata) {
            delete []blockdata;
        }
        
        boost::mutex::scoped_lock lock(m_mutex_gray);
        threads_remaining--;
        m_condition_gray.notify_one();
    }


    DVIDNodeService service;
    string grayscale_name;
    bool use_blocks;
    int request_efficiency;
    int start; int count;
    vector<vector<int> >* spans;
    vector<BinaryDataPtr>* blocks;
    int& threads_remaining;
};

struct FetchLabelBlocks {
    FetchLabelBlocks(DVIDNodeService& service_, string labelsname_, int start_, int count_,
            vector<vector<int> >* spans_, vector<BinaryDataPtr>* blocks_) :
            service(service_), labelsname(labelsname_), start(start_), count(count_),
            spans(spans_), blocks(blocks_) {}

    void operator()()
    {
        size_t blocksize = service.get_blocksize(labelsname);
        uint64* blockdata = 0;
        blockdata = new uint64[blocksize*blocksize*blocksize];
        // iterate only for the threads parts 
        for (int index = start; index < (start+count); ++index) {
            // load span info
            vector<int> span = (*spans)[index];
            int xmin = span[0];
            int y = span[1];
            int z = span[2];
            int curr_runlength = span[3];
            int block_index = span[4];


            Dims_t dims;
            dims.push_back(blocksize*curr_runlength);
            dims.push_back(blocksize);
            dims.push_back(blocksize);
            vector<int> offset;
            offset.push_back(xmin*blocksize);
            offset.push_back(y*blocksize);
            offset.push_back(z*blocksize);

            Labels3D labelvol = service.get_labels3D(labelsname,
                    dims, offset, false); 

            if (curr_runlength == 1) {
                // do a simple copy for just one block
                (*blocks)[block_index] = labelvol.get_binary();
                ++block_index;
            } else {
                const uint64* raw_data = labelvol.get_raw();

                // otherwise create a buffer and do something more complicated 
                for (int j = 0; j < curr_runlength; ++j) {
                    int offsetx = j * blocksize;
                    int offsety = curr_runlength*blocksize;
                    int offsetz = curr_runlength*blocksize*blocksize;
                    uint64* mod_data_iter = blockdata; 

                    for (int ziter = 0; ziter < blocksize; ++ziter) {
                        const uint64* data_iter = raw_data + ziter * offsetz;    
                        data_iter += (offsetx);
                        for (int yiter = 0; yiter < blocksize; ++yiter) {
                            for (int xiter = 0; xiter < blocksize; ++xiter) {
                                *mod_data_iter = *data_iter;
                                ++mod_data_iter;
                                ++data_iter;
                            }
                            data_iter += ((offsety) - blocksize);
                        }
                    }
                    BinaryDataPtr ptr = BinaryData::create_binary_data((const char*) blockdata,
                            sizeof(uint64)*blocksize*blocksize*blocksize);
                    (*blocks)[block_index] = ptr;
                    ++block_index;
                }
            }
        }

        if (blockdata) {
            delete []blockdata;
        }
    }


    DVIDNodeService service;
    string labelsname;
    int start; int count;
    vector<vector<int> >* spans;
    vector<BinaryDataPtr>* blocks;
};

struct WriteLabelBlocks {
    WriteLabelBlocks(DVIDNodeService& service_, string labelsname_, int start_, int count_,
            vector<vector<int> >* spans_, const vector<BinaryDataPtr>* blocks_) :
            service(service_), labelsname(labelsname_), start(start_), count(count_),
            spans(spans_), blocks(blocks_) {}

    void operator()()
    {
        size_t blocksize = service.get_blocksize(labelsname);
        // iterate only for the threads parts 
        for (int index = start; index < (start+count); ++index) {
            // load span info
            vector<int> span = (*spans)[index];
            int xmin = span[0];
            int y = span[1];
            int z = span[2];
            int curr_runlength = span[3];
            int block_index = span[4];

            Dims_t dims;
            dims.push_back(blocksize*curr_runlength);
            dims.push_back(blocksize);
            dims.push_back(blocksize);
            vector<int> offset;
            offset.push_back(xmin*blocksize);
            offset.push_back(y*blocksize);
            offset.push_back(z*blocksize);

            uint64* blockdata = new uint64[blocksize*blocksize*blocksize*curr_runlength];

            // otherwise create a buffer and do something more complicated 
            for (int j = 0; j < curr_runlength; ++j) {
                int offsetx = j * blocksize;
                int offsety = curr_runlength*blocksize;
                int offsetz = curr_runlength*blocksize*blocksize;
                const uint64* copy_data_iter = (uint64*) (*blocks)[block_index]->get_raw();

                for (int ziter = 0; ziter < blocksize; ++ziter) {
                    uint64* data_iter = blockdata + ziter * offsetz;    
                    data_iter += (offsetx);
                    for (int yiter = 0; yiter < blocksize; ++yiter) {
                        for (int xiter = 0; xiter < blocksize; ++xiter) {
                            *data_iter = *copy_data_iter;
                            ++copy_data_iter;
                            ++data_iter;
                        }
                        data_iter += ((offsety) - blocksize);
                    }
                }
                ++block_index;
            }

            // actually put label volume
            Labels3D volume(blockdata, blocksize*blocksize*blocksize*curr_runlength, dims);
            service.put_labels3D(labelsname, volume, offset, false); 
            delete []blockdata;
        }
    }


    DVIDNodeService service;
    string labelsname;
    int start; int count;
    vector<vector<int> >* spans;
    const vector<BinaryDataPtr>* blocks;
};




struct FetchTiles {
    FetchTiles(DVIDNodeService& service_, Slice2D orientation_,
            string instance_,
            unsigned int scaling_, int start_, int count_,
            const vector<vector<int> >& tile_locs_array_,
            vector<BinaryDataPtr>& results_, int& threads_remaining_) :
            service(service_), orientation(orientation_), instance(instance_),
            scaling(scaling_),
            start(start_), count(count_), tile_locs_array(tile_locs_array_),
            results(results_), threads_remaining(threads_remaining_) {}

    void operator()()
    {
        for (int i = start; i < (start+count); ++i) {
            results[i] = 
                service.get_tile_slice_binary(instance, orientation, scaling, tile_locs_array[i]);
        }
        // local and signal
        boost::mutex::scoped_lock lock(m_mutex);
        threads_remaining--;
        m_condition.notify_one();
    }

    DVIDNodeService& service;
    Slice2D orientation;
    string instance;
    unsigned int scaling;
    int start; int count;
    const vector<vector<int> >& tile_locs_array;
    vector<BinaryDataPtr>& results;
    int& threads_remaining;
};

/*!
 * Given a body ID, determines all the X contiguous spans
 * and packs into an array.
*/
int get_block_spans(DVIDNodeService& service, string labelvol_name,
        uint64 bodyid, vector<vector<int> >& spans, int request_efficiency = 1)
{
    vector<BlockXYZ> blockcoords;
    if (!service.get_coarse_body(labelvol_name, bodyid, blockcoords)) {
        throw ErrMsg("Body not found, no grayscale blocks could be retrieved");
    }

    int num_requests = 0;
   
    // !! probably unnecessary copying going on
    // iterate through block coords and call ND or blocks one by one or contig
    int xmin; 
    int curr_runlength = 0;
    int start_index = 0;
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

            // load into queue
            vector<int> span;
            span.push_back(xmin);
            span.push_back(y);
            span.push_back(z);
            span.push_back(curr_runlength);
            span.push_back(start_index);
            start_index += curr_runlength;
            spans.push_back(span);
            curr_runlength = 0;
        }
    }

    return num_requests;
}


vector<BinaryDataPtr> get_body_blocks(DVIDNodeService& service, string labelvol_name,
        string grayscale_name, uint64 bodyid, int num_threads,
        bool use_blocks, int request_efficiency)
{
    vector<vector<int> > spans;
    vector<BinaryDataPtr> blocks;

    int num_requests = get_block_spans(service, labelvol_name, bodyid, spans, request_efficiency);

    if (num_requests < num_threads) {
        num_threads = num_requests;
    }
    
    int num_blocks = 0;
    for (int i = 0; i < spans.size(); ++i) {
        num_blocks += spans[i][3];
    }
    blocks.resize(num_blocks);

    int incr = num_requests / num_threads;
    int start = 0;
    int count_check = 0;
    
    // setup thread pool
    DVIDThreadPool* pool = DVIDThreadPool::get_pool();
    int threads_remaining = num_threads;

    for (int i = 0; i < num_threads; ++i) {
        int count = incr;
        if (i == (num_threads-1)) {
            count = num_requests - start;
        }
        count_check += count;
        
        pool->add_task(FetchGrayBlocks(service, grayscale_name,
                    use_blocks, request_efficiency, start, count, &spans, &blocks, threads_remaining));
        start += incr;
    }
  
    // wait for threads to finish
    boost::mutex::scoped_lock lock(m_mutex_gray);
    while (threads_remaining) {
        m_condition_gray.wait(lock); 
    }

    
    assert(count_check == num_requests);
    std::cout << "Performed " << num_requests << " requests" << std::endl;
    return blocks;
}

vector<BinaryDataPtr> get_body_labelblocks(DVIDNodeService& service, string labelvol_name,
        uint64 bodyid, string labelsname, vector<vector<int> >& spans, int num_threads)
{
    vector<BinaryDataPtr> blocks;
    int num_requests = spans.size();
    
    if (spans.empty()) {
        num_requests = get_block_spans(service, labelvol_name, bodyid, spans);
    }

    // launch threads
    boost::thread_group threads;

    if (num_requests < num_threads) {
        num_threads = num_requests;
    }
    int num_blocks = 0;
    for (int i = 0; i < spans.size(); ++i) {
        num_blocks += spans[i][3];
    }
    blocks.resize(num_blocks);

    int incr = num_requests / num_threads;
    int start = 0;
    int count_check = 0;

    for (int i = 0; i < num_threads; ++i) {
        int count = incr;
        if (i == (num_threads-1)) {
            count = num_requests - start;
        }
        count_check += count;
        threads.create_thread(FetchLabelBlocks(service, labelsname, start, count, &spans, &blocks));
        start += incr;
    }
    threads.join_all();
    assert(count_check == num_requests);
    std::cout << "Performed " << num_requests << " requests" << std::endl;
    return blocks;
}


void put_labelblocks(DVIDNodeService& service, std::string labelsname,
        const vector<BinaryDataPtr>& blocks,
        vector<vector<int> >& spans, int num_threads)
{
    // launch threads
    boost::thread_group threads;
    int num_requests = spans.size();

    if (num_requests < num_threads) {
        num_threads = num_requests;
    }

    int incr = num_requests / num_threads;
    int start = 0;
    int count_check = 0;

    for (int i = 0; i < num_threads; ++i) {
        int count = incr;
        if (i == (num_threads-1)) {
            count = num_requests - start;
        }
        count_check += count;
        threads.create_thread(WriteLabelBlocks(service, labelsname, start, count, &spans, &blocks));
        start += incr;
    }
    threads.join_all();
    assert(count_check == num_requests);
    std::cout << "Performed " << num_requests << " requests" << std::endl;
}


vector<BinaryDataPtr> get_tile_array_binary(DVIDNodeService& service,
        string datatype_instance, Slice2D orientation, unsigned int scaling,
        const vector<vector<int> >& tile_locs_array, int num_threads)
{
    if (!num_threads) {
        num_threads = tile_locs_array.size();
    }
    vector<BinaryDataPtr> results(tile_locs_array.size());
    
    DVIDThreadPool* pool = DVIDThreadPool::get_pool();
   
    DVIDNodePool* nodepool = DVIDNodePool::get_pool();

    int available_nodes = 0;
    if (nodepool->dvidnodes.find(service.get_uuid()) != nodepool->dvidnodes.end()) {
        available_nodes = nodepool->dvidnodes[service.get_uuid()].size();
    }

    while (available_nodes < num_threads) {
        nodepool->dvidnodes[service.get_uuid()].push_back(boost::shared_ptr<DVIDNodeService>(new DVIDNodeService(service)));
        ++available_nodes;
    }


    // not an optimal partitioning
    int incr = tile_locs_array.size() / num_threads;
    int start = 0;

    int threads_remaining = num_threads;

    for (int i = 0; i < num_threads; ++i) {
        int count = incr;
        if (i == (num_threads-1)) {
            count = tile_locs_array.size() - start;
        }
        pool->add_task(FetchTiles(*(nodepool->dvidnodes[service.get_uuid()][i].get()),
                    orientation,
                    datatype_instance, scaling, start, count,
                    tile_locs_array, results, threads_remaining));
        start += incr;
    }

    
    // wait for threads to finish
    boost::mutex::scoped_lock lock(m_mutex);
    while (threads_remaining) {
        m_condition.wait(lock); 
    }


    return results;
}

void for_indiceszyx ( size_t Z, size_t Y, size_t X,
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
int get_sparselabelmask(DVIDNodeService& service, uint64_t bodyid, std::string labelname, std::vector<DVIDCompressedBlock>& maskblocks, unsigned long long maxsize, int scale)
{
    if (scale == -1 && maxsize > 0) {
        // TODO: get label size (need new DVID API)
        throw ErrMsg("DVID label size API not available yet");
        std::ostringstream ss_uri;
        ss_uri << "/" << labelname << "/labelsize/" << bodyid << "/";
        
        // perform query
        auto binary = service.custom_request(ss_uri.str(), BinaryDataPtr(), GET);
        
        // fetch result 
        Json::Value data;
        Json::Reader json_reader;
        if (!json_reader.parse(binary->get_data(), data)) {
            throw ErrMsg("Could not decode JSON");
        }
        scale = data["LabelSize"].asInt();
    }
    
    // set to default max rez if scale still not set
    if (scale == -1) {
        scale = 0;
    }

    // perform custom fetch of labels
    std::ostringstream ss_uri;
    ss_uri << "/" << labelname << "/sparsevol/" << bodyid << "?format=blocks&scale=" << scale;
    auto binary_result = service.custom_request(ss_uri.str(), BinaryDataPtr(), GET);

    const unsigned int blocksize = service.get_blocksize(labelname);

    // extract sparse volume from binary encoding
    int buffer_size = binary_result->length();
    const unsigned char * head = binary_result->get_raw();

    // fetch gx, gy, gz and body id (num sub-blocks in each dim)
    vector<int> gxgygz;
    
  
    // dvid currently missing header  
#if 0
    gxgygz.push_back(extractval<int32_t>(head, buffer_size));
    gxgygz.push_back(extractval<int32_t>(head, buffer_size));
    gxgygz.push_back(extractval<int32_t>(head, buffer_size));

    // sub-block must be isotropic and divide into block size 
    if (gxgygz[0] != gxgygz[1] || gxgygz[0] != gxgygz[2]) {
        throw ErrMsg("Sub-block size must be nxnxn");
    }
    
    if ((blocksize % gxgygz[0]) != 0) {
        throw ErrMsg("Sub-block size must be divide into block size");
    }
    const unsigned int SBW = blocksize / gxgygz[0];
    if ((SBW % 8) != 0) {
        throw ErrMsg("Sub-block size must be a multiple of 8");
    }

    uint64_t bodycheck = extractval<uint64_t>(head, buffer_size);
    assert(bodycheck == bodyid);
#else
    gxgygz.push_back(blocksize/8);
    gxgygz.push_back(blocksize/8);
    gxgygz.push_back(blocksize/8);
    const unsigned int SBW = 8;
#endif

    int oneblocks = 0;
    int zeroblocks = 0;
    int mixblocks = 0;

    // iterate x,y,z block; header; sub-blocks
    while (buffer_size) {
        // retrieve offset
        vector<int> offset;

        // currently dvid is providing voxel coordinates
#if 0
        offset.push_back(extractval<int32_t>(head, buffer_size) * blocksize);
        offset.push_back(extractval<int32_t>(head, buffer_size) * blocksize);
        offset.push_back(extractval<int32_t>(head, buffer_size) * blocksize);
#else
        offset.push_back(extractval<int32_t>(head, buffer_size));
        offset.push_back(extractval<int32_t>(head, buffer_size));
        offset.push_back(extractval<int32_t>(head, buffer_size));
#endif

        // get header 
        unsigned char blockstatus = extractval<uint8_t>(head, buffer_size);

        // create all one block for init case
        unsigned char * blockdata = new unsigned char[blocksize*blocksize*blocksize];
        memset(blockdata, 255, blocksize*blocksize*blocksize); 

        // should not be all blank by construction
        assert(blockstatus != 0);

        if (blockstatus == 2) {
            // parse gx,gy,gz subblocks
            for (int z = 0; z < gxgygz[2]; ++z) {
                for (int y = 0; y < gxgygz[1]; ++y) {
                    for (int x = 0; x < gxgygz[0]; ++x) {
                        // create all 0 subblock for copy over
                        unsigned char * subblockdata = new unsigned char[SBW*SBW*SBW]();
                        
                        unsigned char subblockstatus = extractval<uint8_t>(head, buffer_size);
                        if (int(subblockstatus) == 0) {
                            // zero out subblock
                            write_subblock(blockdata, subblockdata, z, y, x, blocksize, SBW);
                            ++zeroblocks;
                        } else if (int(subblockstatus) == 2) {
                            // write binary block out -- traverse 1 bit encoded subblock
                            ++mixblocks;
                            int index = 0;
                           
                            for (int byteloc = 0; byteloc < (SBW*SBW); ++byteloc) {
                                // each sub-block row is packed in a byte
                                uint8_t val8 = extractval<uint8_t>(head, buffer_size);
                                for (int i = 0; i < 8; ++i) {
                                    subblockdata[index] = (val8 & 1) * 255;
                                    ++index;
                                    val8 = val8 >> 1;
                                }
                            } 
                            write_subblock(blockdata, subblockdata, z, y, x, blocksize, SBW); 
                        } else {
                            ++oneblocks;
                        }
                    }
                }
            }
        }

        // push block back
        BinaryDataPtr blockdata2 = BinaryData::create_binary_data((const char*) blockdata, blocksize*blocksize*blocksize);
        auto m_block = DVIDCompressedBlock(blockdata2, offset, blocksize, 1, DVIDCompressedBlock::uncompressed);
        maskblocks.push_back(m_block);
    }

    //std::cout << "ones: " << oneblocks << " zeros: " << zeroblocks << " mix: " << mixblocks << std::endl;

    // indicate which scale used
    return scale;
}

void erode_sparselabelmask(vector<DVIDCompressedBlock>& maskblocks, unsigned int erosion)
{
    // TODO: erode the body mask
    if (erosion > 0) {
        throw ErrMsg("Erosion not yet implemented");
        // create a hash of blocks and add 0's as necessary for first pass
        // copy block with buffer equal to erosion
        // create thread queue
        // vigra? erode X
        // copy buffers back
    }
}


void decompress_block(vector<DVIDCompressedBlock>* blocks, int id, int num_threads)
{
    BinaryDataPtr uncompressed_data;
    int curr_id = 0;

    for (auto iter = blocks->begin(); iter != blocks->end(); ++iter, ++curr_id) {
        if ((curr_id % num_threads) == id) {
            // create new uncompressed block
            vector<int> toffset = iter->get_offset();
            uncompressed_data = iter->get_uncompressed_data(); 
            size_t bsize = iter->get_blocksize();
            size_t tsize = iter->get_typesize();

            DVIDCompressedBlock temp_block(uncompressed_data, toffset, bsize, tsize, DVIDCompressedBlock::uncompressed);
            (*blocks)[curr_id] = temp_block;
        }
    }
}

struct BlockTupleHash {
    std::size_t operator() (const tuple<int, int,int>& data) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, std::get<0>(data));
        boost::hash_combine(seed, std::get<1>(data));
        boost::hash_combine(seed, std::get<2>(data));
        return seed;
    }
};

void  get_sparsegraymask(DVIDNodeService& service, std::string dataname, const std::vector<DVIDCompressedBlock>& maskblocks, std::vector<DVIDCompressedBlock>& grayblocks, int scale, bool usejpeg)
{
    // TODO: auto determine datatype encoding (would cause extra unnecessary latency now but maybe not important)
    
    if (maskblocks.empty()) {
        return;
    }
    auto blocksize = maskblocks[0].get_blocksize(); 

    // extract intersecting blocks
    vector<int> blockcoords;
    for (int i = 0; i < maskblocks.size(); ++i) {
        auto offset = maskblocks[i].get_offset();
        blockcoords.push_back(offset[0]/blocksize);
        blockcoords.push_back(offset[1]/blocksize);
        blockcoords.push_back(offset[2]/blocksize);
    }

    // get scaled name (grayscale doesn't have an explicit multi-scale type in DVID)
    string dataname_scaled = dataname;
    if (scale > 0) {
        std::ostringstream sstr;
        sstr << dataname << "_" << scale;
        dataname_scaled = sstr.str();
    }


    // fetch specific blocks
    vector<DVIDCompressedBlock> blockstemp;
    service.get_specificblocks3D(dataname_scaled, blockcoords, true, blockstemp, 0, !usejpeg); 

    if (usejpeg) {
        // decompress all jpeg (in parallel)
        boost::thread_group threads; // destructor auto deletes threads
        int num_threads = 8; // TODO allocate dynamically

        vector<boost::thread*> curr_threads;  
        for (int i = 0; i < num_threads; ++i) {
            boost::thread* t = new boost::thread(decompress_block, &blockstemp, i, num_threads);
            threads.add_thread(t);
            curr_threads.push_back(t);
        } 
        threads.join_all();
        grayblocks = blockstemp;
    } else {
        grayblocks = blockstemp;
    }

    // apply mask
    unordered_map<tuple<int, int, int>, DVIDCompressedBlock, BlockTupleHash> maskmap;
    
    // load indices
    for (int i = 0; i < maskblocks.size(); ++i) {
        auto offset = maskblocks[i].get_offset();
        maskmap[make_tuple(offset[0], offset[1], offset[2])] = maskblocks[i];
    }

    for (int i = 0; i < grayblocks.size(); ++i) {
        auto offset = grayblocks[i].get_offset();
        size_t blocksize = grayblocks[i].get_blocksize();
        string& raw_data = grayblocks[i].get_uncompressed_data()->get_data();

        auto mask = maskmap[make_tuple(offset[0], offset[1], offset[2])];
        auto mask_data = mask.get_uncompressed_data()->get_data();
        
        int index = 0;
        for_indiceszyx(blocksize, blocksize, blocksize, [&](size_t z, size_t y, size_t x) {
            raw_data[index] &= mask_data[index];
            ++index;         
        }); 
    }
}










}
