#include "DVIDNodeService.h"
#include "DVIDException.h"
#include "DVIDCache.h"
#include "DVIDLabelCodec.h"
#include "BinaryData.h"

#include <memory>
#include <set>
#include <algorithm>
#include <cstdlib>
#include <unordered_set>
#include <unordered_map>
#include <json/json.h>
#include <boost/assign/list_of.hpp>
#include <boost/bind.hpp>
#include <sys/time.h>
#include <boost/thread/thread.hpp>
#include <tuple>
#include <thread>

using std::set;
using std::vector;
using std::tuple;
using std::make_tuple;
using std::string;
using std::ifstream;
using std::stringstream;
using std::ostringstream;
//Json::Reader json_reader;

//! Gives the limit for how many vertice can be operated on in one call
static const unsigned int TransactionLimit = 1000;

static const bool DVIDNODESERVICE_EXTRA_CHECKS = []{
    char * var = getenv("DVIDNODESERVICE_EXTRA_CHECKS");
    return (var != NULL) && (var == std::string("1"));
}();

namespace libdvid {

DVIDNodeService::DVIDNodeService(string web_addr_, UUID uuid_,
        string user, string app, string resource_server_, int resource_port_) :
    connection(web_addr_, user, app, resource_server_, resource_port_), uuid(uuid_) 
{
}

BinaryDataPtr DVIDNodeService::custom_request(string endpoint,
        BinaryDataPtr payload, ConnectionMethod method, bool compress,
        unsigned long long datasize, int timeout)
{
    // append '/' to the endpoint if it is not provided and there is no
    // query string at the end
    if (!endpoint.empty() && (endpoint[0] != '/')) {
        endpoint = '/' + endpoint;
    }

    if (compress) {
        if (endpoint.find("?") != string::npos) {
            endpoint += "&compression=lz4";
        } else {
            endpoint += "?compression=lz4";
        }

        if (method == POST || method == PUT) {
            payload = BinaryData::compress_lz4(payload); 
        }
    } 

    string respdata;
    string node_endpoint = "/node/" + uuid + endpoint;
    BinaryDataPtr resp_binary = BinaryData::create_binary_data();
    int status_code = connection.make_request(node_endpoint, method, payload,
            resp_binary, respdata, BINARY, timeout, datasize, false);

    // FIXME: For some reason, DVID sometimes returns status 206 for ROI requests.
    //        For now, treat 206 as if it were 200.
    //if (status_code != 200) {
    if (status_code != 200 && status_code != 206) {
        throw DVIDException("DVIDException for " + node_endpoint + "\n" + respdata + "\n" + resp_binary->get_data(), status_code);
    }

    if (compress) {
        // call safe version of lz4
        if (method == GET) {
            resp_binary = BinaryData::decompress_lz4(resp_binary, 0, glb_buffer.get_buffer(), INT_MAX); 
        }
    }

    return resp_binary; 
}
    
Json::Value DVIDNodeService::get_typeinfo(string datatype_name)
{
    BinaryDataPtr binary = custom_request("/" + datatype_name + "/info", BinaryDataPtr(), GET);
   
    // read into json from binary string
    return binary->get_json_value();
}
    
size_t DVIDNodeService::get_blocksize(string datatype_name)
{
    if (instance_blocksize_map.find(datatype_name) !=
            instance_blocksize_map.end()) {
        return instance_blocksize_map[datatype_name];
    }
   
    BinaryDataPtr binary = custom_request("/" + datatype_name + "/info", BinaryDataPtr(), GET);
   
    // read into json from binary string 
    Json::Value data = binary->get_json_value();

    // retrieve block information if it exists 
    Json::Value extended_data = data["Extended"];
    if (!extended_data) {
        throw ErrMsg("Instance does not contain blocksize information");
    }
    Json::Value blockdata = extended_data["BlockSize"];
    if (!blockdata) {
        throw ErrMsg("Instance does not contain blocksize information");
    }
     
    // check if all block dimensions are equal
    int x = blockdata[0u].asInt();
    int y = blockdata[1u].asInt();
    int z = blockdata[2u].asInt();

    if (x < 1 || y < 1 || z < 1)
    {
            std::ostringstream ss;
            ss << "Instance metadata specifies invalid BlockSize: [" << x << ", " << y << ", " << z << "]";
            throw ErrMsg(ss.str());
    }

    if (x != y || x != z) {
        throw ErrMsg("Instance does not have a block size with equal dimensions");
    } 

    // set size in cache (assume that metadata cannot change
    // for instance name for the same repo)
    instance_blocksize_map[datatype_name] = x;
    return x; 
}


bool DVIDNodeService::create_grayscale8(string datatype_name, size_t blocksize)
{
    return create_datatype("uint8blk", datatype_name, blocksize);
}

bool DVIDNodeService::create_labelblk(string datatype_name,
        string labelvol_name, size_t blocksize)
{
    bool is_created = create_datatype("labelblk", datatype_name, blocksize);
    bool is_created2 = true;

    if (labelvol_name != "") {
        // create labelvol instance
        // and setup bi-directional sync
        is_created2 =
            create_datatype("labelvol", labelvol_name, blocksize)
            && sync(labelvol_name, datatype_name)
            && sync(datatype_name, labelvol_name);

    }

    return is_created && is_created2;
}

bool DVIDNodeService::create_labelarray(string datatype_name, size_t blocksize)
{
    bool is_created = create_datatype("labelarray", datatype_name, blocksize);
    return is_created;
}

bool DVIDNodeService::create_labelmap(string datatype_name, size_t blocksize)
{
    bool is_created = create_datatype("labelmap", datatype_name, blocksize);
    return is_created;
}

bool DVIDNodeService::create_keyvalue(string keyvalue)
{
    return create_datatype("keyvalue", keyvalue);
}

bool DVIDNodeService::create_graph(string graph_name)
{
    return create_datatype("labelgraph", graph_name);
}

bool DVIDNodeService::create_roi(string name)
{
    return create_datatype("roi", name);
}

Grayscale2D DVIDNodeService::get_tile_slice(string datatype_instance,
        Slice2D slice, unsigned int scaling, vector<int> tile_loc)
{
    BinaryDataPtr binary_response = get_tile_slice_binary(datatype_instance,
            slice, scaling, tile_loc);
    Dims_t dim_size;

    // retrieve JPEG
    try {
        unsigned int width, height;
        binary_response = 
            BinaryData::decompress_jpeg(binary_response, width, height);
        dim_size.push_back(width); dim_size.push_back(height);
        
    } catch (ErrMsg& msg) {
        unsigned int width, height;
        binary_response = 
            BinaryData::decompress_png8(binary_response, width, height);
        dim_size.push_back(width); dim_size.push_back(height);
    }

    Grayscale2D grayimage(binary_response, dim_size);
    return grayimage;
}

BinaryDataPtr DVIDNodeService::get_tile_slice_binary(string datatype_instance,
        Slice2D slice, unsigned int scaling, vector<int> tile_loc)
{
    if (tile_loc.size() != 3) {
        throw ErrMsg("Tile identification requires 3 numbers");
    }

    string tileplane = "XY";
    if (slice == XZ) {
        tileplane = "XZ";
    } else if (slice == YZ) {
        tileplane = "YZ";
    }

    string uri =  "/" + datatype_instance + "/tile/" + tileplane + "/";
    stringstream sstr;
    sstr << uri;
    sstr << scaling << "/" << tile_loc[0];
    for (unsigned int i = 1; i < tile_loc.size(); ++i) {
        sstr << "_" << tile_loc[i];
    }

    string endpoint = sstr.str();
    BinaryDataPtr cachedata = DVIDCache::get_cache()->get(uuid + "/" + endpoint);
    if (cachedata) {
        return cachedata;
    }

    cachedata = custom_request(endpoint, BinaryDataPtr(), GET);
    DVIDCache::get_cache()->set(uuid + "/" + endpoint, cachedata);
    return cachedata;
}


Array8bit3D DVIDNodeService::get_array8bit3D(string datatype_instance, Dims_t sizes,
        vector<int> offset, bool islabels)
{
    BinaryDataPtr data = get_array3D(datatype_instance, sizes, offset, islabels); 

    if (islabels) {
        // determined number of returned bytes
        int decomp_size = sizes[0]*sizes[1]*sizes[2];
        data = BinaryData::decompress_lz4(data, decomp_size);
    }

    Array8bit3D vol(data, sizes);
    return vol; 
}

Array16bit3D DVIDNodeService::get_array16bit3D(string datatype_instance, Dims_t sizes,
        vector<int> offset, bool islabels)
{
    BinaryDataPtr data = get_array3D(datatype_instance, sizes, offset, islabels);

    if (islabels) {
        // determined number of returned bytes
        int decomp_size = sizes[0]*sizes[1]*sizes[2]*2;
        data = BinaryData::decompress_lz4(data, decomp_size);
    }

    Array16bit3D vol(data, sizes);
    return vol; 
}

Array32bit3D DVIDNodeService::get_array32bit3D(string datatype_instance, Dims_t sizes,
        vector<int> offset, bool islabels)
{
    BinaryDataPtr data = get_array3D(datatype_instance, sizes, offset, islabels);

    if (islabels) {
        // determined number of returned bytes
        int decomp_size = sizes[0]*sizes[1]*sizes[2]*4;
        data = BinaryData::decompress_lz4(data, decomp_size);
    }

    Array32bit3D vol(data, sizes);
    return vol; 
}

Array64bit3D DVIDNodeService::get_array64bit3D(string datatype_instance, Dims_t sizes,
        vector<int> offset, bool islabels)
{
    BinaryDataPtr data = get_array3D(datatype_instance, sizes, offset, islabels);

    if (islabels) {
        // determined number of returned bytes
        int decomp_size = sizes[0]*sizes[1]*sizes[2]*8;
        data = BinaryData::decompress_lz4(data, decomp_size);
    } 

    Array64bit3D vol(data, sizes);
    return vol; 
}

void DVIDNodeService::put_array8bit3D(string datatype_instance,
        Array8bit3D const & volume, vector<int> offset, bool islabels)
{
    Dims_t sizes = volume.get_dims();

    // compression only enabled for labels and default lz4
    put_volume(datatype_instance, volume.get_binary(), sizes,
            offset, false, islabels, "", false, false);
}

void DVIDNodeService::put_array16bit3D(string datatype_instance,
        Array16bit3D const & volume, vector<int> offset, bool islabels)
{
    Dims_t sizes = volume.get_dims();

    // compression only enabled for labels and default lz4
    put_volume(datatype_instance, volume.get_binary(), sizes,
            offset, false, islabels, "", false, false);
}

void DVIDNodeService::put_array32bit3D(string datatype_instance,
        Array32bit3D const & volume, vector<int> offset, bool islabels)
{
    Dims_t sizes = volume.get_dims();

    // compression only enabled for labels and default lz4
    put_volume(datatype_instance, volume.get_binary(), sizes,
            offset, false, islabels, "", false, false);
}

void DVIDNodeService::put_array64bit3D(string datatype_instance,
        Array64bit3D const & volume, vector<int> offset, bool islabels)
{
    Dims_t sizes = volume.get_dims();

    // compression only enabled for labels and default lz4
    put_volume(datatype_instance, volume.get_binary(), sizes,
            offset, false, islabels, "", false, false);
}


BinaryDataPtr DVIDNodeService::get_array3D(string datatype_instance, Dims_t sizes,
        vector<int> offset, bool islabels)
{
    vector<unsigned int> axes;
    axes.push_back(0); axes.push_back(1); axes.push_back(2);
    // turn compression on just lz4 for now
    // TODO: support new compression for labels
    BinaryDataPtr data = get_volume3D(datatype_instance,
            sizes, offset, axes, false, islabels, "");
   
    return data; 
}


Grayscale3D DVIDNodeService::get_gray3D(string datatype_instance, Dims_t sizes,
        vector<int> offset, vector<unsigned int> axes,
        bool throttle, bool compress, string roi)
{
    BinaryDataPtr data = get_volume3D(datatype_instance,
            sizes, offset, axes, throttle, compress, roi);
   
    // decompress using lz4
    if (compress) {
        // determined number of returned bytes
        int decomp_size = sizes[0]*sizes[1]*sizes[2];
        data = BinaryData::decompress_lz4(data, decomp_size);
    }

    Grayscale3D grayvol(data, sizes);
    return grayvol; 
}

Grayscale3D DVIDNodeService::get_gray3D(string datatype_instance, Dims_t sizes,
        vector<int> offset, bool throttle, bool compress, string roi)
{
    vector<unsigned int> axes;
    axes.push_back(0); axes.push_back(1); axes.push_back(2);
    return get_gray3D(datatype_instance, sizes, offset, axes,
            throttle, compress, roi);
}

Labels3D DVIDNodeService::get_labels3D(string datatype_instance, Dims_t sizes,
        vector<int> offset, vector<unsigned int> axes,
        bool throttle, bool compress, string roi, bool supervoxels)
{
    BinaryDataPtr data = get_volume3D(datatype_instance,
            sizes, offset, axes, throttle, compress, roi, false, supervoxels);
   
    // decompress using lz4
    if (compress) {
        // determined number of returned bytes
        int decomp_size = sizes[0]*sizes[1]*sizes[2]*8;
        data = BinaryData::decompress_lz4(data, decomp_size);
    }

    try {
        Labels3D labels(data, sizes);
        return labels;
    } catch (std::exception const & ex) {
        std::ostringstream ssMsg;
        ssMsg << "Failed to read labels: " << ex.what() << "\n"
              << "Call was: get_labels3D( "
              << "\"" << datatype_instance << "\", "
              << "(" << sizes[0] << "," << sizes[1] << "," << sizes[2] << "), "
              << "(" << offset[0] << "," << offset[1] << "," << offset[2] << "), "
              << "(" << axes[0] << "," << axes[1] << "," << axes[2] << "), "
              << ( (throttle) ? "true" : "false" )
              << ( (compress) ? "true" : "false" )
              << roi
              << ")";
         throw ErrMsg(ssMsg.str());
    }
}

vector<DVIDCompressedBlock> DVIDNodeService::get_grayblocks3D(string datatype_instance, Dims_t sizes,
        vector<int> offset, bool throttle)
{
    vector<DVIDCompressedBlock> c_blocks;
    get_subvolblocks3D(datatype_instance, sizes, offset, throttle, true, c_blocks, DVIDCompressedBlock::jpeg);
    return c_blocks;
}

vector<DVIDCompressedBlock> DVIDNodeService::get_labelblocks3D(string datatype_instance, Dims_t sizes,
        vector<int> offset, bool throttle)
{
    vector<DVIDCompressedBlock> c_blocks;
    get_subvolblocks3D(datatype_instance, sizes, offset, throttle, false, c_blocks, DVIDCompressedBlock::lz4);
    return c_blocks;
}

Grayscale3D DVIDNodeService::get_grayblocks3D_subvol(string datatype_instance, Dims_t sizes, vector<int> offset, bool throttle)
{
    auto c_blocks = get_grayblocks3D(datatype_instance, sizes, offset, throttle);
    return inflate_compressedblocks<uint8_t>(c_blocks, sizes, offset);
}

void DVIDNodeService::prefetch_specificblocks3D(string datatype_instance,
        vector<int> const & blockcoords)
{
    if ((blockcoords.size() % 3) != 0) {
        throw ErrMsg("Did not specify a multiple of 3 block coords");
    }

    if (blockcoords.size() == 0) {
        return;
    }
    
    // construct query string
    stringstream resloc;
    resloc << "/" + datatype_instance + "/specificblocks?prefetch=on&blocks=";

    for (int i = 0; i < blockcoords.size(); i+=3) {
        if (i != 0) {
            resloc << ",";
        }
        resloc << blockcoords[i] << ",";
        resloc << blockcoords[i+1] << ",";
        resloc << blockcoords[i+2]; 
    }
 
    BinaryDataPtr binary_result = custom_request(resloc.str(), BinaryDataPtr(), GET);
    return;
}

void DVIDNodeService::get_specificblocks3D(string datatype_instance,
                                           vector<int> const & blockcoords,
                                           bool gray,
                                           vector<DVIDCompressedBlock>& c_blocks,
                                           int scale,
                                           bool uncompressed,
                                           bool supervoxels)
{
    size_t blocksize = get_blocksize(datatype_instance);
    if ((blockcoords.size() % 3) != 0) {
        throw ErrMsg("Did not specify a multiple of 3 block coords");
    }

    if (gray && scale > 0) {
        ostringstream ss;
        ss << "get_specificblocks3D(): DVID grayscale datatypes (e.g. uint8blk) do not support a 'scale' parameter.\n";
        ss << " Instead, select the appropriate instance for the data scale you need, ";
        ss << "e.g. '" << datatype_instance << "_" << scale << "',";
        ss << "and leave the 'scale' argument set to 0 when calling this function.";
        throw ErrMsg(ss.str());
    }

    if (blockcoords.size() == 0) {
        return;
    }
    
    // construct query string
    stringstream resloc;
    resloc << "/" + datatype_instance + "/specificblocks?blocks=";

    for (int i = 0; i < blockcoords.size(); i+=3) {
        if (i != 0) {
            resloc << ",";
        }
        resloc << blockcoords[i] << ",";
        resloc << blockcoords[i+1] << ",";
        resloc << blockcoords[i+2]; 
    }

    // add scale param
    if (scale > 0) {
        resloc << "&scale=" << scale;
    }
    
    if (uncompressed) {
        resloc << "&compression=uncompressed";
    }

    if (supervoxels) {
        resloc << "&supervoxels=true";
    }
    
    
    BinaryDataPtr binary_result = custom_request(resloc.str(), BinaryDataPtr(), GET);

    const unsigned char * head = binary_result->get_raw();
    int buffer_size = binary_result->length();

    DVIDCompressedBlock::CompressType ctype = DVIDCompressedBlock::gzip_labelarray;
    if (gray) {
        ctype = DVIDCompressedBlock::jpeg;
    }
    size_t datasize = sizeof(uint64);
    if (gray) {
        datasize = 1;
    }
    if (uncompressed) {
        ctype = DVIDCompressedBlock::uncompressed;
    }

    // it is possible to have less blocks than requested if they are blank
    int block_index = 0;
    while (buffer_size >= 16) {
        vector<int> offset;
        offset.push_back(*((int*)head) * blocksize);
        head += 4;
        buffer_size -= 4;
        offset.push_back(*((int*)head) * blocksize);
        head += 4;
        buffer_size -= 4;
        offset.push_back(*((int*)head) * blocksize);
        head += 4;
        buffer_size -= 4;

        int compressed_size = *((int*)head);
        head += 4;
        buffer_size -= 4;

        if (compressed_size > buffer_size) {
            std::ostringstream ss;
            ss << "Malformed response for " << resloc.str() << "\n"
               << "Block header " << block_index << " claims that the block occupies "
               << compressed_size << " bytes of the response, but only "
               << buffer_size << " bytes of response remain to be consumed.\n";
            throw std::runtime_error(ss.str());
        }

        // DVID is supposed to omit blocks that are not present,
        // but in some old versions it returns the header anyway, followed by 0 bytes of data.
        if (compressed_size > 0) {
            BinaryDataPtr blockdata = BinaryData::create_binary_data((const char*) head, compressed_size);

            DVIDCompressedBlock c_block(blockdata, offset, blocksize, datasize, ctype);

            c_blocks.push_back(c_block);
            head += compressed_size;
            buffer_size -= compressed_size;
        }

        block_index += 1;
    }
    if (buffer_size > 0) {
        std::ostringstream ss;
        ss << "Malformed response for " + resloc.str() + "\n"
           << "Buffer was not fully consumed.";
        throw std::runtime_error(ss.str());
    }
}

Labels3D extract_label_subvol( Labels3D const & vol,
                               Dims_t const & subvol_dims_xyz,
                               std::vector<int> const & subvol_offset_xyz )
{
    int vol_Z = vol.get_dims()[2];
    int vol_Y = vol.get_dims()[1];
    int vol_X = vol.get_dims()[0];

    int sv_Z = subvol_dims_xyz[2];
    int sv_Y = subvol_dims_xyz[1];
    int sv_X = subvol_dims_xyz[0];

    int off_z = subvol_offset_xyz[2];
    int off_y = subvol_offset_xyz[1];
    int off_x = subvol_offset_xyz[0];

    // Unfortunately, the design of BinaryData doesn't allow us to avoid a copy here.
    std::vector<uint64_t> subvol_data(sv_Z * sv_Y * sv_X, 0);
    auto const & vol_raw_data = vol.get_raw();

    size_t sv_offset = 0;
    for (size_t sv_z = 0; sv_z < sv_Z; ++sv_z)
    {
        for (size_t sv_y = 0; sv_y < sv_Y; ++sv_y)
        {
            for (size_t sv_x = 0; sv_x < sv_X; ++sv_x)
            {
                // Convert from subvol coords to volume coords
                int z = off_z + sv_z;
                int y = off_y + sv_y;
                int x = off_x + sv_x;

                // Convert to buffer position
                int z_offset = z * vol_X * vol_Y;
                int y_offset = y * vol_X;
                int x_offset = x;

                subvol_data[sv_offset] = vol_raw_data[z_offset + y_offset + x_offset];
                sv_offset += 1;
            }
        }
    }

    BinaryDataPtr subvol_binary = BinaryData::create_binary_data(reinterpret_cast<uint8_t const*>(&subvol_data[0]), subvol_data.size() * sizeof(uint64_t));
    Labels3D subvol(subvol_binary, subvol_dims_xyz);
    return subvol;
}


Labels3D DVIDNodeService::get_labelarray_blocks3D(string datatype_instance,
                                                  Dims_t sizes,
                                                  std::vector<int> offset,
                                                  bool throttle,
                                                  int scale,
                                                  bool supervoxels)
{
    vector<DVIDCompressedBlock> c_blocks;
    get_subvolblocks3D(datatype_instance, sizes, offset, throttle, false, c_blocks, DVIDCompressedBlock::gzip_labelarray, scale, supervoxels);
    return inflate_compressedblocks<uint64_t>(c_blocks, sizes, offset);
}

Labels3D DVIDNodeService::inflate_labelarray_blocks3D_from_raw(BinaryDataPtr raw_block_data, Dims_t sizes, std::vector<int> offset, size_t blocksize)
{
    auto c_blocks = load_compressed_blocks(raw_block_data, blocksize, false, DVIDCompressedBlock::gzip_labelarray);
    return inflate_compressedblocks<uint64_t>(c_blocks, sizes, offset);
}

void DVIDNodeService::put_labelblocks3D(string datatype_instance, Labels3D const & volume, vector<int> volume_offset_xyz, bool throttle, int scale, bool noindexing)
{
    unsigned int blocksize = (unsigned int)get_blocksize(datatype_instance);

    int vol_Z = volume.get_dims()[2];
    int vol_Y = volume.get_dims()[1];
    int vol_X = volume.get_dims()[0];

    // make sure volume specified is legal and block aligned
    if (volume_offset_xyz.size() != 3) {
        throw ErrMsg("Did not correctly specify 3D volume");
    }

    if (   (volume_offset_xyz[0] % blocksize != 0)
        || (volume_offset_xyz[1] % blocksize != 0)
        || (volume_offset_xyz[2] % blocksize != 0) )
    {
        throw ErrMsg("Label block POST error: Not block aligned");
    }

    if ((vol_Z % blocksize != 0) || (vol_Y % blocksize != 0) || (vol_X % blocksize != 0)) {
        throw ErrMsg("Label block POST error: Region is not a multiple of block size");
    }

    EncodedData full_data;

    for (int z_block = 0; z_block < vol_Z / blocksize; ++z_block)
    {
        for (int y_block = 0; y_block < vol_Y / blocksize; ++y_block)
        {
            for (int x_block = 0; x_block < vol_X / blocksize; ++x_block)
            {
                try
                {
                    std::vector<int> subvol_offset_xyz = { x_block * (int)blocksize,
                                                           y_block * (int)blocksize,
                                                           z_block * (int)blocksize };

                    const Dims_t subvol_dims = { blocksize, blocksize, blocksize };
                    Labels3D subvol = extract_label_subvol(volume, subvol_dims, subvol_offset_xyz);

                    BinaryDataPtr encoded_block;
                    
                    if (!DVIDNODESERVICE_EXTRA_CHECKS) // From environment -- see above
                    {
                        encoded_block = BinaryData::compress_gzip_labelarray_block(subvol.get_binary(), blocksize);
                    }
                    else
                    {
                        // Perfom labelarray compression and gzip in two steps, so we can examine the block before gzipping.
                        BinaryDataPtr encoded_block_before_gzip = BinaryData::compress_labelarray_block(subvol.get_binary(), blocksize);
                        uint32_t table_size = reinterpret_cast<uint32_t const *>(encoded_block_before_gzip->get_raw())[3];

                        //
                        // This consistency check was helpful for ironing out some bugs in libdvid and/or dvid,
                        // but it is expensive and it's no longer necessary, so it's commented out.
                        //
                        // // Re-scan the entire block to count the number of unique labels.
                        // std::unordered_set<uint64_t> unique_voxels( subvol.get_raw(), subvol.get_raw() + subvol.count() );
                        // if (unique_voxels.size() != table_size)
                        // {
                        //    std::ostringstream ss;
                        //    ss << "Block has " << unique_voxels.size() << " unique labels, but the encoded block table has " << table_size << " entries.";
                        //    throw std::runtime_error(ss.str());
                        // }

                        encoded_block = BinaryData::compress_gzip(encoded_block_before_gzip);
                    }
                
                    encode_int<int32_t>(full_data, (volume_offset_xyz[0] + subvol_offset_xyz[0]) / blocksize);
                    encode_int<int32_t>(full_data, (volume_offset_xyz[1] + subvol_offset_xyz[1]) / blocksize);
                    encode_int<int32_t>(full_data, (volume_offset_xyz[2] + subvol_offset_xyz[2]) / blocksize);
                    encode_int<int32_t>(full_data, static_cast<int32_t>(encoded_block->length()));
                    
                    encode_binary_data(full_data, encoded_block);
                }
                catch (std::exception const & ex)
                {
                    std::ostringstream ss;
                    ss << "Error while encoding volume at offset: "
                       << "(" << volume_offset_xyz[0] << ", " << volume_offset_xyz[1] << ", " << volume_offset_xyz[2] << "), "
                       << "block index: " << "[" << x_block << ", " << y_block << ", " << z_block << "]\n"
                       << ex.what();

                    throw std::runtime_error( ss.str() );
                }
            }
        }
    }

    BinaryDataPtr payload = BinaryData::create_binary_data( &full_data[0], full_data.size() );

    if (throttle) {
        // set random number
        timeval t1;
        gettimeofday(&t1, NULL);
        srand(t1.tv_usec * t1.tv_sec);
    }

    bool waiting = true;
    int status_code;
    string respdata;

     // construct query string
    std::ostringstream ss_uri;
    ss_uri << "/node/" << uuid << "/" << datatype_instance << "/blocks";
    if (throttle)
    {
        ss_uri << "?throttle=on";
        ss_uri << "&scale=" << scale;
    } else {
        ss_uri << "?scale=" << scale;
    }

    if (noindexing) {
    		ss_uri << "&noindexing=true";
    }

    std::string endpoint = ss_uri.str();

    // make instance random (create random seed based on time of day)
    int timeout = 20;
    int timeout_max = 600;

    if (throttle) {
        // set random number
        timeval t1;
        gettimeofday(&t1, NULL);
        srand(t1.tv_usec * t1.tv_sec);
    }

    // try posting until DVID is available (no contention)
    BinaryDataPtr binary_response;
    while (waiting) {
        binary_response = BinaryData::create_binary_data();
        status_code = connection.make_request(endpoint, POST, payload,
                binary_response, respdata, BINARY, DVIDConnection::DEFAULT_TIMEOUT, 1, false);

        // wait if server is busy
        if (status_code == 503) {
            // random backoff
            sleep(rand()%timeout);
            // capped exponential backoff
            if (timeout < timeout_max) {
                timeout *= 2;
            }
        } else {
            waiting = false;
        }
    }

    if (status_code != 200) {
        throw DVIDException("DVIDException for " + endpoint + "\n" + respdata + "\n" + binary_response->get_data(),
                status_code);
    }
}


BinaryDataPtr DVIDNodeService::get_subvolblocks3D_rawbuffer(string datatype_instance,
                                                            Dims_t sizes,
                                                            vector<int> offset,
                                                            bool throttle,
                                                            bool gray,
                                                            DVIDCompressedBlock::CompressType ctype,
                                                            int scale,
                                                            bool supervoxels)
{
    // make sure volume specified is legal and block aligned
    if ((sizes.size() != 3) || (offset.size() != 3)) {
        throw ErrMsg("Did not correctly specify 3D volume");
    }
   
     // construct query string
    string uri = "/node/" + uuid + "/"
                    + datatype_instance;
    if (gray) {
        uri += "/subvolblocks/";
    } else {
        uri += "/blocks/";
    }

    stringstream sstr;
    sstr << uri;
    sstr << sizes[0];
    for (unsigned int i = 1; i < sizes.size(); ++i) {
        sstr << "_" << sizes[i];
    }
    sstr << "/" << offset[0];
    for (unsigned int i = 1; i < offset.size(); ++i) {
        sstr << "_" << offset[i];
    }

    std::unordered_map<DVIDCompressedBlock::CompressType, std::string, std::hash<int> > compression_strings;
    compression_strings[DVIDCompressedBlock::jpeg] = "jpeg";
    compression_strings[DVIDCompressedBlock::lz4] = "lz4";
    compression_strings[DVIDCompressedBlock::gzip_labelarray] = "blocks";

    sstr << "?compression=" << compression_strings[ctype];

    if (throttle) {
        sstr << "&throttle=on";
    }

    if (supervoxels) {
        sstr << "&supervoxels=true";
    }
    
    sstr << "&scale=" << scale;
 
    // try get until DVID is available (no contention)
    BinaryDataPtr binary_result; 
    string respdata;
    bool waiting = true;
    int timeout = 20;
    int timeout_max = 600;
    while (waiting) {
        binary_result = BinaryData::create_binary_data();
        int status_code = connection.make_request(sstr.str(), GET, BinaryDataPtr(),
                binary_result, respdata, BINARY);
       
        // wait if server is busy
        if (status_code == 503) {
            // random backoff
            sleep(rand()%timeout);
            // capped exponential backoff
            if (timeout < timeout_max) {
                timeout *= 2;
            }
        } else {
            waiting = false;
        }
    }

    // put compressed blocks into vector
    //int num_blocks = (sizes[0]/blocksize)*(sizes[1]/blocksize)*(sizes[2]/blocksize);
    return binary_result;
}

std::vector<DVIDCompressedBlock> DVIDNodeService::load_compressed_blocks( BinaryDataPtr block_data,
                                                                          size_t blocksize,
                                                                          bool gray,
                                                                          DVIDCompressedBlock::CompressType ctype )
{
    std::vector<DVIDCompressedBlock> c_blocks;
    
    const unsigned char * head = block_data->get_raw();
    int buffer_size = block_data->length();

    // it is possible to have less blocks than requested if they are blank
    while (buffer_size) {
        // retrieve offset
        vector<int> offset;
        offset.push_back(*((int*)head) * blocksize);
        head += 4;
        buffer_size -= 4;
        offset.push_back(*((int*)head) * blocksize);
        head += 4;
        buffer_size -= 4;
        offset.push_back(*((int*)head) * blocksize);
        head += 4;
        buffer_size -= 4;
        int lz4_bytes = *((int*)head);
        head += 4;
        buffer_size -= 4;

        if (lz4_bytes > 0) {
            BinaryDataPtr blockdata = BinaryData::create_binary_data((const char*) head, lz4_bytes);

            size_t datasize = sizeof(uint64);
            if (gray) {
                datasize = 1;
            }

            DVIDCompressedBlock c_block(blockdata, offset, blocksize, datasize, ctype);

            c_blocks.push_back(c_block);
            head += lz4_bytes;
            buffer_size -= lz4_bytes;
        }
    }
    return c_blocks;
}

void DVIDNodeService::get_subvolblocks3D(string datatype_instance, Dims_t sizes,
                                         vector<int> offset, bool throttle, bool gray, vector<DVIDCompressedBlock>& c_blocks,
                                         DVIDCompressedBlock::CompressType ctype, int scale, bool supervoxels)
{
    size_t blocksize = get_blocksize(datatype_instance);
    
    if (   (offset[0] % blocksize != 0)
        || (offset[1] % blocksize != 0)
        || (offset[2] % blocksize != 0) ) {
        throw ErrMsg("Label block GET error: Not block aligned");
    }
    
    if (   (sizes[0] % blocksize != 0)
        || (sizes[1] % blocksize != 0)
        || (sizes[2] % blocksize != 0) ) {
        throw ErrMsg("Label block GET error: Region is not a multiple of block size");
    }

    BinaryDataPtr raw_data = get_subvolblocks3D_rawbuffer( datatype_instance,
                                                           sizes,
                                                           offset,
                                                           throttle,
                                                           gray,
                                                           ctype,
                                                           scale,
                                                           supervoxels );

    c_blocks = load_compressed_blocks(raw_data, blocksize, gray, ctype);
}


Labels3D DVIDNodeService::get_labels3D(string datatype_instance, Dims_t sizes,
        vector<int> offset, bool throttle, bool compress, string roi, bool supervoxels)
{
    vector<unsigned int> axes;
    axes.push_back(0); axes.push_back(1); axes.push_back(2);
    return get_labels3D(datatype_instance, sizes, offset, axes,
            throttle, compress, roi, supervoxels);
}

uint64 DVIDNodeService::get_label_by_location(std::string datatype_instance, unsigned int x,
            unsigned int y, unsigned int z, bool supervoxels)
{
    Dims_t sizes; sizes.push_back(1);
    sizes.push_back(1); sizes.push_back(1);
    vector<int> start; start.push_back(x); start.push_back(y); start.push_back(z);
    Labels3D labels = get_labels3D(datatype_instance, sizes, start, false, supervoxels);
    const uint64* ptr = (const uint64*) labels.get_raw();
    return *ptr;
}

void DVIDNodeService::put_labels3D(string datatype_instance, Labels3D const & volume,
            vector<int> offset, bool throttle, bool compress, string roi, bool mutate)
{
    Dims_t sizes = volume.get_dims();
    put_volume(datatype_instance, volume.get_binary(), sizes,
            offset, throttle, compress, roi, mutate, false);
}

void DVIDNodeService::put_gray3D(string datatype_instance, Grayscale3D const & volume,
            vector<int> offset, bool throttle, bool compress)
{
    Dims_t sizes = volume.get_dims();
    put_volume(datatype_instance, volume.get_binary(), sizes,
            offset, throttle, compress, "", false, false);
}

GrayscaleBlocks DVIDNodeService::get_grayblocks(string datatype_instance,
        vector<int> block_coords, unsigned int span)
{
    int ret_span = span;
    BinaryDataPtr data = get_blocks(datatype_instance, block_coords, span);
    size_t blocksize = get_blocksize(datatype_instance);

    // make sure this data encodes blocks of grayscale
    if (data->length() !=
            (blocksize*blocksize*blocksize*sizeof(uint8)*span)) {
        throw ErrMsg("Expected 1-byte values from " + datatype_instance);
    }
 
    return GrayscaleBlocks(data, ret_span, blocksize);
} 

LabelBlocks DVIDNodeService::get_labelblocks(string datatype_instance,
           vector<int> block_coords, unsigned int span)
{
    int ret_span = span;
    BinaryDataPtr data = get_blocks(datatype_instance, block_coords, span);
    size_t blocksize = get_blocksize(datatype_instance);

    // make sure this data encodes blocks of grayscale
    if (data->length() !=
            (blocksize*blocksize*blocksize*sizeof(uint64)*ret_span)) {
        throw ErrMsg("Expected 8-byte values from " + datatype_instance);
    }
 
    return LabelBlocks(data, ret_span, blocksize);
}
    
void DVIDNodeService::put_grayblocks(string datatype_instance,
            GrayscaleBlocks blocks, vector<int> block_coords)
{
    put_blocks(datatype_instance, blocks.get_binary(),
            blocks.get_num_blocks(), block_coords);
}


void DVIDNodeService::put_labelblocks(string datatype_instance,
            LabelBlocks blocks, vector<int> block_coords)
{
    put_blocks(datatype_instance, blocks.get_binary(),
            blocks.get_num_blocks(), block_coords);
}

void DVIDNodeService::put(string keyvalue, string key, ifstream& fin)
{
    BinaryDataPtr data = BinaryData::create_binary_data(fin);
    put(keyvalue, key, data);
}

void DVIDNodeService::put(string keyvalue, string key, Json::Value& data)
{
    stringstream datastr;
    datastr << data;
    BinaryDataPtr bdata = BinaryData::create_binary_data(datastr.str().c_str(),
            datastr.str().length());
    put(keyvalue, key, bdata);
}


void DVIDNodeService::put(string keyvalue, string key, BinaryDataPtr value)
{
    string endpoint = "/" + keyvalue + "/key/" + key;
    custom_request(endpoint, value, POST);
}


BinaryDataPtr DVIDNodeService::get(string keyvalue, string key)
{
    return custom_request("/" + keyvalue + "/key/" + key, BinaryDataPtr(), GET);
}

Json::Value DVIDNodeService::get_json(string keyvalue, string key)
{
    BinaryDataPtr binary = get(keyvalue, key);
    return binary->get_json_value();
}

vector<string> DVIDNodeService::get_keys(std::string keyvalue)
{
    BinaryDataPtr response_binary = custom_request("/" + keyvalue + "/keys", BinaryDataPtr(), GET);

    // read into json from binary string
    Json::Value response_json = response_binary->get_json_value();

    vector<string> keys;
    std::transform(response_json.begin(), response_json.end(),
                   std::back_inserter(keys),
                   boost::bind(&Json::Value::asString, _1));
    return keys;

}

void DVIDNodeService::get_subgraph(string graph_name,
        const std::vector<Vertex>& vertices, Graph& graph)
{
    // make temporary graph for query
    Graph temp_graph;
    for (unsigned int i = 0; i < vertices.size(); ++i) {
        temp_graph.vertices.push_back(vertices[i]);
    }   
    Json::Value data;
    temp_graph.export_json(data);

    stringstream datastr;
    datastr << data;
    BinaryDataPtr binary_data = BinaryData::create_binary_data(
            datastr.str().c_str(), datastr.str().length());

    BinaryDataPtr binary = custom_request("/" + graph_name + "/subgraph",
           binary_data, GET);

    // read json from binary string into graph    
    Json::Value returned_data = binary->get_json_value();
    graph.import_json(returned_data);
}

void DVIDNodeService::get_vertex_neighbors(string graph_name, Vertex vertex,
        Graph& graph)
{
    stringstream uri_ending;
    uri_ending << "/neighbors/" << vertex.id;
    
    BinaryDataPtr binary = custom_request("/" + graph_name + uri_ending.str(),
            BinaryDataPtr(), GET);
    
    // read into json from binary string 
    Json::Value data = binary->get_json_value();
    graph.import_json(data);
}

void DVIDNodeService::update_vertices(string graph_name,
        const std::vector<Vertex>& vertices)
{
    unsigned int num_examined = 0;

    while (num_examined < vertices.size()) {
        Graph graph;
        
        // grab 1000 vertices at a time
        unsigned int max_size = ((num_examined + TransactionLimit) > 
                vertices.size()) ? vertices.size() : (num_examined + TransactionLimit); 
        for (; num_examined < max_size; ++num_examined) {
            graph.vertices.push_back(vertices[num_examined]);
        }

        Json::Value data;
        graph.export_json(data);

        stringstream datastr;
        datastr << data;
        BinaryDataPtr binary = BinaryData::create_binary_data(
                datastr.str().c_str(), datastr.str().length());
        custom_request("/" + graph_name + "/weight", binary, POST);
    }
}
    
void DVIDNodeService::update_edges(string graph_name,
        const std::vector<Edge>& edges)
{

    unsigned int num_examined = 0;
    while (num_examined < edges.size()) {
        VertexSet examined_vertices; 
        Graph graph;
        
        for (; num_examined < edges.size(); ++num_examined) {
            // break if it is not possible to add another edge transaction
            // (assuming that both vertices of that edge will be new vertices
            // for simplicity
            // for simplicity
            // for simplicity)
            if (examined_vertices.size() >= (TransactionLimit - 1)) {
                break;
            }

            graph.edges.push_back(edges[num_examined]);
            examined_vertices.insert(edges[num_examined].id1);
            examined_vertices.insert(edges[num_examined].id2);
            
        }

        Json::Value data;
        graph.export_json(data);

        stringstream datastr;
        datastr << data;
        BinaryDataPtr binary = BinaryData::create_binary_data(
                datastr.str().c_str(), datastr.str().length());
        custom_request("/" + graph_name + "/weight", binary, POST);
    }
}

// vertex list is modified, just use a copy
void DVIDNodeService::get_properties(string graph_name,
        std::vector<Vertex> vertices, string key,
        std::vector<BinaryDataPtr>& properties,
        VertexTransactions& transactions)
{
    unsigned int num_examined = 0;
        
    #ifdef __clang__
    std::unordered_map<VertexID, BinaryDataPtr> properties_map;
    #else
    std::unordered_map<VertexID, BinaryDataPtr> properties_map;
    #endif
    
    int num_verts = vertices.size();

    // keep extending vertices with failed ones
    while (num_examined < vertices.size()) {
        // grab 1000 vertices at a time
        unsigned int max_size = ((num_examined + TransactionLimit) > 
                vertices.size()) ? vertices.size() : (num_examined + TransactionLimit); 

        VertexTransactions current_transactions;
        // serialize data to push
        for (; num_examined < max_size; ++num_examined) {
            current_transactions[vertices[num_examined].id] = 0;
        }
        BinaryDataPtr transaction_binary = 
            write_transactions_to_binary(current_transactions);
        
        // add vertex list to get properties
        uint64 * vertex_array =
            new uint64 [(current_transactions.size()+1)];
        int pos = 0;
        vertex_array[pos] = current_transactions.size();
        ++pos;
        for (VertexTransactions::iterator iter = current_transactions.begin();
                iter != current_transactions.end(); ++iter) {
            vertex_array[pos] = iter->first;
            ++pos;
        }
        string& str_append = transaction_binary->get_data();

        str_append += string((char*)vertex_array, (current_transactions.size()+1)*8);
        
        delete []vertex_array;

        // get data associated with given graph and 'key'
        BinaryDataPtr binary = custom_request("/" + graph_name +
                "/propertytransaction/vertices/" + key + "/", transaction_binary, GET);

        VertexSet bad_vertices;
        size_t byte_pos = load_transactions_from_binary(binary->get_data(),
                transactions, bad_vertices);
       
        // failed vertices should be re-examined 
        for (VertexSet::iterator iter = bad_vertices.begin();
                iter != bad_vertices.end(); ++iter) {
            vertices.push_back(Vertex(*iter, 0));
        }

        // load properties
        const unsigned char* bytearray = binary->get_raw();
       
        // get number of properties
        uint64* num_transactions = (uint64 *)(bytearray+byte_pos);
        byte_pos += 8;

        // iterate through all properties
        for (int i = 0; i < int(*num_transactions); ++i) {
            VertexID* vertex_id = (VertexID *)(bytearray+byte_pos);
            byte_pos += 8;
            uint64* data_size = (uint64*)(bytearray+byte_pos);
            byte_pos += 8;
            properties_map[*vertex_id] = BinaryData::create_binary_data(
                    (const char *)(bytearray+byte_pos), *data_size);      
            byte_pos += *data_size;
        }
    }

    for (int i = 0; i < num_verts; ++i) {
        properties.push_back(properties_map[vertices[i].id]);
    }
}

// edge list is modified, just use a copy
void DVIDNodeService::get_properties(string graph_name, std::vector<Edge> edges, string key,
            std::vector<BinaryDataPtr>& properties, VertexTransactions& transactions)
{
    unsigned int num_examined = 0;
    
    #ifdef __clang__
    std::unordered_map<Edge, BinaryDataPtr, Edge> properties_map;
    #else
    std::unordered_map<Edge, BinaryDataPtr, Edge> properties_map;
    #endif
    int num_edges = edges.size();

    // keep extending vertices with failed ones
    while (num_examined < edges.size()) {
        VertexSet examined_vertices; 

        unsigned int num_current_edges = 0;
        unsigned int starting_num = num_examined;
        for (; num_examined < edges.size(); ++num_current_edges, ++num_examined) {
            // break if it is not possible to add another edge transaction
            // (assuming that both vertices of that edge will be new vertices)
            assert(TransactionLimit > 0);
            if (examined_vertices.size() >= (TransactionLimit - 1)) {
                break;
            }

            examined_vertices.insert(edges[num_examined].id1);
            examined_vertices.insert(edges[num_examined].id2);
            
        }

        // write out transactions for current edge
        VertexTransactions current_transactions;
        for (VertexSet::iterator iter = examined_vertices.begin();
                iter != examined_vertices.end(); ++iter) {
            current_transactions[*iter] = 0;
        }
        BinaryDataPtr transaction_binary = write_transactions_to_binary(current_transactions);
       
        // add edge list to get properties
        uint64* edge_array =
            new uint64 [(num_current_edges*2+1)*8];
        unsigned int pos = 0;
        edge_array[pos] = num_current_edges;
        ++pos;
        for (unsigned int iter = starting_num; iter < num_examined; ++iter) {
            edge_array[pos] = edges[iter].id1;
            ++pos;
            edge_array[pos] = edges[iter].id2;
            ++pos;
        }
        string& str_append = transaction_binary->get_data();
        str_append += string((char*)edge_array, (num_current_edges*2+1)*8);
        delete []edge_array;

        // get data associated with given graph and 'key'
        BinaryDataPtr binary = custom_request("/" + graph_name +
                "/propertytransaction/edges/" + key + "/", transaction_binary, GET);

        VertexSet bad_vertices;
        size_t byte_pos = load_transactions_from_binary(
                binary->get_data(), transactions, bad_vertices);
       
        // failed vertices should cause corresponding edges to be re-examined 
        for (unsigned int iter = starting_num; iter < num_examined; ++iter) {
            if (bad_vertices.find(edges[iter].id1) != bad_vertices.end()) {
                edges.push_back(edges[iter]);
            } else if (bad_vertices.find(edges[iter].id2) != bad_vertices.end()) {
                edges.push_back(edges[iter]);
            }
        }
        
        // load properties
        const unsigned char* bytearray = binary->get_raw();
        
        // get number of properties
        uint64* num_transactions = 
            (uint64*)(bytearray+byte_pos);
        byte_pos += 8;

        // iterate through all properties
        for (int i = 0; i < int(*num_transactions); ++i) {
            VertexID* vertex_id1 = (VertexID *)(bytearray+byte_pos);
            byte_pos += 8;
            VertexID* vertex_id2 = (VertexID *)(bytearray+byte_pos);
            byte_pos += 8;
            uint64* data_size = (uint64*)(bytearray+byte_pos);
            byte_pos += 8;
            properties_map[Edge(*vertex_id1, *vertex_id2, 0)] =
                BinaryData::create_binary_data((const char*)
                        (bytearray+byte_pos), *data_size);      
            byte_pos += *data_size;
        }
    }

    for (int i = 0; i < num_edges; ++i) {
        properties.push_back(properties_map[edges[i]]);
    }
}

void DVIDNodeService::set_properties(string graph_name, std::vector<Vertex>& vertices,
        string key, std::vector<BinaryDataPtr>& properties,
        VertexTransactions& transactions, std::vector<Vertex>& leftover_vertices)
{
    unsigned int num_examined = 0;

    // only post 1000 properties at a time
    while (num_examined < vertices.size()) {
        // add vertex list and properties
        
        // grab 1000 vertices at a time
        unsigned int max_size = ((num_examined + TransactionLimit) > 
                vertices.size()) ? vertices.size() : (num_examined + TransactionLimit); 
   
        VertexTransactions temp_transactions;
        for (unsigned int i = num_examined; i < max_size; ++i) {
            temp_transactions[vertices[i].id] = transactions[vertices[i].id];
        }
        BinaryDataPtr binary = write_transactions_to_binary(temp_transactions);

        // write out number of trans
        string& str_append = binary->get_data();
        uint64 num_trans = temp_transactions.size();
        str_append += string((char*)&num_trans, 8);

        for (; num_examined < max_size; ++num_examined) {
            uint64 id = vertices[num_examined].id;
            BinaryDataPtr databin = properties[num_examined];
            string& data = databin->get_data();
            uint64 data_size = data.size();

            str_append += string((char*)&id, 8);
            str_append += string((char*)&data_size, 8);
            str_append += data;
        } 

        // put the data
        VertexSet failed_trans;
        VertexTransactions succ_trans;
        BinaryDataPtr result_binary = custom_request("/" + graph_name +
                "/propertytransaction/vertices/" + key, binary, POST);
        load_transactions_from_binary(result_binary->get_data(),
            succ_trans, failed_trans);

        // update transaction ids for successful trans
        for (VertexTransactions::iterator iter = succ_trans.begin();
                iter != succ_trans.end(); ++iter) {
            transactions[iter->first] = iter->second;
        }

        // add failed vertices to leftover vertices
        for (VertexSet::iterator iter = failed_trans.begin();
                iter != failed_trans.end(); ++iter) {
            leftover_vertices.push_back(Vertex(*iter, 0));
        }
    }
}

void DVIDNodeService::set_properties(string graph_name, std::vector<Edge>& edges,
        string key, std::vector<BinaryDataPtr>& properties,
        VertexTransactions& transactions, std::vector<Edge>& leftover_edges)
{
    unsigned int num_examined = 0;

    // only post 1000 properties at a time
    while (num_examined < edges.size()) {
        VertexSet examined_vertices; 
        
        unsigned int num_current_edges = 0;
        unsigned int starting_num = num_examined;
        for (; num_examined < edges.size(); ++num_current_edges, ++num_examined) {
            // break if it is not possible to add another edge transaction
            // (assuming that both vertices of that edge will be new vertices)
            assert(TransactionLimit > 0);
            if (examined_vertices.size() >= (TransactionLimit - 1)) {
                break;
            }

            examined_vertices.insert(edges[num_examined].id1);
            examined_vertices.insert(edges[num_examined].id2);
        }
    
        // write out transactions for current edge
        VertexTransactions current_transactions;
        for (VertexSet::iterator iter = examined_vertices.begin();
                iter != examined_vertices.end(); ++iter) {
            current_transactions[*iter] = transactions[*iter];
        }
        BinaryDataPtr binary = write_transactions_to_binary(current_transactions);
        
        // add vertex list and properties
        string& str_append = binary->get_data();
        
        uint64 num_trans = num_current_edges;
        str_append += string((char*)&num_trans, 8);

        for (unsigned int iter = starting_num; iter < num_examined; ++iter) {
            uint64 id1 = edges[iter].id1;
            uint64 id2 = edges[iter].id2;
            BinaryDataPtr databin = properties[iter];
            string& data = databin->get_data();
            uint64 data_size = data.size();

            str_append += string((char*)&id1, 8);
            str_append += string((char*)&id2, 8);
            str_append += string((char*)&data_size, 8);
            str_append += data;
        } 

        // put the data
        VertexSet failed_trans;
        VertexTransactions succ_trans;
        BinaryDataPtr result_binary = custom_request("/" + graph_name +
                "/propertytransaction/edges/" + key, binary, POST);
        load_transactions_from_binary(result_binary->get_data(),
            succ_trans, failed_trans);

        // update transaction ids for successful trans
        for (VertexTransactions::iterator iter = succ_trans.begin();
                iter != succ_trans.end(); ++iter) {
            transactions[iter->first] = iter->second;
        }

        // add leftover edges from failed vertices
        for (unsigned int iter = starting_num; iter < num_examined; ++iter) {
            if (failed_trans.find(edges[iter].id1) != failed_trans.end()) {
                leftover_edges.push_back(edges[iter]);
            } else if (failed_trans.find(edges[iter].id2) != failed_trans.end()) {
                leftover_edges.push_back(edges[iter]);
            }
        }
    }
}

void DVIDNodeService::post_roi(std::string roi_name,
        const std::vector<BlockXYZ>& blockcoords)
{
    // Do not assume the blocks are sorted, first
    // sort, deduplicate, and then encode as runlengths in X.
    // Note that comparison of BlockXYZ objects is in z-y-x order.
    vector<BlockXYZ> sorted_blocks( blockcoords );
    std::sort( sorted_blocks.begin(), sorted_blocks.end() );
    sorted_blocks.erase( std::unique(sorted_blocks.begin(), sorted_blocks.end()),
                         sorted_blocks.end() );

    // encode JSON as z,y,x0,x1 (inclusive)
    int z = INT_MAX;
    int y = INT_MAX;
    int xmin = 0; int xmax = 0;
    Json::Value blocks_data(Json::arrayValue);
    unsigned int blockrle_count = 0;
    for (vector<BlockXYZ>::iterator iter = sorted_blocks.begin();
            iter != sorted_blocks.end(); ++iter) {
        if (iter->z != z || iter->y != y || (iter->x != xmax+1)) {
            if (z != INT_MAX) {
                // add run length
                Json::Value block_data(Json::arrayValue);
                block_data[0] = z;
                block_data[1] = y;
                block_data[2] = xmin;
                block_data[3] = xmax;
                blocks_data[blockrle_count] = block_data;
                ++blockrle_count;
            }
        
            z = iter->z;
            y = iter->y;
            xmin = iter->x;
            xmax = xmin;
        } else {
            xmax = iter->x;
        }
    }
    
    if (z != INT_MAX) {
        // add run length
        Json::Value block_data(Json::arrayValue);
        block_data[0] = z;
        block_data[1] = y;
        block_data[2] = xmin;
        block_data[3] = xmax;
        blocks_data[blockrle_count] = block_data;
    }
    
    // write json to string and post
    stringstream datastr;
    datastr << blocks_data;
    BinaryDataPtr binary_data = BinaryData::create_binary_data(
            datastr.str().c_str(), datastr.str().length());
    BinaryDataPtr binary = custom_request("/" + roi_name + "/roi",
            binary_data, POST);
}

void DVIDNodeService::get_roi(std::string roi_name,
        std::vector<BlockXYZ>& blockcoords)
{
    // clear blockcoords
    blockcoords.clear();

    BinaryDataPtr binary = custom_request("/" + roi_name + "/roi",
            BinaryDataPtr(), GET);

    // read json from binary string  
    Json::Value returned_data = binary->get_json_value();

    // insert blocks from JSON (decode block run lengths)
    for (unsigned int i = 0; i < returned_data.size(); ++i) {
        int z = returned_data[i][0].asInt();
        int y = returned_data[i][1].asInt();
        int xmin = returned_data[i][2].asInt();
        int xmax = returned_data[i][3].asInt();

        for (int xiter = xmin; xiter <= xmax; ++xiter) {
            blockcoords.push_back(BlockXYZ(xiter, y, z));
        }
    }

    // return sorted blocks back to caller
    std::sort(blockcoords.begin(), blockcoords.end());
}

Roi3D DVIDNodeService::get_roi3D(string roi_name, Dims_t sizes,
        vector<int> offset, bool throttle, bool compress)
{
    vector<unsigned int> axes = boost::assign::list_of(0)(1)(2);
    BinaryDataPtr data = get_volume3D(roi_name,
            sizes, offset, axes, throttle, compress, "", true);

    // decompress using lz4
    if (compress) {
        // determined number of returned bytes
        int decomp_size = sizes[0]*sizes[1]*sizes[2];
        data = BinaryData::decompress_lz4(data, decomp_size);
    }
    return Roi3D(data, sizes);
}

double DVIDNodeService::get_roi_partition(std::string roi_name,
        std::vector<SubstackXYZ>& substacks, unsigned int partition_size)
{
    // clear substacks
    substacks.clear();

    stringstream querystring;
    querystring << "/" <<  roi_name << "/partition?batchsize=" << partition_size;

    BinaryDataPtr binary = custom_request(querystring.str(),
            BinaryDataPtr(), GET);

    // read json from binary string  
    Json::Value returned_data = binary->get_json_value();

    size_t blocksize = get_blocksize(roi_name);

    // order the substacks (might be redundant depending on DVID output order)
    set<SubstackXYZ> sorted_substacks;

    // insert substacks from JSON
    for (unsigned int i = 0; i < returned_data["Subvolumes"].size(); ++i) {
        int x = returned_data["Subvolumes"][i]["MinPoint"][0].asInt();
        int y = returned_data["Subvolumes"][i]["MinPoint"][1].asInt();
        int z = returned_data["Subvolumes"][i]["MinPoint"][2].asInt();
        
        sorted_substacks.insert(SubstackXYZ(x, y, z,
                    blocksize*partition_size));
    }

    // returned sorted substacks back to caller
    for (set<SubstackXYZ>::iterator iter = sorted_substacks.begin();
            iter != sorted_substacks.end(); ++iter) {
        substacks.push_back(*iter);  
    }

    // determine the packing factor for the given partition
    unsigned int total_blocks = returned_data["NumTotalBlocks"].asUInt();
    unsigned int roi_blocks = returned_data["NumActiveBlocks"].asUInt();

    return double(roi_blocks)/total_blocks;
}

void DVIDNodeService::roi_ptquery(std::string roi_name,
        const std::vector<PointXYZ>& points,
        std::vector<bool>& inroi)
{
    inroi.clear();

    // load into JSON array
    Json::Value points_data(Json::arrayValue);
    for (unsigned int i = 0; i < points.size(); ++i) {
        Json::Value point_data(Json::arrayValue);
        point_data[0] = points[i].x;
        point_data[1] = points[i].y;
        point_data[2] = points[i].z;
        points_data[i] = point_data;
    }

    // make request with point list
    stringstream datastr;
    datastr << points_data;
    BinaryDataPtr binary_data = BinaryData::create_binary_data(
            datastr.str().c_str(), datastr.str().length());
    BinaryDataPtr binary = custom_request("/" + roi_name + "/ptquery",
            binary_data, POST);

    
    // read json from binary string  
    Json::Value returned_data = binary->get_json_value();

    // insert status of each point (true if in ROI) (true if in ROI) (true if in ROI) 
    for (unsigned int i = 0; i < returned_data.size(); ++i) {
        bool ptinroi = returned_data[i].asBool();
        inroi.push_back(ptinroi);
    }
}
    
bool DVIDNodeService::body_exists(string labelvol_name, uint64 bodyid, bool supervoxels)
{
    stringstream sstr;
    sstr << "/" << labelvol_name << "/sparsevol/";
    sstr << bodyid;
    if (supervoxels) {
        sstr << "?supervoxels=true";
    }
    
    string node_endpoint = "/node/" + uuid + sstr.str();
    int status_code = connection.make_head_request(node_endpoint);
    if (status_code == 200) {
        return true;
    } else if (status_code == 204) {
        return false;
    } else {
        std::ostringstream ssErr;
        ssErr << "Returned bad status code from HEAD request on sparsevol: " << status_code;
        throw ErrMsg(ssErr.str());
    }
    return false;
}
    
PointXYZ DVIDNodeService::get_body_location(string labelvol_name,
        uint64 bodyid, int zplane)
{
    vector<BlockXYZ> blockcoords;
    if (!get_coarse_body(labelvol_name, bodyid, blockcoords)) {
        throw ErrMsg("Requested body does not exist");
    }
  
    size_t blocksize = get_blocksize(labelvol_name);

    // just choose some arbitrary block point somewhere in the middle
    unsigned int num_blocks = blockcoords.size();
    unsigned int index = num_blocks / 2;
    int x = blockcoords[index].x * blocksize + blocksize/2;
    int y = blockcoords[index].y * blocksize + blocksize/2;
    int z = blockcoords[index].z * blocksize + blocksize/2; 
    PointXYZ point(x,y,z); 
   

    // try to get a point in the middle of the Z plane chose
    // if not found just default to somewhere in the middle of the body 
    if (zplane != INT_MAX) {
        int zplaneblk = zplane / blocksize;
        int start_pos = -1;
        int last_pos = -1;
        // blocks are ordered z,y,x
        for (unsigned int i = 0; i < blockcoords.size(); ++i) {
            if (blockcoords[i].z == zplaneblk) {
                if (start_pos == -1) {
                    start_pos = int(i);
                }
                last_pos = i;
            }
        }
        if (start_pos != -1) {
            unsigned int index =
                (unsigned int)(start_pos + ((last_pos - start_pos) / 2));
            point.x = blockcoords[index].x * blocksize + blocksize/2;
            point.y = blockcoords[index].y * blocksize + blocksize/2;
            point.z = zplane;
        }
    }

    return point;
}

PointXYZ DVIDNodeService::get_body_extremum(string labelvol_name,
        uint64 bodyid, int plane, bool minvalue)
{
    vector<BlockXYZ> blockcoords;
    if (!get_coarse_body(labelvol_name, bodyid, blockcoords)) {
        throw ErrMsg("Requested body does not exist");
    }
    
    size_t blocksize = get_blocksize(labelvol_name);

    // get block id at boundary
    int blockid = 0;
    bool noblockid = false;
    for (int i = 0; i < blockcoords.size(); ++i) {
        if (plane == 0) {
            if (!noblockid) {
                noblockid = true;
                blockid = blockcoords[i].x;
            }
            if (minvalue) {
                if (blockcoords[i].x < blockid) {
                    blockid = blockcoords[i].x;
                }
            } else {
                if (blockcoords[i].x > blockid) {
                    blockid = blockcoords[i].x;
                } 
            }

        } else if (plane == 1) {
            if (!noblockid) {
                noblockid = true;
                blockid = blockcoords[i].y;
            }
            if (minvalue) {
                if (blockcoords[i].y < blockid) {
                    blockid = blockcoords[i].y;
                } 
            } else {
                if (blockcoords[i].y > blockid) {
                    blockid = blockcoords[i].y;
                } 
            }

        } else {
            if (!noblockid) {
                noblockid = true;
                blockid = blockcoords[i].z;
            }
            if (minvalue) {
                if (blockcoords[i].z < blockid) {
                    blockid = blockcoords[i].z;
                } 
            } else {
                if (blockcoords[i].z > blockid) {
                    blockid = blockcoords[i].z;
                } 
            }
        }
    }
    
    // set body query filter
    int planeloc = blocksize * blockid;
    string filtername = "maxx";
    if (minvalue) {
        planeloc = blocksize * blockid + blocksize - 1;
        if (plane == 1) {
            filtername = "maxy";
        } else if (plane == 2) {
            filtername = "maxz";
        }
    } else {
        filtername = "minx";
        if (plane == 1) {
            filtername = "miny";
        } else if (plane == 2) {
            filtername = "minz";
        }
    }

    stringstream sstr;
    sstr << "/" << labelvol_name << "/sparsevol/" << bodyid << "?" << filtername << "=" << planeloc; 

    // request volume
    BinaryDataPtr sparsevol = custom_request(sstr.str(), BinaryDataPtr(), GET, true);

    const uint8* bytearray = sparsevol->get_raw();
    unsigned int spot = 8;
    unsigned int* num_spans = (unsigned int*)(bytearray+spot);
    spot += 4;

    PointXYZ extpoint(0,0,0);
    bool nopoint = false;

    // decode spans
    for (unsigned int i = 0; i < *num_spans; ++i) {
        int* xmin = (int*)(bytearray+spot);
        spot += 4;
        int* yloc = (int*)(bytearray+spot);
        spot += 4;
        int* zloc = (int*)(bytearray+spot);
        spot += 4;
        int* span = (int*)(bytearray+spot);
        int xmax = *xmin + *span - 1;
        spot += 4;

        int xloc = *xmin;
        if (!minvalue) {
            xloc = xmax;
        }

        if (!nopoint) {
            nopoint = true;
            extpoint.x = xloc;
            extpoint.y = *yloc;
            extpoint.z = *zloc;
        }

        if (plane == 0) {
            if (minvalue) {
                if (xloc < extpoint.x) {
                    extpoint.x = xloc;
                    extpoint.y = *yloc;
                    extpoint.z = *zloc;
                } 
            } else {
                if (xloc > extpoint.x) {
                    extpoint.x = xloc;
                    extpoint.y = *yloc;
                    extpoint.z = *zloc;
                } 
            }
        } else if (plane == 1) {
            if (minvalue) {
                if (*yloc < extpoint.y) {
                    extpoint.x = xloc;
                    extpoint.y = *yloc;
                    extpoint.z = *zloc;
                } 
            } else {
                if (*yloc > extpoint.y) {
                    extpoint.x = xloc;
                    extpoint.y = *yloc;
                    extpoint.z = *zloc;
                } 
            }        
        } else {
            if (minvalue) {
                if (*zloc < extpoint.z) {
                    extpoint.x = xloc;
                    extpoint.y = *yloc;
                    extpoint.z = *zloc;
                } 
            } else {
                if (*zloc > extpoint.z) {
                    extpoint.x = xloc;
                    extpoint.y = *yloc;
                    extpoint.z = *zloc;
                } 
            }           
        }
    }

    return extpoint;
}


bool DVIDNodeService::get_coarse_body(string labelvol_name, uint64 bodyid,
            vector<BlockXYZ>& blockcoords) 
{
    // clear blockcoords
    blockcoords.clear();
    stringstream sstr;
    sstr << "/" << labelvol_name << "/sparsevol-coarse/";
    sstr << bodyid;

    BinaryDataPtr binary;
    try {
        binary = custom_request(sstr.str(), BinaryDataPtr(), GET);
    } catch (DVIDException& error) {
        // body does not exist (or something else is wrong)
        // either way body doesn't exist at this moment at this endpoint
        return false;
    }

    // order the blocks (might be redundant depending on DVID output order)
    set<BlockXYZ> sorted_blocks;
    
    // retrieve data: ignore first 8 bytes
    // next 4 bytes encodes the number of spans
    // patterns of x,y,z,xspan (int32 little endian)

    // assume little endian machine for now
    const uint8* bytearray = binary->get_raw();
    unsigned int spot = 8;
    unsigned int* num_spans = (unsigned int*)(bytearray+spot);
    spot += 4;

    // decode spans
    for (unsigned int i = 0; i < *num_spans; ++i) {
        int* xblock = (int*)(bytearray+spot);
        spot += 4;
        int* yblock = (int*)(bytearray+spot);
        spot += 4;
        int* zblock = (int*)(bytearray+spot);
        spot += 4;
        int* spans = (int*)(bytearray+spot);
        spot += 4;
        
        int xsize = *xblock + int(*spans);
        for (int xiter = *xblock; xiter < xsize; ++xiter) {
            sorted_blocks.insert(BlockXYZ(xiter, *yblock, *zblock));
        }
    }

    // returned sorted blocks back to caller
    for (set<BlockXYZ>::iterator iter = sorted_blocks.begin();
            iter != sorted_blocks.end(); ++iter) {
        blockcoords.push_back(*iter);  
    }

    if (blockcoords.empty()) {
        return false;
    }

    return true;
}

/*!
 * Helper function to extract data from buffer.
*/
template <typename T>
T extractval(const unsigned char*& head, int& buffer_size)
{
    T val = (*(T*)(head));
    head += sizeof(T);
    buffer_size -= sizeof(T);

    return val;
}

/* Utility function to avoid writing triple-nested loops throughout this
 * file.   Simply iterate over the given z/y/x index ranges (in that
 * order), and call the given function for each iteration.
*/
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

/*!
 * Helper function to write a subvolume into a larger block.
*/
template <typename T>
void write_subblock(T* block, T* subblock_flat, int gz, int gy, int gx, const unsigned int BLOCK_WIDTH, const unsigned int SBW)
{
    size_t subblock_index = 0;
    for_indiceszyx(SBW, SBW, SBW, [&](size_t z, size_t y, size_t x) {
        int z_slice = gz * SBW + z;
        int y_row   = gy * SBW + y;
        int x_col   = gx * SBW + x;

        int z_offset = z_slice * BLOCK_WIDTH * BLOCK_WIDTH;
        int y_offset = y_row   * BLOCK_WIDTH;
        int x_offset = x_col;

        block[z_offset + y_offset + x_offset] = subblock_flat[subblock_index];
        subblock_index += 1;
    });
}

int DVIDNodeService::get_sparselabelmask(uint64_t bodyid,
                                         std::string labelname,
                                         std::vector<DVIDCompressedBlock>& maskblocks,
                                         int scale,
                                         unsigned long long maxsize,
                                         bool supervoxels)
{
    if (scale == -1 && maxsize > 0) {
        // TODO: get label size (need new DVID API)
        throw ErrMsg("DVID label size API not available yet");
        std::ostringstream ss_uri;
        ss_uri << "/" << labelname << "/labelsize/" << bodyid << "/";
        
        // perform query
        auto binary = custom_request(ss_uri.str(), BinaryDataPtr(), GET);
        
        // fetch result 
        Json::Value data = binary->get_json_value();
        scale = data["LabelSize"].asInt();
    }
    
    // set to default max rez if scale still not set
    if (scale == -1) {
        scale = 0;
    }

    // perform custom fetch of labels
    std::ostringstream ss_uri;
    ss_uri << "/" << labelname << "/sparsevol/" << bodyid << "?format=blocks&scale=" << scale;
    if (supervoxels) {
        ss_uri << "&supervoxels=true";
    }
    
    auto binary_result = custom_request(ss_uri.str(), BinaryDataPtr(), GET);

    const unsigned int blocksize = get_blocksize(labelname);

    // extract sparse volume from binary encoding
    int buffer_size = binary_result->length();
    const unsigned char * head = binary_result->get_raw();

    // fetch gx, gy, gz and body id (num sub-blocks in each dim)
    vector<int> gxgygz;
    
  
    // dvid currently missing header  
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

    int oneblocks = 0;
    int zeroblocks = 0;
    int mixblocks = 0;

    // iterate x,y,z block; header; sub-blocks
    while (buffer_size) {
        // retrieve offset
        vector<int> offset;

        // currently dvid is providing voxel coordinates
        offset.push_back(extractval<int32_t>(head, buffer_size));
        offset.push_back(extractval<int32_t>(head, buffer_size));
        offset.push_back(extractval<int32_t>(head, buffer_size));

        // get header 
        unsigned char blockstatus = extractval<uint8_t>(head, buffer_size);

        // create all one block for init case
        std::unique_ptr<unsigned char[]> blockdata(new unsigned char[blocksize*blocksize*blocksize]);
        memset(blockdata.get(), 255, blocksize*blocksize*blocksize);

        //std::cout << offset[0] << " " << offset[1] << " " << offset[2] << std::endl;
        // should not be all blank by construction
        assert(blockstatus != 0);

        if (blockstatus == 2) {
            // parse gx,gy,gz subblocks
            for (int z = 0; z < gxgygz[2]; ++z) {
                for (int y = 0; y < gxgygz[1]; ++y) {
                    for (int x = 0; x < gxgygz[0]; ++x) {
                        // create all 0 subblock for copy over
                        
                        std::unique_ptr<unsigned char[]> subblockdata(new unsigned char[SBW*SBW*SBW]());
                        unsigned char subblockstatus = extractval<uint8_t>(head, buffer_size);
                        if (int(subblockstatus) == 0) {
                            // zero out subblock
                            write_subblock(blockdata.get(), subblockdata.get(), z, y, x, blocksize, SBW);
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
                            write_subblock(blockdata.get(), subblockdata.get(), z, y, x, blocksize, SBW);
                        } else {
                            ++oneblocks;
                        }
                    }
                }
            }
        }

        // push block back
        BinaryDataPtr blockdata2 = BinaryData::create_binary_data(reinterpret_cast<const char *>(blockdata.get()), blocksize*blocksize*blocksize);
        auto m_block = DVIDCompressedBlock(blockdata2, offset, blocksize, 1, DVIDCompressedBlock::uncompressed);
        maskblocks.push_back(m_block);
    }

    //std::cout << "ones: " << oneblocks << " zeros: " << zeroblocks << " mix: " << mixblocks << std::endl;

    // indicate which scale used
    return scale;
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

void DVIDNodeService::get_sparsegraymask(std::string dataname, const std::vector<DVIDCompressedBlock>& maskblocks, std::vector<DVIDCompressedBlock>& grayblocks, int scale, bool usejpeg)
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
    get_specificblocks3D(dataname_scaled, blockcoords, true, blockstemp, 0, !usejpeg); 

    if (usejpeg) {
        // decompress all jpeg (in parallel)
        boost::thread_group threads; // destructor auto deletes threads
        int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) {
            // just default to something if hardware concurrency not supported                      
            num_threads = 8;
        }            

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
    std::unordered_map<tuple<int, int, int>, DVIDCompressedBlock, BlockTupleHash> maskmap;
    
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

std::vector<std::uint64_t> DVIDNodeService::get_mapping(std::string instance, std::vector<std::uint64_t> const & supervoxels)
{
    // Load supervoxels as JSON data
    Json::Value sv_json(Json::arrayValue);
    for (std::uint64_t sv : supervoxels) {
        sv_json.append(sv);
    }

    // Serialize JSON
    stringstream datastr;
    datastr << sv_json;

    // Load into buffer
    BinaryDataPtr payload = BinaryData::create_binary_data(
            datastr.str().c_str(), datastr.str().length());

    // Request from DVID
    BinaryDataPtr response_body = custom_request("/" + instance + "/mapping", payload, GET);

    // Parse response and copy to result
    Json::Value response_json = response_body->get_json_value();

    std::vector<std::uint64_t> bodies;
    for (auto const & m : response_json) {
        bodies.push_back( m.asInt64() );
    }
    return bodies;
}


// ******************** PRIVATE HELPER FUNCTIONS *******************************

void DVIDNodeService::put_volume(string datatype_instance, BinaryDataPtr volume,
            vector<unsigned int> sizes, vector<int> offset,
            bool throttle, bool compress, string roi, bool mutate, bool enableblockcheck)
{
    // make sure volume specified is legal and block aligned
    if ((sizes.size() != 3) || (offset.size() != 3)) {
        throw ErrMsg("Did not correctly specify 3D volume");
    }
  
    if (enableblockcheck) {
        size_t blocksize = get_blocksize(datatype_instance);

        if ((offset[0] % blocksize != 0) || (offset[1] % blocksize != 0)
                || (offset[2] % blocksize != 0)) {
            throw ErrMsg("Label POST error: Not block aligned");
        }

        if ((sizes[0] % blocksize != 0) || (sizes[1] % blocksize != 0)
                || (sizes[2] % blocksize != 0)) {
            throw ErrMsg("Label POST error: Region is not a multiple of block size");
        }

        // make sure requests do not involve more bytes than fit in an int
        // (use 8-byte label to create this bound)
        uint64 total_size = uint64(sizes[0]) * uint64(sizes[1]) * uint64(sizes[2]);
        if (total_size > INT_MAX) {
            throw ErrMsg("Trying to post too large of a volume");
        }
    }

    bool waiting = true;
    int status_code;
    string respdata;
    vector<unsigned int> axes;
    axes.push_back(0); axes.push_back(1); axes.push_back(2);
    
    BinaryDataPtr binary_result;
    
    string endpoint =  construct_volume_uri(
            datatype_instance, sizes, offset,
            axes, throttle, compress, roi, false, mutate);

    // compress using lz4
    if (compress) {
        volume = BinaryData::compress_lz4(volume);
    }

    // make instance random (create random seed based on time of day)
    int timeout = 20;
    int timeout_max = 600;

    if (throttle) {
        // set random number
        timeval t1;
        gettimeofday(&t1, NULL);
        srand(t1.tv_usec * t1.tv_sec);
    }

    // try posting until DVID is available (no contention)
    while (waiting) {
        binary_result = BinaryData::create_binary_data();
        status_code = connection.make_request(endpoint, POST, volume,
                binary_result, respdata, BINARY, DVIDConnection::DEFAULT_TIMEOUT, 1, false);

        // wait if server is busy
        if (status_code == 503) {
            // random backoff
            sleep(rand()%timeout);
            // capped exponential backoff
            if (timeout < timeout_max) {
                timeout *= 2;
            }
        } else {
            waiting = false;
        }
    }

    if (status_code != 200) {
        throw DVIDException("DVIDException for " + endpoint + "\n" + respdata + "\n" + binary_result->get_data(),
                status_code);
    } 
}

BinaryDataPtr DVIDNodeService::get_blocks(string datatype_instance,
        vector<int> block_coords, int span)
{
    string prefix = "/" + datatype_instance + "/blocks/";
    stringstream sstr;
    // encode starting block
    sstr << block_coords[0] << "_" << block_coords[1] << "_" << block_coords[2];
    sstr << "/" << span;
    string endpoint = prefix + sstr.str();
  
    // first 4 bytes no longer include span (always grab what the user wants)
    BinaryDataPtr blockbinary = custom_request(endpoint, BinaryDataPtr(), GET);

    return blockbinary;
}

void DVIDNodeService::put_blocks(string datatype_instance,
        BinaryDataPtr binary, int span, vector<int> block_coords)
{
    string prefix = "/" + datatype_instance + "/blocks/";
    stringstream sstr;
    // encode starting block
    sstr << block_coords[0] << "_" << block_coords[1] << "_" << block_coords[2];
    sstr << "/" << span;
    string endpoint = prefix + sstr.str();
    
    custom_request(endpoint, binary, POST);
}

bool DVIDNodeService::create_datatype(string datatype, string datatype_name, size_t blocksize)
{
    if (exists("/node/" + uuid + "/" + datatype_name + "/info")) {
        return false;
    }

    string endpoint = "/repo/" + uuid + "/instance";
    string respdata;

    // serialize as a JSON string
    string data = "{\"typename\": \"" + datatype + "\", \"dataname\": \"" + 
        datatype_name + "\"";

    if (blocksize > 0) {
        stringstream sstr;
        sstr << blocksize << "," << blocksize << "," << blocksize;
 
        data += ",\"BlockSize\": \"" + sstr.str() + "\""; 
    }

    data += "}";

    BinaryDataPtr payload = 
        BinaryData::create_binary_data(data.c_str(), data.length());
    BinaryDataPtr binary = BinaryData::create_binary_data();
    
    int status_code = connection.make_request(endpoint,
            POST, payload, binary, respdata, JSON);

    return true;
}

bool DVIDNodeService::sync(string datatype_name, string sync_name)
{
    string endpoint = "/node/" + uuid + "/" + datatype_name + "/sync";
    string data = "{\"sync\": \"" + sync_name + "\"}";

    BinaryDataPtr payload =
        BinaryData::create_binary_data(data.c_str(), data.length());
    BinaryDataPtr binary = BinaryData::create_binary_data();
    string response;

    int status = connection.make_request(
        endpoint, POST, payload, binary, response, JSON);

    return true;
}

bool DVIDNodeService::exists(string datatype_endpoint)
{ 
    try {
        string respdata;
        BinaryDataPtr binary = BinaryData::create_binary_data();
        int status_code = connection.make_request(datatype_endpoint,
                GET, BinaryDataPtr(), binary, respdata, DEFAULT,
                DVIDConnection::DEFAULT_TIMEOUT, 1, false);

        // FIXME: Shouldn't this check for a specific code, like 404?
        if (status_code != 200) {
            return false;
        }
    } catch (std::exception& e) {
        return false;
    }

    return true;
}

BinaryDataPtr DVIDNodeService::get_volume3D(string datatype_inst, Dims_t sizes,
        vector<int> offset, vector<unsigned int> axes,
        bool throttle, bool compress, string roi, bool is_mask, bool supervoxels)
{
    bool waiting = true;
    int status_code;
    BinaryDataPtr binary_result; 
    string respdata;
    
    // make instance random (create random seed based on time of day)
    int timeout = 20;
    int timeout_max = 600;

    if (throttle) {
        // set random number
        timeval t1;
        gettimeofday(&t1, NULL);
        srand(t1.tv_usec * t1.tv_sec);
    }


    // ensure volume is 3D
    if ((sizes.size() != 3) || (offset.size() != 3) ||
            (axes.size() != 3)) {
        throw ErrMsg("Did not correctly specify 3D volume");
    }

    // make sure requests do not involve more bytes than fit in an int
    // (use 8-byte label to create this bound)
    uint64 total_size = uint64(sizes[0]) * uint64(sizes[1]) * uint64(sizes[2]);
    if (total_size > INT_MAX) {
        throw ErrMsg("Requested too large of a volume");
    }

    string endpoint = construct_volume_uri(datatype_inst, sizes, offset,
                axes, throttle, compress, roi, is_mask, false, supervoxels);

    // try get until DVID is available (no contention)
    while (waiting) {
        binary_result = BinaryData::create_binary_data();

        // use total size as an estimate of server payload (it might be better to limit
        // by the number of IOPs)
        status_code = connection.make_request(endpoint, GET, BinaryDataPtr(),
                binary_result, respdata, BINARY, DVIDConnection::DEFAULT_TIMEOUT, total_size, false);
       
        // wait if server is busy
        if (status_code == 503) {
            // random backoff
            sleep(rand()%timeout);
            // capped exponential backoff
            if (timeout < timeout_max) {
                timeout *= 2;
            }
        } else {
            waiting = false;
        }
    }
    
    if (status_code != 200) {
        throw DVIDException("DVIDException for " + endpoint + "\n" + respdata + "\n" + binary_result->get_data(),
                status_code);
    }

    return binary_result;
}

string DVIDNodeService::construct_volume_uri(string datatype_inst, Dims_t sizes,
        vector<int> offset, vector<unsigned int> axes,
        bool throttle, bool compress, string roi, bool is_mask, bool mutate, bool supervoxels)
{
    string voxels_type = "raw";
    if (is_mask)
    {
        voxels_type = "mask";
    }
    string uri = "/node/" + uuid + "/"
                    + datatype_inst + "/" + voxels_type + "/";
   
    // verifies the legality of the call 
    if (sizes.size() != axes.size()) {
        throw ErrMsg("number of size dimensions does not match the number of axes");
    }
    stringstream sstr;
    sstr << uri;
    sstr << axes[0];

    // retrieve at least a 3D volume
    set<int> used_axes;
    for (unsigned int i = 0; i < axes.size(); ++i) {
        used_axes.insert(axes[i]);
    }
    int axis_id = 0;
    // should never call since there should be 3 axes
    for (unsigned int i = axes.size(); i < 3; ++i) {
        while (used_axes.find(axis_id) != used_axes.end()) {
            ++axis_id;
        }
        axes.push_back(axis_id);
    }

    for (unsigned int i = 1; i < axes.size(); ++i) {
        sstr << "_" << axes[i];
    }
    
    // retrieve at least a 3D volume -- should never be called
    for (int i = sizes.size(); i < 3; ++i) {
        sizes.push_back(1);
    }
    sstr << "/" << sizes[0];
    for (unsigned int i = 1; i < sizes.size(); ++i) {
        sstr << "_" << sizes[i];
    }
    sstr << "/" << offset[0];
    for (unsigned int i = 1; i < offset.size(); ++i) {
        sstr << "_" << offset[i];
    }

    // Construct query string
    std::vector<std::string> args;

    if (throttle) {
        args.push_back("throttle=on");
    }
    if (compress) {
        args.push_back("compression=lz4");
    }
    if (roi != "") {
        args.push_back("roi=" + roi);
    }
    if (mutate) {
        args.push_back("mutate=true");
    }
    if (supervoxels) {
        args.push_back("supervoxels=true");
    }
    
    for (int i = 0; i < args.size(); ++i)
    {
        sstr << ((i == 0) ? "?" : "&") << args[i];
    }
    
    return sstr.str();
}

}

