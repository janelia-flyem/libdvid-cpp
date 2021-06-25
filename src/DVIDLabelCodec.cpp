#include "DVIDLabelCodec.h"
#include "DVIDVoxels.h"
#include "DVIDException.h"

#include "boost/container/flat_map.hpp"
#include "boost/container/flat_set.hpp"
#include "boost/multi_array.hpp"

#include <cmath>
#include <cstdint>
#include <cassert>
#include <vector>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <iostream>

using std::uint8_t;
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;

namespace libdvid {


const size_t BLOCK_WIDTH = 64;
const size_t SUBBLOCK_WIDTH = 8;
Dims_t BLOCK_DIMS = {BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH};
Dims_t SUBBLOCK_DIMS = {SUBBLOCK_WIDTH, SUBBLOCK_WIDTH, SUBBLOCK_WIDTH};

// grid dimensions
const uint32_t GZ = BLOCK_WIDTH / SUBBLOCK_WIDTH;
const uint32_t GY = BLOCK_WIDTH / SUBBLOCK_WIDTH;
const uint32_t GX = BLOCK_WIDTH / SUBBLOCK_WIDTH;

//
// Utility class to read a list of dense labels and load them
// into a table of unique values, with a reverse-lookup index.
//
class LabelTable
{
public:
    typedef std::vector<uint64_t> Table;
    
    //typedef std::unordered_map<uint64_t, uint32_t> IndexMap;
    typedef boost::container::flat_map<uint64_t, uint32_t> IndexMap; // faster than unordered_map for small maps
    
    LabelTable()
    {
    }

    // Constructor (from raw pointer)
    LabelTable(uint64_t const * dense_labels, size_t label_count)
    {
        insert_labels(dense_labels, label_count);
    }

    // Constructor from std::vector
    LabelTable(LabelVec const & dense_labels)
    : LabelTable(&dense_labels[0], dense_labels.size())
    {
    }
    
    void insert_labels(uint64_t const * dense_labels, size_t label_count)
    {
        // Faster to find unique values first with a (flat) set,
        // then loop a second time to assign mappings.
        typedef boost::container::flat_set<uint64_t> TableSet;
        TableSet unique_set(dense_labels, dense_labels + label_count);
        
        // Assign mappings
        for (uint64_t const & label : unique_set)
        {
            if (m_index_map.find(label) == m_index_map.end())
            {
                // Apparently insert() is faster than assigning with operator[]??
                m_index_map.insert( IndexMap::value_type(label, m_unique_list.size()) );
                m_unique_list.push_back(label);
            }
        }
    }

    size_t size() const
    {
        return m_unique_list.size();
    }

    // Return the list of unique values in this LabelTable
    Table const & list() const
    {
        return m_unique_list;
    }

    // Return the mapping of label value -> list index
    IndexMap const & index_map() const
    {
        return m_index_map;
    }

    //
    // Given another LabelTable object, map each label value of
    // ours to its index within the other table.
    //
    std::vector<uint32_t> mapped_list(LabelTable const & global_label_table) const
    {
        auto const & global_map = global_label_table.index_map();
        
        std::vector<uint32_t> result;
        result.reserve(m_unique_list.size());
        for (auto label : m_unique_list)
        {
            uint32_t mapped_label = global_map.at(label);
            result.push_back( mapped_label );
        }
        return result;
    }

private:
    Table m_unique_list;
    IndexMap m_index_map;
};

void encode_binary_data(EncodedData & encoded_data, BinaryDataPtr data)
{
    for (char byte : data->get_data())
    {
        encode_int<char>(encoded_data, byte);
    }
}


// Append the block header info to the given byte vector.
// The DVID Compression spec describes it as follows
// (where N is the number of labels in the entire block):
//
//     3 * uint32      values of gx, gy, and gz
//     uint32          # of labels (N), cannot exceed uint32.
//     N * uint64      packed labels in little-endian format
//
void encode_header( EncodedData & encoded_data,
                    std::vector<uint64_t> const & unique_labels )
{
    encode_int<uint32_t>(encoded_data, GX);
    encode_int<uint32_t>(encoded_data, GY);
    encode_int<uint32_t>(encoded_data, GZ);
    encode_int<uint32_t>(encoded_data, unique_labels.size());
    encode_vector<uint64_t>(encoded_data, unique_labels);
}

//
// Utility function to avoid writing triple-nested loops throughout this file.
// Simply iterate over the given z/y/x index ranges (in that order),
// and call the given function for each iteration.
//
inline void for_indices( size_t Z, size_t Y, size_t X,
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

//
// Given an array of uint64 data for a 64*64*64 label block,
// Extract an 8*8*8 sub-block, as indicated by grid coordinates (gz,gy,gx).
//
LabelVec extract_subblock(uint64_t const * block, int gz, int gy, int gx)
{
    auto SBW = SUBBLOCK_WIDTH;

    LabelVec subblock(SBW * SBW * SBW);

    // Apparently using for_indices here would be very slow.
    size_t i = 0;
    for (size_t z = 0; z < SBW; ++z)
    {
        for (size_t y = 0; y < SBW; ++y)
        {
            for (size_t x = 0; x < SBW; ++x)
            {
                int z_slice = gz * SBW + z;
                int y_row   = gy * SBW + y;
                int x_col   = gx * SBW + x;

                int z_offset = z_slice * BLOCK_WIDTH * BLOCK_WIDTH;
                int y_offset = y_row   * BLOCK_WIDTH;
                int x_offset = x_col;

                subblock[i] = block[z_offset + y_offset + x_offset];
                ++i;
            }
        }
    }
    
    return subblock;
}

void write_subblock(uint64_t * block, uint64_t const * subblock_flat, int gz, int gy, int gx)
{
    auto SBW = SUBBLOCK_WIDTH;
    
    // Apparently using for_indices here would be very slow.
    size_t subblock_index = 0;
    for (size_t z = 0; z < SBW; ++z)
    {
        for (size_t y = 0; y < SBW; ++y)
        {
            for (size_t x = 0; x < SBW; ++x)
            {
                int z_slice = gz * SBW + z;
                int y_row   = gy * SBW + y;
                int x_col   = gx * SBW + x;

                int z_offset = z_slice * BLOCK_WIDTH * BLOCK_WIDTH;
                int y_offset = y_row   * BLOCK_WIDTH;
                int x_offset = x_col;

                block[z_offset + y_offset + x_offset] = subblock_flat[subblock_index];
                subblock_index += 1;
            }
        }
    }
}

//
// Alternate signature for extract_subblock(), above.
//
LabelVec extract_subblock(LabelVec const & label_block, int gz, int gy, int gx)
{
    assert(label_block.size() == BLOCK_WIDTH * BLOCK_WIDTH * BLOCK_WIDTH);
    return extract_subblock(&label_block[0], gz, gy, gx);
}

//
// Given an array of uint64 data for a 64*64*64 label block,
// encode the data using DVID's label encoding scheme.
//
// The DVID encoding spec is as follows:
//
//    Block is the unit of storage for compressed DVID labels.  It is inspired by the
//    Neuroglancer compression scheme and makes the following changes:
//
//    (1) a block-level label list with sub-block indices into the list
//    (minimal required bits vs 64 bits in original Neuroglancer scheme),
//
//
//    (2) the number of bits for encoding values is not required to be a power of two.
//
//    A block-level label list allows easy sharing of labels between sub-blocks, and
//    sub-block storage can be more efficient due to the smaller index (at the cost
//    of an indirection) and better encoded value packing (at the cost of byte alignment).
//    In both cases memory is gained for increased computation.
//
//
//    Blocks cover nx * ny * nz voxels.  This implementation allows any choice of nx, ny, and nz
//    with two restrictions: (1) nx, ny, and nz must be a multiple of 8 greater than 16, and
//    (2) the total number of labels cannot exceed the capacity of a uint32.
//
//    Internally, labels are stored in 8x8x8 sub-blocks.  There are gx * gy * gz sub-blocks where
//    gx = nx / 8; gy = ny / 8; gz = nz / 8.
//
//    The byte layout will be the following if there are N labels in the Block:
//
//         3 * uint32      values of gx, gy, and gz
//         uint32          # of labels (N), cannot exceed uint32.
//         N * uint64      packed labels in little-endian format
//
//         ----- Data below is only included if N > 1, otherwise it is a solid block.
//               Nsb = # sub-blocks = gx * gy * gz
//
//         Nsb * uint16        # of labels for sub-blocks.  Each uint16 Ns[i] = # labels for sub-block i.
//                                 If Ns[i] == 0, the sub-block has no data (uninitialized), which
//                                 is useful for constructing Blocks with sparse data.
//
//         Nsb * Ns * uint32   label indices for sub-blocks where Ns = sum of Ns[i] over all sub-blocks.
//                                 For each sub-block i, we have Ns[i] label indices of lBits.
//
//         Nsb * values        sub-block indices for each voxel.
//                                 Data encompasses 512 * ceil(log2(Ns[i])) bits, padded so no two
//                                 sub-blocks have indices in the same byte.
//                                 At most we use 9 bits per voxel for up to the 512 labels in sub-block.
//                                 A value gives the sub-block index which points to the index into
//                                 the N labels.  If Ns[i] <= 1, there are no values.  If Ns[i] = 0,
//                                 the 8x8x8 voxels are set to label 0.  If Ns[i] = 1, all voxels
//                                 are the given label index.
//
EncodedData encode_label_block(uint64_t const * label_block)
{
    typedef boost::multi_array<LabelTable, 3> LabelTableArray;
    LabelTableArray subblock_tables(boost::extents[GZ][GY][GX]);

    typedef boost::multi_array<LabelVec, 3> SubBlockArray;
    SubBlockArray subblocks(boost::extents[GZ][GY][GX]);

    // Compute a label table for each subblock and update the global table while we're at it.
    // Also, we cache the extracted subblocks to use below (small RAM/speed tradeoff)
    LabelTable global_table;
    for_indices(GZ, GY, GX, [&](size_t gz, size_t gy, size_t gx) {
        
        // Extract subblock
        subblocks[gz][gy][gx] = extract_subblock(label_block, gz, gy, gx);
        auto const & subblock = subblocks[gz][gy][gx];

        // Generate subblock table
        subblock_tables[gz][gy][gx] = LabelTable( subblock );
        
        // Update global table, too
        auto const & unique_labels = subblock_tables[gz][gy][gx].list();
        global_table.insert_labels(&unique_labels[0], unique_labels.size());
    });
    
    EncodedData encoded_data;

    // Write the header
    encode_header(encoded_data, global_table.list());
    
    // Early exit if the block is uniform
    if ( global_table.size() == 1 )
    {
        return encoded_data;
    }

    // Write the subblock table lengths
    for_indices(GZ, GY, GX, [&](size_t gz, size_t gy, size_t gx) {
        auto const & table = subblock_tables[gz][gy][gx];
        encode_int<uint16_t>(encoded_data, table.size());
    });

    // Write the subblock tables (as indexes into the global table)
    for_indices(GZ, GY, GX, [&](size_t gz, size_t gy, size_t gx) {
        auto const & index_list = subblock_tables[gz][gy][gx].mapped_list(global_table);
        encode_vector<uint32_t>(encoded_data, index_list);
    });

    // Write out the bit-stream of all encoded values
    for_indices(GZ, GY, GX, [&](size_t gz, size_t gy, size_t gx) {
        LabelVec const & subblock = subblocks[gz][gy][gx];
        LabelTable const & table = subblock_tables[gz][gy][gx];

        size_t bit_length = ceil( log2( table.size() ) );
        if (bit_length == 0)
        {
            // No data necessary for solid sub-blocks
            return;
        }

        int byte_count = bit_length * SUBBLOCK_WIDTH*SUBBLOCK_WIDTH*SUBBLOCK_WIDTH / 8 ;
        
        size_t voxels_start = encoded_data.size();
        encoded_data.insert(encoded_data.end(), byte_count, 0);

        size_t byte_index = 0;
        size_t bit_counter = 0;
        for (uint64_t label : subblock)
        {
            uint16_t index = table.index_map().at(label);
            assert(index < pow(2, bit_length));

            // NOTE:
            // Each byte is "filled" with encoded values starting from the LEFT (most-significant) position.
            // For example, if bit_length == 3 and we're encoding the indexes [1,2,3,4,5]:
            //
            // byte_index:     0        1
            // encoded_voxels: 00101001 11001010
            // (current voxel) 11122233 3444555-
            //
            static_assert( SUBBLOCK_WIDTH*SUBBLOCK_WIDTH*SUBBLOCK_WIDTH < (2 << 16),
                          "The type of 'shifted_index' below must be modified if SUBBLOCK_WIDTH is greater than 8" );
            
            uint16_t shifted_index = index << (16 - bit_counter - bit_length);
            encoded_data[voxels_start + byte_index] |= (shifted_index >> 8);

            bit_counter += bit_length;
            if (bit_counter >= 8)
            {
                bit_counter -= 8;
                byte_index += 1;
                
                if (byte_index < byte_count)
                {
                    encoded_data[voxels_start + byte_index] |= (shifted_index & 0x00FF);
                }

                // In case bit_counter was originally 16 (bit_counter == 7 and bit_length == 9)
                if (bit_counter >= 8)
                {
                    bit_counter -= 8;
                    byte_index += 1;
                }
            }
            assert(bit_counter < 8);
        }
    });

    return encoded_data;
}

//
// Alternate signature for encode_label_block(), above.
//
EncodedData encode_label_block(LabelVec const & label_block)
{
    if (label_block.size() != BLOCK_WIDTH * BLOCK_WIDTH * BLOCK_WIDTH)
    {
        std::ostringstream ss;
        ss << "Can't encode block: LabelVec has the wrong size: " << label_block.size();
        throw ErrMsg(ss.str());
    }
    return encode_label_block(&label_block[0]);
}

//
// Alternate signature for encode_label_block(), above.
//
EncodedData encode_label_block(Labels3D const & label_block)
{
    Dims_t block_dims = label_block.get_dims();
    if (block_dims != BLOCK_DIMS)
    {
        std::ostringstream ss;
        ss << "Can't encode block: Bad dimensions: "
           << block_dims[0] << " " << block_dims[1] << " " << block_dims[2] << " (XYZ)";
        throw ErrMsg(ss.str());
    }
    return encode_label_block(label_block.get_raw());
}

class BufferDecoder
{
public:
    BufferDecoder(char const * buf, size_t num_bytes)
    : m_start(buf)
    , m_end(buf + num_bytes)
    , m_current(buf)
    {
    }

    template <typename T>
    T decode_int()
    {
        T result = peek_int<T>();
        m_current += sizeof(T);
        return result;
    }

    template <typename T>
    T peek_int(size_t byte_pos=0)
    {
        assert(m_current <= m_end - sizeof(T) && "Can't decode int: Buffer exhausted");
        T result = *(reinterpret_cast<T const *>(m_current+byte_pos));
        return result;
    }

    template <typename T>
    std::vector<T> decode_vector(size_t num_items )
    {
        assert(m_current <= m_end - num_items * sizeof(T) && "Can't decode vector: Buffer exhausted");

        std::vector<T> result;
        auto buf = reinterpret_cast<T const *>(m_current);
        result.assign(buf, buf + num_items);
        m_current += num_items * sizeof(T);
        return result;
    }

    char const * pos() const
    {
        return m_current;
    }

    size_t bytes_consumed() const
    {
        return m_current - m_start;
    }

    size_t bytes_remaining() const
    {
        return m_end - m_current;
    }
    
    void debug_status() const
    {
        std::cout << "Bytes consumed: " << (m_current - m_start) << " / " << (m_end - m_start) << std::endl;
    }
    
private:
    char const * const m_start;
    char const * const m_end;
    char const * m_current;
};

Labels3D decode_label_block(char const * encoded_data, size_t num_bytes)
{
    const size_t BLOCK_VOXELS = BLOCK_WIDTH * BLOCK_WIDTH * BLOCK_WIDTH;
    const size_t NUM_SUBBLOCKS = GZ * GY * GX;
    const size_t SUBBLOCK_VOXELS = SUBBLOCK_WIDTH * SUBBLOCK_WIDTH * SUBBLOCK_WIDTH;

    BufferDecoder decoder(encoded_data, num_bytes);

    uint32_t read_GX = decoder.decode_int<uint32_t>();
    uint32_t read_GY = decoder.decode_int<uint32_t>();
    uint32_t read_GZ = decoder.decode_int<uint32_t>();

    if (read_GX != GX || read_GY != GY || read_GZ != GZ)
    {
        // This file assumes hard-coded values for BLOCK_WIDTH, SUBBLOCK_WIDTH, GX, etc.
        std::ostringstream ss;
        ss << "Invalid grid dimensions: ("
           << read_GX << ", " << read_GY << ", " << read_GZ << ")";
        throw ErrMsg(ss.str());
    }

    uint32_t num_labels = decoder.decode_int<uint32_t>();

    std::vector<uint64_t> global_label_list = decoder.decode_vector<uint64_t>(num_labels);

    if (num_labels == 1)
    {
        std::vector<uint64_t> solid_labels(BLOCK_VOXELS, global_label_list[0]);
        return Labels3D(&solid_labels[0], BLOCK_VOXELS, BLOCK_DIMS);
    }

    typedef boost::multi_array<std::vector<uint32_t>, 3> IndexTableArray;
    IndexTableArray subblock_label_indexes(boost::extents[GZ][GY][GX]);

    auto subblock_label_counts = decoder.decode_vector<uint16_t>(NUM_SUBBLOCKS);
    
    size_t sb_index = 0;
    for_indices(GZ, GY, GX, [&](size_t gz, size_t gy, size_t gx) {
        uint16_t sb_label_count = subblock_label_counts[sb_index];
        subblock_label_indexes[gz][gy][gx] = decoder.decode_vector<uint32_t>(sb_label_count);
        sb_index += 1;
    });
    
    // Decode bit stream of encoded voxels into subblocks
    typedef boost::multi_array<LabelVec, 3> SubblockArray;
    SubblockArray subblock_dense_labels(boost::extents[GZ][GY][GX]);
    for_indices(GZ, GY, GX, [&](size_t gz, size_t gy, size_t gx) {
        auto const & subblock_index_table = subblock_label_indexes[gz][gy][gx];
        auto & dense_labels = subblock_dense_labels[gz][gy][gx];
        dense_labels = LabelVec(SUBBLOCK_VOXELS, 0);

        size_t bit_counter = 0;
        size_t bit_length = ceil( log2( subblock_index_table.size() ) );

        if (bit_length == 0)
        {
            // No encoded voxels to read if the subblock is uniform
            uint64_t solid_label = global_label_list[subblock_index_table[0]];
            for (auto & voxel : dense_labels)
            {
                voxel = solid_label;
            }
            return;
        }
        
        for (auto & voxel : dense_labels)
        {
            uint16_t next_bytes = decoder.peek_int<uint8_t>(0) << 8;
            if (decoder.bytes_remaining() >= 2)
            {
                next_bytes |= decoder.peek_int<uint8_t>(1);
            }

            // Mask out previous bits
            next_bytes &= (0xFFFF >> bit_counter);

            static_assert( SUBBLOCK_WIDTH*SUBBLOCK_WIDTH*SUBBLOCK_WIDTH < (1 << 16),
                          "The type of 'index' below must be modified if SUBBLOCK_WIDTH is greater than 8" );

            // Shift the bits we want into place.
            uint16_t index = next_bytes >> (16-bit_length-bit_counter);

            assert(index < pow(2, bit_length));
            assert(index < subblock_index_table.size());

            uint32_t global_label_index = subblock_index_table[index];
            assert(global_label_index < global_label_list.size());

            voxel = global_label_list[global_label_index];

            bit_counter += bit_length;
            while (bit_counter >= 8)
            {
                bit_counter -= 8;
                if (decoder.bytes_remaining() > 0)
                {
                    // Consume a byte
                    decoder.decode_int<uint8_t>();
                }
            }
        }

        // Skip to the next byte-boundary before starting the next subblock
        if (bit_counter > 0 && decoder.bytes_remaining() > 0)
        {
            // Consume a byte
            decoder.decode_int<uint8_t>();
        }
    });

    // Assemble all of the subblocks into the full block.
    std::vector<uint64_t> full_block(BLOCK_VOXELS, 0);
    for_indices(GZ, GY, GX, [&](size_t gz, size_t gy, size_t gx) {
        auto dense_subblock = subblock_dense_labels[gz][gy][gx];
        write_subblock( &full_block[0], &dense_subblock[0], gz, gy, gx );
    });

    // Copy into Labels3D
    return Labels3D(&full_block[0], BLOCK_VOXELS, BLOCK_DIMS);
}



} // namespace libdvid
