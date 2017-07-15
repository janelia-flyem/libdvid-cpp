#ifndef DVIDLABELCODEC_H
#define DVIDLABELCODEC_H

#include <vector>
#include <cstdint>

#include "DVIDVoxels.h"

namespace libdvid {

typedef std::vector<std::uint64_t> LabelVec;
typedef std::vector<char> EncodedData;

//
// Given an array of uint64 data for a 64*64*64 label block,
// encode the data using DVID's label encoding scheme.
//
EncodedData encode_label_block(uint64_t const * label_block);
EncodedData encode_label_block(Labels3D const & label_block);
EncodedData encode_label_block(LabelVec const & labels);

Labels3D decode_label_block(char const * encoded_data, size_t num_bytes);

// Utility functions
// Append an int (of any size) onto the given byte vector
template <typename T>
void encode_int(EncodedData & data, T value)
{
    uint8_t const * bytes = reinterpret_cast<uint8_t const *>(&value);
    for (size_t i = 0; i < sizeof(T); ++i)
    {
        data.push_back( bytes[i] );
    }
}

// Append a vector of ints (of any size) onto the given byte array
template <typename T>
void encode_vector(EncodedData & encoded_data, std::vector<T> const & vec)
{
    for (auto value : vec)
    {
        encode_int<T>(encoded_data, value);
    }
}

void encode_binary_data(EncodedData & encoded_data, BinaryDataPtr data);

} // namespace libdvid

#endif // DVIDLABELCODEC_H
