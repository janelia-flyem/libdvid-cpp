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

} // namespace libdvid

#endif // DVIDLABELCODEC_H
