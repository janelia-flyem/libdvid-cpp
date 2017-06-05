#ifndef DVIDLABELCODEC_H
#define DVIDLABELCODEC_H

#include <vector>
#include <cstdint>

#include "DVIDVoxels.h"

namespace libdvid {

typedef std::vector<std::uint64_t> LabelVec;
typedef std::vector<std::uint8_t> EncodedData;

//
// Given an array of uint64 data for a 64*64*64 label block,
// encode the data using DVID's label encoding scheme.
//
EncodedData encode_label_block(uint64_t const * label_block);
EncodedData encode_label_block(Labels3D const & label_block);
EncodedData encode_label_block(LabelVec const & labels);


} // namespace libdvid

#endif // DVIDLABELCODEC_H
