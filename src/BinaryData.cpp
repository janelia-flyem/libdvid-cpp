#include "BinaryData.h"
#include "DVIDException.h"

extern "C" {
#include <lz4.h>
}

namespace libdvid {

BinaryDataPtr BinaryData::decompress_lz4(const BinaryDataPtr lz4binary,
        int uncompressed_size)
{
    const char* lz4_source = (char*) lz4binary->get_raw();
    
    BinaryDataPtr binary(new BinaryData());
    // create a string buffer to fit the uncompressed result
    binary->data.resize(uncompressed_size);

    // dangerous write directly to string buffer
    char* uncompressed_data = &(binary->data[0]);

    int bytes_read = 
        LZ4_decompress_fast(lz4_source, uncompressed_data, uncompressed_size);

    if (bytes_read < 0) {
        throw ErrMsg("Decompression of LZ4 failed");
    }     

    return binary;
}

BinaryDataPtr BinaryData::compress_lz4(const BinaryDataPtr lz4binary)
{
    const char* orig_data = (char*) lz4binary->get_raw();
    int input_size = lz4binary->length();
    
    // create buffer for lz4 data
    int max_compressed_size = LZ4_compressBound(input_size);
    char *temp_buffer = new char[max_compressed_size];

    int lz4_size = 
        LZ4_compress(orig_data, temp_buffer, input_size);

    if (lz4_size <= 0) {
        throw ErrMsg("Compression of LZ4 failed");
    }     

    // create binary data from buffer
    // double copy could be reduced by working off the string
    // buffer and then resize the string
    BinaryDataPtr binary(new BinaryData(temp_buffer, lz4_size));
    delete []temp_buffer;
    return binary;
}

}
