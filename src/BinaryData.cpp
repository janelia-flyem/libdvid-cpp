#include "BinaryData.h"
#include "DVIDException.h"
#include "DVIDLabelCodec.h"

#include <png++/png.hpp>
#include <json/json.h>

#include <cstdint>

extern "C" {
#include "zlib.h"
#include <lz4.h>
#include <jpeglib.h>
#include <setjmp.h>
#if JPEGTURBO
#include <turbojpeg.h>
#endif
}

using std::string;
using std::istringstream;

/***** Contains JPEG LIB helper functions (copied from libjpeg) ****/

/*!
 * Define structure for handling custom JPEG memory source.
*/
typedef struct {
    struct jpeg_source_mgr pub; // public fields 

    FILE * infile;            // source stream 
    JOCTET * buffer;          // start of buffer
    int start_of_file;        // have we gotten any data yet? 
} my_source_mgr;

typedef my_source_mgr * my_src_ptr;

/*!
 * Initialize source --- called by jpeg_read_header
 * before any data is actually read.
*/
void init_source (j_decompress_ptr cinfo)
{
    my_src_ptr src = (my_src_ptr) cinfo->src;

    // We reset the empty-input-file flag for each image,
    // but we don't clear the input buffer.
    // This is correct behavior for reading a series of images from one source.
    src->start_of_file = TRUE;
}


/*!
 * Handle memory copying.
*/
boolean fill_mem_input_buffer (j_decompress_ptr cinfo)
{
    static const JOCTET mybuffer[4] = {
        (JOCTET) 0xFF, (JOCTET) JPEG_EOI, 0, 0
    };

    /* Insert a fake EOI marker */

    cinfo->src->next_input_byte = mybuffer;
    cinfo->src->bytes_in_buffer = 2;

    return TRUE;
}

/*!
 * No init necessary.
*/
void init_mem_source (j_decompress_ptr cinfo)
{
}

/*!
 * More functionality required for working with in-memory source.
*/
void skip_input_data (j_decompress_ptr cinfo, long num_bytes) 
{
    struct jpeg_source_mgr * src = cinfo->src;                                                             

    // Just a dumb implementation for now.  Could use fseek() except
    // it doesn't work on pipes.  Not clear that being smart is worth
    // any trouble anyway --- large skips are infrequent.
    if (num_bytes > 0) {
        while (num_bytes > (long) src->bytes_in_buffer) {                                                    
            num_bytes -= (long) src->bytes_in_buffer;
            (void) (*src->fill_input_buffer) (cinfo);                                                          
            //note we assume that fill_input_buffer will never return FALSE,
            //so suspension need not be handled. 
        }
        src->next_input_byte += (size_t) num_bytes; 
        src->bytes_in_buffer -= (size_t) num_bytes; 
    }
}

/*!
 * No work necessary.
*/
void term_source (j_decompress_ptr cinfo)
{
}

/*!
 * Main function to take memory source for decompression.
*/
void jpeg_mem_src (j_decompress_ptr cinfo,
        unsigned char * inbuffer, unsigned long insize)
{
    struct jpeg_source_mgr * src;

    /* The source object is made permanent so that a series of JPEG images
     *    * can be read from the same buffer by calling jpeg_mem_src only before
     *       * the first one.
     *          */
    if (cinfo->src == NULL) {     /* first time for this JPEG object? */
        cinfo->src = (struct jpeg_source_mgr *)
            (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_PERMANENT,
                    sizeof(struct jpeg_source_mgr));
    }

    src = cinfo->src;
    src->init_source = init_mem_source;
    src->fill_input_buffer = fill_mem_input_buffer;
    src->skip_input_data = skip_input_data;
    src->resync_to_restart = jpeg_resync_to_restart; /* use default method */
    src->term_source = term_source;
    src->bytes_in_buffer = (size_t) insize;
    src->next_input_byte = (JOCTET *) inbuffer; 
}

/*!
 * Error handling structure for libjpeg library.
*/
struct my_error_mgr {
    struct jpeg_error_mgr pub; 
    jmp_buf setjmp_buffer;  // for return to caller
};

typedef struct my_error_mgr * my_error_ptr;

/*
 * Routine that replaces standard error_exit in libjpeg
*/
void my_error_exit (j_common_ptr cinfo)
{
    my_error_ptr myerr = (my_error_ptr) cinfo->err;

    // return control to the setjmp point 
    longjmp(myerr->setjmp_buffer, 1);
}


//***** HELPER FUNCTIONS FOR GZIP COMPRESSION ****//
//
// zlib's compress() and uncompress() functions don't
// use gzip headers (they use 'zlib' headers).
// These helper functions are copied from the zlib
// source, but modified to use the gzip options instead.
//
//************************************************//

// Copied from https://github.com/madler/zlib/blob/master/compress.c and modified for gzip
int zlib_compress2_gzip(Bytef *dest, uLongf *destLen, const Bytef *source, uLong sourceLen, int level)
{
    z_stream stream;
    int err;
    const uInt max = (uInt)-1;
    uLong left;

    left = *destLen;
    *destLen = 0;

    stream.zalloc = (alloc_func)0;
    stream.zfree = (free_func)0;
    stream.opaque = (voidpf)0;

    // zlib manual (https://www.zlib.net/manual.html) says:
    //   Add 16 to windowBits to write a simple gzip header and trailer
    //   around the compressed data instead of a zlib wrapper.
    int windowBits = MAX_WBITS + 16;

    // These are the default values
    int method = Z_DEFLATED;
    int memLevel = 8;
    int strategy = Z_DEFAULT_STRATEGY;
    err = deflateInit2(&stream, level, method, windowBits, memLevel, strategy);

    if (err != Z_OK) return err;

    stream.next_out = dest;
    stream.avail_out = 0;
    stream.next_in = (z_const Bytef *)source;
    stream.avail_in = 0;

    do {
        if (stream.avail_out == 0) {
            stream.avail_out = left > (uLong)max ? max : (uInt)left;
            left -= stream.avail_out;
        }
        if (stream.avail_in == 0) {
            stream.avail_in = sourceLen > (uLong)max ? max : (uInt)sourceLen;
            sourceLen -= stream.avail_in;
        }
        err = deflate(&stream, sourceLen ? Z_NO_FLUSH : Z_FINISH);
    } while (err == Z_OK);

    *destLen = stream.total_out;
    deflateEnd(&stream);
    return err == Z_STREAM_END ? Z_OK : err;
}

// Copied from https://github.com/madler/zlib/blob/master/uncompr.c and modified for gzip
int zlib_uncompress2_gzip(Bytef *dest, uLongf *destLen, const Bytef *source, uLong *sourceLen)
{
    z_stream stream;
    int err;
    const uInt max = (uInt)-1;
    uLong len, left;
    Byte buf[1];    /* for detection of incomplete stream when *destLen == 0 */

    len = *sourceLen;
    if (*destLen) {
        left = *destLen;
        *destLen = 0;
    }
    else {
        left = 1;
        dest = buf;
    }

    stream.next_in = (z_const Bytef *)source;
    stream.avail_in = 0;
    stream.zalloc = (alloc_func)0;
    stream.zfree = (free_func)0;
    stream.opaque = (voidpf)0;

    // zlib manual (https://www.zlib.net/manual.html) says:
    //   Add 32 to windowBits to enable zlib and gzip decoding with
    //   automatic header detection, or add 16 to decode only the
    //   gzip format (the zlib format will return a Z_DATA_ERROR).
    int windowBits = MAX_WBITS + 16;
    err = inflateInit2(&stream, windowBits);
    if (err != Z_OK) return err;

    stream.next_out = dest;
    stream.avail_out = 0;

    do {
        if (stream.avail_out == 0) {
            stream.avail_out = left > (uLong)max ? max : (uInt)left;
            left -= stream.avail_out;
        }
        if (stream.avail_in == 0) {
            stream.avail_in = len > (uLong)max ? max : (uInt)len;
            len -= stream.avail_in;
        }
        err = inflate(&stream, Z_NO_FLUSH);
    } while (err == Z_OK);

    *sourceLen -= len + stream.avail_in;
    if (dest != buf)
        *destLen = stream.total_out;
    else if (stream.total_out && err == Z_BUF_ERROR)
        left = 1;

    inflateEnd(&stream);
    return err == Z_STREAM_END ? Z_OK :
           err == Z_NEED_DICT ? Z_DATA_ERROR  :
           err == Z_BUF_ERROR && left + stream.avail_out ? Z_DATA_ERROR :
           err;
}


/************** Start of libdvid specific functions *****************/

namespace libdvid {

BinaryDataPtr BinaryData::decompress_gzip( const BinaryDataPtr compressed_data,
                                           int max_uncompressed_size )
{
    if (compressed_data->length() == 0) {
        throw ErrMsg("Cannot decompress empty buffer");
    }
    
    std::vector<uint8_t> destination;
    destination.resize(max_uncompressed_size);

    uint8_t const * src_buf = compressed_data->get_raw();
    uint8_t * dest_buf = &destination[0];

    unsigned long dest_length = max_uncompressed_size;
    unsigned long src_length = compressed_data->length();

    // zlib's uncompress2() will modify dest_length to match the length of the buffer used,
    // and modify src_length to indicate the length of data consumed.
    int err_code = zlib_uncompress2_gzip(dest_buf, &dest_length, src_buf, &src_length);

    if (err_code != Z_OK)
    {   std::ostringstream ss;
        ss << "zlib decompression failed: " << err_code;
        throw ErrMsg(ss.str());
    }

    if ( src_length != compressed_data->length() )
    {
        // If the user passed more data than zlib consumed, then something is probably wrong.
        throw ErrMsg("decompress_gzip(): Did not consume all input data.");
    }

    return BinaryData::create_binary_data(dest_buf, dest_length);
}

BinaryDataPtr BinaryData::compress_gzip(const BinaryDataPtr uncompressed_data)
{
    unsigned long src_length = uncompressed_data->length();
    unsigned long dest_length = 1.5*uncompressed_data->length(); // Some inputs don't compress, so be super conservative here.

    std::vector<uint8_t> destination;
    destination.resize(dest_length);

    uint8_t const * src_buf = uncompressed_data->get_raw();
    uint8_t * dest_buf = reinterpret_cast<uint8_t *>(&destination[0]);

    int level = Z_DEFAULT_COMPRESSION; // 6 by default

    // zlib's gzip_deflate will modify dest_length to match the length of the buffer used,
    int err_code = zlib_compress2_gzip(dest_buf, &dest_length, src_buf, src_length, level);

    if (err_code != Z_OK)
    {   std::ostringstream ss;
        ss << "zlib compression failed: " << err_code;
        throw ErrMsg(ss.str());
    }

    return BinaryData::create_binary_data(dest_buf, dest_length);
}


BinaryDataPtr BinaryData::decompress_lz4(const BinaryDataPtr lz4binary,
        int uncompressed_size, char* buffer, int bufsize)
{
    if (lz4binary->length() == 0) {
        throw ErrMsg("Cannot decompress empty buffer");
    }

    const char* lz4_source = (char*) lz4binary->get_raw();
    
    if (!buffer) {
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
    } else {
        int bytes_read = 
            LZ4_decompress_safe(lz4_source, buffer, lz4binary->data.size(), bufsize);
        if (bytes_read < 0) {
            throw ErrMsg("Decompression of LZ4 failed");
        }     
         
        BinaryDataPtr binary = BinaryData::create_binary_data(buffer, bytes_read);
    
        return binary;
    }

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

BinaryDataPtr BinaryData::compress_labelarray_block(const BinaryDataPtr full_block, unsigned int block_width )
{
    if (block_width != 64)
    {
        std::ostringstream ss;
        ss << "Can't handle block widths other than 64px (You gave " << block_width << ")";
        throw ErrMsg(ss.str());
    }

    if (full_block->length() != sizeof(uint64_t)*block_width*block_width*block_width)
    {
        std::ostringstream ss;
        ss << "Can't compress labelarray block: Buffer size (" << full_block->length()
        << ") does not correspond to expected size (" << block_width*block_width*block_width
        << ") for the block width (" << block_width << ").";
        throw ErrMsg(ss.str());
    }

    Dims_t blockdims = {block_width, block_width, block_width};
    EncodedData encoded_data = encode_label_block( Labels3D(full_block, blockdims) );

    // Must copy the data, unfortunately.
    return BinaryData::create_binary_data(&encoded_data[0], encoded_data.size());
}

BinaryDataPtr BinaryData::compress_lz4_labelarray_block(const BinaryDataPtr full_bock, unsigned int block_width )
{
    BinaryDataPtr encoded_data = BinaryData::compress_labelarray_block(full_bock, block_width);
    return BinaryData::compress_lz4(encoded_data);
}

BinaryDataPtr BinaryData::compress_gzip_labelarray_block(const BinaryDataPtr full_bock, unsigned int block_width )
{
    BinaryDataPtr encoded_data = BinaryData::compress_labelarray_block(full_bock, block_width);
    return BinaryData::compress_gzip(encoded_data);
}

BinaryDataPtr BinaryData::decompress_png8(const BinaryDataPtr pngbinary,
        unsigned int& width, unsigned int& height)
{
    if (pngbinary->length() == 0) {
        throw ErrMsg("Cannot decompress empty buffer");
    }
    
    // ?! currently no check if it is grayscale
    
    // retrieve PNG
    string& png_image = pngbinary->get_data();
    istringstream sstr2(png_image);
    png::image<png::gray_pixel> image;         
    image.read(sstr2);

    width = image.get_width();
    height = image.get_height();

    // direct access of string buffer
    BinaryDataPtr binary(new BinaryData());
    binary->data.resize(image.get_width()*image.get_height());
    char* ptr = &binary->data[0];

    for (unsigned int y = 0; y < image.get_height(); ++y) {
        for (unsigned int x = 0; x < image.get_width(); ++x) {
            *ptr = image[y][x];
            ++ptr;
        }
    }

    return binary;
}

BinaryDataPtr BinaryData::decompress_jpeg(const BinaryDataPtr jpegbinary,
        unsigned int& width, unsigned int& height)
{
    if (jpegbinary->length() == 0) {
        throw ErrMsg("Cannot decompress empty buffer");
    }
    
#if JPEGTURBO
    long unsigned int _jpegSize = jpegbinary->length();
    unsigned char* _compressedImage = (unsigned char*) jpegbinary->get_raw();


    int jpegSubsamp, width2, height2;
    tjhandle _jpegDecompressor = tjInitDecompress();
    tjDecompressHeader2(_jpegDecompressor, _compressedImage, _jpegSize, &width2, &height2, &jpegSubsamp);

    width = width2;
    height = height2;
    
    BinaryDataPtr binary(new BinaryData());
    // create a string buffer to fit the uncompressed result
    binary->data.resize(width*height);

    // dangerous write directly to string buffer
    char* uncompressed_data = &(binary->data[0]);
    unsigned char* buffer = (unsigned char*) uncompressed_data;

    tjDecompress2(_jpegDecompressor, _compressedImage, _jpegSize, buffer, width2, 0/*pitch*/, height2, TJPF_GRAY, TJFLAG_FASTDCT);


    tjDestroy(_jpegDecompressor);

    return binary;
#else
    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr       jerr;
    cinfo.err = jpeg_std_error((jpeg_error_mgr*)&jerr);
    jerr.pub.error_exit = my_error_exit;
    
    // handle error handling for libjpeg library
    if (setjmp(jerr.setjmp_buffer)) {
        // destroy datastructures and indicate error
        //cinfo.err->output_message(cinfo);
        jpeg_destroy_decompress(&cinfo);
        throw ErrMsg("Invalid JPEG");
    }
    
    jpeg_create_decompress(&cinfo);

    // specify data source
    jpeg_mem_src(&cinfo, (unsigned char*) jpegbinary->get_raw(),
            jpegbinary->length());

    // read the header
    jpeg_read_header(&cinfo, TRUE);

    // decompress
    jpeg_start_decompress(&cinfo);

    // get image width
    int row_stride = cinfo.output_width * cinfo.output_components;
    
    BinaryDataPtr binary(new BinaryData());
    // create a string buffer to fit the uncompressed result
    binary->data.resize(row_stride * cinfo.output_height);

    // dangerous write directly to string buffer
    char* uncompressed_data = &(binary->data[0]);
    unsigned char* buffer = (unsigned char*) uncompressed_data;

    width = cinfo.output_width;
    height = cinfo.output_height;

    // use the library's state variable cinfo.output_scanline
    // as the loop counter
    while (cinfo.output_scanline < cinfo.output_height) {
        // try to read as many lines as possible
        int lines_read = jpeg_read_scanlines(&cinfo, &buffer, 1);

        // update buffer position
        buffer += (row_stride * lines_read);
    }

    // finish decompression
    (void) jpeg_finish_decompress(&cinfo);

    // release memory
    jpeg_destroy_decompress(&cinfo);

    return binary;
#endif
}

BinaryDataPtr BinaryData::decompress_labelarray_block(const BinaryDataPtr blockbinary, unsigned int block_width)
{
    // TODO: Use block_width once decode_label_block() is changed to use it!
    assert(block_width == 64);

    Labels3D inflated_block = decode_label_block(blockbinary->get_raw_char(), blockbinary->length());
    return inflated_block.get_binary();
}

BinaryDataPtr BinaryData::decompress_lz4_labelarray_block(const BinaryDataPtr lz4_compressed, unsigned int block_width)
{
    int decomp_size = sizeof(uint64_t) * block_width * block_width * block_width;
    BinaryDataPtr lz4_inflated = BinaryData::decompress_lz4(lz4_compressed, decomp_size);

    Labels3D inflated_block = decode_label_block(lz4_inflated->get_raw_char(), lz4_inflated->length());
    return inflated_block.get_binary();
}

BinaryDataPtr BinaryData::decompress_gzip_labelarray_block(const BinaryDataPtr gzip_compressed, unsigned int block_width)
{
    int decomp_size = sizeof(uint64_t) * block_width * block_width * block_width;
    BinaryDataPtr gzip_inflated = BinaryData::decompress_gzip(gzip_compressed, decomp_size);

    Labels3D inflated_block = decode_label_block(gzip_inflated->get_raw_char(), gzip_inflated->length());
    return inflated_block.get_binary();
}

Json::Value BinaryData::get_json_value() {
    Json::CharReaderBuilder builder;
    std::unique_ptr<Json::CharReader> const reader(builder.newCharReader());
    Json::Value json_val;
    std::string errs;

    if (!reader->parse(data.c_str(), data.c_str() + data.size(), &json_val, &errs)) {
        throw ErrMsg("Could not decode json");
    }

    return json_val;
}

} // namespace: libdvid
