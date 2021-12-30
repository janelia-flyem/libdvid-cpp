/*!
 * This file provides functionality for working with binary data.
 * The goal is to ensure that users of libdvid are not responsible
 * for heap management.
 *
 * TODO: consider maintaining a custom buffer rather than use a string.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/
#ifndef BINARYDATA
#define BINARYDATA

#include <boost/shared_ptr.hpp>
#include <json/json.h>
#include <fstream>
#include <string>


namespace libdvid {

typedef unsigned char byte;

//! Declares smart pointer type to access binary data
class BinaryData;
typedef boost::shared_ptr<BinaryData> BinaryDataPtr;

/*!
 * Wraps a string object in an object that can only be allocated
 * on the heap.
*/
class BinaryData {
  public:
    /*!
     * Create binary data from a byte array.
     * \param data_ Constant source data
     * \param length Number of bytes in data_
     * \return smart pointer to new binary data
    */
    static BinaryDataPtr create_binary_data(const char* data_, unsigned int length)
    {
        return BinaryDataPtr(new BinaryData(data_, length));
    }

    static BinaryDataPtr create_binary_data(const unsigned char* data_, unsigned int length)
    {
        return BinaryDataPtr(new BinaryData(reinterpret_cast<const char*>(data_), length));
    }

    //! Create from scratch
    static BinaryDataPtr create_binary_data(unsigned int length)
    {
        return BinaryDataPtr(new BinaryData(length));
    }

    /*!
     * Create an empty binary data object for later modifications.
     * \return smart pointer to new binary data
    */
    static BinaryDataPtr create_binary_data()
    {
        return BinaryDataPtr(new BinaryData(0, 0));
    }
    
    /*!
     * Read a file and load the data into binary format.
     * \param fin input file
     * \return smart pointer to new binary data
    */
    static BinaryDataPtr create_binary_data(std::ifstream& fin)
    {
        return BinaryDataPtr(new BinaryData(fin));
    }

    /*!
     * Decompress and load from lz4 format.
     * TODO: decompress from lz4 streaming interface
     * \param lz4binary binary that contains lz4 data
     * \param the size of the resulting uncompresed buffer
     * \param buffer if not 0 than unknown uncompressed size
     * \param bufsize size of buffer
     * \return smart pointer to new binary data (uncompressed)
    */
    static BinaryDataPtr decompress_lz4(const BinaryDataPtr lz4binary,
            int uncompressed_size, char* buffer = 0, int bufsize = 0);

    static BinaryDataPtr decompress_gzip( const BinaryDataPtr compressed_data,
                                          int max_uncompressed_size );

    /*!
     * Load data and compress to lz4 format.
     * \param binary data to compress
     * \return smart pointer to new binary data (compressed)
    */
    static BinaryDataPtr compress_lz4(const BinaryDataPtr lz4binary);

    /*!
     * Load data and compress to DVID labelarray native block format.
     * \param binary data to compress (must be a SINGLE label block)
     * \return smart pointer to new binary data (compressed)
    */
    static BinaryDataPtr compress_labelarray_block(const BinaryDataPtr full_bock, unsigned int block_width=64);

    /*!
     * Encode (with DVID labelarray native block format) and then lz4-compress.
     * \param binary data to compress (must be a SINGLE label block)
     * \return smart pointer to new binary data (encoded and lz4-compressed)
    */
    static BinaryDataPtr compress_lz4_labelarray_block(const BinaryDataPtr full_bock, unsigned int block_width=64);
    static BinaryDataPtr compress_gzip_labelarray_block(const BinaryDataPtr full_bock, unsigned int block_width=64 );

    /*!
     * Decompress and load from jpeg format.
     * \param jpegbinary binary that contains jpeg data
     * \param width returns width of decompressed image
     * \param height returns height of decompressed image
     * \return smart pointer to new binary data (uncompressed)
    */ 
    static BinaryDataPtr decompress_jpeg(const BinaryDataPtr jpegbinary,
            unsigned int& width, unsigned int& height);

    /*!
     * Decompress and load from png format (must be 8-bit!).
     * TODO: allow loading of non-8bt images
     * \param pngbinary binary that contains png data
     * \param width returns width of decompressed image
     * \param height returns height of decompressed image
     * \return smart pointer to new binary data (uncompressed)
    */ 
    static BinaryDataPtr decompress_png8(const BinaryDataPtr pngbinary,
        unsigned int& width, unsigned int& height);


    /*!
     * Decompress and load from DVID labelarray native block format.
     * \param blockbinary binary that contains encoded labelarray data for a single block.
     * \return smart pointer to new binary data (uncompressed)
    */
    static BinaryDataPtr decompress_labelarray_block(const BinaryDataPtr blockbinary, unsigned int block_width=64);

    /*!
     * Decompress and load from lz4-compressed DVID labelarray native block format.
     * \param blockbinary binary that contains lz4-compressed encoded labelarray data for a single block.
     * \return smart pointer to new binary data (completely inflated, from lz4 and then dvid native compression)
    */
    static BinaryDataPtr decompress_lz4_labelarray_block(const BinaryDataPtr lz4_compressed, unsigned int block_width=64);
    static BinaryDataPtr decompress_gzip_labelarray_block(const BinaryDataPtr gzip_compressed, unsigned int block_width=64);

    static BinaryDataPtr compress_gzip(const BinaryDataPtr uncompressed_data);

    /*!
     * Allows modification of underlying buffer data.
     * \return string reference
    */
    std::string& get_data()
    {
        return data;
    }

    /*!
     * Parse data and return json value
     * throws error on failure to parse json
     */
    Json::Value get_json_value();

    /*!
     * Returns the length of the array.
     * \return length of array
    */
    int length() const
    {
        return data.length();
    }

    /*!
     * Retrieves constant pointer to underlying buffer.
     * \return constant byte array
    */ 
    const byte * get_raw() const
    {
        if (data.size() == 0) {
            return nullptr;
        }
        return reinterpret_cast<const byte *>(data.c_str());
    }

    //! Same as above, but returns char instead of byte (signed vs. unsigned)
    const char * get_raw_char() const
    {
        if (data.size() == 0) {
            return nullptr;
        }
        return reinterpret_cast<const char *>(data.c_str());
    }
    
    /*!
     * Default destruction of string is sufficient.
    */  
     ~BinaryData() {}

  private:
    /*!
     * Private constructor to prevent stack allocation of binary data.
     * This creates a string with the data in the buffer.
     * \param data_ Constant source data
     * \param length Number of bytes in data_
    */
    BinaryData(const char* data_, unsigned int length) : data(data_, length) {}
   

    /*!
     * Private constructor to prevent stack allocation of binary data.
     * This creates an uninitialized string of the requested length.
     * \param length Number of bytes in data_
    */
    BinaryData(unsigned int length) : data(length, '\0') {}

    /*!
     * Private empty constructor.
    */
    BinaryData() {}

    /*!
     * Private constructor to prevent stack allocation of binary data.
     * Read a file and load the data into binary format.
     * \param fin input file
    */
    explicit BinaryData(std::ifstream& fin)
    {
        data.assign( (std::istreambuf_iterator<char>(fin) ),
                (std::istreambuf_iterator<char>()    ) ); 
    }

    //! store binary array
    std::string data;
};

}

#endif

