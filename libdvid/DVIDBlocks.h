/*!
 * This file defines a class that holds an array of DVID blocks.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#ifndef DVIDBLOCKS_H
#define DVIDBLOCKS_H

#include "BinaryData.h"
#include "Globals.h"
#include "DVIDException.h"

#include <vector>
#include <string>

namespace libdvid {

/*!
 * Class holds reference to a compressed block (assume lz4 or jpeg compression).
 * Assumes that the block size is less than INT_MAX.
 * TODO: create hash index function and add comparators
*/
class DVIDCompressedBlock {
  public:

    //! Defines compression used by DVIDBlocks
    enum CompressType { lz4, jpeg, uncompressed };

    /*!
     * Constructor takes compressed data, offset, blocksize.
     * \param data lz4 or jpeg compressed data
     * \param offset offset in global coordinates
     * \param blocksize size of block (needed for lz4/jpeg conversion)
     * \param typesize size of data in bytes
    */
    DVIDCompressedBlock(BinaryDataPtr data, std::vector<int> offset_,
            size_t blocksize_, size_t typesize_, CompressType ctype_=lz4) : cdata(data),
            offset(offset_), blocksize(blocksize_), typesize(typesize_), ctype(ctype_) {}
 
    DVIDCompressedBlock() {}
  
    /*!
     * Sets data value.
     * \param data compressed block data
    */ 
    void set_data(BinaryDataPtr data) { cdata = data; }
    
    BinaryDataPtr get_data() const { return cdata; }

    /*!
     * Gets block offset in global coordinates.
     * \return vector of x,y,z offset
    */
    std::vector<int> get_offset() const { return offset; }

    /*!
     * Gets block size.
     * \return block size
    */
    size_t get_blocksize() const { return blocksize; }

    /*!
     * Gets type size.
     * \return type size
    */
    size_t get_typesize() const { return typesize; }

    /*!
     * Decompress data.
     * \return decompressed binary
    */
    BinaryDataPtr get_uncompressed_data()
    {
        // lz4 decompress
        switch (ctype) {
          case jpeg:
          {
            unsigned int width, height;
            return BinaryData::decompress_jpeg(cdata, width, height);
          }
          case lz4:
          {
              int decomp_size = blocksize*blocksize*blocksize*typesize;
              return BinaryData::decompress_lz4(cdata, decomp_size);
          }
          case uncompressed:
          {
            return cdata;
          }
        }
        throw ErrMsg("Unknown compression type");
    }

    /*!
     * Get size of compressed buffer.
     * \return size of buffer.
    */
    size_t get_datasize() const { return cdata->length(); }

  private:
    
    BinaryDataPtr cdata;
    std::vector<int> offset;
    size_t blocksize;
    size_t typesize;
    CompressType ctype;
};



/*!
 * Class to access DVID blocks.  The block size is almost always
 * 32x32x32 as dictated by DVID.  For flexiblity purposes, the
 * user can define a specific blocksize in case DVID is initialized
 * with a different blocksize.  The blocks object contains a buffer
 * to a contiguous chunk of memory.
*/
template <typename T>
class DVIDBlocks {
  public:
    /*!
     * Constructor takes a constant buffer corresponding
     * to an array of blocks.  The memory should be laid
     * so that block one is before block two, etc.
     * \param array_ buffer to be copied
     * \param num_blocks_ number of blocks in buffer
     * \param N block size (DEFAULT=32)
    */
    DVIDBlocks(const T* array_, int num_blocks_, size_t N_=DEFBLOCKSIZE) : num_blocks(num_blocks_), N(N_)
    {
        uint64 total_size = uint64(N)*uint64(N)*uint64(N)*
            uint64(sizeof(T))*uint64(num_blocks); 
        if (total_size > INT_MAX) {
            throw ErrMsg("Cannot allocate larger than INT_MAX");
        }
        data = BinaryData::create_binary_data((const char*) array_, total_size);
    }

    /*!
     * Empty constructor.
    */
    DVIDBlocks() : num_blocks(0), N(DEFBLOCKSIZE)
    {
        data = BinaryData::create_binary_data();
    }

    /*!
     * Constructor takes a binary object corresponding
     * to an array of blocks.  The memory should be laid
     * so that block one is before block two, etc.  The binary
     * blob is referenced, not copied.
     * \param ptr_ binary buffer to be stored
     * \param num_blocks_ number of blocks in buffer
     * \param N block size (DEFAULT=32)
    */
    DVIDBlocks(BinaryDataPtr ptr_, int num_blocks_, size_t N_=DEFBLOCKSIZE) : data(ptr_),
            num_blocks(num_blocks_), N(N_) {}

    /*!
     * Returns number of blocks in structure.
     * \returns num blocks
    */
    int get_num_blocks() const
    {
        return num_blocks;
    }
  
    /*!
     * Grabs pointer for block in the array.
     * \return constant buffer
    */
    const T* operator[](const int index) const
    {
        if (index >= num_blocks) {
            throw ErrMsg("Block index out-of-bounds");
        }
        return (const T*) &(data->get_raw()[N*N*N*sizeof(T)*index]);
    }

    /*!
     * Copies blocks to end of current array.
     * \param block constant buffer to be copied
    */
    void push_back(const T* block)
    {
        uint64 total_size = uint64(num_blocks+1)*uint64(N)*uint64(N)*
            uint64(N)*uint64(sizeof(T)); 

        if (total_size > INT_MAX) {
            throw ErrMsg("Cannot allocate larger than INT_MAX");
        }

        std::string& dataint = data->get_data();
        dataint.append((char*) block, N*N*N*sizeof(T));
        ++num_blocks;
    }
    
    /*!
     * Retrieve binary data for the given volume.
     * \return binary data for volume
    */
    BinaryDataPtr get_binary()
    {
        return data;
    }


    /*!
     * Retrieve raw buffer from binary pointer.
     * \return constant blocks buffer
    */
    const T* get_raw() const
    {
        return (T*) data->get_raw();
    }

    /*!
     * Retrive block size.
     * \return block size
    */
    int block_size() const
    {
        return N;
    }

  private:

    //! Holds binary data
    BinaryDataPtr data;

    //! Number of blocks
    int num_blocks;

    //! Block size
    size_t N;
};

//! Label blocks
typedef DVIDBlocks<uint64> LabelBlocks;

//! Grayscale blocks
typedef DVIDBlocks<uint8> GrayscaleBlocks;

}

#endif

