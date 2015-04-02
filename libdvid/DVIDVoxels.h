/*!
 * The file defines class for wrapping ND volumes.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#ifndef DVIDVOXELS_H
#define DVIDVOXELS_H

#include "BinaryData.h"
#include "DVIDException.h"
#include "Globals.h"

#include <string>
#include <vector>

namespace libdvid {

//! Represents dimension sizes for ND volumes
//! TODO: create a special class for dims and offset
typedef std::vector<unsigned int> Dims_t;

/*!
 * Wrapper for binary data that is part of an n-D array.  This
 * just associates a shape with a binary blob.  For a 3D volume,
 * dim1, dim2, dim3 means that dim1 represents the number of matrix
 * columns.  Typically, dim1, dim2, and dim3 corresponds to x,y,z.
 * This templated class requires the number of dimensions
 * to be specified.
*/
template <typename T, unsigned int N>
class DVIDVoxels {
  public:
    /*!
     * Construtor takes a constant buffer, creates a binary
     * buffer of a certain length and associates it with
     * a volume of the provided dimensions.
     * \param array_ buffer to be copied
     * \param length number of bytes in buffer
     * \param dims_ dimension sizes for the volume
    */ 
    DVIDVoxels(const T* array_, unsigned int length, Dims_t& dims_)
               : dims(dims_) {
        uint64 total_size = uint64(dims_[0])*uint64(dims_[1])*
            uint64(dims_[2])*uint64(sizeof(T)); 
        if (total_size > INT_MAX) {
            throw ErrMsg("Cannot allocate larger than INT_MAX");
        }

        data = BinaryData::create_binary_data((const char*) array_,
                                                length*sizeof(T));
        if (dims.size() != N) {
            throw ErrMsg("Incorrect dimensions provided");
        }

        unsigned long long total = 0;
        for (int i = 0; i < dims.size(); ++i) {
            if (i == 0) {
                total = dims[0];
            } else {
                total *= dims[i];
            }
        }
        if (total*sizeof(T) != data->length()) {
            throw ErrMsg("Dimension mismatch with buffer size");
        }
    }

    /*!
     * Constructor takes a a binary buffer and associates
     * it with a volume of the provided dimensions.
     * \param data_ binary buffer referenced by the volume
     * \param dims_ dimension sizes for the volume
    */ 
    DVIDVoxels(BinaryDataPtr data_, Dims_t& dims_) :
            data(data_), dims(dims_) {
        
        if (dims.size() != N) {
            throw ErrMsg("Incorrect dimensions provided");
        }
        unsigned long long total = 0;
        for (int i = 0; i < dims.size(); ++i) {
            if (i == 0) {
                total = dims[0];
            } else {
                total *= dims[i];
            }
        }
        if (total*sizeof(T) != data->length()) {
            throw ErrMsg("Dimension mismatch with buffer size");
        }
    }

    /*!
     * Constructor that produces an empty object.
    */
    DVIDVoxels()
    {
        data = BinaryData::create_binary_data();
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
     * \return constant volume buffer
    */
    const T* get_raw() const
    {
        return (T*) data->get_raw();
    }

    /*!
     * Get the dimensions of this volume.
     * \return volume dimensions
    */
    Dims_t get_dims() const
    {
        return dims;
    }

  private:
    //! Holds binary data
    BinaryDataPtr data;

    //! Dimensions for volume
    Dims_t dims;
};

//! 3D label volume
typedef DVIDVoxels<uint64, 3> Labels3D;

//! 3D 8-bit volume (corresponding to grayscale)
typedef DVIDVoxels<uint8, 3> Grayscale3D;

//! 2D label volume
typedef DVIDVoxels<uint64, 2> Labels2D;

//! 2D 8-bit volume (corresponding to grayscale)
typedef DVIDVoxels<uint8, 2> Grayscale2D;

}

#endif
