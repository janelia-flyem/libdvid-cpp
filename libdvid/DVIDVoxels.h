#ifndef DVIDVOXELS_H
#define DVIDVOXELS_H

#include "BinaryData.h"
#include "DVIDException.h"
#include <string>
#include <vector>

namespace libdvid {

typedef unsigned long long uint64;
typedef unsigned char uint8;
typedef std::vector<unsigned int> Dims_t;


/*!
 * Wrapper for binary data that is part of an n-D array.  This
 * just associates a shape with a binary blob.  Dim1, dim2, dim3
 * typically refer to x,y,z with x being the lowest order byte 
*/
template <typename T, unsigned int N>
class DVIDVoxels {
  public:
    DVIDVoxels(T* array_, unsigned int length, Dims_t& dims_)
               : dims(dims_) {
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

    DVIDVoxels(Dims_t& dims_) : dims(dims_)
    {
        if (dims.size() != N) {
            throw ErrMsg("Incorrect dimensions provided");
        }
    }

    void set_binary(BinaryDataPtr data_)
    {
        data = data_;
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

    BinaryDataPtr get_binary()
    {
        return data;
    }

    const T* get_raw() const
    {
        return (T*) data->get_raw();
    }

    Dims_t get_dims() const
    {
        return dims;
    }

  private:
    BinaryDataPtr data;
    Dims_t dims;
};

typedef DVIDVoxels<uint64, 3> Labels3D;
typedef DVIDVoxels<uint8, 3> Grayscale3D;

typedef DVIDVoxels<uint64, 2> Labels2D;
typedef DVIDVoxels<uint8, 2> Grayscale2D;

}

#endif
