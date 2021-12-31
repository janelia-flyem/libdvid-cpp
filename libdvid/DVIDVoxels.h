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
#include <sstream>
#include <vector>
#include <boost/foreach.hpp>

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

    typedef T voxel_type;
    const static int num_dims = N;

    /*!
     * Construtor takes a constant buffer, creates a binary
     * buffer of a certain length and associates it with
     * a volume of the provided dimensions.
     * \param array_ buffer to be copied
     * \param length number of bytes in buffer
     * \param dims_ dimension sizes for the volume
    */ 
    DVIDVoxels(const T* array_, unsigned int length, Dims_t& dims_)
    : dims(dims_)
    {
        check_inputs(length*sizeof(T), dims_);
        data = BinaryData::create_binary_data(reinterpret_cast<const char*>(array_), length*sizeof(T));
    }

    /*!
     * Constructor takes a a binary buffer and associates
     * it with a volume of the provided dimensions.
     * \param data_ binary buffer referenced by the volume
     * \param dims_ dimension sizes for the volume
    */ 
    DVIDVoxels(BinaryDataPtr data_, Dims_t const& dims_)
    : data(data_)
    , dims(dims_)
    {
        check_inputs(data_->length(), dims_);
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
    BinaryDataPtr get_binary() const
    {
        return data;
    }

    /*!
     * Retrieve raw buffer from binary pointer.
     * \return constant volume buffer
    */
    const T* get_raw() const
    {
        return reinterpret_cast<T const *>(data->get_raw());
    }

    /*!
     * Get the dimensions of this volume.
     * \return volume dimensions
    */
    Dims_t const & get_dims() const
    {
        return dims;
    }
    
    /*!
     * Get the total count of voxels in the volume
    */
    size_t count() const
    {
        return dims[0] * dims[1] * dims[2];
    }

  private:
    //! Holds binary data
    BinaryDataPtr data;

    //! Dimensions for volume
    Dims_t dims;

    //! Check input length/dimensions and throw an exception is something is wrong.
    void check_inputs(size_t length_bytes, Dims_t const & dims_)
    {
        if (dims_.size() != N) {
            throw ErrMsg("Incorrect dimensions provided");
        }
        
        uint64 total_size = sizeof(T);
        for (auto dim : dims_) {
            total_size *= dim;
        }
        
        if (total_size > INT_MAX) {
            throw ErrMsg("Cannot allocate larger than INT_MAX");
        }
        
        if (total_size != length_bytes) {
            std::stringstream ssMsg;
            ssMsg << "Dimensions ( ";
            for ( auto d : dims ) {
                ssMsg << d << ", ";
            }
            ssMsg << ") do not match buffer size (" << length_bytes << ").";
            throw ErrMsg( ssMsg.str() );
        }
    }
};

//! 3D label volume
typedef DVIDVoxels<uint64, 3> Labels3D;

//! 3D array types
typedef DVIDVoxels<uint8, 3> Array8bit3D;
typedef DVIDVoxels<uint16, 3> Array16bit3D;
typedef DVIDVoxels<uint32, 3> Array32bit3D;
typedef DVIDVoxels<uint64, 3> Array64bit3D;

//! 3D 8-bit volume (corresponding to grayscale)
typedef DVIDVoxels<uint8, 3> Grayscale3D;

//! 3D bool volume (corresponding to dense roi data)
typedef DVIDVoxels<uint8, 3> Roi3D;

//! 2D label volume
typedef DVIDVoxels<uint64, 2> Labels2D;

//! 2D array of coordinates
typedef DVIDVoxels<int32, 2> Coords2D;

//! 2D 8-bit volume (corresponding to grayscale)
typedef DVIDVoxels<uint8, 2> Grayscale2D;

template <typename VoxelType>
void overwrite_subvol( DVIDVoxels<VoxelType, 3> & vol,
                       DVIDVoxels<VoxelType, 3> const & subvol,
                       std::vector<int> const & subvol_offset_xyz )
{
    int vol_Z = vol.get_dims()[2];
    int vol_Y = vol.get_dims()[1];
    int vol_X = vol.get_dims()[0];

    int sv_Z = subvol.get_dims()[2];
    int sv_Y = subvol.get_dims()[1];
    int sv_X = subvol.get_dims()[0];

    int off_z = subvol_offset_xyz[2];
    int off_y = subvol_offset_xyz[1];
    int off_x = subvol_offset_xyz[0];

    auto & vol_data = vol.get_binary()->get_data();
    char * vol_buf = &vol_data[0];

    BinaryDataPtr subvol_binary_data = subvol.get_binary();
    auto const & subvol_data = subvol_binary_data->get_data();
    char const * subvol_buf = subvol_data.c_str();

    // x-offset is constant for all YZ iterations
    int x_offset = off_x;
    int x_offset_bytes = x_offset * sizeof(VoxelType);

    size_t sv_offset = 0;
    for (size_t sv_z = 0; sv_z < sv_Z; ++sv_z)
    {
        // Convert from subvol coords to volume coords
        int z = off_z + sv_z;
        int z_offset = z * vol_X * vol_Y;

        for (size_t sv_y = 0; sv_y < sv_Y; ++sv_y)
        {
            int y = off_y + sv_y;
            int y_offset = y * vol_X;
            size_t y_offset_bytes = y_offset * sizeof(VoxelType);

            auto vol_offset_bytes = sizeof(VoxelType) * (z_offset + y_offset + x_offset);
            auto subvol_offset_bytes = sv_offset * sizeof(VoxelType);

            // Copy X-row
            std::memcpy( vol_buf + vol_offset_bytes,
                         subvol_buf + subvol_offset_bytes,
                         sv_X * sizeof(VoxelType) );

            sv_offset += sv_X;
        }
    }
}

}

#endif
