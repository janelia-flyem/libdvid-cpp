/*!
 * This file defines functionality for datatypes used in interacting
 * with the DVID ROI datatype.  This includes wrapping tuples
 * for substack coordinates and block coordinates.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#ifndef DVIDROI_H
#define DVIDROI_H

#include <algorithm>
#include <boost/operators.hpp>

namespace libdvid {

/*!
 * Defines a substack at a given offset and size.
*/
struct SubstackXYZ : boost::totally_ordered<SubstackXYZ> {
    /*!
     * Constructs a substack based on the smallest coordinate that
     * the substack intersects and its size.
     * \param x_ starting x coordinate
     * \param y_ starting y coordinate
     * \param z_ starting z coordinate
     * \param size_ size along one dimension (same for all dimensions)
    */
    SubstackXYZ(int x_, int y_, int z_, int size_) : x(x_), y(y_), z(z_), size(size_) {}
   
    /*!
     * Used to order substacks by Z then Y then X.
     * \param sbustack2 substack being compared to
     * \returns true if less than other block
    */ 
    bool operator<(SubstackXYZ const & other) const
    {
    	int params[] = {z, y, x, size};
    	int other_params[] = {other.z, other.y, other.x, other.size};
		return std::lexicographical_compare( &params[0], 	   &params[4],
											 &other_params[0], &other_params[4] );
    }

    bool operator==(SubstackXYZ const & other) const
    {
    	int params[] = {z, y, x, size};
    	int other_params[] = {other.z, other.y, other.x, other.size};
		return std::equal( &params[0], &params[4], &other_params[0] );
    }

    //! public access to member data
    int x,y,z,size;
};

/*
 * Defines a block position in block coordinate space (assume DEFBLOCKSIZE).
*/
struct BlockXYZ : boost::totally_ordered<BlockXYZ> {
    /*!
     * Constructs a block location using block coordinates
     * which is dvid voxel coordinates / DEFBLOCKSIZE.
     * \param x_ starting x block coordinate
     * \param y_ starting y block coordinate
     * \param z_ starting z block coordinate
    */
    BlockXYZ(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
   
    /*!
     * Used to order blocks by Z then Y then X.
     * \param block2 block being compared to
     * \returns true if less than other block
    */ 
    bool operator<(BlockXYZ const & other) const
    {
    	int params[] = {z, y, x};
    	int other_params[] = {other.z, other.y, other.x};
		return std::lexicographical_compare( &params[0], 	   &params[3],
											 &other_params[0], &other_params[3] );
    }

    bool operator==(BlockXYZ const & other) const
    {
    	int params[] = {z, y, x};
    	int other_params[] = {other.z, other.y, other.x};
		return std::equal( &params[0], &params[3], &other_params[0] );
    }

    //! public access to member data
    int x,y,z;
};

/*
 * Defines a voxel point in voxel coordinate space.
*/
struct PointXYZ : boost::totally_ordered<PointXYZ>{
    /*!
     * Constructs a voxel location using voxel coordinates.
     * \param x_ x coordinate
     * \param y_ y coordinate
     * \param z_ z coordinate
    */
    PointXYZ(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
    
    //! public access to member data
    int x,y,z;

    // Like blocks, points are ordered z-y-x, not x-y-z
    bool operator<(PointXYZ const & other) const
    {
    	int params[] = {z, y, x};
    	int other_params[] = {other.z, other.y, other.x};
		return std::lexicographical_compare( &params[0], 	   &params[3],
											 &other_params[0], &other_params[3] );
    }

    bool operator==(PointXYZ const & other) const
    {
    	int params[] = {z, y, x};
    	int other_params[] = {other.z, other.y, other.x};
		return std::equal( &params[0], &params[3], &other_params[0] );
    }
};

}

#endif

