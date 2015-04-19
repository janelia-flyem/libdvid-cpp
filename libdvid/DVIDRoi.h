/*!
 * This file defines functionality for datatypes used in interacting
 * with the DVID ROI datatype.  This includes wrapping tuples
 * for substack coordinates and block coordinates.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/

#ifndef DVIDROI_H
#define DVIDROI_H

namespace libdvid {

/*!
 * Defines a substack at a given offset and size.
*/
struct SubstackXYZ {
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
    bool operator<(const SubstackXYZ& substack2) const
    {
        if ((z < substack2.z) ||
            ((z == substack2.z) && (y < substack2.y)) ||
            ((z == substack2.z) && (y == substack2.y) && (x < substack2.x))) {
            return true;
        }

        return false;
    }

    //! public access to member data
    int x,y,z,size;
};

/*
 * Defines a block position in block coordinate space (assume DEFBLOCKSIZE).
*/
struct BlockXYZ {
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
    bool operator<(const BlockXYZ& block2) const
    {
        if ((z < block2.z) ||
            ((z == block2.z) && (y < block2.y)) ||
            ((z == block2.z) && (y == block2.y) && (x < block2.x))) {
            return true;
        }

        return false;
    }

    //! public access to member data
    int x,y,z;
};

/*
 * Defines a voxel point in voxel coordinate space.
*/
struct PointXYZ {
    /*!
     * Constructs a voxel location using voxel coordinates.
     * \param x_ x coordinate
     * \param y_ y coordinate
     * \param z_ z coordinate
    */
    PointXYZ(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
    
    //! public access to member data
    int x,y,z;
};

}

#endif

