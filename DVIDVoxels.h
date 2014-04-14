#ifndef DVIDVOXELS_H
#define DVIDVOXELS_H

#include <boost/shared_ptr.hpp>

template <typename T> 
class DVIDVoxels {
  public:
    static boost::shared_ptr<DVIDVoxels<T> > get_dvid_voxels(T* array_)
    {
        return boost::shared_ptr<DVIDVoxels<T> >(new DVIDVoxels(array_));
    }
    ~DVIDVoxels()
    {
        delete array;
    }
    T* get_raw()
    {
        return array;
    }

  private:
    DVIDVoxels(T* array_) : array(array_) {}
    T* array;
};


typedef boost::shared_ptr<DVIDVoxels<unsigned char> > DVIDGrayPtr;
typedef boost::shared_ptr<DVIDVoxels<unsigned long long> > DVIDLabelPtr;
