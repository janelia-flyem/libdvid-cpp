#ifndef DVIDVOXELS_H
#define DVIDVOXELS_H

#include <boost/shared_ptr.hpp>
#include <string>

namespace libdvid {

template <typename T> 
class DVIDVoxels {
  public:
    static boost::shared_ptr<DVIDVoxels<T> > get_dvid_voxels(T* array_)
    {
        return boost::shared_ptr<DVIDVoxels<T> >(new DVIDVoxels<T>(array_));
    }
    static boost::shared_ptr<DVIDVoxels<T> > get_dvid_voxels(std::string& data_str)
    {
        return boost::shared_ptr<DVIDVoxels<T> >(new DVIDVoxels<T>(data_str));
    }

    ~DVIDVoxels()
    {
        delete []array;
    }
    T* get_raw()
    {
        return array;
    }

  private:
    DVIDVoxels(T* array_) : array(array_) {}
    DVIDVoxels(std::string& data_str)
    {
        T * byte_array = (T*) data_str.c_str();
        int incr = sizeof(T);
        array = new T[data_str.size()];
        for (int i = 0, pos = 0; i < data_str.size(); ++i) {
            array[i] = T(byte_array[i]); 
        }
    }
    
    T* array;
};


typedef boost::shared_ptr<DVIDVoxels<unsigned char> > DVIDGrayPtr;
typedef boost::shared_ptr<DVIDVoxels<unsigned long long> > DVIDLabelPtr;

}

#endif
