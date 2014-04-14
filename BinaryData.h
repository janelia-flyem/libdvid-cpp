#ifndef BINARYDATA
#define BINARYDATA

#include <boost/shared_ptr.hpp>

typedef unsigned char byte

class BinaryData;
typedef boost::shared_ptr<BinaryData> BinaryDataPtr;

class BinaryData {
  public:
    static BinaryDataPtr create_binary_data(byte* data_)
    {
        return BinaryDataPtr(new BinaryData(data_));
    }
    char * get_raw()
    {
        return data;
    }
    ~BinaryData()
    {
        delete data;
    }

  private:
    BinaryData(byte data_) : data(data_) {}
    byte data;
};








#endif

