#ifndef BINARYDATA
#define BINARYDATA

#include <boost/shared_ptr.hpp>
#include <fstream>
#include <string>

namespace libdvid {

typedef unsigned char byte;

class BinaryData;
typedef boost::shared_ptr<BinaryData> BinaryDataPtr;

class BinaryData {
  public:
    static BinaryDataPtr create_binary_data(const char* data_, unsigned int length)
    {
        return BinaryDataPtr(new BinaryData(data_, length));
    }
    static BinaryDataPtr create_binary_data(std::ifstream& fin)
    {
        return BinaryDataPtr(new BinaryData(fin));
    }

    byte * get_raw()
    {
        return (byte *)(data.c_str());
    }
    std::string& get_data()
    {
        return data;
    }
    int length()
    {
        return data.length();
    }

    ~BinaryData() {}
  private:
    BinaryData(const char* data_, unsigned int length) : data(data_, length) {}
    BinaryData(std::ifstream& fin)
    {
        data.assign( (std::istreambuf_iterator<char>(fin) ),
                (std::istreambuf_iterator<char>()    ) ); 
    }
    std::string data;
};

}

#endif

