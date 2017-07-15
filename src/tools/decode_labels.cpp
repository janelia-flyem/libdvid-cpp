#include <iostream>
#include <fstream>

#include "libdvid/DVIDLabelCodec.h"

int main(int argc, char * argv[])
{
    using libdvid::Labels3D;
    using libdvid::LabelVec;
    using libdvid::EncodedData;
    using libdvid::decode_label_block;

    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <encoded_labels.bin>" << std::endl;
        return 1;
    }

    std::cout << "Reading from " << argv[1] << std::endl;
    std::ifstream file(argv[1], std::ios::in | std::ios::binary);
    file.unsetf(std::ios::skipws);

    file.seekg(0, std::ios::end);
    size_t file_bytecount = file.tellg();
    file.seekg(0, std::ios::beg);

    EncodedData encoded_data(file_bytecount, 0);
    file.read(&encoded_data[0], file_bytecount);

    Labels3D inflated;
    inflated = decode_label_block(&encoded_data[0], file_bytecount);

    std::string out_path = std::string(argv[1]) + ".inflated";
    std::ofstream encoded_file(out_path, std::ios::out | std::ios::binary);
    for (auto byte : inflated.get_binary()->get_data())
    {
        encoded_file << byte;
    }
    std::cout << "Wrote " << out_path << std::endl;

    return 0;
}
