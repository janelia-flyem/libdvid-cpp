#include <iostream>
#include <fstream>

#include "libdvid/DVIDLabelCodec.h"

int main(int argc, char * argv[])
{
    using libdvid::LabelVec;
    using libdvid::EncodedData;

    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <labels.bin>" << std::endl;
        return 1;
    }

    std::cout << "Reading from " << argv[1] << std::endl;
    std::ifstream labelvol_file(argv[1], std::ios::in | std::ios::binary);
    labelvol_file.unsetf(std::ios::skipws);

    LabelVec labels;
    labels.resize(64*64*64);

    char * labels_buf = reinterpret_cast<char *>(&labels[0]);
    labelvol_file.read(labels_buf, labels.size() * sizeof(uint64_t));

    labels.insert( labels.begin(),
                   std::istream_iterator<uint64_t>(labelvol_file),
                   std::istream_iterator<uint64_t>() );

    std::cout << "Loaded " << labels.size() << " labels." << std::endl;
    EncodedData encoded = libdvid::encode_label_block(labels);

    std::string out_path = std::string(argv[1]) + ".encoded";
    std::ofstream encoded_file(out_path, std::ios::out | std::ios::binary);
    for (auto byte : encoded)
    {
        encoded_file << byte;
    }
    std::cout << "Wrote " << out_path << std::endl;

    return 0;
}
