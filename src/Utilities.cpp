#include "Utilities.h"

using std::ostream;

namespace libdvid {

ostream& operator<<(ostream& os, ErrMsg& err)
{
    os << err.what(); 
    return os;
}

}
