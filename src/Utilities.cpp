#include "Utilities.h"

using std::ostream;

namespace libdvid {

ostream& operator<<(ostream& os, ErrMsg& err)
{
    os << err.get_msg(); 
    return os;
}

}
