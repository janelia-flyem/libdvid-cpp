/*!
 * Caches requests from DVID.
*/

#ifndef DVIDCACHE
#define DVIDCACHE

#include <unordered_map>
#include <boost/thread/mutex.hpp>
#include "BinaryData.h"

namespace libdvid {

class DVIDCache {
  public:
    static DVIDCache* get_cache();
    void set_cache_size(unsigned long long max_size_);
    void set(std::string key, BinaryDataPtr data);
    BinaryDataPtr get(std::string key);

  private:
    DVIDCache() : cache_stamp(0), max_size(0), curr_size(0) {}
    void delete_old_cache();
    void clear_cache();

    static boost::mutex mutex;

    std::unordered_map<std::string, BinaryDataPtr> cache;
    std::unordered_map<std::string, unsigned long long> cache_stamps;
    unsigned long long cache_stamp;
    unsigned long long max_size;
    unsigned long long curr_size;

};

}


#endif
