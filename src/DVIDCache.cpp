#include "DVIDCache.h"
#include <map>

using std::map; using std::string;
using std::unordered_map;

namespace libdvid {

boost::mutex DVIDCache::mutex;

DVIDCache* DVIDCache::get_cache() 
{
    boost::mutex::scoped_lock lock(mutex);
    static DVIDCache cache;
    return &cache;
}

void DVIDCache::set_cache_size(unsigned long long max_size_)
{
    boost::mutex::scoped_lock lock(mutex);
    clear_cache();
    max_size = max_size_;
}

void DVIDCache::set(std::string key, BinaryDataPtr data)
{
    boost::mutex::scoped_lock lock(mutex);

    if (max_size > 0) {
        BinaryDataPtr ptr = cache[key];

        if (ptr) {
            curr_size -= ptr->length();
        }
        curr_size += data->length();
        cache[key] = data;
        cache_stamps[key] = cache_stamp;
        ++cache_stamp;
    
        if (curr_size > max_size) {
            delete_old_cache();
        }
    }
}

BinaryDataPtr DVIDCache::get(std::string key)
{
    boost::mutex::scoped_lock lock(mutex);
    BinaryDataPtr ptr;

    if (max_size > 0) {
        if (cache.find(key) != cache.end()) {
            ptr = cache[key]; 
            cache_stamps[key] = cache_stamp;
            ++cache_stamp;
        }
    }
    return ptr;
}

void DVIDCache::clear_cache()
{
    cache.clear();
    cache_stamps.clear();
    cache_stamp = 0;
    max_size = 0;
    curr_size = 0;
}


void DVIDCache::delete_old_cache()
{
    map<unsigned long long, string> cache_order;
    unsigned long long delsize = 0;

    for (unordered_map<string, unsigned long long>::iterator iter = cache_stamps.begin(); iter != cache_stamps.end(); ++iter) {
        cache_order[iter->second] = iter->first;
    } 

    for (map<unsigned long long, string>::iterator iter = cache_order.begin(); iter != cache_order.end(); ++iter) {
        delsize += cache[iter->second]->length();
        cache.erase(iter->second);
        cache_stamps.erase(iter->second);
        if (delsize > (max_size/2)) {
            break;
        }
    }
}

}
