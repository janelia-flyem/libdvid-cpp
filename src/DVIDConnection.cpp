#include "DVIDConnection.h"
#include "DVIDException.h"

extern "C" {
#include <curl/curl.h>
}

using std::string;

/*!
 * Initializes libcurl libraries.  This could probably be a singleton
 * but the static scope make unauthorized access unlikely
*/
struct CurlEnv {
    CurlEnv() { curl_global_init(CURL_GLOBAL_ALL); }
    ~CurlEnv() { curl_global_cleanup(); }
};
static CurlEnv curl_env;

//! Function for libcurl that writes results into a string buffer
static size_t
WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    size_t realsize = size * nmemb;
    string* str = (string*) userp;
    str->append((const char*) contents, realsize);
    return realsize;
}

namespace libdvid {
    
string DVIDConnection::DVID_PREFIX = "/api";

DVIDConnection::DVIDConnection(string addr_) : addr(addr_)
{
    curl_connection = curl_easy_init();
}

DVIDConnection::DVIDConnection(const DVIDConnection& copy_connection)
{
    addr = copy_connection.addr;
    curl_connection = curl_easy_init();
}

DVIDConnection::~DVIDConnection()
{
//    curl_easy_cleanup(curl_connection);
}

int DVIDConnection::make_request(string endpoint, ConnectionMethod method,
        BinaryDataPtr payload, BinaryDataPtr results, string& error_msg,
        ConnectionType type, int timeout)
{
    CURLcode result;

    // pass the custom headers
    struct curl_slist *headers=0;
    if (type == JSON) {
        headers = curl_slist_append(headers, "Content-Type: application/json");
    } else if (type == BINARY) {
        headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
    } 
    curl_easy_setopt(curl_connection, CURLOPT_HTTPHEADER, headers);
    
    // load url
    string url = get_uri_root() + endpoint;
    curl_easy_setopt(curl_connection, CURLOPT_URL, url.c_str());
        
    // set the method
    if (method == GET) {
        curl_easy_setopt(curl_connection, CURLOPT_CUSTOMREQUEST, "GET");
    }
    if (method == POST) {
        curl_easy_setopt(curl_connection, CURLOPT_CUSTOMREQUEST, "POST");
    }
    if (method == PUT) {
        curl_easy_setopt(curl_connection, CURLOPT_CUSTOMREQUEST, "PUT");
    }
    if (method == DELETE) {
        curl_easy_setopt(curl_connection, CURLOPT_CUSTOMREQUEST, "DELETE");
    }

    // set to 0 for infinite
    curl_easy_setopt(curl_connection, CURLOPT_TIMEOUT, long(timeout));

    // post binary data
    if (payload) {
        // set binary payload and indicate size
        curl_easy_setopt(curl_connection, CURLOPT_POSTFIELDS, payload->get_raw());
        curl_easy_setopt(curl_connection, CURLOPT_POSTFIELDSIZE, long(payload->length()));
    } else {
        curl_easy_setopt(curl_connection, CURLOPT_POSTFIELDS, 0);
        curl_easy_setopt(curl_connection, CURLOPT_POSTFIELDSIZE, long(0));
    }
    
    // results should be an empty binary array
    assert(results);
    assert(results->length() == 0);

    // pass the raw pointer to the write data command
    string& raw_data = results->get_data();

    // set callback for writing data
    curl_easy_setopt(curl_connection, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl_connection, CURLOPT_WRITEDATA, (void *)&raw_data);

    // set verbose only for debug
    //curl_easy_setopt(curl_connection, CURLOPT_VERBOSE, 1L);

    // set error buffer
    char error_buf[CURL_ERROR_SIZE]; // size of the error buffer
    curl_easy_setopt(curl_connection, CURLOPT_ERRORBUFFER, error_buf);

    // actually perform the request
    result = curl_easy_perform(curl_connection);
    
    // get the error code
    long http_code = 0;
    curl_easy_getinfo (curl_connection, CURLINFO_RESPONSE_CODE, &http_code);
    
    // throw exception if connection doesn't work
    if (result != CURLE_OK) {
        throw DVIDException("DVIDConnection error: " + string(url), http_code);
    }

    // load error if there is one
    error_msg = error_buf;

    // return status
    return int(http_code);
}

}
