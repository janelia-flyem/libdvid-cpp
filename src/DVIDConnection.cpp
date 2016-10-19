#include <cstring>

#include "DVIDConnection.h"
#include "DVIDException.h"
#include <sstream>
#include <json/json.h>

extern "C" {
#include <curl/curl.h>
#include <zmq.h>
}

using std::string;
using std::stringstream;

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

const int DVIDConnection::DEFAULT_TIMEOUT;

//! Defines DVID prefix -- this might have a version ID eventually 
const char* DVIDConnection::DVID_PREFIX = "/api";

DVIDConnection::DVIDConnection(string addr_, string user, string app,
        string resource_server_, int resource_port_) : 
    addr(addr_), username(user), appname(app), resource_server(resource_server_),
    resource_port(resource_port_)
{
    // trim trailing slash, e.g. http://localhost:8000/
    while (*addr.rbegin() == '/')
    {
        addr = addr.substr(0, addr.size()-1);
    }

    curl_connection = curl_easy_init();
    
    if (resource_server != "") {
        setup_0mq();
    }
}

void DVIDConnection::setup_0mq()
{
    zmq_context = zmq_ctx_new();
    zmq_commsocket = zmq_socket(zmq_context, ZMQ_REQ);
    stringstream addr0mq;
    addr0mq << "tcp://" << resource_server << ":" << resource_port;
    zmq_connect(zmq_commsocket, addr0mq.str().c_str()); 
}

DVIDConnection::DVIDConnection(const DVIDConnection& copy_connection)
{
    addr = copy_connection.addr;
    username = copy_connection.username;
    appname = copy_connection.appname;
    directaddress = copy_connection.directaddress;
    resource_server = copy_connection.resource_server;
    resource_port = copy_connection.resource_port;
    curl_connection = curl_easy_init();
    if (resource_server != "") {
        setup_0mq();
    }
}

DVIDConnection::~DVIDConnection()
{
    curl_easy_cleanup(curl_connection);

    if (resource_server != "") {
        zmq_close(zmq_commsocket);
        zmq_ctx_destroy(zmq_context);
    }
}

int DVIDConnection::make_head_request(string endpoint) {
	std::string error_msg;
	return make_request(endpoint, HEAD, BinaryDataPtr(), BinaryData::create_binary_data(), error_msg);
}

int DVIDConnection::make_request(string endpoint, ConnectionMethod method,
        BinaryDataPtr payload, BinaryDataPtr results, string& error_msg,
        ConnectionType type, int timeout, unsigned long long datasize)
{
    CURLcode result;
    char buffer[100];

    // pass the custom headers
    struct curl_slist *headers=0;
    if (type == JSON) {
        headers = curl_slist_append(headers, "Content-Type: application/json");
    } else if (type == BINARY) {
        headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
    } 
    curl_easy_setopt(curl_connection, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl_connection, CURLOPT_NOSIGNAL, 1);

    // handle any special characters
    char * user_enc = curl_easy_escape(curl_connection, username.c_str(), username.length());
    char * app_enc = curl_easy_escape(curl_connection, appname.c_str(), appname.length());

    // append user and app info for dvid requests
    if (endpoint.find('?') == string::npos) {
        // create query string
        endpoint += "?u=" + string(user_enc) + "&app=" + string(app_enc);
    } else {
        // add to query string
        endpoint += "&u=" + string(user_enc) + "&app=" + string(app_enc);
    }

    // load url
    string url = get_uri_root() + endpoint;
    curl_easy_setopt(curl_connection, CURLOPT_URL, url.c_str());
        
    // set the method
    if (method == HEAD) {
        curl_easy_setopt(curl_connection, CURLOPT_CUSTOMREQUEST, "HEAD");
    }
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

    // HEAD request has no body.
    curl_easy_setopt(curl_connection, CURLOPT_NOBODY, long(method == HEAD));

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
    memset(error_buf, 0, CURL_ERROR_SIZE);
    curl_easy_setopt(curl_connection, CURLOPT_ERRORBUFFER, error_buf);

    // request resource if resource server is available 
    int client_id; 
    if (resource_server != "") {
        Json::Value request;
        request["type"] = "request"; 
        if (directaddress == "") {
            request["resource"] = addr; 
        } else {
            request["resource"] = directaddress; 
        }
        request["numopts"] = 1;
        if ((method == POST) || (method == PUT)) {
            unsigned long long payload_size = 0;
            request["read"] = false;
            if (payload) {
                payload_size = (unsigned long long)(payload->length());
            }
            request["datasize"] = payload_size; 
        } else {
            request["read"] = true;
            request["datasize"] = datasize; // need to pass if read unless really tiny 
        }
        stringstream ss;
        ss << request;
        string request_str = ss.str();
   
        // send request
        zmq_send(zmq_commsocket, request_str.c_str(), request_str.length(), 0);
    
        // retrieve request
        zmq_recv(zmq_commsocket, buffer, 100, 0);

        // parse response
        Json::Value response;
        stringstream stream_buffer;
        stream_buffer << string(buffer);
        stream_buffer >> response;
        client_id = response["id"].asInt();
        bool resp_val = response["available"].asBool();
        if (!resp_val) {
            // if not availabe, create SUB, filter on port, wait for message
            void* zmq_sub = zmq_socket(zmq_context, ZMQ_SUB);
            stringstream addr0mq;
            addr0mq << "tcp://" << resource_server << ":" << resource_port + 1;
            stringstream cidstream;
            cidstream << client_id;
            zmq_setsockopt(zmq_sub, ZMQ_SUBSCRIBE, cidstream.str().c_str(), cidstream.str().length());
            zmq_connect(zmq_sub, addr0mq.str().c_str());

            // listen for events
            zmq_recv(zmq_sub, buffer, 100, 0);
            zmq_close(zmq_sub);

            // when message is received, send reservation requst
            stringstream holdstream;
            holdstream << "{\"type\": \"hold\", \"id\": " << client_id << '}';
            string holdstr = holdstream.str();
            zmq_send(zmq_commsocket, holdstr.c_str(), holdstr.length(), 0);
            zmq_recv(zmq_commsocket, buffer, 100, 0);
        }
    } 
    
    
    
    // actually perform the request
    result = curl_easy_perform(curl_connection);

    // release resource if resource server is available 
    if (resource_server != "") {
        stringstream cidstream;
        cidstream << "{\"type\": \"release\", \"id\": " << client_id << '}';
        string cidstr = cidstream.str();
        zmq_send(zmq_commsocket, cidstr.c_str(), cidstr.length(), 0);
        zmq_recv(zmq_commsocket, buffer, 100, 0);
    } 
    
    // get the error code
    long http_code = 0;
    curl_easy_getinfo (curl_connection, CURLINFO_RESPONSE_CODE, &http_code);
   
    if (directaddress == "") {
        // get IP address to bypass dns on first invocation
        char *ipaddr = 0;
        curl_easy_getinfo (curl_connection, CURLINFO_PRIMARY_IP, &ipaddr);
        stringstream tempstr;
        tempstr << ipaddr << ":";        
        long port = 0;
        curl_easy_getinfo (curl_connection, CURLINFO_PRIMARY_PORT, &port);
        tempstr << port;

        directaddress = tempstr.str();
    }

    // throw exception if connection doesn't work
    if (result != CURLE_OK) {
        curl_free(user_enc);
        curl_free(app_enc);
        throw DVIDException("DVIDConnection error: " + string(url), http_code);
    }

    // load error if there is one
    error_msg = error_buf;

    // free allocated string 
    curl_free(user_enc);
    curl_free(app_enc);

    // return status
    return int(http_code);
}

}
